//===- MemRefToEmitC.cpp - MemRef to EmitC conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert memref ops into emitc ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"

#include "mlir/Conversion/ConvertToEmitC/ToEmitCInterface.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>
#include <numeric>

using namespace mlir;

static bool isMemRefTypeLegalForEmitC(MemRefType memRefType) {
  // Rank-0 memrefs are always legal (layout doesn't apply to scalars)
  if (memRefType.getRank() == 0)
    return memRefType.hasStaticShape();

  return memRefType.hasStaticShape() && memRefType.getLayout().isIdentity() &&
         !llvm::is_contained(memRefType.getShape(), 0);
}

namespace {
/// Implement the interface to convert MemRef to EmitC.
struct MemRefToEmitCDialectInterface : public ConvertToEmitCPatternInterface {
  using ConvertToEmitCPatternInterface::ConvertToEmitCPatternInterface;

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToEmitCConversionPatterns(
      ConversionTarget &target, TypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateMemRefToEmitCTypeConversion(typeConverter);
    populateMemRefToEmitCConversionPatterns(patterns, typeConverter);
  }
};
} // namespace

void mlir::registerConvertMemRefToEmitCInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    dialect->addInterfaces<MemRefToEmitCDialectInterface>();
  });
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
struct ConvertAlloca final : public OpConversionPattern<memref::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocaOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    if (!op.getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "cannot transform alloca with dynamic shape");
    }

    if (op.getAlignment().value_or(1) > 1) {
      // TODO: Allow alignment if it is not more than the natural alignment
      // of the C array.
      return rewriter.notifyMatchFailure(
          op.getLoc(), "cannot transform alloca with alignment requirement");
    }

    MemRefType memrefType = op.getType();
    auto resultTy = getTypeConverter()->convertType(memrefType);
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(), "cannot convert type");
    }

    // Handle rank-0 specially: create variable of element type, then take address
    if (memrefType.getRank() == 0) {
      Type elementType =
          getTypeConverter()->convertType(memrefType.getElementType());
      if (!elementType) {
        return rewriter.notifyMatchFailure(op.getLoc(),
                                           "cannot convert element type");
      }
      auto noInit = emitc::OpaqueAttr::get(getContext(), "");
      emitc::LValueType lvalueType = emitc::LValueType::get(elementType);
      emitc::VariableOp var =
          rewriter.create<emitc::VariableOp>(op.getLoc(), lvalueType, noInit);
      rewriter.replaceOpWithNewOp<emitc::ApplyOp>(
          op, resultTy, rewriter.getStringAttr("&"), var);
      return success();
    }

    // Higher-rank case: create array variable
    auto noInit = emitc::OpaqueAttr::get(getContext(), "");
    rewriter.replaceOpWithNewOp<emitc::VariableOp>(op, resultTy, noInit);
    return success();
  }
};

Type convertMemRefType(MemRefType opTy, const TypeConverter *typeConverter) {
  Type resultTy;
  if (opTy.getRank() == 0) {
    resultTy = typeConverter->convertType(mlir::getElementTypeOrSelf(opTy));
  } else {
    resultTy = typeConverter->convertType(opTy);
  }
  return resultTy;
}

/// Strip const qualifier from a type if present.
/// Converts `!emitc.opaque<"const TYPE">` back to the original type.
static Type stripConstQualifier(Type type, OpBuilder &builder) {
  auto opaqueType = dyn_cast<emitc::OpaqueType>(type);
  if (!opaqueType)
    return type;

  StringRef value = opaqueType.getValue();
  if (!value.starts_with("const "))
    return type;

  // Extract the type string after "const "
  std::string unconst = value.substr(6).str();

  // Try to convert back to a non-opaque type if possible
  // Common mappings from C type strings to MLIR types
  if (unconst == "int8_t")
    return builder.getIntegerType(8);
  if (unconst == "int16_t")
    return builder.getIntegerType(16);
  if (unconst == "int32_t")
    return builder.getI32Type();
  if (unconst == "int64_t")
    return builder.getI64Type();
  if (unconst == "float")
    return builder.getF32Type();
  if (unconst == "double")
    return builder.getF64Type();

  // If we can't convert back, return an opaque type without const
  return emitc::OpaqueType::get(builder.getContext(), unconst);
}

static Value calculateMemrefTotalSizeBytes(Location loc, MemRefType memrefType,
                                           OpBuilder &builder) {
  assert(isMemRefTypeLegalForEmitC(memrefType) &&
         "incompatible memref type for EmitC conversion");
  emitc::CallOpaqueOp elementSize = emitc::CallOpaqueOp::create(
      builder, loc, emitc::SizeTType::get(builder.getContext()),
      builder.getStringAttr("sizeof"), ValueRange{},
      ArrayAttr::get(builder.getContext(),
                     {TypeAttr::get(memrefType.getElementType())}));

  IndexType indexType = builder.getIndexType();
  int64_t numElements = llvm::product_of(memrefType.getShape());
  emitc::ConstantOp numElementsValue = emitc::ConstantOp::create(
      builder, loc, indexType, builder.getIndexAttr(numElements));

  Type sizeTType = emitc::SizeTType::get(builder.getContext());
  emitc::MulOp totalSizeBytes = emitc::MulOp::create(
      builder, loc, sizeTType, elementSize.getResult(0), numElementsValue);

  return totalSizeBytes.getResult();
}

static emitc::AddressOfOp
createPointerFromEmitcArray(Location loc, OpBuilder &builder,
                            TypedValue<emitc::ArrayType> arrayValue) {

  emitc::ConstantOp zeroIndex = emitc::ConstantOp::create(
      builder, loc, builder.getIndexType(), builder.getIndexAttr(0));

  emitc::ArrayType arrayType = arrayValue.getType();
  llvm::SmallVector<mlir::Value> indices(arrayType.getRank(), zeroIndex);
  emitc::SubscriptOp subPtr =
      emitc::SubscriptOp::create(builder, loc, arrayValue, ValueRange(indices));
  emitc::AddressOfOp ptr = emitc::AddressOfOp::create(
      builder, loc, emitc::PointerType::get(arrayType.getElementType()),
      subPtr);

  return ptr;
}

struct ConvertAlloc final : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::AllocOp allocOp, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = allocOp.getLoc();
    MemRefType memrefType = allocOp.getType();
    if (!isMemRefTypeLegalForEmitC(memrefType)) {
      return rewriter.notifyMatchFailure(
          loc, "incompatible memref type for EmitC conversion");
    }

    Type sizeTType = emitc::SizeTType::get(rewriter.getContext());
    Type elementType = memrefType.getElementType();
    IndexType indexType = rewriter.getIndexType();
    emitc::CallOpaqueOp sizeofElementOp = emitc::CallOpaqueOp::create(
        rewriter, loc, sizeTType, rewriter.getStringAttr("sizeof"),
        ValueRange{},
        ArrayAttr::get(rewriter.getContext(), {TypeAttr::get(elementType)}));

    int64_t numElements = 1;
    for (int64_t dimSize : memrefType.getShape()) {
      numElements *= dimSize;
    }
    Value numElementsValue = emitc::ConstantOp::create(
        rewriter, loc, indexType, rewriter.getIndexAttr(numElements));

    Value totalSizeBytes =
        emitc::MulOp::create(rewriter, loc, sizeTType,
                             sizeofElementOp.getResult(0), numElementsValue);

    emitc::CallOpaqueOp allocCall;
    StringAttr allocFunctionName;
    Value alignmentValue;
    SmallVector<Value, 2> argsVec;
    if (allocOp.getAlignment()) {
      allocFunctionName = rewriter.getStringAttr(alignedAllocFunctionName);
      alignmentValue = emitc::ConstantOp::create(
          rewriter, loc, sizeTType,
          rewriter.getIntegerAttr(indexType,
                                  allocOp.getAlignment().value_or(0)));
      argsVec.push_back(alignmentValue);
    } else {
      allocFunctionName = rewriter.getStringAttr(mallocFunctionName);
    }

    argsVec.push_back(totalSizeBytes);
    ValueRange args(argsVec);

    allocCall = emitc::CallOpaqueOp::create(
        rewriter, loc,
        emitc::PointerType::get(
            emitc::OpaqueType::get(rewriter.getContext(), "void")),
        allocFunctionName, args);

    emitc::PointerType targetPointerType = emitc::PointerType::get(elementType);
    emitc::CastOp castOp = emitc::CastOp::create(
        rewriter, loc, targetPointerType, allocCall.getResult(0));

    rewriter.replaceOp(allocOp, castOp);
    return success();
  }
};

struct ConvertCopy final : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp copyOp, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = copyOp.getLoc();
    MemRefType srcMemrefType = cast<MemRefType>(copyOp.getSource().getType());
    MemRefType targetMemrefType =
        cast<MemRefType>(copyOp.getTarget().getType());

    if (!isMemRefTypeLegalForEmitC(srcMemrefType))
      return rewriter.notifyMatchFailure(
          loc, "incompatible source memref type for EmitC conversion");

    if (!isMemRefTypeLegalForEmitC(targetMemrefType))
      return rewriter.notifyMatchFailure(
          loc, "incompatible target memref type for EmitC conversion");

    // Handle rank-0 (scalar) copies differently - use assign instead of memcpy
    if (srcMemrefType.getRank() == 0) {
      // For rank-0, operands are pointers, not arrays
      auto srcPtr = cast<TypedValue<emitc::PointerType>>(operands.getSource());
      auto targetPtr =
          cast<TypedValue<emitc::PointerType>>(operands.getTarget());

      // Extract the element type and strip const qualifier if present
      // (source might be const-qualified, but the value type should not be)
      Type elementType =
          cast<emitc::PointerType>(srcPtr.getType()).getPointee();
      elementType = stripConstQualifier(elementType, rewriter);

      // Use subscript[0] to get lvalue from source pointer
      emitc::ConstantOp zeroIndex = emitc::ConstantOp::create(
          rewriter, loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
      emitc::LValueType lvalueType = emitc::LValueType::get(elementType);
      Value srcLValue = emitc::SubscriptOp::create(rewriter, loc, lvalueType,
                                                    srcPtr, ValueRange{zeroIndex})
                            .getResult();
      Value srcValue =
          emitc::LoadOp::create(rewriter, loc, elementType, srcLValue);

      // Use subscript[0] to get lvalue from target pointer
      Value targetLValue = emitc::SubscriptOp::create(rewriter, loc, lvalueType,
                                                      targetPtr,
                                                      ValueRange{zeroIndex})
                               .getResult();

      // Assign value to target
      rewriter.replaceOpWithNewOp<emitc::AssignOp>(copyOp, targetLValue,
                                                    srcValue);
      return success();
    }

    // Higher-rank case - use memcpy
    auto srcArrayValue =
        cast<TypedValue<emitc::ArrayType>>(operands.getSource());
    emitc::AddressOfOp srcPtr =
        createPointerFromEmitcArray(loc, rewriter, srcArrayValue);

    auto targetArrayValue =
        cast<TypedValue<emitc::ArrayType>>(operands.getTarget());
    emitc::AddressOfOp targetPtr =
        createPointerFromEmitcArray(loc, rewriter, targetArrayValue);

    emitc::CallOpaqueOp memCpyCall = emitc::CallOpaqueOp::create(
        rewriter, loc, TypeRange{}, "memcpy",
        ValueRange{
            targetPtr.getResult(), srcPtr.getResult(),
            calculateMemrefTotalSizeBytes(loc, srcMemrefType, rewriter)});

    rewriter.replaceOp(copyOp, memCpyCall.getResults());

    return success();
  }
};

struct ConvertGlobal final : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GlobalOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType opTy = op.getType();
    if (!op.getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "cannot transform global with dynamic shape");
    }

    if (op.getAlignment().value_or(1) > 1) {
      // TODO: Extend GlobalOp to specify alignment via the `alignas` specifier.
      return rewriter.notifyMatchFailure(
          op.getLoc(), "global variable with alignment requirement is "
                       "currently not supported");
    }

    Type resultTy = convertMemRefType(opTy, getTypeConverter());

    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cannot convert result type");
    }

    SymbolTable::Visibility visibility = SymbolTable::getSymbolVisibility(op);
    if (visibility != SymbolTable::Visibility::Public &&
        visibility != SymbolTable::Visibility::Private) {
      return rewriter.notifyMatchFailure(
          op.getLoc(),
          "only public and private visibility is currently supported");
    }
    // We are explicit in specifing the linkage because the default linkage
    // for constants is different in C and C++.
    bool staticSpecifier = visibility == SymbolTable::Visibility::Private;
    bool externSpecifier = !staticSpecifier;

    Attribute initialValue = operands.getInitialValueAttr();
    if (opTy.getRank() == 0 && op.getInitialValue()) {
      auto elementsAttr = llvm::cast<ElementsAttr>(*op.getInitialValue());
      initialValue = elementsAttr.getSplatValue<Attribute>();
    }
    if (isa_and_present<UnitAttr>(initialValue))
      initialValue = {};

    rewriter.replaceOpWithNewOp<emitc::GlobalOp>(
        op, operands.getSymName(), resultTy, initialValue, externSpecifier,
        staticSpecifier, operands.getConstant());
    return success();
  }
};

struct ConvertGetGlobal final
    : public OpConversionPattern<memref::GetGlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GetGlobalOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    MemRefType opTy = op.getType();
    Type resultTy = convertMemRefType(opTy, getTypeConverter());

    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cannot convert result type");
    }

    if (opTy.getRank() == 0) {
      emitc::LValueType lvalueType = emitc::LValueType::get(resultTy);
      emitc::GetGlobalOp globalLValue = emitc::GetGlobalOp::create(
          rewriter, op.getLoc(), lvalueType, operands.getNameAttr());

      // Determine the pointer element type
      Type pointerElementType = resultTy;

      // Check if the global is const
      auto globalOp = SymbolTable::lookupNearestSymbolFrom<emitc::GlobalOp>(
          globalLValue, operands.getNameAttr());
      if (globalOp && globalOp.getConstSpecifier()) {
        // Create a const pointer type using opaque type
        std::string cTypeString = emitc::getCTypeString(pointerElementType);
        if (!cTypeString.empty()) {
          pointerElementType = emitc::OpaqueType::get(rewriter.getContext(),
                                                      "const " + cTypeString);
        }
      }

      emitc::PointerType pointerType =
          emitc::PointerType::get(pointerElementType);
      rewriter.replaceOpWithNewOp<emitc::ApplyOp>(
          op, pointerType, rewriter.getStringAttr("&"), globalLValue);
      return success();
    }
    rewriter.replaceOpWithNewOp<emitc::GetGlobalOp>(op, resultTy,
                                                    operands.getNameAttr());
    return success();
  }
};

/// Helper to obtain an lvalue from either pointer or array memref operands.
/// For rank-0 (pointer), uses emitc.subscript with index 0.
/// For higher-rank (array), uses emitc.subscript with indices.
/// Note: elementType should be the unqualified element type, not const-qualified.
static Value getLValueFromMemRef(Location loc, OpBuilder &builder, Value memref,
                                  ValueRange indices, Type elementType) {
  // Check if this is a pointer type (rank-0 memref case)
  if (auto ptrType = dyn_cast<emitc::PointerType>(memref.getType())) {
    // Treat pointer as 1-element array and subscript with 0
    assert(indices.empty() && "rank-0 memref should have no indices");
    emitc::ConstantOp zeroIndex = emitc::ConstantOp::create(
        builder, loc, builder.getIndexType(), builder.getIndexAttr(0));

    // Strip const qualifier from element type if present
    // (elementType should already be unqualified, but be defensive)
    Type unconst = stripConstQualifier(elementType, builder);
    emitc::LValueType lvalueType = emitc::LValueType::get(unconst);
    return emitc::SubscriptOp::create(builder, loc, lvalueType, memref,
                                      ValueRange{zeroIndex})
        .getResult();
  }

  // Array type (higher-rank memref case) - use subscript
  auto arrayValue = cast<TypedValue<emitc::ArrayType>>(memref);
  return emitc::SubscriptOp::create(builder, loc, arrayValue, indices)
      .getResult();
}

struct ConvertLoad final : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(), "cannot convert type");
    }

    Value lvalue = getLValueFromMemRef(op.getLoc(), rewriter,
                                       operands.getMemref(),
                                       operands.getIndices(), resultTy);

    rewriter.replaceOpWithNewOp<emitc::LoadOp>(op, resultTy, lvalue);
    return success();
  }
};

struct ConvertStore final : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the element type from the value being stored
    Type elementType = operands.getValue().getType();

    Value lvalue = getLValueFromMemRef(op.getLoc(), rewriter,
                                       operands.getMemref(),
                                       operands.getIndices(), elementType);

    rewriter.replaceOpWithNewOp<emitc::AssignOp>(op, lvalue,
                                                 operands.getValue());
    return success();
  }
};
} // namespace

void mlir::populateMemRefToEmitCTypeConversion(TypeConverter &typeConverter) {
  typeConverter.addConversion(
      [&](MemRefType memRefType) -> std::optional<Type> {
        if (!isMemRefTypeLegalForEmitC(memRefType)) {
          return {};
        }
        Type convertedElementType =
            typeConverter.convertType(memRefType.getElementType());
        if (!convertedElementType)
          return {};

        // Rank-0 memrefs convert to pointers
        if (memRefType.getRank() == 0) {
          return emitc::PointerType::get(convertedElementType);
        }

        // Higher-rank memrefs convert to arrays
        return emitc::ArrayType::get(memRefType.getShape(),
                                     convertedElementType);
      });

  auto materializeAsUnrealizedCast = [](OpBuilder &builder, Type resultType,
                                        ValueRange inputs,
                                        Location loc) -> Value {
    if (inputs.size() != 1)
      return Value();

    return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
        .getResult(0);
  };

  // Target materialization for const pointer to non-const pointer conversion
  auto materializeConstPointerCast = [](OpBuilder &builder, Type resultType,
                                        ValueRange inputs,
                                        Location loc) -> Value {
    if (inputs.size() != 1)
      return Value();

    Value input = inputs[0];
    auto inputPtrType = dyn_cast<emitc::PointerType>(input.getType());
    auto resultPtrType = dyn_cast<emitc::PointerType>(resultType);

    // Check if this is a const pointer to non-const pointer conversion
    if (inputPtrType && resultPtrType) {
      auto inputPointee = inputPtrType.getPointee();

      // Check if input is const-qualified opaque type
      if (auto inputOpaque = dyn_cast<emitc::OpaqueType>(inputPointee)) {
        StringRef value = inputOpaque.getValue();
        if (value.starts_with("const ")) {
          // Use emitc.cast for const-to-non-const pointer conversion
          // This represents casting away const, which is semantically
          // allowed for read-only operations
          return emitc::CastOp::create(builder, loc, resultType, input)
              .getResult();
        }
      }
    }

    // Fall back to unrealized cast for other cases
    return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
        .getResult(0);
  };

  typeConverter.addSourceMaterialization(materializeAsUnrealizedCast);
  typeConverter.addTargetMaterialization(materializeConstPointerCast);
}

void mlir::populateMemRefToEmitCConversionPatterns(
    RewritePatternSet &patterns, const TypeConverter &converter) {
  patterns.add<ConvertAlloca, ConvertAlloc, ConvertCopy, ConvertGlobal,
               ConvertGetGlobal, ConvertLoad, ConvertStore>(
      converter, patterns.getContext());
}
