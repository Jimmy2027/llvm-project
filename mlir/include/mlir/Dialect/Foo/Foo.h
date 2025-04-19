#ifndef MLIR_DIALECT_FOO_H_
#define MLIR_DIALECT_FOO_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

//===----------------------------------------------------------------------===//
// Foo Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Foo/FooOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Foo Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Foo/FooOps.h.inc"

#endif // MLIR_DIALECT_FOO_H_
