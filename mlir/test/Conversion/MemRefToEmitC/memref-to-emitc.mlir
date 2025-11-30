// RUN: mlir-opt -convert-memref-to-emitc %s -split-input-file | FileCheck %s
// RUN: mlir-opt -convert-to-emitc="filter-dialects=memref" %s -split-input-file | FileCheck %s

// CHECK-LABEL: alloca()
func.func @alloca() {
  // CHECK-NEXT: %[[ALLOCA:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<2xf32>
  %0 = memref.alloca() : memref<2xf32>
  return
}

// -----

// CHECK-LABEL: alloca_rank0
func.func @alloca_rank0() {
  // CHECK-NEXT: %[[VAR:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
  // CHECK-NEXT: %[[PTR:.*]] = emitc.apply "&"(%[[VAR]]) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
  %0 = memref.alloca() : memref<i32>
  return
}

// -----

// CHECK-LABEL: memref_store
// CHECK-SAME:  %[[buff:.*]]: memref<4x8xf32>, %[[v:.*]]: f32, %[[i:.*]]: index, %[[j:.*]]: index
func.func @memref_store(%buff : memref<4x8xf32>, %v : f32, %i: index, %j: index) {
  // CHECK-NEXT: %[[BUFFER:.*]] = builtin.unrealized_conversion_cast %[[buff]] : memref<4x8xf32> to !emitc.array<4x8xf32>
  
  // CHECK-NEXT: %[[SUBSCRIPT:.*]] = emitc.subscript %[[BUFFER]][%[[i]], %[[j]]] : (!emitc.array<4x8xf32>, index, index) -> !emitc.lvalue<f32>
  // CHECK-NEXT: emitc.assign %[[v]] : f32 to %[[SUBSCRIPT]] : <f32>
  memref.store %v, %buff[%i, %j] : memref<4x8xf32>
  return
}

// -----

// CHECK-LABEL: memref_load
// CHECK-SAME:  %[[buff:.*]]: memref<4x8xf32>, %[[i:.*]]: index, %[[j:.*]]: index
func.func @memref_load(%buff : memref<4x8xf32>, %i: index, %j: index) -> f32 {
  // CHECK-NEXT: %[[BUFFER:.*]] = builtin.unrealized_conversion_cast %[[buff]] : memref<4x8xf32> to !emitc.array<4x8xf32>
  
  // CHECK-NEXT: %[[SUBSCRIPT:.*]] = emitc.subscript %[[BUFFER]][%[[i]], %[[j]]] : (!emitc.array<4x8xf32>, index, index) -> !emitc.lvalue<f32>
  // CHECK-NEXT: %[[LOAD:.*]] = emitc.load %[[SUBSCRIPT]] : <f32>
  %1 = memref.load %buff[%i, %j] : memref<4x8xf32>
  // CHECK-NEXT: return %[[LOAD]] : f32
  return %1 : f32
}

// -----

// CHECK-LABEL: globals
module @globals {
  memref.global "private" constant @internal_global : memref<3x7xf32> = dense<4.0>
  // CHECK-NEXT: emitc.global static const @internal_global : !emitc.array<3x7xf32> = dense<4.000000e+00>
  memref.global "private" constant @__constant_xi32 : memref<i32> = dense<-1>
  // CHECK-NEXT: emitc.global static const @__constant_xi32 : i32 = -1
  memref.global @public_global : memref<3x7xf32>
  // CHECK-NEXT: emitc.global extern @public_global : !emitc.array<3x7xf32>
  memref.global @uninitialized_global : memref<3x7xf32> = uninitialized
  // CHECK-NEXT: emitc.global extern @uninitialized_global : !emitc.array<3x7xf32>

  // CHECK-LABEL: use_global
  func.func @use_global() {
    // CHECK-NEXT: emitc.get_global @public_global : !emitc.array<3x7xf32>
    %0 = memref.get_global @public_global : memref<3x7xf32>
    // CHECK-NEXT: emitc.get_global @__constant_xi32 : !emitc.lvalue<i32>
    // CHECK-NEXT: emitc.address_of %1 : !emitc.lvalue<i32>
    %1 = memref.get_global @__constant_xi32 : memref<i32>
    return
  }
}

// -----

// CHECK-LABEL: rank0_load
module @rank0_load {
  memref.global "private" constant @scalar_constant : memref<f32> = dense<42.0>
  // CHECK: emitc.global static const @scalar_constant : f32 = 4.200000e+01

  // CHECK-LABEL: load_from_rank0
  func.func @load_from_rank0() -> f32 {
    // CHECK-NEXT: %[[GLOBAL:.*]] = emitc.get_global @scalar_constant : !emitc.lvalue<f32>
    // CHECK-NEXT: %[[PTR:.*]] = emitc.apply "&"(%[[GLOBAL]]) : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
    %0 = memref.get_global @scalar_constant : memref<f32>
    // CHECK-NEXT: %[[ZERO:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
    // CHECK-NEXT: %[[LVALUE:.*]] = emitc.subscript %[[PTR]][%[[ZERO]]] : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
    // CHECK-NEXT: %[[VALUE:.*]] = emitc.load %[[LVALUE]] : <f32>
    %1 = memref.load %0[] : memref<f32>
    // CHECK-NEXT: return %[[VALUE]] : f32
    return %1 : f32
  }
}

// -----

// CHECK-LABEL: rank0_store
module @rank0_store {
  memref.global "private" @scalar_global : memref<i32>
  // CHECK: emitc.global static @scalar_global : i32

  // CHECK-LABEL: store_to_rank0
  func.func @store_to_rank0(%val : i32) {
    // CHECK-NEXT: %[[GLOBAL:.*]] = emitc.get_global @scalar_global : !emitc.lvalue<i32>
    // CHECK-NEXT: %[[PTR:.*]] = emitc.apply "&"(%[[GLOBAL]]) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    %0 = memref.get_global @scalar_global : memref<i32>
    // CHECK-NEXT: %[[ZERO:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
    // CHECK-NEXT: %[[LVALUE:.*]] = emitc.subscript %[[PTR]][%[[ZERO]]] : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
    // CHECK-NEXT: emitc.assign %{{.*}} : i32 to %[[LVALUE]] : <i32>
    memref.store %val, %0[] : memref<i32>
    return
  }
}

// -----

// CHECK-LABEL: extract_aligned_pointer_as_index
// CHECK-SAME:  %[[ARG:.*]]: memref<4xf32>
func.func @extract_aligned_pointer_as_index(%arg : memref<4xf32>) -> index {
  // CHECK-NEXT: %[[ARRAY:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : memref<4xf32> to !emitc.array<4xf32>
  // CHECK-NEXT: %[[ZERO:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> index
  // CHECK-NEXT: %[[SUB:.*]] = emitc.subscript %[[ARRAY]][%[[ZERO]]] : (!emitc.array<4xf32>, index) -> !emitc.lvalue<f32>
  // CHECK-NEXT: %[[PTR:.*]] = emitc.apply "&"(%[[SUB]]) : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
  // CHECK-NEXT: %[[CAST:.*]] = emitc.cast %[[PTR]] : !emitc.ptr<f32> to index
  %0 = memref.extract_aligned_pointer_as_index %arg : memref<4xf32> -> index
  // CHECK-NEXT: return %[[CAST]] : index
  return %0 : index
}

// -----

// CHECK-LABEL: extract_aligned_pointer_as_index_rank0
func.func @extract_aligned_pointer_as_index_rank0() -> index {
  // CHECK-NEXT: %[[VAR:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
  // CHECK-NEXT: %[[PTR:.*]] = emitc.apply "&"(%[[VAR]]) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
  %0 = memref.alloca() : memref<i32>
  // CHECK-NEXT: %[[CAST:.*]] = emitc.cast %[[PTR]] : !emitc.ptr<i32> to index
  %1 = memref.extract_aligned_pointer_as_index %0 : memref<i32> -> index
  // CHECK-NEXT: return %[[CAST]] : index
  return %1 : index
}
