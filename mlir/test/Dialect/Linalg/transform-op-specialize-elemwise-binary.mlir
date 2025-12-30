// RUN: mlir-opt --transform-interpreter --split-input-file --verify-diagnostics %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @specialize_add(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: specialize_add
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>,  %[[ARG2:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.add ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>

func.func @specialize_sub(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.subf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: specialize_sub
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>,  %[[ARG2:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.sub ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>

func.func @specialize_sub_swapped_operands(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.subf %in_0, %in : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: specialize_sub
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>,  %[[ARG2:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.sub ins(%[[ARG1]], %[[ARG0]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>

func.func @specialize_mul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: specialize_mul
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>,  %[[ARG2:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.mul ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>

func.func @specialize_div(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.divf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: specialize_div
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>,  %[[ARG2:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.div ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// Integer addition specialization
func.func @specialize_addi(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%arg2 : tensor<?x?xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %1 = arith.addi %in, %in_0 : i32
    linalg.yield %1 : i32
  } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
// CHECK-LABEL: specialize_addi
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xi32>, %[[ARG1:.+]]: tensor<?x?xi32>, %[[ARG2:.+]]: tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.add ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xi32>, tensor<?x?xi32>) outs(%[[ARG2]] : tensor<?x?xi32>) -> tensor<?x?xi32>

func.func @specialize_subi(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%arg2 : tensor<?x?xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %1 = arith.subi %in, %in_0 : i32
    linalg.yield %1 : i32
  } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
// CHECK-LABEL: specialize_subi
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xi32>, %[[ARG1:.+]]: tensor<?x?xi32>, %[[ARG2:.+]]: tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.sub ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xi32>, tensor<?x?xi32>) outs(%[[ARG2]] : tensor<?x?xi32>) -> tensor<?x?xi32>

func.func @specialize_muli(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%arg2 : tensor<?x?xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %1 = arith.muli %in, %in_0 : i32
    linalg.yield %1 : i32
  } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
// CHECK-LABEL: specialize_muli
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xi32>, %[[ARG1:.+]]: tensor<?x?xi32>, %[[ARG2:.+]]: tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.mul ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xi32>, tensor<?x?xi32>) outs(%[[ARG2]] : tensor<?x?xi32>) -> tensor<?x?xi32>

func.func @specialize_divsi(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%arg2 : tensor<?x?xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %1 = arith.divsi %in, %in_0 : i32
    linalg.yield %1 : i32
  } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
// CHECK-LABEL: specialize_divsi
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xi32>, %[[ARG1:.+]]: tensor<?x?xi32>, %[[ARG2:.+]]: tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.div ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xi32>, tensor<?x?xi32>) outs(%[[ARG2]] : tensor<?x?xi32>) -> tensor<?x?xi32>

// Test with i64 to ensure different integer widths work
func.func @specialize_addi_i64(%arg0: tensor<?x?xi64>, %arg1: tensor<?x?xi64>, %arg2: tensor<?x?xi64>) -> tensor<?x?xi64> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xi64>, tensor<?x?xi64>) outs(%arg2 : tensor<?x?xi64>) {
  ^bb0(%in: i64, %in_0: i64, %out: i64):
    %1 = arith.addi %in, %in_0 : i64
    linalg.yield %1 : i64
  } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}
// CHECK-LABEL: specialize_addi_i64
// CHECK-NOT: linalg.generic
// CHECK: linalg.add ins({{.*}} : tensor<?x?xi64>, tensor<?x?xi64>) outs({{.*}} : tensor<?x?xi64>)

// Test with i8 (common in quantized models)
func.func @specialize_addi_i8(%arg0: tensor<?x?xi8>, %arg1: tensor<?x?xi8>, %arg2: tensor<?x?xi8>) -> tensor<?x?xi8> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xi8>, tensor<?x?xi8>) outs(%arg2 : tensor<?x?xi8>) {
  ^bb0(%in: i8, %in_0: i8, %out: i8):
    %1 = arith.addi %in, %in_0 : i8
    linalg.yield %1 : i8
  } -> tensor<?x?xi8>
  return %0 : tensor<?x?xi8>
}
// CHECK-LABEL: specialize_addi_i8
// CHECK-NOT: linalg.generic
// CHECK: linalg.add ins({{.*}} : tensor<?x?xi8>, tensor<?x?xi8>) outs({{.*}} : tensor<?x?xi8>)


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.specialize %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
