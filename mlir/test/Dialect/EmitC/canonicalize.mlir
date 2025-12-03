// RUN: mlir-opt %s -canonicalize -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: module @deduplicate_standard_includes
module @deduplicate_standard_includes {
  // CHECK: emitc.include <"stdio.h">
  // CHECK-NOT: emitc.include <"stdio.h">
  emitc.include <"stdio.h">
  emitc.include <"stdio.h">
  emitc.include <"stdio.h">

  func.func @foo() {
    return
  }
}

// -----

// CHECK-LABEL: module @deduplicate_non_standard_includes
module @deduplicate_non_standard_includes {
  // CHECK: emitc.include "myheader.h"
  // CHECK-NOT: emitc.include "myheader.h"
  emitc.include "myheader.h"
  emitc.include "myheader.h"

  func.func @bar() {
    return
  }
}

// -----

// CHECK-LABEL: module @standard_and_non_standard_not_deduplicated
module @standard_and_non_standard_not_deduplicated {
  // Both should be kept since one is standard (<>) and one is non-standard ("")
  // CHECK: emitc.include <"header.h">
  // CHECK: emitc.include "header.h"
  emitc.include <"header.h">
  emitc.include "header.h"

  func.func @baz() {
    return
  }
}

// -----

// CHECK-LABEL: module @different_headers_not_deduplicated
module @different_headers_not_deduplicated {
  // All different headers should be kept
  // CHECK: emitc.include <"stdio.h">
  // CHECK: emitc.include <"stdlib.h">
  // CHECK: emitc.include <"string.h">
  emitc.include <"stdio.h">
  emitc.include <"stdlib.h">
  emitc.include <"string.h">

  func.func @qux() {
    return
  }
}

// -----

// CHECK-LABEL: module @preserve_order_of_first_occurrences
module @preserve_order_of_first_occurrences {
  // The first occurrence of each unique include should be preserved in order
  // CHECK: emitc.include <"first.h">
  // CHECK-NEXT: emitc.include <"second.h">
  // CHECK-NEXT: emitc.include <"third.h">
  // CHECK-NOT: emitc.include <"first.h">
  // CHECK-NOT: emitc.include <"second.h">
  emitc.include <"first.h">
  emitc.include <"second.h">
  emitc.include <"first.h">
  emitc.include <"third.h">
  emitc.include <"second.h">

  func.func @quux() {
    return
  }
}

// -----

// CHECK-LABEL: module @mixed_standard_and_non_standard_deduplication
module @mixed_standard_and_non_standard_deduplication {
  // Duplicates within each category should be removed
  // CHECK: emitc.include <"stdio.h">
  // CHECK: emitc.include "local.h"
  // CHECK: emitc.include <"stdlib.h">
  // CHECK-NOT: emitc.include <"stdio.h">
  // CHECK-NOT: emitc.include "local.h"
  emitc.include <"stdio.h">
  emitc.include "local.h"
  emitc.include <"stdio.h">
  emitc.include <"stdlib.h">
  emitc.include "local.h"

  func.func @mixed() {
    return
  }
}

// -----

// CHECK-LABEL: module @single_include_unchanged
module @single_include_unchanged {
  // A single include should remain unchanged
  // CHECK: emitc.include <"unique.h">
  emitc.include <"unique.h">

  func.func @single() {
    return
  }
}

// -----

// CHECK-LABEL: module @no_includes
module @no_includes {
  // Module with no includes should remain unchanged
  func.func @empty() {
    return
  }
}

// -----

// CHECK-LABEL: module @includes_with_functions_interleaved
module @includes_with_functions_interleaved {
  // Includes scattered among functions should still be deduplicated
  // CHECK: emitc.include <"header1.h">
  emitc.include <"header1.h">

  func.func @func1() {
    return
  }

  // CHECK-NOT: emitc.include <"header1.h">
  emitc.include <"header1.h">

  // CHECK: emitc.include <"header2.h">
  emitc.include <"header2.h">

  func.func @func2() {
    return
  }

  // CHECK-NOT: emitc.include <"header2.h">
  emitc.include <"header2.h">
}

// -----

// CHECK-LABEL: module @many_duplicates
module @many_duplicates {
  // Test with many duplicates of the same include
  // CHECK: emitc.include <"common.h">
  // CHECK-NOT: emitc.include <"common.h">
  emitc.include <"common.h">
  emitc.include <"common.h">
  emitc.include <"common.h">
  emitc.include <"common.h">
  emitc.include <"common.h">
  emitc.include <"common.h">
  emitc.include <"common.h">
  emitc.include <"common.h">

  func.func @many() {
    return
  }
}

// -----

// CHECK-LABEL: module @nested_modules_separate_scopes
module @nested_modules_separate_scopes {
  // Parent module includes should be deduplicated within parent scope
  // CHECK: emitc.include <"parent.h">
  // CHECK-NEXT: emitc.include <"shared.h">
  // CHECK-NOT: emitc.include <"parent.h">
  emitc.include <"parent.h">
  emitc.include <"parent.h">
  emitc.include <"shared.h">

  // CHECK: module @nested_child
  module @nested_child {
    // Nested module should maintain its own include scope.
    // Even if "shared.h" appears in parent, the nested module keeps its copy
    // because they represent separate translation units.
    // CHECK: emitc.include <"shared.h">
    // CHECK-NEXT: emitc.include <"child.h">
    // CHECK-NOT: emitc.include <"child.h">
    emitc.include <"shared.h">
    emitc.include <"child.h">
    emitc.include <"child.h">

    func.func @child_func() {
      return
    }
  }

  // CHECK: module @another_nested
  module @another_nested {
    // Another nested module with its own scope
    // CHECK: emitc.include <"other.h">
    emitc.include <"other.h">
  }

  func.func @parent_func() {
    return
  }
}
