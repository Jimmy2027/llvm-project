#ifndef FOO_OPS
#define FOO_OPS

include "mlir/Dialect/Foo/FooBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def BarOp : Foo_Op<"bar"> {
  let summary = "bar operation";
  // Additional operation definitions
}

#endif // FOO_OPS

