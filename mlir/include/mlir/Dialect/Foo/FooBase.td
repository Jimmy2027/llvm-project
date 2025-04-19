
//===- FooBase.td - Foo dialect base definitions ----------------*- tablegen -*-===//
//
// This file defines the base of the Foo dialect.
//
//===----------------------------------------------------------------------===//

#ifndef FOO_BASE
#define FOO_BASE

// Include the base definitions for MLIR dialects.
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Foo Dialect Definition
//===----------------------------------------------------------------------===//

def Foo_Dialect : Dialect {
  let name = "foo";
  let cppNamespace = "::mlir::foo";
  let summary = "Foo dialect";
  let description = [{
    This dialect serves as an example for defining custom MLIR dialects.
  }];
}

//===----------------------------------------------------------------------===//
// Base class for Foo operations
//===----------------------------------------------------------------------===//

class Foo_Op<string mnemonic, list<Trait> traits = []> :
    Op<Foo_Dialect, mnemonic, traits>;

#endif // FOO_BASE

