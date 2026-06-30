# Phase 1 Micro Plan

This is the implementation breakdown for [plan.md](plan.md). It treats Phase 1 as the first bridge slice from C++ expression capture to a verified TinyTensor `mlir::ModuleOp`.

This is Phase 1A relative to [../roadmap_v3.md](../roadmap_v3.md): it establishes the framework-to-MLIR boundary. Runtime tensors, lowering, JIT, and numeric execution remain outside this micro plan.

## Implementation Log

| Micro phase | Status | Date | Notes |
| --- | --- | --- | --- |
| 1A.0 Build Skeleton And Targets | Done | 2026-06-18 | Added skeleton CMake targets and `tt-emit` placeholder. |
| 1A.1 Frontend Types And Stable IDs | Done | 2026-06-18 | Added MLIR-free frontend type, spec, and ID headers with unit tests. |
| 1A.2 Minimal Graph Inputs And Outputs | Done | 2026-06-18 | Added symbolic `Tensor`, input/output `Graph`, graph values, and verifier tests. |
| 1A.3 Binary Operation Graph Nodes | Done | 2026-06-29 | Added add/mul graph nodes, free op helpers, type checks, producer links, and topological verifier checks. |
| 1A.4 Dense Constants And `full` | Done | 2026-06-29 | Added dense float constant payloads, `Graph::constant`, `full`, constant verifier checks, and constant tests. |
| 1A.5 TinyTensor ODS Skeleton | Not started |  |  |
| 1A.6 Compiler Context And Type Conversion | Not started |  |  |
| 1A.7 Import Identity Graph | Not started |  |  |
| 1A.8 Import Constants And Binary Operations | Not started |  |  |
| 1A.9 Framework-Facing Compiler API | Not started |  |  |
| 1A.10 `tt-emit` And Integration Examples | Not started |  |  |
| 1A.11 Documentation And Exit Checklist | Not started |  |  |

## Phase 1A.0: Build Skeleton And Targets

Goal:

```text
Create empty build targets and directory ownership before adding behavior.
```

Implement:

- `tinytensor_frontend` library target.
- `tinytensor_compiler` library target, gated behind LLVM/MLIR availability.
- `tinytensor_ttdialect` library target, gated behind LLVM/MLIR availability.
- `tt-emit` tool target, gated behind LLVM/MLIR availability.
- Test placeholders for `Frontend`, `Dialect`, and `Integration`.

Files:

```text
include/tinytensor/Frontend/
lib/Frontend/
include/tinytensor/Compiler/
lib/Compiler/
include/tinytensor/Dialect/TinyTensor/
lib/Dialect/TinyTensor/
tools/tt-emit/
tests/Frontend/
tests/Dialect/
tests/Integration/
```

Acceptance:

```bash
cmake -S . -B build -DTINYTENSOR_BUILD_TT_OPT=OFF
cmake --build build
ctest --test-dir build
```

No frontend behavior is required yet. The value is a stable place to add it.

## Phase 1A.1: Frontend Types And Stable IDs

Goal:

```text
Represent symbolic tensor value identity without MLIR dependencies.
```

Implement:

- `DType` with only `F32`.
- `TensorType` with static shape and dtype.
- `TensorSpec` with shape, dtype, and optional name.
- `ValueId` and `NodeId`.
- Basic validation for positive dimensions.

Tests:

- Construct `TensorType({4}, F32)`.
- Reject zero or negative dimensions.
- Rank-0 scalar type is allowed.
- `TensorSpec` preserves name and type metadata.

Acceptance:

```text
Frontend type tests pass without linking MLIR.
```

## Phase 1A.2: Minimal Graph Inputs And Outputs

Goal:

```text
Build an identity graph with no operation nodes.
```

Implement:

- `Tensor` as `Graph* + ValueId`.
- `GraphValue` for input and produced values.
- `Graph::input(const TensorSpec&)`.
- `Graph::setOutputs(ArrayRef<Tensor>)`.
- `Graph::value(ValueId)`, `inputs()`, `outputs()`, and `nodes()` accessors.
- `Graph::verify()` for basic value existence and ownership checks.

Tests:

```cpp
Graph graph("identity");
Tensor x = graph.input({.shape = {4}, .dtype = DType::F32, .name = "x"});
graph.setOutputs({x});
EXPECT_TRUE(graph.verify());
```

Completed tests in `tests/Frontend/test_graph.cpp`:

- `BuildsIdentityGraph`: creates one input, sets it as output, checks zero nodes, IDs, and verifier success.
- `InputCreatesGraphValueMetadata`: checks value ID, tensor type, missing producer, and input name metadata.
- `VerifyRejectsGraphWithoutOutputs`: verifies missing outputs fail with a useful diagnostic string.
- `SetOutputsRejectsInvalidTensor`: rejects default/invalid tensor handles.
- `SetOutputsRejectsTensorFromDifferentGraph`: rejects tensors owned by another graph.
- `ValueRejectsOutOfRangeId`: rejects invalid value lookup.
- `RejectsEmptyGraphName`: rejects unnamed graphs.

This file is part of the `tinytensor-tests` executable through `tests/CMakeLists.txt`.

Acceptance:

```text
Identity graph has one input, one output, zero nodes, and verifies.
```

## Phase 1A.3: Binary Operation Graph Nodes

Goal:

```text
Represent add and mul as frontend graph nodes with type-checked operands.
```

Implement:

- `OpKind::Add` and `OpKind::Mul`.
- `Node` with operands, results, and payload.
- `Graph::createBinary(OpKind, Tensor, Tensor)`.
- Free functions `add(lhs, rhs)` and `mul(lhs, rhs)`.
- Frontend error type or diagnostic return path.

Tests:

- `add(x, y)` creates one node and one produced value.
- `add(mul(x, y), y)` preserves insertion/topological order.
- Mismatched shapes are rejected.
- Cross-graph operands are rejected.
- Result value records the correct producer node.

Completed tests in `tests/Frontend/test_ops.cpp`:

- `AddCreatesBinaryNodeAndProducedValue`: checks `OpKind::Add`, operands, result, propagated type, and producer ID.
- `MulCreatesBinaryNodeAndProducedValue`: checks `OpKind::Mul`, result type, and graph verification.
- `NestedExpressionPreservesTopologicalOrder`: checks `mul` is produced before dependent `add`.
- `RejectsMismatchedOperandTypes`: rejects different tensor shapes and leaves the graph unchanged.
- `RejectsCrossGraphOperands`: rejects operands from different graph owners.
- `RejectsInvalidTensorOperand`: rejects default/invalid tensor handles.

Acceptance:

```text
Frontend graph can express add/mul dataflow and verify producer/result invariants.
```

## Phase 1A.4: Dense Constants And `full`

Goal:

```text
Represent tensor constants without introducing runtime storage.
```

Implement:

- `OpKind::Constant`.
- `DenseFloatConstant` payload.
- `Graph::constant(TensorType, ArrayRef<float>)`.
- `full(Graph&, ArrayRef<int64_t> shape, float value)`.
- Constant element-count validation.

Tests:

- `full(graph, {4}, 2.0f)` stores four values.
- Constant result type matches requested shape and dtype.
- Wrong element count is rejected.
- Constant can feed `mul(x, two)`.

Completed tests in `tests/Frontend/test_constants.cpp`:

- `FullCreatesDenseConstantNode`: checks `OpKind::Constant`, no operands, one result, dense payload values, result type, and producer ID.
- `ExplicitConstantPreservesProvidedValues`: checks `Graph::constant` stores caller-provided dense values.
- `RejectsWrongConstantElementCount`: rejects constants whose value count does not match tensor shape and leaves the graph unchanged.
- `FullSupportsScalarTensor`: verifies rank-0 tensors store one scalar value.
- `ConstantCanFeedBinaryOperation`: checks a constant result can be consumed by `mul` and graph verification still passes.

Acceptance:

```text
Frontend graph can express add/mul/full expression from plan.md.
```

## Phase 1A.5: TinyTensor ODS Skeleton

Goal:

```text
Introduce the `tt` dialect with primitive tensor operations.
```

Implement:

- `TinyTensorDialect.td`.
- Shared `TT_Op` base.
- `tt.constant` with `ElementsAttr`.
- `tt.add` and `tt.mul` with ranked tensor operands/results.
- Generated headers and CMake TableGen plumbing.
- Register TinyTensor dialect in `tt-opt`.

Tests:

```text
tests/Dialect/add.mlir
tests/Dialect/mul.mlir
tests/Dialect/constant.mlir
tests/Dialect/invalid-add-type.mlir
tests/Dialect/invalid-constant-type.mlir
```

Acceptance:

```bash
build-mlir/tools/tt-opt/tt-opt tests/Dialect/add.mlir --verify-diagnostics
```

Valid IR parses and verifies. Invalid IR emits useful diagnostics.

## Phase 1A.6: Compiler Context And Type Conversion

Goal:

```text
Own MLIR context setup in one compiler-facing object.
```

Implement:

- `CompilerContext` owning `DialectRegistry` and `MLIRContext`.
- Registration for `func` and TinyTensor dialects.
- `GraphToMLIR::convertType(const TensorType&)`.

Tests:

- `DType::F32` converts to `f32`.
- `{4}` converts to `tensor<4xf32>`.
- Scalar shape converts to `tensor<f32>`.

Acceptance:

```text
Type conversion tests pass and do not require graph import yet.
```

## Phase 1A.7: Import Identity Graph

Goal:

```text
Convert graph inputs and outputs to a `func.func` boundary.
```

Implement:

- `GraphToMLIR::import(const Graph&)` initial module creation.
- Function type creation from graph inputs and outputs.
- Entry block argument mapping.
- `func.return` creation.
- Final `mlir::verify(module)`.

Input:

```cpp
Graph graph("identity");
Tensor x = graph.input({.shape = {4}, .dtype = DType::F32, .name = "x"});
graph.setOutputs({x});
```

Expected IR shape:

```mlir
module {
  func.func @identity(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
}
```

Acceptance:

```text
Importer handles function boundaries before operation import exists.
```

## Phase 1A.8: Import Constants And Binary Operations

Goal:

```text
Map frontend `ValueId` values to MLIR SSA values while walking graph nodes.
```

Implement:

- Temporary `DenseMap<ValueId, mlir::Value>` inside importer.
- `DenseMapInfo<ValueId>` only if needed by LLVM DenseMap.
- Constant node import to `tt.constant`.
- Add node import to `tt.add`.
- Mul node import to `tt.mul`.
- Missing operand mapping diagnostics.

Input:

```cpp
Tensor two = full(graph, {4}, 2.0f);
Tensor result = add(mul(x, two), y);
graph.setOutputs({result});
```

Expected IR shape:

```mlir
module {
  func.func @forward(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %cst = tt.constant dense<2.000000e+00> : tensor<4xf32>
    %0 = tt.mul %arg0, %cst : tensor<4xf32>
    %1 = tt.add %0, %arg1 : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
```

Acceptance:

```text
The full Phase 1 expression imports to verified TinyTensor MLIR.
```

## Phase 1A.9: Framework-Facing Compiler API

Goal:

```text
Expose the importer through the public compiler boundary without pretending to compile executable code.
```

Implement:

- `Compiler` with owned `CompilerContext`.
- `Compiler::importToMLIR(const Graph&)`.
- Error propagation from frontend verification and MLIR verification.

Tests:

- `Compiler::importToMLIR(identityGraph)` succeeds.
- `Compiler::importToMLIR(addMulGraph)` succeeds.
- Malformed graph fails before MLIR import.

Acceptance:

```text
User code can call one compiler-facing API and receive an owning ModuleOp.
```

## Phase 1A.10: `tt-emit` And Integration Examples

Goal:

```text
Create a repeatable CLI path for inspecting C++-generated MLIR.
```

Implement:

- `tools/tt-emit/tt-emit.cpp`.
- `--example=identity`.
- `--example=add`.
- `--example=add-mul`.
- Optional `--output=<path>`.
- Optional `--verify-each` or `--verify` flag.
- `examples/emit_expression.cpp` using the public API.

Tests:

- CLI emits identity module.
- CLI emits add/mul module.
- Integration test checks output contains `tt.constant`, `tt.mul`, `tt.add`, and `func.return`.

Acceptance:

```bash
build-mlir/tools/tt-emit/tt-emit --example=add-mul
```

prints verified MLIR generated from C++ graph construction.

## Phase 1A.11: Documentation And Exit Checklist

Goal:

```text
Make the boundary and limitations explicit before moving to execution/lowering work.
```

Update:

- README build section if `tt-emit` adds a new build target.
- `docs/phase_1/plan.md` with a link to this micro plan.
- Add examples to the Phase 1 docs.

Exit criteria:

- Frontend unit tests pass.
- Dialect parser/verifier tests pass.
- Integration import tests pass.
- `tt-emit --example=add-mul` prints verified MLIR.
- No runtime tensor storage, eager execution, JIT, autodiff, Linalg lowering, or bufferization has been added.

## Recommended Implementation Order

```text
1. Build skeleton and CMake targets
2. Frontend types and IDs
3. Graph input/output identity graph
4. Add/mul graph nodes
5. Constants/full
6. TinyTensor dialect ODS
7. CompilerContext and type conversion
8. Import identity graph
9. Import constants/add/mul
10. Compiler::importToMLIR
11. tt-emit and integration tests
12. Documentation cleanup
```

## Dependency Rules

- Frontend library must not include MLIR headers.
- TinyTensor dialect may include MLIR headers but must not depend on frontend graph classes.
- Compiler bridge may depend on both frontend and dialect libraries.
- `Tensor` must not store `mlir::Value`.
- `Graph` owns stable frontend IDs; importer owns temporary MLIR SSA mapping.
- Phase 1A stops at `ModuleOp`; execution belongs to the next vertical slice.
