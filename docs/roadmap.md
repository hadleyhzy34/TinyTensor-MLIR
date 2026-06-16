# Project Roadmap & Execution Plan

This document maps out the timeline, objectives, deliverables, and current progress of the TinyTensor-MLIR compiler.

---

## Roadmap at a Glance

| Phase                                                                                          | Goal                  | Key Objectives                                                                                                           | Status        |
| :--------------------------------------------------------------------------------------------- | :-------------------- | :----------------------------------------------------------------------------------------------------------------------- | :------------ |
| **[Phase 0](file:///home/hadley/Developments/TinyTensor-MLIR/docs/phase0_llvm_foundation.md)** | **LLVM Foundation**   | Custom ADT headers (`StringRef`, `SmallVector`, `DenseMap`), GTest suite, and skeleton `tt-opt`.                         | **Completed** |
| **Phase 1**                                                                                    | **Tensor Runtime**    | Implement core Graph and Tensor Runtime structures in C++ (without MLIR).                                                | _Planned_     |
| **Phase 2**                                                                                    | **MLIR Basics**       | Define `TinyTensorDialect` operations: `tt.constant`, `tt.input`, `tt.add`, `tt.mul`.                                    | _Planned_     |
| **Phase 3**                                                                                    | **Constant Folding**  | Implement pattern rewriting rules using `RewritePattern` and `PatternRewriter`.                                          | _Planned_     |
| **Phase 4**                                                                                    | **Algebraic Simp.**   | Optimize identity operations: $x + 0 \rightarrow x$, $x \times 1 \rightarrow x$, $x \times 0 \rightarrow 0$.             | _Planned_     |
| **Phase 5**                                                                                    | **DCE**               | Implement use-def chain analysis and eliminate dead operations.                                                          | _Planned_     |
| **Phase 6**                                                                                    | **Shape Inference**   | Implement dynamic dimension shape propagation across operations.                                                         | _Planned_     |
| **Phase 7**                                                                                    | **BatchNorm Folding** | Fold batch normalization layers into basic weight scaling operators.                                                     | _Planned_     |
| **Phase 8**                                                                                    | **Fusion**            | Fuse Conv + BatchNorm operations to optimize inference graph processing.                                                 | _Planned_     |
| **Phase 9**                                                                                    | **Lowering Pipeline** | Construct full lowering pipeline: TinyTensor $\rightarrow$ Linalg $\rightarrow$ SCF $\rightarrow$ GPU $\rightarrow$ PTX. | _Planned_     |

---

## Current Focus: Phase 1 — Tensor Runtime

The next phase targets constructing a pure C++ graph evaluation engine. This provides context on tensor data layout, execution graph representations, and inputs/outputs management without invoking MLIR's complexity.

### Next Steps:

- Implement `Shape` class to represent dynamic and static dimensions.
- Implement `Tensor` memory buffer allocator and wrapper.
- Implement `Operation` base class and basic math operators (add, multiply).
- Implement `Graph` execution driver executing forward topological runs.
