# TinyTensor-MLIR Architecture Overview

TinyTensor-MLIR is a custom deep learning compiler designed to lowering high-level tensor operations into optimized GPU code using the MLIR framework. 

---

## Compiler Flow

```
                      +-------------------+
                      |   Tensor Graph    |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |   TinyTensor IR   |  (TinyTensor Dialect: tt.add, tt.matmul, etc.)
                      +---------+---------+
                                |
        +-----------------------+-----------------------+
        |                       |                       |
        v                       v                       v
 +--------------+       +---------------+       +---------------+
 | ConstantFold |       | AlgebraicSimp |       |      DCE      |
 +------+-------+       +-------+-------+       +-------+-------+
        |                       |                       |
        +-----------------------+-----------------------+
                                |
                                v
                      +-------------------+
                      |   BatchNormFold   |  (Fuses BatchNorm parameters into weights)
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |    TTToLinalg     |  (Lowers TinyTensor Dialect to Linalg)
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |    LinalgToGPU    |  (Converts loops to parallel GPU grids)
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |     CUDA PTX      |  (Generates assembly code for execution)
                      +-------------------+
```

---

## Component Architecture

1. **[LLVM ADT Foundation](file:///home/hadley/Developments/TinyTensor-MLIR/docs/phase0_llvm_foundation.md)**: Simplified implementations of `StringRef`, `SmallVector`, and `DenseMap` which provide zero-allocation string views, stack-allocated dynamic arrays, and cache-friendly open-addressed hash tables respectively.
2. **TinyTensor Dialect**: Declares high-level machine learning operations (e.g. convolution, matrix multiply, element-wise add/mul, batch norm) with strict shape constraints and type safety.
3. **Optimizations passes**:
   * Constant folding: Combines static operations.
   * Algebraic simplification: Rewrites identity operations ($x + 0 \rightarrow x$, $x \times 1 \rightarrow x$).
   * Dead code elimination (DCE): Evicts unused operations.
   * Shape inference: Dynamically propagates size properties across multi-layer graphs.
   * Batch Normalization folding: Fuses batch-norm parameters directly into preceding convolution/dense layer weights.
4. **Lowering Pipeline**: Implements the Dialect Conversion infrastructure mapping TinyTensor Dialect into Linalg, converting loops to SCF (Structured Control Flow), mapping to GPU blocks, and translating to PTX via the NVVM dialect.
