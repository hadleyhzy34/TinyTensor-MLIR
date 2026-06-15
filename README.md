# TinyTensor-MLIRI think this project is good enough that it deserves to be treated almost like a mini open-source compiler project rather than a “tutorial.”

Given your background (strong C++, LLVM ADT learning, compiler interest, eventual goal of DL graph optimization and memory optimization), I would write the project README like this.

⸻

TinyTensor-MLIR

A tiny deep learning compiler built from scratch to learn MLIR through practical implementation.

Project Goal

This project is not intended to build a production deep learning framework.

The purpose is to understand:

- Compiler IR design
- SSA representation
- MLIR dialect development
- Pattern rewriting
- Compiler optimization passes
- Graph transformations
- Lowering pipelines
- CUDA code generation

By implementing a minimal tensor framework and compiler stack.

The final result should be able to:

Tensor Graph
↓
TinyTensor Dialect
↓
Optimization Passes
↓
Linalg
↓
GPU
↓
NVVM
↓
CUDA/PTX

⸻

Learning Objectives

After finishing this project, you should understand:

LLVM

SmallVector
DenseMap
StringRef
ArrayRef
Twine
Casting System

MLIR

Context
Dialect
Operation
Type
Attribute
Region
Block
Value
SSA
Pass
Pattern Rewrite
Lowering

Compiler Concepts

Constant Folding
Canonicalization
Dead Code Elimination
Shape Inference
Graph Rewrite
Operator Fusion
Lowering
Code Generation

⸻

Project Architecture

                 ┌───────────────┐
                 │ Tensor Graph  │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ TinyTensor IR │
                 └───────┬───────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼

ConstantFold AlgebraicSimplify DCE
└───────────────┬───────────────┘
▼
BatchNormFold
▼
TTToLinalg
▼
LinalgToGPU
▼
PTX

⸻

Repository Structure

tinytensor-mlir/
├── CMakeLists.txt
├── docs/
│ ├── architecture.md
│ ├── passes.md
│ ├── lowering.md
│ ├── roadmap.md
│
├── runtime/
│ ├── include/
│ │ ├── Tensor.h
│ │ ├── Shape.h
│ │ └── Graph.h
│ │
│ └── src/
│ ├── Tensor.cpp
│ └── Graph.cpp
│
├── dialect/
│ ├── TinyTensorDialect.td
│ ├── TinyTensorDialect.cpp
│ ├── TinyTensorDialect.h
│ │
│ ├── TinyTensorOps.td
│ ├── TinyTensorOps.cpp
│ ├── TinyTensorOps.h
│
├── passes/
│ ├── ConstantFold.cpp
│ ├── AlgebraicSimplify.cpp
│ ├── DCE.cpp
│ ├── ShapeInference.cpp
│ ├── BatchNormFold.cpp
│
├── lowering/
│ ├── TTToLinalg.cpp
│ ├── LinalgToSCF.cpp
│ ├── SCFToGPU.cpp
│
├── tools/
│ ├── tt-opt/
│ └── tt-translate/
│
├── examples/
│ ├── add.mlir
│ ├── fold.mlir
│ ├── bn.mlir
│
└── tests/
├── constant-fold.mlir
├── simplify.mlir
├── batchnorm.mlir

⸻

Implementation Schedule

Phase 0 — LLVM Foundation

Estimated:

1 week

Read:

llvm/ADT

Implement:

SmallVector
DenseMap
StringRef

Rewrite several Leetcode solutions.

Goal:

Read LLVM code comfortably.

⸻

Phase 1 — Tensor Runtime

Estimated:

1 week

Implement:

Tensor
Shape
Graph
Operation

Example:

Tensor x({32,128});
auto y =
Add(
Mul(x, Constant(2)),
Constant(3)
);

Goal:

Understand graph representation.

Do NOT use MLIR yet.

⸻

Phase 2 — MLIR Basics

Estimated:

1 week

Create:

TinyTensorDialect

Operations:

tt.constant
tt.input
tt.add
tt.mul

Learn:

Operation
Type
Attribute
SSA

Success criterion:

tt-opt example.mlir

parses successfully.

⸻

Phase 3 — Constant Folding

Estimated:

3 days

Input:

%0 = tt.constant 2
%1 = tt.constant 3
%2 = tt.add %0, %1

Output:

%2 = tt.constant 5

Learn:

RewritePattern
PatternRewriter
replaceOp

Most important MLIR learning milestone.

⸻

Phase 4 — Algebraic Simplification

Estimated:

3 days

Implement:

x + 0 → x
x _ 1 → x
x _ 0 → 0

Learn:

replace operation with existing SSA value

This is a different rewrite pattern from constant folding.

⸻

Phase 5 — Dead Code Elimination

Estimated:

2 days

Input:

%0 = tt.constant 3

unused.

Remove it.

Learn:

Use-def chain
Users
Operands

⸻

Phase 6 — Shape Inference

Estimated:

1 week

Example:

[32,128]

- [32,128]

infer:

[32,128]

Learn:

TensorType
Shape propagation

Critical for future DL compiler work.

⸻

Phase 7 — BatchNorm Folding

Estimated:

1 week

Input:

BatchNorm

with constant parameters.

Transform into:

Mul
Add

Learn:

real graph optimization

This is your first genuinely ML-compiler-style pass.

⸻

Phase 8 — Conv + BatchNorm Fusion

Estimated:

1 week

Transform:

Conv
↓
BN

into:

Conv

with updated weights.

Learn:

multi-op matching

This is how production compilers optimize inference graphs.

⸻

Phase 9 — Lowering

Estimated:

2 weeks

Pipeline:

TinyTensor
↓
Linalg
↓
SCF
↓
GPU
↓
NVVM

Learn:

Dialect Conversion
Type Conversion
ConversionTarget

⸻

Important Design Rules

Rule 1

Never add an operation until a pass requires it.

Bad:

Add
Mul
Div
Pow
MatMul
Conv
BatchNorm
ReLU
Softmax
LayerNorm

on day one.

Good:

Add
Mul

only.

⸻

Rule 2

Every phase must produce something runnable.

Bad:

Read 10 MLIR chapters.

Good:

Implement one pass.
Run one test.
Observe one transformation.

⸻

Rule 3

Every pass must have tests.

Example:

tests/
constant-fold.mlir

Input:

%0 = tt.constant 2
%1 = tt.constant 3
%2 = tt.add %0, %1

Expected:

%2 = tt.constant 5

⸻

Rule 4

Prefer understanding compiler mechanics over optimization complexity.

This project is not about making BatchNorm faster.

It is about understanding:

Pattern Matching
IR Transformation
Lowering
Compiler Design

because those skills later transfer directly to:

Memory Planning
Tensor Fusion
Scheduling
Kernel Generation
Custom DL Compilers

which are the areas you ultimately want to explore.

