# TinyTensor-MLIR

TinyTensor-MLIR is a learning compiler for tensor programs. The project is intentionally small: the C++ layer exists to build and drive MLIR, while the main learning path is dialect design, pass construction, autodiff, lowering, and runtime invocation.

The target vertical path is:

```text
C++ expression API
    -> frontend graph
    -> TinyTensor MLIR
    -> differentiation and graph optimization
    -> Linalg / bufferization / loops
    -> LLVM or GPU lowering
    -> C++ runtime invocation
```

This is not a production deep learning framework. Runtime tensors, storage, modules, and training state should stay thin until they are needed to exercise a compiler boundary.

## Current Focus

The repository is being organized around two immediate milestones:

- Phase 0: build a reliable MLIR development loop and keep the LLVM-style ADT exercises isolated as support code.
- Phase 1: import a small C++ tensor expression graph into a verified `mlir::ModuleOp` using the TinyTensor dialect.

Read these first:

- [Phase 0 plan](docs/phase_0/plan.md)
- [Phase 1 plan](docs/phase_1/plan.md)
- [Roadmap v3 and architecture](docs/roadmap_v3.md)
- [Roadmap index](docs/roadmap.md)

Older planning notes are kept in [docs/archive/roadmaps](docs/archive/roadmaps) for context. `roadmap_v3.md` is not archived; it is the current architecture plan.

## Repository Layout

```text
tinytensor-mlir/
|-- CMakeLists.txt
|-- include/tinytensor/
|   |-- Support/ADT/          # Phase 0 LLVM-style ADT learning code
|   |-- Frontend/             # Planned framework/frontend headers
|   |-- Compiler/             # Planned compiler bridge headers
|   `-- Dialect/TinyTensor/   # Planned TinyTensor dialect headers
|-- lib/
|   |-- Framework/
|   |-- Frontend/
|   |-- Dialect/TinyTensor/
|   |-- Transforms/
|   |-- Conversion/
|   |-- Compiler/
|   `-- Runtime/
|-- tools/
|   |-- tt-opt/               # MLIR optimizer/debug driver
|   `-- tt-run/               # Planned runtime driver
|-- examples/                 # Planned vertical demos
|-- tests/
|   |-- Support/ADT/          # Current Phase 0 tests
|   |-- Dialect/
|   |-- Transforms/
|   |-- Conversion/
|   |-- Frontend/
|   `-- Integration/
`-- docs/
    |-- phase_0/
    |-- phase_1/
    |-- architecture.md
    |-- roadmap.md
    |-- roadmap_v3.md
    `-- archive/roadmaps/
```

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

`tt-opt` and the Phase 1A `tt-emit` skeleton are built only when LLVM and MLIR CMake packages are found. Set `TINYTENSOR_BUILD_TT_OPT=OFF` to build only the support tests.

## Phase 0 Status

Implemented support exercises:

- `SmallVector`
- `StringRef`
- `DenseMap`
- compiler-style mock use cases for value mapping and shape propagation

These live under `Support/ADT` because they are learning scaffolding, not the long-term TinyTensor frontend. The future public framework headers listed in `roadmap_v3.md` should be added when their corresponding vertical slice starts.
