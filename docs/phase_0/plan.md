# Phase 0 Concrete Plan

Phase 0 establishes the workflow needed to develop MLIR code comfortably, under the architecture described in `docs/roadmap_v3.md`. It should be short and concrete; do not expand it into a separate framework project.

## Goal

Create a reliable edit-build-run-inspect loop for compiler work:

```text
edit operation or pass
    -> rebuild
    -> run tt-opt or unit test
    -> inspect IR
    -> check expected output
```

## Scope

Implement or keep:

- LLVM-style support exercises: `SmallVector`, `StringRef`, and `DenseMap`.
- Unit tests for the support exercises.
- A minimal `tt-opt` driver that can load MLIR, register standard dialects and passes, and print IR before or after passes.
- CMake structure that cleanly separates support code, tools, tests, and future compiler libraries.

Postpone:

- TinyTensor dialect definitions.
- C++ tensor frontend objects.
- Graph-to-MLIR import.
- Linalg lowering, bufferization, JIT, GPU, runtime tensors, autodiff, and training state.

## Repository Shape At The End Of Phase 0

```text
include/tinytensor/Support/ADT/
  DenseMap.h
  SmallVector.h
  StringRef.h

tests/Support/ADT/
  test_compiler_usecases.cpp
  test_densemap.cpp
  test_smallvector.cpp
  test_stringref.cpp

tools/tt-opt/
  tt-opt.cpp
```

The Phase 1 directories may exist, but they should stay empty until the C++ expression-to-MLIR bridge begins.

## Checks

Run:

```bash
cmake -S . -B build -DTINYTENSOR_BUILD_TT_OPT=OFF
cmake --build build
ctest --test-dir build
```

If LLVM/MLIR is available, also run:

```bash
cmake -S . -B build -DTINYTENSOR_BUILD_TT_OPT=ON
cmake --build build --target tt-opt
build/tools/tt-opt/tt-opt --help
```

## Exit Criteria

Phase 0 is done when:

- Support ADT tests pass.
- `tt-opt` can be built in an LLVM/MLIR environment.
- README and docs point to the MLIR-centered roadmap.
- The repository layout matches the planned `Support`, `Frontend`, `Compiler`, `Dialect`, `tools`, and `tests` split.
