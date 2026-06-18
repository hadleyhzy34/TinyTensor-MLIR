# TinyTensor-MLIR Roadmap

The current roadmap and general architecture design is [roadmap_v3.md](roadmap_v3.md).

Use that document as the source of truth for the project shape:

```text
C++ framework expression
    -> Frontend Graph
    -> C++ MLIR importer
    -> mlir::ModuleOp
    -> MLIR passes invoked from C++
    -> lowered executable
    -> C++ runtime invocation
```

`tt-opt` remains a debugging and pass-testing tool, but the main product path is the vertical C++ -> MLIR -> C++ execution flow described in `roadmap_v3.md`.

## Immediate Concrete Work

- [Phase 0 plan](phase_0/plan.md): support ADT exercises, build/test layout, and `tt-opt` inspection workflow.
- [Phase 1 plan](phase_1/plan.md): first C++ expression-to-TinyTensor-MLIR bridge.

Older planning variants are archived in [archive/roadmaps](archive/roadmaps).
