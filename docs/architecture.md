# General Architecture

The general architecture is defined by [roadmap_v3.md](roadmap_v3.md). The key correction in that plan is that TinyTensor-MLIR should not become two disconnected projects: a C++ tensor framework on one side and standalone MLIR pass experiments on the other.

The central path is vertical:

```text
C++ framework expression
    -> Frontend Graph
    -> Graph-to-MLIR importer
    -> mlir::ModuleOp
    -> compiler pipelines invoked from C++
    -> executable object
    -> C++ runtime invocation
```

This means each major feature should connect a framework concept to an MLIR representation or transformation and then to runtime behavior.

For the full design, phase sequence, callable ABI discussion, and repository organization, read [roadmap_v3.md](roadmap_v3.md).
