# **Phase 1 — C++ Expression → TinyTensor MLIR**

I would make this phase end at a **verified** **`mlir::ModuleOp`** **constructed directly from the C++ expression API**:

```text
C++ Tensor expression
    ↓
frontend Graph
    ↓
Graph-to-MLIR importer
    ↓
verified TinyTensor ModuleOp
    ↓
printed .mlir for inspection and tests
```

Do **not** include Linalg lowering, bufferization, JIT, runtime tensors, eager execution, autodiff, or training state yet. Those belong to later vertical increments. The purpose here is to establish the framework/compiler boundary cleanly.

This narrows the earlier Phase 1 proposal—`Tensor`, `Graph`, `GraphBuilder`, and the TinyTensor primitive operations—into an implementation you can finish and inspect end to end.  

For a smaller implementation checklist, see [micro_plan.md](micro_plan.md).

---

## **1. Phase deliverable**

The public C++ program should look like this:

```cpp
#include "tinytensor/Frontend/Graph.h"
#include "tinytensor/Frontend/Ops.h"
#include "tinytensor/Compiler/Compiler.h"

using namespace tinytensor;

int main() {
    Graph graph("forward");

    Tensor x = graph.input(
        TensorSpec{
            .shape = {4},
            .dtype = DType::F32,
            .name = "x",
        });

    Tensor y = graph.input(
        TensorSpec{
            .shape = {4},
            .dtype = DType::F32,
            .name = "y",
        });

    Tensor two = full(graph, {4}, 2.0f);

    Tensor result = add(mul(x, two), y);

    graph.setOutputs({result});

    Compiler compiler;
    auto module = compiler.importToMLIR(graph);

    module->print(llvm::outs());
}
```

Expected output:

```mlir
module {
  func.func @forward(
      %arg0: tensor<4xf32>,
      %arg1: tensor<4xf32>
  ) -> tensor<4xf32> {
    %cst = tt.constant dense<2.000000e+00>
        : tensor<4xf32>
    %0 = tt.mul %arg0, %cst
        : tensor<4xf32>
    %1 = tt.add %0, %arg1
        : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
```

Inputs should become `func.func` arguments rather than `tt.input` operations. This keeps function boundaries explicit and avoids inventing an operation for something MLIR already represents naturally.  

---

# **2. Strict scope**

## **Implement**

```text
Frontend:
    DType
    TensorType
    TensorSpec
    ValueId
    NodeId
    Tensor
    Node
    Graph
    add
    mul
    full / constant

TinyTensor dialect:
    tt.constant
    tt.add
    tt.mul

Compiler bridge:
    CompilerContext
    GraphToMLIR
    Compiler::importToMLIR()

Tools/tests:
    tt-emit
    frontend unit tests
    dialect verifier tests
    integration tests
```

## **Deliberately postpone**

```text
TensorImpl shared runtime handle
Storage and allocator
runtime data pointers
eager CPU kernels
CompiledFunction
Linalg lowering
bufferization
LLVM lowering
ExecutionEngine
dynamic dimensions
broadcasting
autodiff
Parameter and Module
```

The frontend `Tensor` in this phase is a **symbolic graph handle**, not a runtime tensor. Do not force one class to solve symbolic identity and runtime storage before either requirement is mature.

---

# **3. Repository structure**

```text
tinytensor-mlir/
├── include/tinytensor/
│   ├── Frontend/
│   │   ├── DType.h
│   │   ├── TensorType.h
│   │   ├── Tensor.h
│   │   ├── Node.h
│   │   ├── Graph.h
│   │   └── Ops.h
│   │
│   ├── Compiler/
│   │   ├── Compiler.h
│   │   ├── CompilerContext.h
│   │   └── GraphToMLIR.h
│   │
│   └── Dialect/TinyTensor/
│       ├── TinyTensorDialect.h
│       └── TinyTensorOps.h
│
├── lib/
│   ├── Frontend/
│   │   ├── Tensor.cpp
│   │   ├── Graph.cpp
│   │   └── Ops.cpp
│   │
│   ├── Compiler/
│   │   ├── Compiler.cpp
│   │   ├── CompilerContext.cpp
│   │   └── GraphToMLIR.cpp
│   │
│   └── Dialect/TinyTensor/
│       ├── TinyTensorDialect.cpp
│       ├── TinyTensorOps.cpp
│       ├── TinyTensorDialect.td
│       └── TinyTensorOps.td
│
├── tools/
│   ├── tt-emit/
│   │   └── tt-emit.cpp
│   └── tt-opt/
│       └── tt-opt.cpp
│
├── tests/
│   ├── Frontend/
│   ├── Dialect/
│   └── Integration/
│
└── examples/
    └── emit_expression.cpp
```

The separation between `Frontend`, `Compiler`, and `Dialect` is important: it makes the importer a visible bridge rather than quietly embedding MLIR construction inside every frontend operation. That bridge-oriented organization is also the architectural correction highlighted in the revised project plan.  

---

# **4. Frontend data model**

##

## **4.1**

**`DType`**

Keep it framework-owned:

```cpp
#pragma once

#include <cstdint>

namespace tinytensor {

enum class DType : std::uint8_t {
    F32,
};

} // namespace tinytensor
```

Only `F32` is needed in Phase 1. Supporting five element types now mostly creates verifier branches without teaching you anything useful.

---

##

## **4.2**

**`TensorType`**

```cpp
#pragma once

#include "tinytensor/Frontend/DType.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace tinytensor {

class TensorType {
public:
    TensorType(
        llvm::ArrayRef<std::int64_t> shape,
        DType dtype
    )
        : shape_(shape.begin(), shape.end()),
          dtype_(dtype) {}

    llvm::ArrayRef<std::int64_t> shape() const {
        return shape_;
    }

    DType dtype() const {
        return dtype_;
    }

    bool operator==(const TensorType&) const = default;

private:
    llvm::SmallVector<std::int64_t, 4> shape_;
    DType dtype_;
};

} // namespace tinytensor
```

For Phase 1:

```text
all dimensions must be positive
all tensor ranks are allowed
rank-0 means scalar
no dynamic dimensions
```

Static shapes keep the importer and operation verification crisp.

---

## **4.3 Stable IDs**

```cpp
struct ValueId {
    std::uint32_t value = 0;

    bool operator==(const ValueId&) const = default;
};

struct NodeId {
    std::uint32_t value = 0;

    bool operator==(const NodeId&) const = default;
};
```

These IDs are the permanent frontend identities.

Never store `mlir::Value` in `Tensor`. The importer creates a temporary mapping:

```cpp
llvm::DenseMap<ValueId, mlir::Value> valueMap;
```

That protects the C++ graph from MLIR context, operation, and module lifetimes.

Add `DenseMapInfo<ValueId>` only when you actually need the map:

```cpp
namespace llvm {

template <>
struct DenseMapInfo<tinytensor::ValueId> {
    static inline tinytensor::ValueId getEmptyKey() {
        return {std::numeric_limits<std::uint32_t>::max()};
    }

    static inline tinytensor::ValueId getTombstoneKey() {
        return {std::numeric_limits<std::uint32_t>::max() - 1};
    }

    static unsigned getHashValue(tinytensor::ValueId id) {
        return hash_value(id.value);
    }

    static bool isEqual(
        tinytensor::ValueId lhs,
        tinytensor::ValueId rhs
    ) {
        return lhs == rhs;
    }
};

} // namespace llvm
```

---

##

## **4.4**

**`Tensor`**

```cpp
class Graph;

class Tensor {
public:
    Tensor() = default;

    const TensorType& type() const;
    ValueId valueId() const;
    const Graph& graph() const;

    explicit operator bool() const {
        return graph_ != nullptr;
    }

private:
    friend class Graph;

    Tensor(Graph* graph, ValueId value)
        : graph_(graph), value_(value) {}

    Graph* graph_ = nullptr;
    ValueId value_{};
};
```

Keep it tiny:

```text
Tensor = Graph pointer + ValueId
```

A copied `Tensor` refers to the same symbolic SSA-like frontend value.

Do not add:

```text
storage
data()
device
grad
requiresGrad
mlir::Value
shared_ptr<TensorImpl>
```

Those concepts are valid later, but none is needed to express or import a graph in Phase 1.

---

# **5. Graph representation**

## **5.1 Operation kinds**

```cpp
enum class OpKind : std::uint8_t {
    Constant,
    Add,
    Mul,
};
```

For constants:

```cpp
struct DenseFloatConstant {
    llvm::SmallVector<float, 4> values;
};
```

A generic attribute dictionary would be premature. Use a typed representation:

```cpp
using NodePayload = std::variant<
    std::monostate,
    DenseFloatConstant
>;
```

---

## **5.2 Values**

```cpp
struct GraphValue {
    ValueId id;
    TensorType type;
    std::optional<NodeId> producer;
    std::string name;
};
```

Inputs have no producer. Operation results do.

---

## **5.3 Nodes**

```cpp
struct Node {
    NodeId id;
    OpKind kind;

    llvm::SmallVector<ValueId, 2> operands;
    llvm::SmallVector<ValueId, 1> results;

    NodePayload payload;
};
```

Although every Phase 1 operation has one result, keeping `results` as a vector avoids baking single-result assumptions into the graph too deeply.

---

##

## **5.4**

**`Graph`**

```cpp
class Graph {
public:
    explicit Graph(std::string name);

    Tensor input(const TensorSpec& spec);

    Tensor createUnary(/* later */);

    Tensor createBinary(
        OpKind kind,
        const Tensor& lhs,
        const Tensor& rhs
    );

    Tensor constant(
        TensorType type,
        llvm::ArrayRef<float> values
    );

    void setOutputs(llvm::ArrayRef<Tensor> outputs);

    llvm::StringRef name() const;
    llvm::ArrayRef<ValueId> inputs() const;
    llvm::ArrayRef<ValueId> outputs() const;
    llvm::ArrayRef<Node> nodes() const;

    const GraphValue& value(ValueId id) const;

    llvm::LogicalResult verify() const;

private:
    ValueId addValue(
        TensorType type,
        std::optional<NodeId> producer,
        llvm::StringRef name = {}
    );

    NodeId nextNodeId();
    ValueId nextValueId();

    std::string name_;
    llvm::SmallVector<GraphValue> values_;
    llvm::SmallVector<Node> nodes_;
    llvm::SmallVector<ValueId> inputs_;
    llvm::SmallVector<ValueId> outputs_;
};
```

Use append-only storage. Since `ValueId.value` can index directly into `values_`, lookup is simple:

```cpp
const GraphValue& Graph::value(ValueId id) const {
    assert(id.value < values_.size());
    return values_[id.value];
}
```

No map is needed there.

---

# **6. Expression-building API**

Use free functions for user-facing operations:

```cpp
Tensor add(const Tensor& lhs, const Tensor& rhs);
Tensor mul(const Tensor& lhs, const Tensor& rhs);

Tensor full(
    Graph& graph,
    llvm::ArrayRef<std::int64_t> shape,
    float value
);
```

Implementation:

```cpp
Tensor add(const Tensor& lhs, const Tensor& rhs) {
    if (&lhs.graph() != &rhs.graph()) {
        throw FrontendError(
            "add operands belong to different graphs"
        );
    }

    return const_cast<Graph&>(lhs.graph())
        .createBinary(OpKind::Add, lhs, rhs);
}
```

I would expose a non-const graph accessor internally rather than retain the `const_cast`; the example only shows the intended ownership check.

`createBinary` performs frontend type checking:

```cpp
Tensor Graph::createBinary(
    OpKind kind,
    const Tensor& lhs,
    const Tensor& rhs
) {
    const TensorType& lhsType = value(lhs.valueId()).type;
    const TensorType& rhsType = value(rhs.valueId()).type;

    if (lhsType != rhsType) {
        throw FrontendError(
            "binary operands must have identical types"
        );
    }

    NodeId nodeId = nextNodeId();
    ValueId resultId = addValue(lhsType, nodeId);

    nodes_.push_back(Node{
        .id = nodeId,
        .kind = kind,
        .operands = {lhs.valueId(), rhs.valueId()},
        .results = {resultId},
        .payload = std::monostate{},
    });

    return Tensor(this, resultId);
}
```

For Phase 1, require exact shape and dtype equality. Broadcasting belongs in a phase where you can intentionally design and test its semantics.

Operator sugar can come at the very end:

```cpp
Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    return add(lhs, rhs);
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
    return mul(lhs, rhs);
}
```

First make explicit functions correct. Sugary syntax has a funny habit of making unfinished systems look more finished than they are.

---

# **7. Frontend verification**

`Graph::verify()` should check invariants independently of MLIR:

```text
1. Graph has at least one output.
2. Every input ValueId exists.
3. Every output ValueId exists.
4. Every node operand exists.
5. Every node result exists.
6. Each produced value names the correct producer.
7. Nodes appear after their operand producers.
8. Add/Mul operands and result have identical types.
9. Constant element count matches tensor shape.
10. Every Tensor belongs to this graph.
```

Example:

```cpp
llvm::LogicalResult Graph::verify() const {
    for (const Node& node : nodes_) {
        for (ValueId operand : node.operands) {
            if (operand.value >= values_.size()) {
                return llvm::failure();
            }

            const auto& producer = values_[operand.value].producer;
            if (producer && producer->value >= node.id.value) {
                return llvm::failure();
            }
        }
    }

    return llvm::success();
}
```

Provide useful diagnostic text through an error collector or `llvm::raw_ostream`; a bare failure is fine internally but miserable at the CLI boundary.

---

# **8. TinyTensor dialect**

## **8.1 Dialect definition**

`TinyTensorDialect.td`:

```tablegen
include "mlir/IR/DialectBase.td"

def TinyTensor_Dialect : Dialect {
  let name = "tt";
  let cppNamespace = "::tinytensor::mlir";

  let summary = "TinyTensor high-level tensor dialect";
  let description = [{
    TinyTensor represents primitive, pure tensor computations imported
    from the TinyTensor C++ frontend graph.
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}
```

You do not need custom types in Phase 1. Use builtin `tensor` types.

---

## **8.2 Shared operation base**

```tablegen
include "mlir/IR/OpBase.td"

class TT_Op<string mnemonic, list<Trait> traits = []>
    : Op<TinyTensor_Dialect, mnemonic, traits>;
```

---

##

## **8.3**

**`tt.add`**

```tablegen
def TT_AddOp : TT_Op<"add", [
    Pure,
    SameOperandsAndResultType
]> {
  let summary = "Elementwise tensor addition";

  let arguments = (ins AnyRankedTensor:$lhs,
                        AnyRankedTensor:$rhs);

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}
```

Verifier:

```cpp
LogicalResult AddOp::verify() {
    auto lhsType =
        llvm::dyn_cast<RankedTensorType>(getLhs().getType());
    auto rhsType =
        llvm::dyn_cast<RankedTensorType>(getRhs().getType());
    auto resultType =
        llvm::dyn_cast<RankedTensorType>(getResult().getType());

    if (!lhsType || !rhsType || !resultType) {
        return emitOpError("requires ranked tensor types");
    }

    if (lhsType != rhsType || lhsType != resultType) {
        return emitOpError(
            "requires identical operand and result types"
        );
    }

    return success();
}
```

`SameOperandsAndResultType` already enforces much of this, but writing the verifier once is useful hands-on MLIR practice and gives you control over diagnostics.

---

##

## **8.4**

**`tt.mul`**

Identical type rules:

```tablegen
def TT_MulOp : TT_Op<"mul", [
    Pure,
    SameOperandsAndResultType
]> {
  let summary = "Elementwise tensor multiplication";

  let arguments = (ins AnyRankedTensor:$lhs,
                        AnyRankedTensor:$rhs);

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}
```

---

##

## **8.5**

**`tt.constant`**

```tablegen
def TT_ConstantOp : TT_Op<"constant", [
    Pure,
    ConstantLike
]> {
  let summary = "Dense tensor constant";

  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyRankedTensor:$result);

  let builders = [
    OpBuilder<(ins "ElementsAttr":$value)>
  ];

  let assemblyFormat = [{
    $value attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}
```

Builder:

```cpp
void ConstantOp::build(
    OpBuilder& builder,
    OperationState& state,
    ElementsAttr value
) {
    state.addAttribute("value", value);
    state.addTypes(value.getType());
}
```

Verifier:

```cpp
LogicalResult ConstantOp::verify() {
    if (getValue().getType() != getResult().getType()) {
        return emitOpError(
            "value attribute type must match result type"
        );
    }

    return success();
}
```

Do not define `tt.return`; use `func.return`. The earlier plan listed a TinyTensor return operation, but using the Func dialect’s terminator is cleaner for a normal `func.func` body.

---

# **9. Compiler context ownership**

Create a class that owns MLIR’s long-lived registration and context state:

```cpp
class CompilerContext {
public:
    CompilerContext();

    mlir::MLIRContext& mlirContext() {
        return context_;
    }

private:
    mlir::DialectRegistry registry_;
    mlir::MLIRContext context_;
};
```

Implementation:

```cpp
CompilerContext::CompilerContext() {
    registry_.insert<
        mlir::func::FuncDialect,
        tinytensor::mlir::TinyTensorDialect
    >();

    context_.appendDialectRegistry(registry_);
    context_.loadAllAvailableDialects();
}
```

In a larger compiler you may avoid loading everything, but explicit simplicity wins here.

---

# **10. Graph-to-MLIR importer**

## **Interface**

```cpp
class GraphToMLIR {
public:
    explicit GraphToMLIR(mlir::MLIRContext& context);

    mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
    import(const Graph& graph);

private:
    mlir::Type convertType(const TensorType& type);

    mlir::FailureOr<mlir::Value>
    importNode(
        const Graph& graph,
        const Node& node,
        mlir::OpBuilder& builder,
        llvm::DenseMap<ValueId, mlir::Value>& valueMap
    );

    mlir::MLIRContext& context_;
};
```

## **Import algorithm**

```text
1. Verify frontend Graph.
2. Create ModuleOp.
3. Convert graph input types.
4. Convert graph output types.
5. Create func.func named after Graph.
6. Add one entry block.
7. Map Graph input ValueIds to block arguments.
8. Walk nodes in insertion/topological order.
9. Create one TinyTensor operation per node.
10. Map each result ValueId to its mlir::Value.
11. Resolve graph outputs.
12. Create func.return.
13. Run mlir::verify(module).
14. Return OwningOpRef<ModuleOp>.
```

---

## **Type conversion**

```cpp
mlir::Type GraphToMLIR::convertType(
    const TensorType& type
) {
    mlir::Type elementType;

    switch (type.dtype()) {
    case DType::F32:
        elementType =
            mlir::Float32Type::get(&context_);
        break;
    }

    return mlir::RankedTensorType::get(
        type.shape(),
        elementType
    );
}
```

---

## **Function creation**

```cpp
auto location = mlir::UnknownLoc::get(&context_);
auto module = mlir::ModuleOp::create(location);

llvm::SmallVector<mlir::Type> inputTypes;
for (ValueId input : graph.inputs()) {
    inputTypes.push_back(
        convertType(graph.value(input).type)
    );
}

llvm::SmallVector<mlir::Type> outputTypes;
for (ValueId output : graph.outputs()) {
    outputTypes.push_back(
        convertType(graph.value(output).type)
    );
}

auto functionType = mlir::FunctionType::get(
    &context_,
    inputTypes,
    outputTypes
);

mlir::OpBuilder moduleBuilder(&context_);
moduleBuilder.setInsertionPointToStart(module.getBody());

auto function = moduleBuilder.create<mlir::func::FuncOp>(
    location,
    graph.name(),
    functionType
);

mlir::Block* entry = function.addEntryBlock();
mlir::OpBuilder bodyBuilder =
    mlir::OpBuilder::atBlockBegin(entry);
```

---

## **Map inputs**

```cpp
llvm::DenseMap<ValueId, mlir::Value> valueMap;

for (auto [valueId, argument] :
     llvm::zip(graph.inputs(), entry->getArguments())) {
    valueMap.try_emplace(valueId, argument);
}
```

This temporary map is the exact intended bridge:

```text
frontend ValueId → MLIR SSA Value
```

---

## **Import binary operations**

```cpp
case OpKind::Add: {
    mlir::Value lhs = valueMap.lookup(node.operands[0]);
    mlir::Value rhs = valueMap.lookup(node.operands[1]);

    auto op = builder.create<tinytensor::mlir::AddOp>(
        location,
        convertType(graph.value(node.results[0]).type),
        lhs,
        rhs
    );

    valueMap.try_emplace(
        node.results[0],
        op.getResult()
    );

    break;
}
```

Do the same for `MulOp`.

Do not silently use `lookup` without first checking presence in production code:

```cpp
auto lhsIt = valueMap.find(node.operands[0]);
if (lhsIt == valueMap.end()) {
    return emitError(...);
}
```

A missing operand mapping means either malformed graph ordering or an importer bug.

---

## **Import constants**

```cpp
const auto& constant =
    std::get<DenseFloatConstant>(node.payload);

auto resultType =
    llvm::cast<mlir::RankedTensorType>(
        convertType(graph.value(node.results[0]).type)
    );

auto denseAttr = mlir::DenseFPElementsAttr::get(
    resultType,
    llvm::ArrayRef<float>(constant.values)
);

auto op = builder.create<tinytensor::mlir::ConstantOp>(
    location,
    resultType,
    denseAttr
);

valueMap.try_emplace(node.results[0], op.getResult());
```

---

## **Return and verification**

```cpp
llvm::SmallVector<mlir::Value> outputValues;

for (ValueId output : graph.outputs()) {
    auto it = valueMap.find(output);
    if (it == valueMap.end()) {
        return mlir::failure();
    }

    outputValues.push_back(it->second);
}

bodyBuilder.create<mlir::func::ReturnOp>(
    location,
    outputValues
);

if (mlir::failed(mlir::verify(module))) {
    return mlir::failure();
}

return mlir::OwningOpRef<mlir::ModuleOp>(module);
```

---

# **11. Framework-facing compiler API**

```cpp
class Compiler {
public:
    Compiler();

    mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
    importToMLIR(const Graph& graph);

private:
    CompilerContext context_;
};
```

Implementation:

```cpp
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
Compiler::importToMLIR(const Graph& graph) {
    GraphToMLIR importer(context_.mlirContext());
    return importer.import(graph);
}
```

At this stage, calling it `compile()` would overstate what it does. Use `importToMLIR()` now; introduce `compile()` when the class actually runs a lowering pipeline and creates an executable.

---

# **12. Development milestones**

## **Milestone 1 — Frontend graph only**

Implement:

```text
DType
TensorType
ValueId
Node
Tensor
Graph::input
Graph::setOutputs
Graph::verify
```

Test graph:

```cpp
Graph graph("identity");
Tensor x = graph.input({{4}, DType::F32, "x"});
graph.setOutputs({x});
```

Acceptance:

```text
one graph input
one graph output
zero operations
frontend verifier succeeds
```

---

##

##

##

## **Milestone 2 — C++**

**`add`**

**and**

**`mul`**

Implement:

```text
Graph::createBinary
add()
mul()
type equality checks
cross-graph operand rejection
```

Test expression:

```cpp
Tensor z = add(mul(x, y), y);
```

Acceptance:

```text
correct node order
correct producer/result links
result type propagated
malformed operand types rejected
```

---

## **Milestone 3 — Constants**

Implement:

```text
Graph::constant
full()
constant element-count validation
```

Test:

```cpp
Tensor two = full(graph, {4}, 2.0f);
Tensor z = mul(x, two);
```

Acceptance:

```text
constant node owns its value
constant type matches shape
scalar fill expands to dense frontend values
```

For this tiny phase, expanding `full({4}, 2.0f)` to four values is acceptable. A splat representation can come later.

---

## **Milestone 4 — TinyTensor dialect builds**

Implement TableGen definitions and generated headers.

Acceptance:

```mlir
%0 = tt.add %arg0, %arg1 : tensor<4xf32>
%1 = tt.mul %0, %arg1 : tensor<4xf32>
```

must parse and verify through `tt-opt`.

Also ensure malformed IR fails:

```mlir
%0 = tt.add %lhs, %rhs : tensor<4xf32>
```

where `%rhs` is `tensor<8xf32>`.

---

## **Milestone 5 — Import identity function**

Import:

```cpp
graph.input → func argument
graph.output → func.return
```

Expected:

```mlir
func.func @identity(%arg0: tensor<4xf32>)
    -> tensor<4xf32> {
  return %arg0 : tensor<4xf32>
}
```

This isolates function-boundary import before operations complicate debugging.

---

## **Milestone 6 — Import constants and binary operations**

Import every `OpKind` using a `ValueId → mlir::Value` map.

Acceptance:

```cpp
Tensor result = add(mul(x, two), y);
```

must produce the exact intended dataflow in TinyTensor IR.

---

## **Milestone 7 — Integration CLI**

Create `tt-emit`:

```bash
tt-emit --example=add-mul
```

Output:

```mlir
module {
  ...
}
```

Useful options:

```text
--example=identity
--example=add
--example=add-mul
--verify-each
--output=<path>
```

Do not design a graph serialization format in this phase. Hard-coded example builders are enough for a compiler-development tool.

---

# **13. Test plan**

The project should follow the dual-testing principle from the wider plan: isolated MLIR tests plus C++ integration tests.  

## **Frontend unit tests**

```text
tests/Frontend/GraphTest.cpp
tests/Frontend/BinaryOpsTest.cpp
tests/Frontend/ConstantTest.cpp
tests/Frontend/VerificationTest.cpp
```

Essential cases:

```cpp
TEST(BinaryOps, ResultHasOperandType);
TEST(BinaryOps, RejectsDifferentShapes);
TEST(BinaryOps, RejectsDifferentGraphs);
TEST(Constant, RejectsWrongElementCount);
TEST(Graph, RejectsUnknownOutput);
TEST(Graph, PreservesNodeInsertionOrder);
```

---

## **Dialect tests**

```text
tests/Dialect/add.mlir
tests/Dialect/mul.mlir
tests/Dialect/constant.mlir
tests/Dialect/invalid-add-types.mlir
```

Example:

```mlir
// RUN: tt-opt %s | FileCheck %s

func.func @add(
    %lhs: tensor<4xf32>,
    %rhs: tensor<4xf32>
) -> tensor<4xf32> {
  %0 = tt.add %lhs, %rhs : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @add
// CHECK: %[[RESULT:.*]] = tt.add %arg0, %arg1
// CHECK: return %[[RESULT]]
```

---

## **Integration tests**

```text
tests/Integration/EmitIdentity.cpp
tests/Integration/EmitAdd.cpp
tests/Integration/EmitAddMul.cpp
tests/Integration/EmitConstant.cpp
```

The strongest Phase 1 test:

```cpp
TEST(GraphToMLIR, EmitsAddMulExpression) {
    Graph graph("forward");

    Tensor x = graph.input({{4}, DType::F32, "x"});
    Tensor y = graph.input({{4}, DType::F32, "y"});
    Tensor two = full(graph, {4}, 2.0f);

    graph.setOutputs({add(mul(x, two), y)});

    Compiler compiler;
    auto module = compiler.importToMLIR(graph);

    ASSERT_TRUE(succeeded(module));

    std::string text;
    llvm::raw_string_ostream stream(text);
    (*module)->print(stream);

    EXPECT_THAT(text, HasSubstr("tt.constant"));
    EXPECT_THAT(text, HasSubstr("tt.mul"));
    EXPECT_THAT(text, HasSubstr("tt.add"));
}
```

Prefer structural assertions where practical:

```cpp
auto functions =
    module->getOps<mlir::func::FuncOp>();

auto addOps =
    function.getOps<tinytensor::mlir::AddOp>();
```

Text checks are useful for readable integration snapshots but should not be your only correctness mechanism.

---

# **14. Phase completion checklist**

Phase 1 is complete only when all of these work:

```text
[ ] C++ constructs an expression without mentioning MLIR.
[ ] Tensor is a stable symbolic Graph + ValueId handle.
[ ] Frontend graph verifies its own invariants.
[ ] Graph inputs become func.func arguments.
[ ] Graph outputs become func.return operands.
[ ] Constant nodes become tt.constant.
[ ] Add nodes become tt.add.
[ ] Mul nodes become tt.mul.
[ ] TensorType becomes RankedTensorType.
[ ] Import uses a local ValueId → mlir::Value map.
[ ] Imported module passes mlir::verify.
[ ] tt-opt can parse and print TinyTensor operations.
[ ] Frontend, dialect, and integration tests all exist.
[ ] One example prints inspectable TinyTensor MLIR.
```

The final demonstration should be:

```cpp
Tensor result = x * full(graph, {4}, 2.0f) + y;
```

becoming:

```mlir
%c2 = tt.constant ...
%0 = tt.mul %arg0, %c2 ...
%1 = tt.add %0, %arg1 ...
return %1 ...
```

That is a small milestone, but it establishes the most important architectural contract:

```text
framework-owned symbolic identities
        ↓ temporary translation
MLIR-owned SSA identities
```

Once this contract is solid, Phase 2 can extend metadata and verification without redesigning the frontend, and the later lowering/runtime phases can consume the same verified `ModuleOp`.
