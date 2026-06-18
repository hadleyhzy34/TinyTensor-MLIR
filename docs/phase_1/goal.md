Given your background, **do not spend Phase 1 learning how to design tensors, graphs, or operator APIs**. Keep those pieces intentionally boring. The valuable work is learning how MLIR wants you to represent, construct, verify, inspect, and test IR.

Your main learning target is this boundary:

```text
Your familiar world                    New MLIR world

GraphValue / ValueId          →        mlir::Value
TensorType                    →        RankedTensorType
Graph node                    →        Operation
Graph inputs                  →        Block arguments
Graph outputs                 →        func.return operands
Graph                         →        func.func inside ModuleOp
Frontend validation           →        MLIR verification
```

The phase succeeds when these mappings feel natural—not merely when the code prints some MLIR.

# **1. Focus most on MLIR’s IR object model**

Before implementing many operations, get comfortable with how these objects relate:

```text
MLIRContext
└── ModuleOp
    └── Region
        └── Block
            ├── Block arguments
            ├── Operation
            │   ├── Operands: mlir::Value
            │   ├── Results: mlir::Value
            │   ├── Attributes
            │   └── Regions, optionally
            └── func.return
```

This is the most important conceptual foundation.

For your first function:

```mlir
func.func @forward(
    %x: tensor<4xf32>,
    %y: tensor<4xf32>
) -> tensor<4xf32> {
    %0 = tt.mul %x, %y : tensor<4xf32>
    %1 = tt.add %0, %y : tensor<4xf32>
    return %1 : tensor<4xf32>
}
```

identify every object explicitly:

```text
@forward             func::FuncOp
function body        Region
entry block          Block
%x and %y            BlockArgument, both are mlir::Value
tt.mul               Operation / TT_MulOp
%0                   OpResult, also an mlir::Value
tensor<4xf32>         RankedTensorType
func.return           func::ReturnOp
```

A useful exercise is to inspect these through C++:

```cpp
func::FuncOp function = ...;

mlir::Region& body = function.getBody();
mlir::Block& entry = body.front();

for (mlir::BlockArgument argument : entry.getArguments()) {
    argument.getType().dump();
}

for (mlir::Operation& operation : entry.getOperations()) {
    operation.dump();

    for (mlir::Value operand : operation.getOperands()) {
        operand.dump();
    }

    for (mlir::Value result : operation.getResults()) {
        result.dump();
    }
}
```

You already understand graph topology. The new thing is learning MLIR’s precise ownership and containment model.

#

#

# **2. Internalize that**

**`mlir::Value`**

**is not a stored value**

A common early confusion is treating `mlir::Value` like a tensor object or node.

It is neither.

`mlir::Value` is a lightweight reference to an SSA value defined by either:

```text
an operation result
or
a block argument
```

For example:

```mlir
%x = block argument
%0 = result of tt.mul
```

Both are represented by `mlir::Value`, despite having different definitions.

Practice asking these questions:

```cpp
mlir::Value value = ...;

mlir::Operation* definingOp = value.getDefiningOp();

if (!definingOp) {
    auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value);
}
```

This distinction will matter constantly later in:

```text
pattern rewriting
def-use traversal
constant folding
autodiff
liveness analysis
bufferization
```

Your frontend `ValueId` should remain separate from `mlir::Value`. The importer’s temporary map:

```cpp
llvm::DenseMap<ValueId, mlir::Value> valueMap;
```

is therefore not plumbing to rush through. It is the key conceptual bridge between your framework graph and MLIR SSA. The broader project architecture deliberately keeps these identities separate because `mlir::Value` is tied to the lifetime and structure of its owning IR.  

# **3. Learn SSA through construction, not theory alone**

You probably already understand computational graphs, so SSA will initially look obvious. Still, pay attention to the differences.

Your frontend may store:

```cpp
Node {
    operands = {value3, value7};
    results = {value8};
}
```

MLIR expresses the same relationship through actual use-def links:

```mlir
%8 = tt.add %3, %7
```

When you construct:

```cpp
auto add = builder.create<TT_AddOp>(
    location,
    resultType,
    lhs,
    rhs
);
```

MLIR automatically establishes:

```text
lhs → use by add operand 0
rhs → use by add operand 1
add result → definition of resulting mlir::Value
```

Explore this directly:

```cpp
mlir::Value result = add.getResult();

for (mlir::OpOperand& use : result.getUses()) {
    llvm::outs()
        << "used by: "
        << use.getOwner()->getName()
        << '\n';
}
```

Also experiment with:

```cpp
result.use_empty();
result.hasOneUse();
result.getUsers();
```

These APIs become central in optimization passes. Phase 1 is a good time to understand them on a three-operation graph rather than later on a differentiated training graph containing dozens of values.

# **4. Pay close attention to insertion points**

`OpBuilder` is one of the first genuinely practical MLIR challenges.

This code does not merely create an operation:

```cpp
builder.create<TT_AddOp>(...);
```

It inserts the operation at the builder’s current insertion point.

You need to understand:

```cpp
builder.setInsertionPointToStart(block);
builder.setInsertionPointToEnd(block);
builder.setInsertionPointAfter(op);
builder.setInsertionPointBefore(op);
```

For the importer:

```cpp
mlir::Block* entry = function.addEntryBlock();
mlir::OpBuilder builder =
    mlir::OpBuilder::atBlockBegin(entry);
```

Then operations are inserted sequentially:

```cpp
auto mul = builder.create<TT_MulOp>(...);
auto add = builder.create<TT_AddOp>(...);
builder.create<func::ReturnOp>(...);
```

A good practical mistake to make deliberately:

1. Insert `func.return`.
2. Attempt to insert another operation after it.
3. Run verification.
4. Observe why the block is invalid.

This teaches that MLIR is not simply an append-only list of graph nodes. Blocks have structural rules, terminators, dominance constraints, and operation ordering.

Insertion-point handling will become even more important when writing rewrite patterns.

# **5. Treat locations as part of the IR design**

It is tempting to write this everywhere:

```cpp
auto loc = builder.getUnknownLoc();
```

That is acceptable for the first ten minutes, but do not end Phase 1 without understanding locations.

Operations carry source locations:

```mlir
%0 = tt.add %x, %y
    : tensor<4xf32> loc("example.cpp":20:18)
```

Eventually your C++ frontend should be able to preserve where an expression originated. You do not need full source tracking now, but design the importer so the location is not permanently hard-coded.

For example:

```cpp
struct SourceLocation {
    std::string file;
    unsigned line;
    unsigned column;
};

struct Node {
    ...
    std::optional<SourceLocation> sourceLocation;
};
```

Then convert it:

```cpp
mlir::Location GraphToMLIR::convertLocation(
    const Node& node
) {
    if (!node.sourceLocation) {
        return mlir::UnknownLoc::get(&context_);
    }

    return mlir::FileLineColLoc::get(
        &context_,
        node.sourceLocation->file,
        node.sourceLocation->line,
        node.sourceLocation->column
    );
}
```

Why care this early? Because compiler diagnostics without source locations quickly become:

```text
error at unknown location
```

which is fine in tiny tests and awful in a framework-facing compiler.

# **6. Learn the boundary between builtin IR and your dialect**

Do not make every concept a TinyTensor operation or type.

Use existing MLIR infrastructure where it already models the concept:

```text
Framework concept       MLIR representation

Graph/program            ModuleOp
Callable graph           func.func
Graph input              Function block argument
Graph output             func.return
Tensor type              builtin RankedTensorType
f32                      builtin Float32Type
Dense constant data      DenseElementsAttr
```

Your dialect should initially represent only actual TinyTensor semantics:

```text
tt.constant
tt.add
tt.mul
```

This is an important dialect-design habit:

A dialect should introduce domain semantics, not duplicate the surrounding IR infrastructure.

For example, avoid inventing:

```mlir
tt.module
tt.function
tt.input
tt.output
tt.tensor_type
tt.float32
```

unless the framework eventually has semantics that the builtin forms genuinely cannot express.

# **7. Spend real effort on TableGen**

Given your goals, TableGen is not boilerplate to mechanically copy. It is one of the core MLIR skills to learn during Phase 1.

For each operation, understand what TableGen generates:

```tablegen
def TT_AddOp : TT_Op<"add", [
    Pure,
    SameOperandsAndResultType
]> {
    let arguments = (ins
        AnyRankedTensor:$lhs,
        AnyRankedTensor:$rhs
    );

    let results = (outs
        AnyRankedTensor:$result
    );

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` type($result)
    }];
}
```

Study the generated output enough to answer:

```text
What is TT_AddOp?
Where does getLhs() come from?
Why does getResult() exist?
What builder overloads were generated?
What does SameOperandsAndResultType actually verify?
How does assemblyFormat control parsing and printing?
How is the operation registered with the dialect?
```

You do not need to memorize generated code. You should be able to navigate it when a compiler error refers to a generated `.inc` file—which it absolutely will, usually while looking smug about it.

A strong exercise is to implement `tt.add` three ways:

1. Generic `OperationState` construction.
2. Generated `TT_AddOp` builder.
3. Parsing textual MLIR.

Compare the results:

```cpp
mlir::OperationState state(loc, "tt.add");
state.addOperands({lhs, rhs});
state.addTypes(resultType);
mlir::Operation* generic = builder.create(state);
```

versus:

```cpp
auto typed = builder.create<TT_AddOp>(
    loc,
    resultType,
    lhs,
    rhs
);
```

This clarifies what custom op wrappers actually provide over generic MLIR operations.

# **8. Understand parsing, printing, and verification as separate systems**

These are related but distinct:

```text
Parser:
    text → IR objects

Printer:
    IR objects → text

Verifier:
    checks whether constructed IR obeys invariants
```

An operation may parse successfully and still fail verification.

For example:

```mlir
%0 = tt.add %lhs, %rhs : tensor<4xf32>
```

could syntactically parse while `%rhs` has a mismatched type. The verifier must reject it.

For each operation, test three construction paths:

```text
C++ frontend importer
handwritten textual MLIR
programmatic generic operation creation
```

All should eventually reach the same verifier.

This prevents a subtle bad design where:

```text
frontend checks everything
but the MLIR operation accepts malformed IR
```

An MLIR dialect must defend its own invariants regardless of who created the IR.

# **9. Learn which invariants belong where**

You will have two validation layers.

## **Frontend graph validation**

Examples:

```text
Operands belong to the same Graph.
ValueId exists.
Node ordering is valid.
Constant payload length matches the frontend shape.
Graph has declared outputs.
```

These are framework representation invariants.

## **MLIR operation verification**

Examples:

```text
tt.add operands are ranked tensors.
Operand types match.
Result type matches operands.
tt.constant attribute type matches result type.
```

These are dialect semantic invariants.

Some checks intentionally overlap:

```text
Add operand types match
```

That duplication is healthy. The frontend gives earlier user-facing errors; the dialect remains valid when constructed by other tools, passes, tests, or textual parsing.

A useful question for every check is:

Would this still need to be valid if the C++ frontend did not exist?

If yes, enforce it in the MLIR dialect.

# **10. Learn MLIR diagnostics instead of throwing everywhere**

Your C++ framework may use exceptions or result objects, but inside MLIR prefer MLIR diagnostics and `LogicalResult`.

Operation verification:

```cpp
mlir::LogicalResult AddOp::verify() {
    if (getLhs().getType() != getRhs().getType()) {
        return emitOpError()
            << "expected matching operand types, but got "
            << getLhs().getType()
            << " and "
            << getRhs().getType();
    }

    return mlir::success();
}
```

Importer-level failure:

```cpp
return mlir::emitError(location)
       << "missing imported value for frontend ValueId "
       << operandId.value;
```

Or:

```cpp
mlir::InFlightDiagnostic diagnostic =
    mlir::emitError(location);

diagnostic << "cannot import " << operationName;
diagnostic.attachNote()
    << "frontend operand was not previously defined";

return mlir::failure();
```

Get comfortable with:

```text
LogicalResult
FailureOr<T>
success()
failure()
succeeded(...)
failed(...)
emitError(...)
emitOpError(...)
```

These patterns appear throughout MLIR APIs and passes.

# **11. Study ownership and lifetime carefully**

This will feel less glamorous than dialect design, but it causes many early problems.

Understand the roles of

```text
MLIRContext
    owns uniqued types, attributes, identifiers and dialect state

OwningOpRef<ModuleOp>
    owns the top-level module operation

ModuleOp / FuncOp / AddOp
    lightweight typed handles to operations

mlir::Value
    lightweight reference into existing IR

OpBuilder
    creates operations; does not own the whole IR
```

A dangerous pattern would be returning an `mlir::Value` after destroying its module:

```cpp
mlir::Value badFunction() {
    auto module = buildModule();
    mlir::Value value = ...;
    return value; // Module dies; value is now invalid.
}
```

Likewise, types and attributes are tied to the `MLIRContext` that created them. Ensure your `CompilerContext` outlives the module and all IR inspection.

This is one reason to make ownership explicit:

```cpp
class Compiler {
    CompilerContext context_;
};
```

and return:

```cpp
mlir::OwningOpRef<mlir::ModuleOp>
```

rather than a raw `ModuleOp` whose owner is unclear.

# **12. Do not rush into declarative type inference**

It may be tempting to make:

```cpp
builder.create<TT_AddOp>(loc, lhs, rhs);
```

infer the result type automatically.

That is a good later exercise, but first explicitly provide the result type:

```cpp
builder.create<TT_AddOp>(
    loc,
    resultType,
    lhs,
    rhs
);
```

This forces you to understand operation state:

```text
operands
results
types
attributes
location
```

After that works, add an inference interface or custom builder:

```cpp
auto add = builder.create<TT_AddOp>(loc, lhs, rhs);
```

Then compare:

```text
frontend-computed result type
versus
dialect-inferred result type
```

This becomes useful in Phase 2 when you focus more on framework metadata and type inference.

# **13. Make IR inspection part of your workflow**

Do not debug exclusively through C++.

At every milestone, print the module:

```cpp
module->dump();
```

or:

```cpp
module->print(
    llvm::outs(),
    mlir::OpPrintingFlags()
        .enableDebugInfo()
        .printGenericOpForm()
);
```

Generic operation form is especially educational:

Pretty form:

```mlir
%0 = tt.add %arg0, %arg1 : tensor<4xf32>
```

Generic form:

```mlir
%0 = "tt.add"(%arg0, %arg1)
    : (tensor<4xf32>, tensor<4xf32>)
      -> tensor<4xf32>
```

Generic form exposes MLIR’s underlying universal representation:

```text
operation name
operand list
result type list
attributes
regions
successors
```

Pretty syntax is a dialect-specific presentation layered on top.

A productive debugging loop is:

```text
construct through C++
→ print generic form
→ run verifier
→ print custom form
→ copy into .mlir test
→ run through tt-opt
```

#

#

# **14. Use**

**`tt-opt`**

**even though it is not the product path**

Your actual vertical path should remain:

```text
C++ Graph → ModuleOp
```

But `tt-opt` is invaluable for dialect development.

Use it to answer:

```text
Does the dialect register correctly?
Does parsing work?
Does custom printing round-trip?
Does malformed input fail verification?
Does generic syntax construct the same operation?
```

For example:

```bash
tt-opt test.mlir
tt-opt test.mlir --mlir-print-op-generic
tt-opt test.mlir --verify-diagnostics
```

The framework path and `tt-opt` serve different purposes:

```text
C++ integration test:
    verifies the frontend/compiler bridge

tt-opt test:
    isolates dialect behavior
```

The project plan’s dual testing strategy—MLIR-level tests plus C++ end-to-end integration tests—is particularly important here.  

# **15. Deliberately introduce malformed IR**

A surprisingly effective learning technique is to break every invariant once.

Try:

### **Mismatched operand types**

```mlir
%x: tensor<4xf32>
%y: tensor<8xf32>
%0 = tt.add %x, %y : tensor<4xf32>
```

### **Wrong result type**

```mlir
%x: tensor<4xf32>
%y: tensor<4xf32>
%0 = tt.add %x, %y : tensor<8xf32>
```

### **Return mismatch**

```mlir
func.func @f() -> tensor<4xf32> {
    %c = tt.constant ... : tensor<8xf32>
    return %c : tensor<8xf32>
}
```

### **Operation after terminator**

```mlir
return %x : tensor<4xf32>
%0 = tt.add %x, %x : tensor<4xf32>
```

### **Value defined in the wrong scope**

Later, construct nested regions and attempt to use an unavailable value.

Seeing the verifier reject bad IR is often more educational than only constructing valid examples.

# **16. Explore the generic operation API**

Even though you should use generated typed operations in normal code, inspect the generic API:

```cpp
mlir::Operation* op = typedAdd.getOperation();

op->getName();
op->getNumOperands();
op->getOperand(0);
op->getNumResults();
op->getResult(0);
op->getAttrs();
op->getBlock();
op->getParentOp();
```

Why this matters: transformation infrastructure often works with `Operation*`, not only your generated operation class.

You should understand the relationship:

```text
TT_AddOp
    typed wrapper around
Operation*
```

Similarly:

```text
func::FuncOp
    typed wrapper around
Operation*
```

Typed ops are convenient interfaces, not a separate object hierarchy owning different IR.

# **17. Learn casting in the MLIR context**

Your LLVM casting knowledge will be used constantly:

```cpp
if (auto add = llvm::dyn_cast<TT_AddOp>(operation)) {
    ...
}

auto rankedType =
    llvm::dyn_cast<mlir::RankedTensorType>(
        value.getType()
    );

if (!rankedType) {
    ...
}
```

Notice that MLIR uses casting across several wrapper-like systems:

```text
Operation* → typed operation wrapper
Type       → RankedTensorType
Attribute  → DenseElementsAttr
Value      → BlockArgument or OpResult
```

A good Phase 1 exercise is writing a module inspector that counts and describes:

```text
functions
blocks
tt.add operations
tt.mul operations
ranked tensor values
constant attributes
```

That prepares you for pass implementation more directly than spending more time on the frontend Graph API.

# **18. Keep the frontend deliberately unsophisticated**

Since framework architecture is already familiar, avoid disappearing into:

```text
perfect Tensor ownership
generic attribute maps
advanced graph mutation
operator registries
schema systems
layer abstractions
runtime storage design
eager-versus-lazy mode
```

For Phase 1, this is enough:

```cpp
class Tensor {
    Graph* graph_;
    ValueId value_;
};
```

and:

```cpp
enum class OpKind {
    Constant,
    Add,
    Mul
};
```

Hard-coded importer switch:

```cpp
switch (node.kind) {
case OpKind::Constant:
    ...
case OpKind::Add:
    ...
case OpKind::Mul:
    ...
}
```

Yes, a registry-based importer is more extensible. It also hides the exact MLIR-building mechanics you are currently trying to learn.

Build abstraction after repetition appears, not before.

# **19. Recommended effort distribution**

I would allocate your Phase 1 attention roughly like this:

```text
35% — MLIR object model and SSA/value ownership
25% — ODS/TableGen operation definitions
15% — parsing, printing, verification and diagnostics
15% — Graph-to-MLIR importer and type conversion
10% — frontend graph implementation
```

Your natural tendency may be to spend most of the time polishing the C++ framework side because it is comfortable. Resist that. Make the frontend intentionally minimal and spend the saved attention poking at the generated IR.

# **20. A better learning sequence**

Rather than implementing the repository layer by layer, use this sequence:

## **Step 1: Handwrite valid TinyTensor MLIR**

Before the C++ importer, make this parse:

```mlir
module {
  func.func @forward(
      %x: tensor<4xf32>,
      %y: tensor<4xf32>
  ) -> tensor<4xf32> {
    %0 = tt.mul %x, %y : tensor<4xf32>
    %1 = tt.add %0, %y : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
```

This forces dialect registration, parsing, printing, and verification to work first.

##

## **Step 2: Build the same IR directly with**

**`OpBuilder`**

No frontend graph yet:

```cpp
auto module = buildExampleModule(context);
module->dump();
```

You learn MLIR construction without debugging your graph representation simultaneously.

## **Step 3: Inspect the built IR**

Walk:

```text
ModuleOp
FuncOp
Block
BlockArgument
TT_MulOp
TT_AddOp
ReturnOp
```

Print operands, results, types, uses, owners, and parents.

## **Step 4: Build the minimal frontend graph**

Only then implement:

```cpp
Tensor result = add(mul(x, y), y);
```

## **Step 5: Replace hard-coded MLIR construction with importer traversal**

Use:

```cpp
DenseMap<ValueId, mlir::Value>
```

and confirm that the imported module is structurally equivalent to the hand-built one.

## **Step 6: Add negative tests**

Break types, mappings, terminators, and constant payloads.

This order isolates problems. If you begin by implementing all frontend classes, TableGen, importer, CMake, and CLI together, every bug becomes a five-layer archaeological dig.

# **21. Practical challenges most worth embracing**

The problems you should **not** avoid are:

### **CMake and generated TableGen files**

Learn how:

```text
.td
→ generated operation declarations
→ generated definitions
→ dialect library
→ tool links dialect
```

fits together. Build-system friction is part of real MLIR work.

### **Generated builder mismatch errors**

When `builder.create<TT_AddOp>(...)` fails with a long template error, inspect the generated builders and understand which operand/type arguments it expects.

### **Dialect not registered**

Recognize the difference between:

```text
operation unknown because dialect is unregistered
versus
operation malformed according to its verifier
```

### **Context mismatch**

Learn that types, attributes, locations, and operations need compatible `MLIRContext` ownership.

### **Invalid insertion points**

Understand why an operation ended up outside the function, after a terminator, or in the wrong block.

### **Failed verification**

Read the diagnostic and inspect the nearest operation rather than patching the frontend until the error disappears.

These annoyances are not side quests. They are the hands-on MLIR background you currently lack.

# **22. What you should be able to explain afterward**

At the end of Phase 1, you should confidently answer:

1. What owns an MLIR operation?
2. What is the difference between `Operation*` and `TT_AddOp`?
3. What are the two possible definitions of an `mlir::Value`?
4. Why are graph inputs represented as block arguments?
5. How are SSA use-def relationships stored?
6. What does `OpBuilder`’s insertion point control?
7. What does `MLIRContext` own?
8. Why does the importer use `ValueId → mlir::Value`?
9. What does TableGen generate for an operation?
10. What is the difference between parser, printer, and verifier?
11. Which invariants belong to the frontend versus the dialect?
12. Why use builtin tensor types and Func operations?
13. How does a custom operation differ from generic operation syntax?
14. Why must the module outlive every `mlir::Value` referring into it?
15. How would you traverse users of a result value?

If those answers are solid, Phase 1 has done its job—even if it supports only `constant`, `add`, and `mul`.

The biggest trap is producing attractive MLIR output while treating MLIR as a text-emission backend. Your goal is to understand it as an **in-memory, verified, extensible SSA infrastructure**. Printing the module is just how you look through the window.
