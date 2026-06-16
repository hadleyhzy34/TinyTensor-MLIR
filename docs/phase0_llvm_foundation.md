# Phase 0: LLVM Foundation

This document details the architecture, design principles, memory layouts, and programming tutorials for the custom data structures implemented in Phase 0. These structures mirror the core APIs of LLVM's `StringRef`, `SmallVector`, and `DenseMap`, which are essential to building high-performance deep learning compilers.

---

## 1. StringRef: Zero-Allocation String Views

### Motivation & Design
Standard C++ `std::string` objects own their data and perform heap allocation when constructed or modified (unless small-string optimization applies). In compilers, names of operations, types, functions, and attribute keys are constantly passed around. Allocating heap memory for each lookup or conversion would severely degrade compiler performance.

`StringRef` is a lightweight, non-owning view of a character array. It stores only two fields:
* A pointer to the start of the string (`const char *Data`)
* The length of the string (`size_t Length`)

```
             +-----------------------+
StringRef    |  Data   |   Length    |
             +----|----+------|------+
                  |           |
                  v           v
Memory       ['h','e','l','l','o',' ','w','o','r','l','d']
```

Because it does not own the memory, copying a `StringRef` is extremely fast (just copying a pointer and an integer). It does not require a null-terminated string (`\0`), allowing efficient slicing and substring generation via pointer offsets.

### Usage Example
```cpp
#include "tinytensor/adt/StringRef.h"
#include <iostream>

void processOp(tinytensor::adt::StringRef name) {
  // Slicing without heap allocation
  if (name.startswith("tt.")) {
    tinytensor::adt::StringRef dialect = name.substr(0, 2); // "tt"
    tinytensor::adt::StringRef op = name.substr(3);        // e.g. "add"
    std::cout << "Dialect: " << dialect << ", Op: " << op << "\n";
  }
}
```

---

## 2. SmallVector: Stack-Allocated Dynamic Arrays

### Motivation & Design
Compilers manipulate arrays of variable sizes frequently (e.g., shapes of tensors, operands of an operation, block arguments). However, the majority of tensors in deep learning have a small rank (typically $\le 4$). 

`SmallVector<T, N>` is optimized for this distribution. It allocates an inline stack buffer of size `N * sizeof(T)` inside the object itself. 
* If the number of elements $\le N$, no heap allocations occur.
* If the elements exceed $N$, the vector dynamically allocates memory on the heap (using `std::malloc`) and copies/moves elements over.

### Memory Layout Comparison
#### Under Capacity Limit (Size $\le N$)
The array elements reside entirely inside the `InlineBufferMemory` array on the stack.
```
+-----------------------------------------------------------+
| SmallVector<T, N>                                         |
|  - BeginX --------+                                       |
|  - Size (3)       |                                       |
|  - Capacity (4)   |                                       |
|  - InlineBuffer   |                                       |
|  - InlineCapacity |                                       |
|  - InlineBufferMemory: [ Element 0 | Element 1 | Element 2 | [Empty] ]
|                         ^                                 |
|                         +---------------------------------+
+-----------------------------------------------------------+
```

#### After Heap Spill (Size $> N$)
The vector allocates memory on the heap, and updates `BeginX` to point to it.
```
+---------------------------------------------------+
| SmallVector<T, N>                                 |
|  - BeginX ------------+                           |
|  - Size (5)           |                           |
|  - Capacity (8)       |                           |
|  - InlineBuffer       |                           |      Heap Memory
|  - InlineCapacity     |                           |     +-----------------------------------------+
|  - InlineBufferMemory |                           +---> | Elem 0 | Elem 1 | Elem 2 | Elem 3 | Elem 4|
+-----------------------+                                 +-----------------------------------------+
```

### Type Erasure via `SmallVectorImpl<T>`
To avoid code bloat from instantiating functions for every possible inline capacity `N`, `SmallVector<T, N>` inherits from `SmallVectorImpl<T>`. 
APIs in the compiler should be written to accept `SmallVectorImpl<T> &` instead of `SmallVector<T, N> &`:
```cpp
#include "tinytensor/adt/SmallVector.h"

// Good: compiles into a single function regardless of static capacity N
void printShape(const tinytensor::adt::SmallVectorImpl<int64_t> &shape) {
  for (auto dim : shape) {
    std::cout << dim << " ";
  }
  std::cout << "\n";
}

int main() {
  tinytensor::adt::SmallVector<int64_t, 4> shape4{1, 3, 224, 224};
  tinytensor::adt::SmallVector<int64_t, 2> shape2{32, 128};

  printShape(shape4); // Works
  printShape(shape2); // Works
}
```

---

## 3. DenseMap: Cache-Friendly Open-Addressed Hash Maps

### Motivation & Design
A standard `std::unordered_map` uses bucket chaining (a linked list for each bucket). This results in node allocations on the heap and pointer-chasing, leading to bad cache locality.

`DenseMap<KeyT, ValueT>` uses **open addressing** (all entries are stored in a single flat array of buckets). It resolves collisions using **quadratic (triangular) probing**:
$$Idx = (Idx + ProbeAmt++) \ \& \ (NumBuckets - 1)$$

Because there is no chaining, it relies on two special sentinel keys to distinguish slots:
1. **Empty Key**: Indicates a slot that has never been occupied. Probing stops here.
2. **Tombstone Key**: Indicates an entry that was deleted. Probing must continue past tombstones, but during insertion, tombstones can be recycled.

### DenseMap Bucket Probing Flow
```
Hash(Key) -> [Bucket 1] (Occupied by other key) -> Probe 1
                 |
                 v
             [Bucket 2] (Tombstone) --------------> Probe 2 (Remember index)
                 |
                 v
             [Bucket 5] (Empty Key) --------------> Probe Stop (Insert at Bucket 2)
```

### Sentinel Trait Specifications (`DenseMapInfo`)
We implement sentinel keys via `DenseMapInfo<T>` specializations:
* **Integers**: Empty key = `-1`, Tombstone key = `-2` (or maximum bounds).
* **Pointers**: Empty key = `0xFFFF...FFFF` (`-1`), Tombstone key = `0xFFFF...FFFE` (`-2`).
* **StringRef**: Empty key pointer = `(const char*)-1`, Tombstone key pointer = `(const char*)-2`. Pointer comparison is performed first to guarantee no read violations.

---

## 4. Compiler Use-Case Tutorial

Below is a hands-on tutorial simulating a compiler lowering pass. It uses all three data structures to translate a set of operations and map old SSA values to new SSA values.

```cpp
#include "tinytensor/adt/StringRef.h"
#include "tinytensor/adt/SmallVector.h"
#include "tinytensor/adt/DenseMap.h"
#include <iostream>

using namespace tinytensor::adt;

struct Value {
  int id;
  explicit Value(int id) : id(id) {}
};

struct Op {
  StringRef name;
  SmallVector<Value*, 4> operands;
  SmallVector<Value*, 2> results;
  DenseMap<StringRef, StringRef> attrs;
};

void lowerOp(const Op &oldOp, DenseMap<Value*, Value*> &valueMap) {
  std::cout << "Lowering operation: " << oldOp.name << "\n";

  // Map input operands to their newly created counterpart
  SmallVector<Value*, 4> loweredOperands;
  for (Value* operand : oldOp.operands) {
    if (valueMap.count(operand)) {
      loweredOperands.push_back(valueMap.lookup(operand));
    } else {
      std::cerr << "Error: operand %" << operand->id << " not mapped!\n";
    }
  }

  // Create mock lowered output
  static int nextValueId = 200;
  Value* newResult = new Value(nextValueId++);
  
  // Register output mapping
  valueMap[oldOp.results[0]] = newResult;

  std::cout << "  Lowered operand count: " << loweredOperands.size() << "\n";
  std::cout << "  Mapped old output %" << oldOp.results[0]->id 
            << " to new output %" << newResult->id << "\n";
}

int main() {
  Value v0(0), v1(1), vOut(2);

  Op addOp{"tt.add"};
  addOp.operands.push_back(&v0);
  addOp.operands.push_back(&v1);
  addOp.results.push_back(&vOut);
  addOp.attrs.insert({"device", "gpu"});

  // Setup lowering map
  Value newV0(100), newV1(101);
  DenseMap<Value*, Value*> valueMap;
  valueMap[&v0] = &newV0;
  valueMap[&v1] = &newV1;

  lowerOp(addOp, valueMap);

  delete valueMap[&vOut]; // Clean up dynamic mock Value
  return 0;
}
```
