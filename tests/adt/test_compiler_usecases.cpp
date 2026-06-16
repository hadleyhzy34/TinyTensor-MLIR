#include "tinytensor/adt/StringRef.h"
#include "tinytensor/adt/SmallVector.h"
#include "tinytensor/adt/DenseMap.h"
#include <gtest/gtest.h>

using namespace tinytensor::adt;

// Compiler mock structures
struct MockValue {
  int Id;
  explicit MockValue(int Id) : Id(Id) {}
};

class MockOperation {
public:
  StringRef OpName;
  SmallVector<MockValue*, 4> Operands;
  SmallVector<MockValue*, 2> Results;
  DenseMap<StringRef, StringRef> Attributes;

  MockOperation(StringRef Name) : OpName(Name) {}
};

struct MockTensorType {
  StringRef ElementType;
  SmallVector<int64_t, 4> Shape;

  MockTensorType(StringRef ElemType, std::initializer_list<int64_t> Sh)
      : ElementType(ElemType), Shape(Sh) {}

  size_t getRank() const { return Shape.size(); }
};

TEST(CompilerUseCasesTest, MockOperationCreation) {
  // Simulate creating: %y = "tt.add"(%x0, %x1) {device = "gpu"} : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  MockValue ValX0(0);
  MockValue ValX1(1);
  MockValue ValY(2);

  MockOperation AddOp("tt.add");
  AddOp.Operands.push_back(&ValX0);
  AddOp.Operands.push_back(&ValX1);
  AddOp.Results.push_back(&ValY);
  AddOp.Attributes.insert({"device", "gpu"});
  AddOp.Attributes.insert({"precision", "fp32"});

  EXPECT_EQ(AddOp.OpName, "tt.add");
  EXPECT_EQ(AddOp.Operands.size(), 2U);
  EXPECT_EQ(AddOp.Operands[0]->Id, 0);
  EXPECT_EQ(AddOp.Operands[1]->Id, 1);
  EXPECT_EQ(AddOp.Results.size(), 1U);
  EXPECT_EQ(AddOp.Results[0]->Id, 2);

  EXPECT_TRUE(AddOp.Attributes.count("device"));
  EXPECT_EQ(AddOp.Attributes.lookup("device"), "gpu");
  EXPECT_EQ(AddOp.Attributes.lookup("precision"), "fp32");
}

TEST(CompilerUseCasesTest, ShapeInferenceSimulation) {
  // Elementwise binary shape propagation: input shapes must match (or broadcast)
  MockTensorType LHS("f32", {32, 128});
  MockTensorType RHS("f32", {32, 128});
  
  EXPECT_EQ(LHS.getRank(), 2U);
  EXPECT_EQ(LHS.ElementType, "f32");

  // Infer output shape
  SmallVector<int64_t, 4> OutputShape;
  ASSERT_EQ(LHS.getRank(), RHS.getRank());
  for (size_t i = 0; i < LHS.getRank(); ++i) {
    ASSERT_EQ(LHS.Shape[i], RHS.Shape[i]);
    OutputShape.push_back(LHS.Shape[i]);
  }

  MockTensorType Output("f32", {});
  Output.Shape = OutputShape;

  EXPECT_EQ(Output.getRank(), 2U);
  EXPECT_EQ(Output.Shape[0], 32);
  EXPECT_EQ(Output.Shape[1], 128);
}

TEST(CompilerUseCasesTest, ValueMappingSimulation) {
  // During lowering, we map old IR values to new IR values.
  MockValue OldV0(100);
  MockValue OldV1(101);
  MockValue OldV2(102);

  MockValue NewV0(200);
  MockValue NewV1(201);
  MockValue NewV2(202);

  DenseMap<MockValue*, MockValue*> ValueMap;
  ValueMap[&OldV0] = &NewV0;
  ValueMap[&OldV1] = &NewV1;
  ValueMap[&OldV2] = &NewV2;

  // Verify lookups during transformation
  MockValue* InputOperands[] = {&OldV1, &OldV2};
  SmallVector<MockValue*, 2> LoweredOperands;

  for (auto* Op : InputOperands) {
    ASSERT_TRUE(ValueMap.count(Op));
    LoweredOperands.push_back(ValueMap.lookup(Op));
  }

  EXPECT_EQ(LoweredOperands.size(), 2U);
  EXPECT_EQ(LoweredOperands[0]->Id, 201);
  EXPECT_EQ(LoweredOperands[1]->Id, 202);
}
