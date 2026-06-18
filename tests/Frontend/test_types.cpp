#include <gtest/gtest.h>

#include <stdexcept>

#include "tinytensor/Frontend/DType.h"
#include "tinytensor/Frontend/Ids.h"
#include "tinytensor/Frontend/TensorSpec.h"
#include "tinytensor/Frontend/TensorType.h"

using namespace tinytensor;

TEST(FrontendTypesTest, ConstructsRankedF32TensorType) {
  TensorType type({4}, DType::F32);

  ASSERT_EQ(type.shape().size(), 1U);
  EXPECT_EQ(type.shape()[0], 4);
  EXPECT_EQ(type.rank(), 1);
  EXPECT_EQ(type.dtype(), DType::F32);
  EXPECT_FALSE(type.isScalar());
}

TEST(FrontendTypesTest, AllowsRankZeroScalarTensorType) {
  TensorType scalar({}, DType::F32);

  EXPECT_TRUE(scalar.shape().empty());
  EXPECT_EQ(scalar.rank(), 0);
  EXPECT_TRUE(scalar.isScalar());
  EXPECT_EQ(scalar.dtype(), DType::F32);
}

TEST(FrontendTypesTest, RejectsZeroAndNegativeDimensions) {
  EXPECT_THROW((TensorType({0}, DType::F32)), std::invalid_argument);
  EXPECT_THROW((TensorType({4, -1}, DType::F32)), std::invalid_argument);
}

TEST(FrontendTypesTest, TensorSpecPreservesMetadata) {
  TensorSpec spec = tensorSpec({2, 3}, DType::F32, "input");

  EXPECT_EQ(spec.name, "input");
  EXPECT_EQ(spec.dtype, DType::F32);
  ASSERT_EQ(spec.shape.size(), 2U);
  EXPECT_EQ(spec.shape[0], 2);
  EXPECT_EQ(spec.shape[1], 3);

  TensorType type = spec.type();
  EXPECT_EQ(type, (TensorType({2, 3}, DType::F32)));
}

TEST(FrontendTypesTest, StableIdsCompareByValue) {
  EXPECT_EQ(ValueId{3}, ValueId{3});
  EXPECT_NE(ValueId{3}, ValueId{4});
  EXPECT_EQ(NodeId{7}, NodeId{7});
  EXPECT_NE(NodeId{7}, NodeId{8});
}

TEST(FrontendTypesTest, StableIdsCompareByValueSecond) {
  EXPECT_EQ(ValueId{.value = 3}, ValueId{.value = 3});
  EXPECT_NE(ValueId{3}, ValueId{4});
  EXPECT_EQ(NodeId{7}, NodeId{7});
  EXPECT_NE(NodeId{7}, NodeId{8});
}
