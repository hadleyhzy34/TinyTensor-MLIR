#include "tinytensor/Frontend/Graph.h"

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

using namespace tinytensor;

TEST(FrontendGraphTest, BuildsIdentityGraph) {
  Graph graph("identity");
  Tensor x = graph.input({.shape = {4}, .dtype = DType::F32, .name = "x"});

  graph.setOutputs({x});

  EXPECT_EQ(graph.name(), "identity");
  ASSERT_EQ(graph.inputs().size(), 1U);
  ASSERT_EQ(graph.outputs().size(), 1U);
  EXPECT_TRUE(graph.nodes().empty());
  EXPECT_EQ(graph.inputs()[0], x.valueId());
  EXPECT_EQ(graph.outputs()[0], x.valueId());
  EXPECT_TRUE(graph.verify());
}

TEST(FrontendGraphTest, InputCreatesGraphValueMetadata) {
  Graph graph("metadata");
  Tensor x = graph.input({.shape = {2, 3}, .dtype = DType::F32, .name = "features"});

  const GraphValue &value = graph.value(x.valueId());

  EXPECT_EQ(value.id, x.valueId());
  EXPECT_EQ(value.type, (TensorType({2, 3}, DType::F32)));
  EXPECT_FALSE(value.producer.has_value());
  EXPECT_EQ(value.name, "features");
}

TEST(FrontendGraphTest, VerifyRejectsGraphWithoutOutputs) {
  Graph graph("missing_output");
  graph.input({.shape = {4}, .dtype = DType::F32, .name = "x"});

  std::string error;
  EXPECT_FALSE(graph.verify(&error));
  EXPECT_EQ(error, "graph must have at least one output");
}

TEST(FrontendGraphTest, SetOutputsRejectsInvalidTensor) {
  Graph graph("invalid_output");
  Tensor invalid;

  EXPECT_THROW(graph.setOutputs({invalid}), std::invalid_argument);
}

TEST(FrontendGraphTest, SetOutputsRejectsTensorFromDifferentGraph) {
  Graph lhs("lhs");
  Graph rhs("rhs");
  Tensor x = lhs.input({.shape = {4}, .dtype = DType::F32, .name = "x"});

  EXPECT_THROW(rhs.setOutputs({x}), std::invalid_argument);
}

TEST(FrontendGraphTest, ValueRejectsOutOfRangeId) {
  Graph graph("out_of_range");
  graph.input({.shape = {4}, .dtype = DType::F32, .name = "x"});

  EXPECT_THROW((void)graph.value(ValueId{99}), std::out_of_range);
}

TEST(FrontendGraphTest, RejectsEmptyGraphName) {
  EXPECT_THROW(Graph(""), std::invalid_argument);
}
