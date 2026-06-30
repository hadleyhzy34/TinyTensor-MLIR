#include "tinytensor/Frontend/Graph.h"
#include "tinytensor/Frontend/Ops.h"

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

using namespace tinytensor;

TEST(FrontendOpsTest, AddCreatesBinaryNodeAndProducedValue) {
  Graph graph("add");
  Tensor x = graph.input({.shape = {4}, .dtype = DType::F32, .name = "x"});
  Tensor y = graph.input({.shape = {4}, .dtype = DType::F32, .name = "y"});

  Tensor z = add(x, y);
  graph.setOutputs({z});

  ASSERT_EQ(graph.nodes().size(), 1U);
  const Node &node = graph.nodes()[0];
  EXPECT_EQ(node.id, (NodeId{0}));
  EXPECT_EQ(node.kind, OpKind::Add);
  ASSERT_EQ(node.operands.size(), 2U);
  EXPECT_EQ(node.operands[0], x.valueId());
  EXPECT_EQ(node.operands[1], y.valueId());
  ASSERT_EQ(node.results.size(), 1U);
  EXPECT_EQ(node.results[0], z.valueId());

  const GraphValue &result = graph.value(z.valueId());
  EXPECT_EQ(result.type, (TensorType({4}, DType::F32)));
  ASSERT_TRUE(result.producer.has_value());
  EXPECT_EQ(*result.producer, node.id);
  EXPECT_TRUE(graph.verify());
}

TEST(FrontendOpsTest, MulCreatesBinaryNodeAndProducedValue) {
  Graph graph("mul");
  Tensor x = graph.input({.shape = {4}, .dtype = DType::F32, .name = "x"});
  Tensor y = graph.input({.shape = {4}, .dtype = DType::F32, .name = "y"});

  Tensor z = mul(x, y);
  graph.setOutputs({z});

  ASSERT_EQ(graph.nodes().size(), 1U);
  EXPECT_EQ(graph.nodes()[0].kind, OpKind::Mul);
  EXPECT_EQ(graph.value(z.valueId()).type, (TensorType({4}, DType::F32)));
  EXPECT_TRUE(graph.verify());
}

TEST(FrontendOpsTest, NestedExpressionPreservesTopologicalOrder) {
  Graph graph("nested");
  Tensor x = graph.input({.shape = {4}, .dtype = DType::F32, .name = "x"});
  Tensor y = graph.input({.shape = {4}, .dtype = DType::F32, .name = "y"});

  Tensor product = mul(x, y);
  Tensor result = add(product, y);
  graph.setOutputs({result});

  ASSERT_EQ(graph.nodes().size(), 2U);
  EXPECT_EQ(graph.nodes()[0].kind, OpKind::Mul);
  EXPECT_EQ(graph.nodes()[1].kind, OpKind::Add);
  EXPECT_EQ(graph.nodes()[1].operands[0], product.valueId());
  EXPECT_EQ(graph.nodes()[1].operands[1], y.valueId());

  const GraphValue &productValue = graph.value(product.valueId());
  ASSERT_TRUE(productValue.producer.has_value());
  EXPECT_EQ(productValue.producer->value, 0U);
  EXPECT_TRUE(graph.verify());
}

TEST(FrontendOpsTest, RejectsMismatchedOperandTypes) {
  Graph graph("mismatch");
  Tensor x = graph.input({.shape = {4}, .dtype = DType::F32, .name = "x"});
  Tensor y = graph.input({.shape = {8}, .dtype = DType::F32, .name = "y"});

  EXPECT_THROW(add(x, y), std::invalid_argument);
  EXPECT_TRUE(graph.nodes().empty());
}

TEST(FrontendOpsTest, RejectsCrossGraphOperands) {
  Graph lhsGraph("lhs");
  Graph rhsGraph("rhs");
  Tensor lhs = lhsGraph.input({.shape = {4}, .dtype = DType::F32, .name = "lhs"});
  Tensor rhs = rhsGraph.input({.shape = {4}, .dtype = DType::F32, .name = "rhs"});

  EXPECT_THROW(add(lhs, rhs), std::invalid_argument);
  EXPECT_TRUE(lhsGraph.nodes().empty());
  EXPECT_TRUE(rhsGraph.nodes().empty());
}

TEST(FrontendOpsTest, RejectsInvalidTensorOperand) {
  Graph graph("invalid_operand");
  Tensor x = graph.input({.shape = {4}, .dtype = DType::F32, .name = "x"});
  Tensor invalid;

  EXPECT_THROW(add(x, invalid), std::invalid_argument);
  EXPECT_TRUE(graph.nodes().empty());
}
