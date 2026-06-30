#include "tinytensor/Frontend/Graph.h"
#include "tinytensor/Frontend/Ops.h"

#include <gtest/gtest.h>
#include <stdexcept>
#include <variant>

using namespace tinytensor;

TEST(FrontendConstantsTest, FullCreatesDenseConstantNode) {
  Graph graph("full");

  Tensor two = full(graph, {4}, 2.0f);
  graph.setOutputs({two});

  ASSERT_EQ(graph.nodes().size(), 1U);
  const Node &node = graph.nodes()[0];
  EXPECT_EQ(node.id, (NodeId{0}));
  EXPECT_EQ(node.kind, OpKind::Constant);
  EXPECT_TRUE(node.operands.empty());
  ASSERT_EQ(node.results.size(), 1U);
  EXPECT_EQ(node.results[0], two.valueId());

  ASSERT_TRUE(std::holds_alternative<DenseFloatConstant>(node.payload));
  const DenseFloatConstant &constant = std::get<DenseFloatConstant>(node.payload);
  ASSERT_EQ(constant.values.size(), 4U);
  for (float stored : constant.values) {
    EXPECT_FLOAT_EQ(stored, 2.0f);
  }

  const GraphValue &result = graph.value(two.valueId());
  EXPECT_EQ(result.type, (TensorType({4}, DType::F32)));
  ASSERT_TRUE(result.producer.has_value());
  EXPECT_EQ(*result.producer, node.id);
  EXPECT_TRUE(graph.verify());
}

TEST(FrontendConstantsTest, ExplicitConstantPreservesProvidedValues) {
  Graph graph("constant");

  Tensor c = graph.constant(TensorType({2, 2}, DType::F32), {1.0f, 2.0f, 3.0f, 4.0f});
  graph.setOutputs({c});

  const Node &node = graph.nodes()[0];
  ASSERT_TRUE(std::holds_alternative<DenseFloatConstant>(node.payload));
  const DenseFloatConstant &constant = std::get<DenseFloatConstant>(node.payload);
  EXPECT_EQ(constant.values, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
  EXPECT_EQ(graph.value(c.valueId()).type, (TensorType({2, 2}, DType::F32)));
  EXPECT_TRUE(graph.verify());
}

TEST(FrontendConstantsTest, RejectsWrongConstantElementCount) {
  Graph graph("bad_constant");

  EXPECT_THROW(graph.constant(TensorType({4}, DType::F32), {1.0f, 2.0f}),
               std::invalid_argument);
  EXPECT_TRUE(graph.nodes().empty());
}

TEST(FrontendConstantsTest, FullSupportsScalarTensor) {
  Graph graph("scalar_full");

  Tensor scalar = full(graph, {}, 3.5f);
  graph.setOutputs({scalar});

  ASSERT_EQ(graph.nodes().size(), 1U);
  const DenseFloatConstant &constant =
      std::get<DenseFloatConstant>(graph.nodes()[0].payload);
  ASSERT_EQ(constant.values.size(), 1U);
  EXPECT_FLOAT_EQ(constant.values[0], 3.5f);
  EXPECT_TRUE(graph.value(scalar.valueId()).type.isScalar());
  EXPECT_TRUE(graph.verify());
}

TEST(FrontendConstantsTest, ConstantCanFeedBinaryOperation) {
  Graph graph("constant_mul");
  Tensor x = graph.input({.shape = {4}, .dtype = DType::F32, .name = "x"});
  Tensor two = full(graph, {4}, 2.0f);

  Tensor y = mul(x, two);
  graph.setOutputs({y});

  ASSERT_EQ(graph.nodes().size(), 2U);
  EXPECT_EQ(graph.nodes()[0].kind, OpKind::Constant);
  EXPECT_EQ(graph.nodes()[1].kind, OpKind::Mul);
  EXPECT_EQ(graph.nodes()[1].operands[0], x.valueId());
  EXPECT_EQ(graph.nodes()[1].operands[1], two.valueId());
  EXPECT_TRUE(graph.verify());
}
