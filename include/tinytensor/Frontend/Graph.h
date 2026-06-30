#ifndef TINYTENSOR_FRONTEND_GRAPH_H
#define TINYTENSOR_FRONTEND_GRAPH_H

#include "tinytensor/Frontend/Ids.h"
#include "tinytensor/Frontend/Tensor.h"
#include "tinytensor/Frontend/TensorSpec.h"
#include "tinytensor/Frontend/TensorType.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace tinytensor {

enum class OpKind : std::uint8_t {
  Constant,
  Add,
  Mul,
};

struct DenseFloatConstant {
  std::vector<float> values;
};

using NodePayload = std::variant<std::monostate, DenseFloatConstant>;

struct GraphValue {
  ValueId id;
  TensorType type;
  std::optional<NodeId> producer;
  std::string name;
};

struct Node {
  NodeId id;
  OpKind kind;
  std::vector<ValueId> operands;
  std::vector<ValueId> results;
  NodePayload payload;
};

class Graph {
public:
  explicit Graph(std::string name) : name_(std::move(name)) {
    if (name_.empty()) {
      throw std::invalid_argument("graph name must not be empty");
    }
  }

  Tensor input(const TensorSpec &spec) {
    ValueId id = addValue(spec.type(), std::nullopt, spec.name);
    inputs_.push_back(id);
    return Tensor(this, id);
  }

  Tensor createBinary(OpKind kind, const Tensor &lhs, const Tensor &rhs) {
    if (kind != OpKind::Add && kind != OpKind::Mul) {
      throw std::invalid_argument("unsupported binary operation kind");
    }

    requireOwnedTensor(lhs, "binary lhs tensor");
    requireOwnedTensor(rhs, "binary rhs tensor");

    const TensorType &lhsType = value(lhs.valueId()).type;
    const TensorType &rhsType = value(rhs.valueId()).type;
    if (lhsType != rhsType) {
      throw std::invalid_argument("binary operands must have identical types");
    }

    NodeId nodeId{static_cast<std::uint32_t>(nodes_.size())};
    ValueId resultId = addValue(lhsType, nodeId);
    nodes_.push_back(Node{nodeId,
                          kind,
                          {lhs.valueId(), rhs.valueId()},
                          {resultId},
                          std::monostate{}});
    return Tensor(this, resultId);
  }

  Tensor constant(TensorType type, std::vector<float> values) {
    const std::size_t expectedElements = elementCount(type);
    if (values.size() != expectedElements) {
      throw std::invalid_argument("constant value count must match tensor shape");
    }

    NodeId nodeId{static_cast<std::uint32_t>(nodes_.size())};
    ValueId resultId = addValue(std::move(type), nodeId);
    nodes_.push_back(Node{nodeId,
                          OpKind::Constant,
                          {},
                          {resultId},
                          DenseFloatConstant{std::move(values)}});
    return Tensor(this, resultId);
  }

  Tensor constant(TensorType type, std::initializer_list<float> values) {
    return constant(std::move(type), std::vector<float>(values));
  }

  void setOutputs(std::initializer_list<Tensor> outputs) {
    setOutputs(std::vector<Tensor>(outputs));
  }

  void setOutputs(const std::vector<Tensor> &outputs) {
    outputs_.clear();
    outputs_.reserve(outputs.size());
    for (const Tensor &tensor : outputs) {
      requireOwnedTensor(tensor, "graph output tensor");
      outputs_.push_back(tensor.valueId());
    }
  }

  const std::string &name() const { return name_; }

  const std::vector<ValueId> &inputs() const { return inputs_; }

  const std::vector<ValueId> &outputs() const { return outputs_; }

  const std::vector<Node> &nodes() const { return nodes_; }

  const GraphValue &value(ValueId id) const {
    if (!contains(id)) {
      throw std::out_of_range("graph value id is out of range");
    }
    return values_[id.value];
  }

  static std::size_t elementCount(const TensorType &type) {
    std::size_t count = 1;
    for (std::int64_t dim : type.shape()) {
      count *= static_cast<std::size_t>(dim);
    }
    return count;
  }

  bool verify(std::string *error = nullptr) const {
    if (outputs_.empty()) {
      return fail(error, "graph must have at least one output");
    }

    for (std::size_t index = 0; index < values_.size(); ++index) {
      if (values_[index].id.value != index) {
        return fail(error, "graph value id does not match value storage index");
      }
    }

    for (std::size_t index = 0; index < nodes_.size(); ++index) {
      const Node &node = nodes_[index];
      if (node.id.value != index) {
        return fail(error, "node id does not match node storage index");
      }
      if (!verifyNode(node, error)) {
        return false;
      }
    }

    for (ValueId input : inputs_) {
      if (!contains(input)) {
        return fail(error, "graph input references a missing value");
      }
      if (values_[input.value].producer.has_value()) {
        return fail(error, "graph input must not have a producer");
      }
    }

    for (ValueId output : outputs_) {
      if (!contains(output)) {
        return fail(error, "graph output references a missing value");
      }
    }

    return true;
  }

private:
  ValueId addValue(TensorType type, std::optional<NodeId> producer,
                   std::string name = {}) {
    ValueId id{static_cast<std::uint32_t>(values_.size())};
    values_.push_back(GraphValue{id, std::move(type), producer, std::move(name)});
    return id;
  }

  bool contains(ValueId id) const { return id.value < values_.size(); }

  void requireOwnedTensor(const Tensor &tensor, const char *description) const {
    if (!tensor) {
      throw std::invalid_argument(std::string(description) + " is invalid");
    }
    if (&tensor.graph() != this) {
      throw std::invalid_argument(std::string(description) +
                                  " belongs to a different graph");
    }
    if (!contains(tensor.valueId())) {
      throw std::invalid_argument(std::string(description) +
                                  " references a missing value");
    }
  }

  bool verifyNode(const Node &node, std::string *error) const {
    switch (node.kind) {
    case OpKind::Constant:
      return verifyConstantNode(node, error);
    case OpKind::Add:
    case OpKind::Mul:
      return verifyBinaryNode(node, error);
    }

    return fail(error, "unknown node kind");
  }

  bool verifyConstantNode(const Node &node, std::string *error) const {
    if (!node.operands.empty()) {
      return fail(error, "constant node must not have operands");
    }
    if (node.results.size() != 1) {
      return fail(error, "constant node must have exactly one result");
    }
    if (!std::holds_alternative<DenseFloatConstant>(node.payload)) {
      return fail(error, "constant node must carry dense float payload");
    }

    const ValueId result = node.results[0];
    if (!verifyResultProducer(node, result, error)) {
      return false;
    }

    const DenseFloatConstant &constant = std::get<DenseFloatConstant>(node.payload);
    const TensorType &resultType = values_[result.value].type;
    if (constant.values.size() != elementCount(resultType)) {
      return fail(error, "constant value count must match tensor shape");
    }

    return true;
  }

  bool verifyBinaryNode(const Node &node, std::string *error) const {
    if (node.operands.size() != 2) {
      return fail(error, "binary node must have exactly two operands");
    }
    if (node.results.size() != 1) {
      return fail(error, "binary node must have exactly one result");
    }
    if (!std::holds_alternative<std::monostate>(node.payload)) {
      return fail(error, "binary node must not carry payload");
    }

    for (ValueId operand : node.operands) {
      if (!contains(operand)) {
        return fail(error, "node operand references a missing value");
      }
      const std::optional<NodeId> producer = values_[operand.value].producer;
      if (producer && producer->value >= node.id.value) {
        return fail(error, "node appears before one of its operand producers");
      }
    }

    const ValueId result = node.results[0];
    if (!verifyResultProducer(node, result, error)) {
      return false;
    }

    const TensorType &lhsType = values_[node.operands[0].value].type;
    const TensorType &rhsType = values_[node.operands[1].value].type;
    const TensorType &resultType = values_[result.value].type;
    if (lhsType != rhsType || lhsType != resultType) {
      return fail(error, "binary node operands and result must have identical types");
    }

    return true;
  }

  bool verifyResultProducer(const Node &node, ValueId result,
                            std::string *error) const {
    if (!contains(result)) {
      return fail(error, "node result references a missing value");
    }
    const std::optional<NodeId> producer = values_[result.value].producer;
    if (!producer || *producer != node.id) {
      return fail(error, "node result does not name the correct producer");
    }
    return true;
  }

  static bool fail(std::string *error, const char *message) {
    if (error) {
      *error = message;
    }
    return false;
  }

  std::string name_;
  std::vector<GraphValue> values_;
  std::vector<Node> nodes_;
  std::vector<ValueId> inputs_;
  std::vector<ValueId> outputs_;
};

} // namespace tinytensor

#endif // TINYTENSOR_FRONTEND_GRAPH_H
