#ifndef TINYTENSOR_FRONTEND_GRAPH_H
#define TINYTENSOR_FRONTEND_GRAPH_H

#include "tinytensor/Frontend/Ids.h"
#include "tinytensor/Frontend/Tensor.h"
#include "tinytensor/Frontend/TensorSpec.h"
#include "tinytensor/Frontend/TensorType.h"

#include <cstddef>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tinytensor {

struct GraphValue {
  ValueId id;
  TensorType type;
  std::optional<NodeId> producer;
  std::string name;
};

struct Node {
  NodeId id;
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

  void setOutputs(std::initializer_list<Tensor> outputs) {
    setOutputs(std::vector<Tensor>(outputs));
  }

  void setOutputs(const std::vector<Tensor> &outputs) {
    outputs_.clear();
    outputs_.reserve(outputs.size());
    for (const Tensor &tensor : outputs) {
      requireOwnedTensor(tensor);
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

  bool verify(std::string *error = nullptr) const {
    if (outputs_.empty()) {
      return fail(error, "graph must have at least one output");
    }

    for (std::size_t index = 0; index < values_.size(); ++index) {
      if (values_[index].id.value != index) {
        return fail(error, "graph value id does not match value storage index");
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

  void requireOwnedTensor(const Tensor &tensor) const {
    if (!tensor) {
      throw std::invalid_argument("graph output tensor is invalid");
    }
    if (&tensor.graph() != this) {
      throw std::invalid_argument("graph output tensor belongs to a different graph");
    }
    if (!contains(tensor.valueId())) {
      throw std::invalid_argument("graph output tensor references a missing value");
    }
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
