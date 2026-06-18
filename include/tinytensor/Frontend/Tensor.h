#ifndef TINYTENSOR_FRONTEND_TENSOR_H
#define TINYTENSOR_FRONTEND_TENSOR_H

#include "tinytensor/Frontend/Ids.h"

#include <stdexcept>

namespace tinytensor {

class Graph;

class Tensor {
public:
  Tensor() = default;

  ValueId valueId() const {
    requireValid();
    return value_;
  }

  const Graph &graph() const {
    requireValid();
    return *graph_;
  }

  Graph &graph() {
    requireValid();
    return *graph_;
  }

  explicit operator bool() const { return graph_ != nullptr; }

private:
  friend class Graph;

  Tensor(Graph *graph, ValueId value) : graph_(graph), value_(value) {}

  void requireValid() const {
    if (!graph_) {
      throw std::logic_error("invalid symbolic tensor handle");
    }
  }

  Graph *graph_ = nullptr;
  ValueId value_{};
};

} // namespace tinytensor

#endif // TINYTENSOR_FRONTEND_TENSOR_H
