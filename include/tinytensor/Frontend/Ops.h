#ifndef TINYTENSOR_FRONTEND_OPS_H
#define TINYTENSOR_FRONTEND_OPS_H

#include "tinytensor/Frontend/DType.h"
#include "tinytensor/Frontend/Graph.h"
#include "tinytensor/Frontend/Tensor.h"
#include "tinytensor/Frontend/TensorType.h"

#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

namespace tinytensor {

inline Tensor add(const Tensor &lhs, const Tensor &rhs) {
  Graph &graph = const_cast<Graph &>(lhs.graph());
  return graph.createBinary(OpKind::Add, lhs, rhs);
}

inline Tensor mul(const Tensor &lhs, const Tensor &rhs) {
  Graph &graph = const_cast<Graph &>(lhs.graph());
  return graph.createBinary(OpKind::Mul, lhs, rhs);
}

inline Tensor full(Graph &graph, std::vector<std::int64_t> shape, float value) {
  TensorType type(std::move(shape), DType::F32);
  std::vector<float> values(Graph::elementCount(type), value);
  return graph.constant(std::move(type), std::move(values));
}

inline Tensor full(Graph &graph, std::initializer_list<std::int64_t> shape,
                   float value) {
  return full(graph, std::vector<std::int64_t>(shape), value);
}

} // namespace tinytensor

#endif // TINYTENSOR_FRONTEND_OPS_H
