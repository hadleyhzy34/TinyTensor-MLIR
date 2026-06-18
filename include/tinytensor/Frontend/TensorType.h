#ifndef TINYTENSOR_FRONTEND_TENSORTYPE_H
#define TINYTENSOR_FRONTEND_TENSORTYPE_H

#include "tinytensor/Frontend/DType.h"

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tinytensor {

class TensorType {
public:
  TensorType() = default;

  TensorType(std::initializer_list<std::int64_t> shape, DType dtype)
      : TensorType(std::vector<std::int64_t>(shape), dtype) {}

  TensorType(std::vector<std::int64_t> shape, DType dtype)
      : shape_(std::move(shape)), dtype_(dtype) {
    validateShape(shape_);
  }

  const std::vector<std::int64_t> &shape() const { return shape_; }

  std::int64_t rank() const { return static_cast<std::int64_t>(shape_.size()); }

  DType dtype() const { return dtype_; }

  bool isScalar() const { return shape_.empty(); }

  bool operator==(const TensorType &) const = default;

  static void validateShape(const std::vector<std::int64_t> &shape) {
    for (std::int64_t dim : shape) {
      if (dim <= 0) {
        throw std::invalid_argument("tensor dimensions must be positive");
      }
    }
  }

private:
  std::vector<std::int64_t> shape_;
  DType dtype_ = DType::F32;
};

} // namespace tinytensor

#endif // TINYTENSOR_FRONTEND_TENSORTYPE_H
