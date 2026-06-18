#ifndef TINYTENSOR_FRONTEND_TENSORSPEC_H
#define TINYTENSOR_FRONTEND_TENSORSPEC_H

#include "tinytensor/Frontend/DType.h"
#include "tinytensor/Frontend/TensorType.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tinytensor {

struct TensorSpec {
  std::vector<std::int64_t> shape;
  DType dtype = DType::F32;
  std::string name;

  TensorType type() const { return TensorType(shape, dtype); }
};

inline TensorSpec tensorSpec(std::vector<std::int64_t> shape, DType dtype,
                             std::string name = {}) {
  TensorType::validateShape(shape);
  return TensorSpec{std::move(shape), dtype, std::move(name)};
}

} // namespace tinytensor

#endif // TINYTENSOR_FRONTEND_TENSORSPEC_H
