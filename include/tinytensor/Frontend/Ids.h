#ifndef TINYTENSOR_FRONTEND_IDS_H
#define TINYTENSOR_FRONTEND_IDS_H

#include <cstdint>

namespace tinytensor {

struct ValueId {
  std::uint32_t value = 0;

  bool operator==(const ValueId &) const = default;
};

struct NodeId {
  std::uint32_t value = 0;

  bool operator==(const NodeId &) const = default;
};

} // namespace tinytensor

#endif // TINYTENSOR_FRONTEND_IDS_H
