#ifndef TINYTENSOR_ADT_STRINGREF_H
#define TINYTENSOR_ADT_STRINGREF_H

#include <string>
#include <string_view>
#include <ostream>
#include <cstring>
#include <cassert>

namespace tinytensor {
namespace adt {

class StringRef {
public:
  static constexpr size_t npos = -1ULL;

private:
  const char *Data = nullptr;
  size_t Length = 0;

public:
  // Constructors
  constexpr StringRef() = default;
  
  constexpr StringRef(std::nullptr_t) = delete;

  constexpr StringRef(const char *Str)
      : Data(Str), Length(Str ? std::char_traits<char>::length(Str) : 0) {}

  constexpr StringRef(const char *Str, size_t Length)
      : Data(Str), Length(Length) {}

  StringRef(const std::string &Str)
      : Data(Str.data()), Length(Str.length()) {}

  // Implicit conversion from std::string_view
  constexpr StringRef(std::string_view SV)
      : Data(SV.data()), Length(SV.size()) {}

  // Conversions
  operator std::string_view() const { return std::string_view(Data, Length); }

  // Iterators
  using const_iterator = const char *;
  using iterator = const char *;

  constexpr const_iterator begin() const { return Data; }
  constexpr const_iterator end() const { return Data + Length; }

  // Accessors
  constexpr const char *data() const { return Data; }
  constexpr size_t size() const { return Length; }
  constexpr bool empty() const { return Length == 0; }

  constexpr char operator[](size_t Index) const {
    assert(Index < Length && "Index out of bounds");
    return Data[Index];
  }

  constexpr char front() const {
    assert(Length > 0 && "StringRef is empty");
    return Data[0];
  }

  constexpr char back() const {
    assert(Length > 0 && "StringRef is empty");
    return Data[Length - 1];
  }

  // Operations
  constexpr bool startswith(StringRef Prefix) const {
    return Length >= Prefix.Length &&
           std::string_view(Data, Prefix.Length) == std::string_view(Prefix.Data, Prefix.Length);
  }

  constexpr bool endswith(StringRef Suffix) const {
    return Length >= Suffix.Length &&
           std::string_view(Data + Length - Suffix.Length, Suffix.Length) ==
               std::string_view(Suffix.Data, Suffix.Length);
  }

  constexpr StringRef substr(size_t Start, size_t N = npos) const {
    assert(Start <= Length && "Start index out of bounds");
    return StringRef(Data + Start, std::min(N, Length - Start));
  }

  constexpr int compare(StringRef RHS) const {
    size_t MinLen = std::min(Length, RHS.Length);
    if (int Res = std::string_view(Data, MinLen).compare(std::string_view(RHS.Data, MinLen)))
      return Res;
    if (Length < RHS.Length)
      return -1;
    if (Length > RHS.Length)
      return 1;
    return 0;
  }

  std::string str() const { return std::string(Data, Length); }
};

// Comparison operators
inline bool operator==(StringRef LHS, StringRef RHS) {
  return LHS.size() == RHS.size() &&
         std::string_view(LHS.data(), LHS.size()) == std::string_view(RHS.data(), RHS.size());
}

inline bool operator!=(StringRef LHS, StringRef RHS) { return !(LHS == RHS); }

inline bool operator<(StringRef LHS, StringRef RHS) {
  return LHS.compare(RHS) < 0;
}

inline bool operator<=(StringRef LHS, StringRef RHS) {
  return LHS.compare(RHS) <= 0;
}

inline bool operator>(StringRef LHS, StringRef RHS) {
  return LHS.compare(RHS) > 0;
}

inline bool operator>=(StringRef LHS, StringRef RHS) {
  return LHS.compare(RHS) >= 0;
}

inline std::ostream &operator<<(std::ostream &OS, StringRef Str) {
  OS.write(Str.data(), Str.size());
  return OS;
}

} // namespace adt
} // namespace tinytensor

#endif // TINYTENSOR_ADT_STRINGREF_H
