#ifndef TINYTENSOR_ADT_STRINGREF_H
#define TINYTENSOR_ADT_STRINGREF_H

#include <string>
#include <string_view>
#include <ostream>
#include <cstring>
#include <cassert>
#include <algorithm>

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
  operator std::string_view() const {
    return Length == 0 ? std::string_view() : std::string_view(Data, Length);
  }

  // Iterators
  using const_iterator = const char *;
  using iterator = const char *;

  constexpr const_iterator begin() const { return Data; }
  constexpr const_iterator end() const { return Data ? Data + Length : Data; }

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
    if (Prefix.Length == 0)
      return true;
    return Length >= Prefix.Length &&
           std::string_view(Data, Prefix.Length) == std::string_view(Prefix.Data, Prefix.Length);
  }

  constexpr bool endswith(StringRef Suffix) const {
    if (Suffix.Length == 0)
      return true;
    return Length >= Suffix.Length &&
           std::string_view(Data + Length - Suffix.Length, Suffix.Length) ==
               std::string_view(Suffix.Data, Suffix.Length);
  }

  constexpr StringRef substr(size_t Start, size_t N = npos) const {
    assert(Start <= Length && "Start index out of bounds");
    return StringRef(Data ? Data + Start : Data, std::min(N, Length - Start));
  }

  constexpr int compare(StringRef RHS) const {
    size_t MinLen = std::min(Length, RHS.Length);
    if (MinLen > 0) {
      int Res = std::char_traits<char>::compare(Data, RHS.Data, MinLen);
      if (Res != 0)
        return Res;
    }
    if (Length < RHS.Length)
      return -1;
    if (Length > RHS.Length)
      return 1;
    return 0;
  }

  std::string str() const {
    return Length == 0 ? std::string() : std::string(Data, Length);
  }
};

// Comparison operators
inline bool operator==(StringRef LHS, StringRef RHS) {
  if (LHS.size() != RHS.size())
    return false;
  if (LHS.size() == 0)
    return true;
  return std::string_view(LHS.data(), LHS.size()) ==
         std::string_view(RHS.data(), RHS.size());
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
  if (!Str.empty())
    OS.write(Str.data(), Str.size());
  return OS;
}

} // namespace adt
} // namespace tinytensor

#endif // TINYTENSOR_ADT_STRINGREF_H
