#ifndef TINYTENSOR_ADT_DENSEMAP_H
#define TINYTENSOR_ADT_DENSEMAP_H

#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <new>
#include <utility>
#include <type_traits>
#include "tinytensor/adt/StringRef.h"

namespace tinytensor {
namespace adt {

// Key traits template
template <typename T>
struct DenseMapInfo;

// Specialization for int
template <>
struct DenseMapInfo<int> {
  static inline int getEmptyKey() { return -1; }
  static inline int getTombstoneKey() { return -2; }
  static unsigned getHashValue(const int &Val) {
    return static_cast<unsigned>(Val * 37U);
  }
  static bool isEqual(const int &LHS, const int &RHS) {
    return LHS == RHS;
  }
};

// Specialization for unsigned
template <>
struct DenseMapInfo<unsigned> {
  static inline unsigned getEmptyKey() { return ~0U; }
  static inline unsigned getTombstoneKey() { return ~0U - 1; }
  static unsigned getHashValue(const unsigned &Val) {
    return Val * 37U;
  }
  static bool isEqual(const unsigned &LHS, const unsigned &RHS) {
    return LHS == RHS;
  }
};

// Specialization for long long
template <>
struct DenseMapInfo<long long> {
  static inline long long getEmptyKey() { return -1LL; }
  static inline long long getTombstoneKey() { return -2LL; }
  static unsigned getHashValue(const long long &Val) {
    return static_cast<unsigned>(Val * 37ULL);
  }
  static bool isEqual(const long long &LHS, const long long &RHS) {
    return LHS == RHS;
  }
};

// Specialization for unsigned long long
template <>
struct DenseMapInfo<unsigned long long> {
  static inline unsigned long long getEmptyKey() { return ~0ULL; }
  static inline unsigned long long getTombstoneKey() { return ~0ULL - 1; }
  static unsigned getHashValue(const unsigned long long &Val) {
    return static_cast<unsigned>(Val * 37ULL);
  }
  static bool isEqual(const unsigned long long &LHS, const unsigned long long &RHS) {
    return LHS == RHS;
  }
};

// Specialization for pointers
template <typename T>
struct DenseMapInfo<T*> {
  static inline T* getEmptyKey() {
    return reinterpret_cast<T*>(-1LL);
  }
  static inline T* getTombstoneKey() {
    return reinterpret_cast<T*>(-2LL);
  }
  static unsigned getHashValue(const T* const &Val) {
    return static_cast<unsigned>(reinterpret_cast<uintptr_t>(Val) >> 4);
  }
  static bool isEqual(const T* const &LHS, const T* const &RHS) {
    return LHS == RHS;
  }
};

// Specialization for StringRef
template <>
struct DenseMapInfo<StringRef> {
  static inline StringRef getEmptyKey() {
    return StringRef(reinterpret_cast<const char *>(-1LL), 0);
  }
  static inline StringRef getTombstoneKey() {
    return StringRef(reinterpret_cast<const char *>(-2LL), 0);
  }
  static unsigned getHashValue(const StringRef &Val) {
    unsigned Hash = 2166136261U;
    for (size_t i = 0; i < Val.size(); ++i) {
      Hash ^= static_cast<unsigned char>(Val[i]);
      Hash *= 16777619U;
    }
    return Hash;
  }
  static bool isEqual(const StringRef &LHS, const StringRef &RHS) {
    if (LHS.data() == getEmptyKey().data())
      return RHS.data() == getEmptyKey().data();
    if (LHS.data() == getTombstoneKey().data())
      return RHS.data() == getTombstoneKey().data();
    if (RHS.data() == getEmptyKey().data() || RHS.data() == getTombstoneKey().data())
      return false;
    return LHS == RHS;
  }
};

// DenseMap Iterator
template <typename KeyT, typename ValueT, typename KeyInfoT, bool IsConst>
class DenseMapIterator {
  using BucketT = std::pair<KeyT, ValueT>;
  using PtrT = std::conditional_t<IsConst, const BucketT*, BucketT*>;
  using RefT = std::conditional_t<IsConst, const BucketT&, BucketT&>;

  PtrT Current = nullptr;
  PtrT End = nullptr;

public:
  DenseMapIterator() = default;
  DenseMapIterator(PtrT Current, PtrT End) : Current(Current), End(End) {
    AdvanceToNext();
  }

  RefT operator*() const { return *Current; }
  PtrT operator->() const { return Current; }

  bool operator==(const DenseMapIterator &RHS) const { return Current == RHS.Current; }
  bool operator!=(const DenseMapIterator &RHS) const { return Current != RHS.Current; }

  DenseMapIterator &operator++() {
    assert(Current != End && "Advancing past end");
    ++Current;
    AdvanceToNext();
    return *this;
  }

  DenseMapIterator operator++(int) {
    DenseMapIterator Tmp = *this;
    ++*this;
    return Tmp;
  }

private:
  void AdvanceToNext() {
    while (Current != End &&
           (KeyInfoT::isEqual(Current->first, KeyInfoT::getEmptyKey()) ||
            KeyInfoT::isEqual(Current->first, KeyInfoT::getTombstoneKey()))) {
      ++Current;
    }
  }
};

template <typename KeyT, typename ValueT, typename KeyInfoT = DenseMapInfo<KeyT>>
class DenseMap {
public:
  using BucketT = std::pair<KeyT, ValueT>;
  using iterator = DenseMapIterator<KeyT, ValueT, KeyInfoT, false>;
  using const_iterator = DenseMapIterator<KeyT, ValueT, KeyInfoT, true>;

private:
  BucketT *Buckets = nullptr;
  unsigned NumEntries = 0;
  unsigned NumTombstones = 0;
  unsigned NumBuckets = 0;

public:
  DenseMap() = default;

  // Copy Constructor
  DenseMap(const DenseMap &RHS) {
    if (RHS.NumBuckets == 0) return;
    grow(RHS.NumBuckets);
    for (auto &KV : RHS) {
      insert(KV);
    }
  }

  // Move Constructor
  DenseMap(DenseMap &&RHS) noexcept {
    Buckets = RHS.Buckets;
    NumBuckets = RHS.NumBuckets;
    NumEntries = RHS.NumEntries;
    NumTombstones = RHS.NumTombstones;

    RHS.Buckets = nullptr;
    RHS.NumBuckets = 0;
    RHS.NumEntries = 0;
    RHS.NumTombstones = 0;
  }

  ~DenseMap() {
    clear();
  }

  // Copy Assignment
  DenseMap &operator=(const DenseMap &RHS) {
    if (this == &RHS) return *this;
    clear();
    if (RHS.NumBuckets == 0) return *this;
    grow(RHS.NumBuckets);
    for (auto &KV : RHS) {
      insert(KV);
    }
    return *this;
  }

  // Move Assignment
  DenseMap &operator=(DenseMap &&RHS) noexcept {
    if (this == &RHS) return *this;
    clear();
    Buckets = RHS.Buckets;
    NumBuckets = RHS.NumBuckets;
    NumEntries = RHS.NumEntries;
    NumTombstones = RHS.NumTombstones;

    RHS.Buckets = nullptr;
    RHS.NumBuckets = 0;
    RHS.NumEntries = 0;
    RHS.NumTombstones = 0;
    return *this;
  }

  // Iterators
  iterator begin() { return iterator(Buckets, Buckets + NumBuckets); }
  const_iterator begin() const { return const_iterator(Buckets, Buckets + NumBuckets); }
  iterator end() { return iterator(Buckets + NumBuckets, Buckets + NumBuckets); }
  const_iterator end() const { return const_iterator(Buckets + NumBuckets, Buckets + NumBuckets); }

  // Size details
  unsigned size() const { return NumEntries; }
  bool empty() const { return NumEntries == 0; }

  // Modifiers
  void clear() {
    if (NumBuckets == 0) return;
    for (unsigned i = 0; i < NumBuckets; ++i) {
      Buckets[i].first.~KeyT();
      Buckets[i].second.~ValueT();
    }
    std::free(Buckets);
    Buckets = nullptr;
    NumBuckets = 0;
    NumEntries = 0;
    NumTombstones = 0;
  }

  // Insert LValue
  std::pair<iterator, bool> insert(const std::pair<KeyT, ValueT> &KV) {
    if ((NumEntries + NumTombstones + 1) * 4 > NumBuckets * 3) {
      grow(NumBuckets ? NumBuckets * 2 : 4);
    }

    auto [B, Found] = find_bucket(KV.first);
    if (Found) {
      return {iterator(B, Buckets + NumBuckets), false};
    }

    if (KeyInfoT::isEqual(B->first, KeyInfoT::getTombstoneKey())) {
      --NumTombstones;
    }

    B->first = KV.first;
    B->second = KV.second;
    ++NumEntries;
    return {iterator(B, Buckets + NumBuckets), true};
  }

  // Insert RValue
  std::pair<iterator, bool> insert(std::pair<KeyT, ValueT> &&KV) {
    if ((NumEntries + NumTombstones + 1) * 4 > NumBuckets * 3) {
      grow(NumBuckets ? NumBuckets * 2 : 4);
    }

    auto [B, Found] = find_bucket(KV.first);
    if (Found) {
      return {iterator(B, Buckets + NumBuckets), false};
    }

    if (KeyInfoT::isEqual(B->first, KeyInfoT::getTombstoneKey())) {
      --NumTombstones;
    }

    B->first = std::move(KV.first);
    B->second = std::move(KV.second);
    ++NumEntries;
    return {iterator(B, Buckets + NumBuckets), true};
  }

  // Lookup / search
  iterator find(const KeyT &Key) {
    auto [B, Found] = find_bucket(Key);
    if (Found) return iterator(B, Buckets + NumBuckets);
    return end();
  }

  const_iterator find(const KeyT &Key) const {
    auto [B, Found] = find_bucket(Key);
    if (Found) return const_iterator(B, Buckets + NumBuckets);
    return end();
  }

  ValueT lookup(const KeyT &Key) const {
    auto [B, Found] = find_bucket(Key);
    if (Found) return B->second;
    return ValueT();
  }

  bool count(const KeyT &Key) const {
    return find_bucket(Key).second;
  }

  // Bracket access operator
  ValueT &operator[](const KeyT &Key) {
    if ((NumEntries + NumTombstones + 1) * 4 > NumBuckets * 3) {
      grow(NumBuckets ? NumBuckets * 2 : 4);
    }

    auto [B, Found] = find_bucket(Key);
    if (Found) {
      return B->second;
    }

    if (KeyInfoT::isEqual(B->first, KeyInfoT::getTombstoneKey())) {
      --NumTombstones;
    }

    B->first = Key;
    B->second = ValueT();
    ++NumEntries;
    return B->second;
  }

  // Deletion
  bool erase(const KeyT &Key) {
    auto [B, Found] = find_bucket(Key);
    if (!Found) return false;

    B->first = KeyInfoT::getTombstoneKey();
    B->second = ValueT();
    --NumEntries;
    ++NumTombstones;
    return true;
  }

private:
  // Find bucket helper
  std::pair<BucketT*, bool> find_bucket(const KeyT &Key) const {
    if (NumBuckets == 0) return {nullptr, false};

    unsigned H = KeyInfoT::getHashValue(Key);
    unsigned Mask = NumBuckets - 1;
    unsigned ProbeAmt = 1;
    unsigned Idx = H & Mask;

    BucketT *FirstTombstone = nullptr;

    while (true) {
      BucketT *B = &Buckets[Idx];
      if (KeyInfoT::isEqual(B->first, KeyInfoT::getEmptyKey())) {
        return {FirstTombstone ? FirstTombstone : B, false};
      }
      if (KeyInfoT::isEqual(B->first, KeyInfoT::getTombstoneKey())) {
        if (!FirstTombstone) FirstTombstone = B;
      } else if (KeyInfoT::isEqual(B->first, Key)) {
        return {B, true};
      }

      Idx = (Idx + ProbeAmt++) & Mask;
    }
  }

  // Grow method
  void grow(unsigned AtLeast) {
    unsigned NewNumBuckets = NumBuckets ? NumBuckets : 4;
    while (NewNumBuckets < AtLeast) {
      NewNumBuckets *= 2;
    }

    BucketT *OldBuckets = Buckets;
    unsigned OldNumBuckets = NumBuckets;

    // Allocate & initialize new table
    Buckets = static_cast<BucketT*>(std::malloc(NewNumBuckets * sizeof(BucketT)));
    assert(Buckets && "Memory allocation failed");
    NumBuckets = NewNumBuckets;
    NumEntries = 0;
    NumTombstones = 0;

    for (unsigned i = 0; i < NumBuckets; ++i) {
      new (&Buckets[i].first) KeyT(KeyInfoT::getEmptyKey());
      new (&Buckets[i].second) ValueT();
    }

    // Copy/move elements over
    for (unsigned i = 0; i < OldNumBuckets; ++i) {
      BucketT &OldB = OldBuckets[i];
      if (!KeyInfoT::isEqual(OldB.first, KeyInfoT::getEmptyKey()) &&
          !KeyInfoT::isEqual(OldB.first, KeyInfoT::getTombstoneKey())) {
        auto [B, Found] = find_bucket(OldB.first);
        assert(!Found && "Key should not exist in new table during grow");
        
        B->first = std::move(OldB.first);
        B->second = std::move(OldB.second);
        ++NumEntries;

        OldB.first.~KeyT();
        OldB.second.~ValueT();
      } else {
        OldB.first.~KeyT();
        OldB.second.~ValueT();
      }
    }

    if (OldBuckets) {
      std::free(OldBuckets);
    }
  }
};

} // namespace adt
} // namespace tinytensor

#endif // TINYTENSOR_ADT_DENSEMAP_H
