#ifndef TINYTENSOR_SUPPORT_ADT_SMALLVECTOR_H
#define TINYTENSOR_SUPPORT_ADT_SMALLVECTOR_H

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <utility>
#include <algorithm>
#include <initializer_list>

namespace tinytensor {
namespace adt {

class SmallVectorBase {
protected:
  void *BeginX;
  unsigned Size = 0;
  unsigned Capacity = 0;
  void *InlineBuffer = nullptr;
  unsigned InlineCapacity = 0;

  SmallVectorBase(void *FirstEl, unsigned Capacity)
      : BeginX(FirstEl), Capacity(Capacity), InlineBuffer(FirstEl), InlineCapacity(Capacity) {}
};

template <typename T>
class SmallVectorImpl : public SmallVectorBase {
public:
  using value_type = T;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;
  using iterator = T *;
  using const_iterator = const T *;

protected:
  SmallVectorImpl(void *FirstEl, unsigned Capacity)
      : SmallVectorBase(FirstEl, Capacity) {}

public:
  ~SmallVectorImpl() {
    clear();
    if (BeginX != InlineBuffer) {
      std::free(BeginX);
    }
  }

  // Copy assignment
  SmallVectorImpl &operator=(const SmallVectorImpl &RHS) {
    if (this == &RHS) return *this;

    clear();
    reserve(RHS.size());
    for (const auto &Val : RHS) {
      push_back(Val);
    }
    return *this;
  }

  // Move assignment
  SmallVectorImpl &operator=(SmallVectorImpl &&RHS) {
    if (this == &RHS) return *this;

    clear();
    if (RHS.BeginX != RHS.InlineBuffer) {
      // RHS is dynamic, we can steal its heap buffer
      if (BeginX != InlineBuffer) {
        std::free(BeginX);
      }
      BeginX = RHS.BeginX;
      Size = RHS.Size;
      Capacity = RHS.Capacity;

      // Reset RHS to empty inline state
      RHS.BeginX = RHS.InlineBuffer;
      RHS.Size = 0;
      RHS.Capacity = RHS.InlineCapacity;
    } else {
      // RHS is inline, we must move elements
      reserve(RHS.size());
      for (auto &Val : RHS) {
        push_back(std::move(Val));
      }
      RHS.clear();
    }
    return *this;
  }

  // Iterators
  iterator begin() { return static_cast<iterator>(BeginX); }
  const_iterator begin() const { return static_cast<const_iterator>(BeginX); }
  iterator end() { return begin() + Size; }
  const_iterator end() const { return begin() + Size; }

  // Size details
  size_type size() const { return Size; }
  size_type capacity() const { return Capacity; }
  bool empty() const { return Size == 0; }

  // Accessors
  reference operator[](size_type Index) {
    assert(Index < Size && "Index out of bounds");
    return begin()[Index];
  }

  const_reference operator[](size_type Index) const {
    assert(Index < Size && "Index out of bounds");
    return begin()[Index];
  }

  reference front() {
    assert(Size > 0 && "Vector is empty");
    return begin()[0];
  }

  const_reference front() const {
    assert(Size > 0 && "Vector is empty");
    return begin()[0];
  }

  reference back() {
    assert(Size > 0 && "Vector is empty");
    return begin()[Size - 1];
  }

  const_reference back() const {
    assert(Size > 0 && "Vector is empty");
    return begin()[Size - 1];
  }

  pointer data() { return begin(); }
  const_pointer data() const { return begin(); }

  // Modifiers
  void clear() {
    for (size_t i = 0; i < Size; ++i) {
      begin()[i].~T();
    }
    Size = 0;
  }

  void push_back(const T &Val) {
    if (Size >= Capacity) {
      grow();
    }
    new (static_cast<void*>(begin() + Size)) T(Val);
    ++Size;
  }

  void push_back(T &&Val) {
    if (Size >= Capacity) {
      grow();
    }
    new (static_cast<void*>(begin() + Size)) T(std::move(Val));
    ++Size;
  }

  template <typename... Args>
  reference emplace_back(Args&&... args) {
    if (Size >= Capacity) {
      grow();
    }
    T *Ptr = begin() + Size;
    new (static_cast<void*>(Ptr)) T(std::forward<Args>(args)...);
    ++Size;
    return *Ptr;
  }

  void pop_back() {
    assert(Size > 0 && "Vector is empty");
    --Size;
    begin()[Size].~T();
  }

  void reserve(size_type N) {
    if (N > Capacity) {
      grow(N);
    }
  }

  void resize(size_type N) {
    if (N < Size) {
      while (Size > N) {
        pop_back();
      }
    } else if (N > Size) {
      reserve(N);
      while (Size < N) {
        new (static_cast<void*>(begin() + Size)) T();
        ++Size;
      }
    }
  }

  void resize(size_type N, const T &Val) {
    if (N < Size) {
      while (Size > N) {
        pop_back();
      }
    } else if (N > Size) {
      reserve(N);
      while (Size < N) {
        new (static_cast<void*>(begin() + Size)) T(Val);
        ++Size;
      }
    }
  }

private:
  void grow(size_t MinCapacity = 0) {
    size_t NewCapacity = std::max(MinCapacity, size_t(Capacity) * 2);
    if (NewCapacity < 4) NewCapacity = 4;

    T *NewElts = static_cast<T*>(std::malloc(NewCapacity * sizeof(T)));
    assert(NewElts && "Memory allocation failed");
    T *OldElts = static_cast<T*>(BeginX);

    for (size_t i = 0; i < Size; ++i) {
      new (static_cast<void*>(NewElts + i)) T(std::move(OldElts[i]));
      OldElts[i].~T();
    }

    if (BeginX != InlineBuffer) {
      std::free(BeginX);
    }

    BeginX = NewElts;
    Capacity = NewCapacity;
  }
};

template <typename T, unsigned N>
class SmallVector : public SmallVectorImpl<T> {
  alignas(T) char InlineBufferMemory[N > 0 ? N * sizeof(T) : 1];

public:
  SmallVector() : SmallVectorImpl<T>(InlineBufferMemory, N) {}

  SmallVector(const SmallVector &RHS) : SmallVectorImpl<T>(InlineBufferMemory, N) {
    this->reserve(RHS.size());
    for (const auto &Val : RHS) {
      this->push_back(Val);
    }
  }

  SmallVector(SmallVector &&RHS) : SmallVectorImpl<T>(InlineBufferMemory, N) {
    *this = std::move(RHS);
  }

  template <typename InputIt>
  SmallVector(InputIt First, InputIt Last) : SmallVectorImpl<T>(InlineBufferMemory, N) {
    while (First != Last) {
      this->push_back(*First);
      ++First;
    }
  }

  SmallVector(std::initializer_list<T> IL) : SmallVectorImpl<T>(InlineBufferMemory, N) {
    this->reserve(IL.size());
    for (const auto &Val : IL) {
      this->push_back(Val);
    }
  }

  SmallVector &operator=(const SmallVector &RHS) {
    SmallVectorImpl<T>::operator=(RHS);
    return *this;
  }

  SmallVector &operator=(SmallVector &&RHS) {
    SmallVectorImpl<T>::operator=(std::move(RHS));
    return *this;
  }

  template <unsigned M>
  SmallVector(const SmallVector<T, M> &RHS) : SmallVectorImpl<T>(InlineBufferMemory, N) {
    this->reserve(RHS.size());
    for (const auto &Val : RHS) {
      this->push_back(Val);
    }
  }

  template <unsigned M>
  SmallVector(SmallVector<T, M> &&RHS) : SmallVectorImpl<T>(InlineBufferMemory, N) {
    *this = std::move(RHS);
  }

  template <unsigned M>
  SmallVector &operator=(const SmallVector<T, M> &RHS) {
    SmallVectorImpl<T>::operator=(RHS);
    return *this;
  }

  template <unsigned M>
  SmallVector &operator=(SmallVector<T, M> &&RHS) {
    SmallVectorImpl<T>::operator=(std::move(RHS));
    return *this;
  }
};

} // namespace adt
} // namespace tinytensor

#endif // TINYTENSOR_SUPPORT_ADT_SMALLVECTOR_H
