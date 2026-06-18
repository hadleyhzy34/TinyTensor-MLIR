#include "tinytensor/Support/ADT/SmallVector.h"
#include <gtest/gtest.h>

using namespace tinytensor::adt;

TEST(SmallVectorTest, InitialState) {
  SmallVector<int, 4> V;
  EXPECT_TRUE(V.empty());
  EXPECT_EQ(V.size(), 0U);
  EXPECT_EQ(V.capacity(), 4U);
}

TEST(SmallVectorTest, InlineStorageUse) {
  SmallVector<int, 4> V;
  V.push_back(1);
  V.push_back(2);
  V.push_back(3);
  V.push_back(4);
  EXPECT_EQ(V.size(), 4U);
  EXPECT_EQ(V.capacity(), 4U);
  // Verify address is inside the vector object (inline storage)
  EXPECT_GE(V.data(), reinterpret_cast<int*>(&V));
  EXPECT_LT(V.data(), reinterpret_cast<int*>(reinterpret_cast<char*>(&V) + sizeof(V)));
}

TEST(SmallVectorTest, HeapSpill) {
  SmallVector<int, 4> V;
  for (int i = 0; i < 5; ++i) {
    V.push_back(i);
  }
  EXPECT_EQ(V.size(), 5U);
  EXPECT_GT(V.capacity(), 4U);
  // Verify address is outside the vector object (heap storage)
  EXPECT_TRUE(V.data() < reinterpret_cast<int*>(&V) ||
              V.data() >= reinterpret_cast<int*>(reinterpret_cast<char*>(&V) + sizeof(V)));
}

TEST(SmallVectorTest, Accessors) {
  SmallVector<int, 4> V{10, 20, 30};
  EXPECT_EQ(V.front(), 10);
  EXPECT_EQ(V.back(), 30);
  EXPECT_EQ(V[1], 20);
}

TEST(SmallVectorTest, PopBack) {
  SmallVector<int, 4> V{1, 2};
  V.pop_back();
  EXPECT_EQ(V.size(), 1U);
  EXPECT_EQ(V.back(), 1);
}

TEST(SmallVectorTest, ResizeReserve) {
  SmallVector<int, 2> V;
  V.reserve(10);
  EXPECT_GE(V.capacity(), 10U);
  EXPECT_TRUE(V.empty());

  V.resize(5, 42);
  EXPECT_EQ(V.size(), 5U);
  EXPECT_EQ(V[0], 42);
  EXPECT_EQ(V[4], 42);

  V.resize(2);
  EXPECT_EQ(V.size(), 2U);
}

// Track object construction and destruction
struct LifecycleTracker {
  static int Copies;
  static int Moves;
  static int Destructors;

  int Value;

  LifecycleTracker(int V = 0) : Value(V) {}
  LifecycleTracker(const LifecycleTracker &O) : Value(O.Value) { ++Copies; }
  LifecycleTracker(LifecycleTracker &&O) noexcept : Value(O.Value) { ++Moves; }
  ~LifecycleTracker() { ++Destructors; }

  LifecycleTracker &operator=(const LifecycleTracker &O) {
    Value = O.Value;
    ++Copies;
    return *this;
  }
  LifecycleTracker &operator=(LifecycleTracker &&O) noexcept {
    Value = O.Value;
    ++Moves;
    return *this;
  }

  static void Reset() {
    Copies = 0;
    Moves = 0;
    Destructors = 0;
  }
};

int LifecycleTracker::Copies = 0;
int LifecycleTracker::Moves = 0;
int LifecycleTracker::Destructors = 0;

TEST(SmallVectorTest, ObjectLifecycles) {
  LifecycleTracker::Reset();
  {
    SmallVector<LifecycleTracker, 2> V;
    V.push_back(LifecycleTracker(1));
    V.push_back(LifecycleTracker(2));
    // Trigger spill to heap
    V.push_back(LifecycleTracker(3));
  }
  // All elements constructed (3 temporaries + moves) must be destructed
  EXPECT_EQ(LifecycleTracker::Destructors, LifecycleTracker::Copies + LifecycleTracker::Moves + 3);
}

TEST(SmallVectorTest, CopyAndMove) {
  SmallVector<int, 2> V1{1, 2, 3}; // Dynamic
  SmallVector<int, 2> V2(V1); // Copy
  EXPECT_EQ(V2.size(), 3U);
  EXPECT_EQ(V2[0], 1);

  SmallVector<int, 2> V3(std::move(V1)); // Move
  EXPECT_EQ(V3.size(), 3U);
  EXPECT_TRUE(V1.empty()); // V1 should have been reset

  SmallVector<int, 4> V4;
  V4 = V3; // Copy Assign
  EXPECT_EQ(V4.size(), 3U);
  EXPECT_EQ(V4[2], 3);

  SmallVector<int, 4> V5;
  V5 = std::move(V3); // Move Assign
  EXPECT_EQ(V5.size(), 3U);
  EXPECT_TRUE(V3.empty());
}
