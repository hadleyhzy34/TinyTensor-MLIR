#include "tinytensor/Support/ADT/DenseMap.h"
#include <gtest/gtest.h>
#include <memory>

using namespace tinytensor::adt;

TEST(DenseMapTest, InitialState) {
  DenseMap<int, std::string> Map;
  EXPECT_TRUE(Map.empty());
  EXPECT_EQ(Map.size(), 0U);
}

TEST(DenseMapTest, InsertAndLookup) {
  DenseMap<int, std::string> Map;
  auto [It1, Inserted1] = Map.insert({42, "hello"});
  EXPECT_TRUE(Inserted1);
  EXPECT_EQ(It1->first, 42);
  EXPECT_EQ(It1->second, "hello");

  auto [It2, Inserted2] = Map.insert({42, "world"});
  EXPECT_FALSE(Inserted2);
  EXPECT_EQ(It2->second, "hello"); // Should not overwrite

  EXPECT_EQ(Map.lookup(42), "hello");
  EXPECT_EQ(Map.lookup(99), ""); // Default constructed value
}

TEST(DenseMapTest, BracketOperator) {
  DenseMap<int, int> Map;
  Map[1] = 100;
  Map[2] = 200;
  EXPECT_EQ(Map[1], 100);
  EXPECT_EQ(Map[2], 200);
  EXPECT_EQ(Map.size(), 2U);

  // Read non-existent key
  EXPECT_EQ(Map[3], 0);
  EXPECT_EQ(Map.size(), 3U);
}

TEST(DenseMapTest, FindAndCount) {
  DenseMap<int, double> Map;
  Map.insert({10, 1.0});
  Map.insert({20, 2.0});

  auto It = Map.find(10);
  EXPECT_NE(It, Map.end());
  EXPECT_DOUBLE_EQ(It->second, 1.0);

  auto ItEnd = Map.find(99);
  EXPECT_EQ(ItEnd, Map.end());

  EXPECT_TRUE(Map.count(20));
  EXPECT_FALSE(Map.count(99));
}

TEST(DenseMapTest, EraseAndTombstone) {
  DenseMap<int, int> Map;
  Map[1] = 10;
  Map[2] = 20;
  Map[3] = 30;

  EXPECT_TRUE(Map.erase(2));
  EXPECT_FALSE(Map.erase(2)); // Already deleted
  EXPECT_EQ(Map.size(), 2U);

  // We should not find 2
  EXPECT_EQ(Map.find(2), Map.end());
  EXPECT_FALSE(Map.count(2));

  // Insert again in same or different slot (tombstone recycling)
  Map[2] = 40;
  EXPECT_EQ(Map.size(), 3U);
  EXPECT_EQ(Map[2], 40);
}

TEST(DenseMapTest, GrowthAndRehash) {
  DenseMap<int, int> Map;
  // Insert many values to trigger growth multiple times
  for (int i = 0; i < 100; ++i) {
    Map[i] = i * 10;
  }

  EXPECT_EQ(Map.size(), 100U);
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(Map.lookup(i), i * 10);
  }
}

TEST(DenseMapTest, StringRefKeys) {
  DenseMap<StringRef, int> Map;
  Map["apple"] = 1;
  Map["banana"] = 2;

  EXPECT_EQ(Map["apple"], 1);
  EXPECT_EQ(Map["banana"], 2);
  EXPECT_TRUE(Map.count("apple"));
  EXPECT_FALSE(Map.count("cherry"));
}

TEST(DenseMapTest, IteratorTraverse) {
  DenseMap<int, int> Map;
  Map[10] = 1;
  Map[20] = 2;
  Map[30] = 3;
  Map.erase(20);

  int Sum = 0;
  int KeySum = 0;
  for (auto &KV : Map) {
    KeySum += KV.first;
    Sum += KV.second;
  }

  EXPECT_EQ(KeySum, 40); // 10 + 30 (20 deleted)
  EXPECT_EQ(Sum, 4);     // 1 + 3
}

TEST(DenseMapTest, CopyAndMove) {
  DenseMap<int, int> Map1;
  Map1[1] = 10;
  Map1[2] = 20;

  DenseMap<int, int> Map2(Map1);
  EXPECT_EQ(Map2.size(), 2U);
  EXPECT_EQ(Map2[1], 10);

  DenseMap<int, int> Map3(std::move(Map1));
  EXPECT_EQ(Map3.size(), 2U);
  EXPECT_TRUE(Map1.empty());

  DenseMap<int, int> Map4;
  Map4 = Map2;
  EXPECT_EQ(Map4.size(), 2U);

  DenseMap<int, int> Map5;
  Map5 = std::move(Map3);
  EXPECT_EQ(Map5.size(), 2U);
  EXPECT_TRUE(Map3.empty());
}

TEST(DenseMapTest, NonTrivialValueEraseReinsertAndGrow) {
  DenseMap<int, std::unique_ptr<int>> Map;
  Map[1] = std::make_unique<int>(10);
  Map[2] = std::make_unique<int>(20);

  EXPECT_TRUE(Map.erase(1));
  EXPECT_FALSE(Map.count(1));

  Map[1] = std::make_unique<int>(30);
  EXPECT_EQ(*Map[1], 30);

  for (int i = 3; i < 80; ++i) {
    Map[i] = std::make_unique<int>(i);
  }

  EXPECT_EQ(Map.size(), 79U);
  EXPECT_EQ(*Map[2], 20);
  EXPECT_EQ(*Map[79], 79);
}
