#include "tinytensor/adt/StringRef.h"
#include <gtest/gtest.h>
#include <sstream>

using namespace tinytensor::adt;

TEST(StringRefTest, EmptyConstructor) {
  StringRef S;
  EXPECT_TRUE(S.empty());
  EXPECT_EQ(S.size(), 0U);
  EXPECT_EQ(S.data(), nullptr);
}

TEST(StringRefTest, CStrConstructor) {
  StringRef S("hello");
  EXPECT_FALSE(S.empty());
  EXPECT_EQ(S.size(), 5U);
  EXPECT_STREQ(S.data(), "hello");
}

TEST(StringRefTest, LengthConstructor) {
  StringRef S("hello world", 5);
  EXPECT_FALSE(S.empty());
  EXPECT_EQ(S.size(), 5U);
  EXPECT_EQ(S[0], 'h');
  EXPECT_EQ(S[4], 'o');
}

TEST(StringRefTest, StdStringConstructor) {
  std::string StdStr = "compiler";
  StringRef S(StdStr);
  EXPECT_EQ(S.size(), 8U);
  EXPECT_EQ(S.str(), "compiler");
}

TEST(StringRefTest, Iterators) {
  StringRef S("abc");
  auto It = S.begin();
  EXPECT_EQ(*It, 'a');
  ++It;
  EXPECT_EQ(*It, 'b');
  ++It;
  EXPECT_EQ(*It, 'c');
  ++It;
  EXPECT_EQ(It, S.end());
}

TEST(StringRefTest, Elements) {
  StringRef S("xyz");
  EXPECT_EQ(S.front(), 'x');
  EXPECT_EQ(S.back(), 'z');
  EXPECT_EQ(S[1], 'y');
}

TEST(StringRefTest, StartsWithEndsWith) {
  StringRef S("tinytensor");
  EXPECT_TRUE(S.startswith("tiny"));
  EXPECT_FALSE(S.startswith("tensor"));
  EXPECT_TRUE(S.endswith("tensor"));
  EXPECT_FALSE(S.endswith("tiny"));
}

TEST(StringRefTest, Substr) {
  StringRef S("tinytensor");
  EXPECT_EQ(S.substr(0, 4), StringRef("tiny"));
  EXPECT_EQ(S.substr(4), StringRef("tensor"));
  EXPECT_EQ(S.substr(10), StringRef(""));
}

TEST(StringRefTest, Compare) {
  StringRef A("abc");
  StringRef B("def");
  StringRef C("abc");
  EXPECT_LT(A.compare(B), 0);
  EXPECT_GT(B.compare(A), 0);
  EXPECT_EQ(A.compare(C), 0);

  EXPECT_TRUE(A == C);
  EXPECT_TRUE(A != B);
  EXPECT_TRUE(A < B);
  EXPECT_TRUE(B > A);
}

TEST(StringRefTest, StreamInsertion) {
  StringRef S("mlir-opt");
  std::stringstream SS;
  SS << S;
  EXPECT_EQ(SS.str(), "mlir-opt");
}

TEST(StringRefTest, EmptyOperationsAreSafe) {
  StringRef Empty;
  EXPECT_TRUE(Empty.startswith(""));
  EXPECT_TRUE(Empty.endswith(""));
  EXPECT_EQ(Empty.compare(""), 0);
  EXPECT_EQ(Empty.str(), "");

  std::stringstream SS;
  SS << Empty;
  EXPECT_EQ(SS.str(), "");
}
