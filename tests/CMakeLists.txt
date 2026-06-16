add_executable(tinytensor-tests
  adt/test_stringref.cpp
  adt/test_smallvector.cpp
  adt/test_densemap.cpp
  adt/test_compiler_usecases.cpp
)

# Link against the system GoogleTest libraries
target_link_libraries(tinytensor-tests PRIVATE
  GTest::gtest
  GTest::gtest_main
)

# Register tests with CTest
add_test(NAME TinyTensorTests COMMAND tinytensor-tests)
