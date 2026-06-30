add_executable(tinytensor-tests
  Support/ADT/test_stringref.cpp
  Support/ADT/test_smallvector.cpp
  Support/ADT/test_densemap.cpp
  Support/ADT/test_compiler_usecases.cpp
  Frontend/test_types.cpp
  Frontend/test_graph.cpp
  Frontend/test_ops.cpp
  Frontend/test_constants.cpp
)

# Link against the system GoogleTest libraries
target_link_libraries(tinytensor-tests PRIVATE
  tinytensor_support
  tinytensor_frontend
  GTest::gtest
  GTest::gtest_main
)

# Register tests with CTest
add_test(NAME TinyTensorTests COMMAND tinytensor-tests)
