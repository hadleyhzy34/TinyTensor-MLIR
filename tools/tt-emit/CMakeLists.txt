add_executable(tt-emit tt-emit.cpp)

target_link_libraries(tt-emit PRIVATE
  tinytensor_frontend
  tinytensor_compiler
  tinytensor_ttdialect
  MLIRIR
  MLIRSupport
)

set_target_properties(tt-emit PROPERTIES
  INSTALL_RPATH "${LLVM_LIBRARY_DIRS}"
  BUILD_WITH_INSTALL_RPATH ON
)
