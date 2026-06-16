add_executable(tt-opt tt-opt.cpp)

# Link against MLIR's MlirOptMain and standard IR libraries
target_link_libraries(tt-opt PRIVATE
  MLIRMlirOptMain
  MLIRIR
  MLIRSupport
  MLIRParser
  MLIRTransforms
  # Add other standard dialects as needed
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRControlFlowDialect
)

# Ensure proper RPATH so the binary can find LLVM/MLIR shared libraries at runtime
set_target_properties(tt-opt PROPERTIES
  INSTALL_RPATH "${LLVM_LIBRARY_DIRS}"
  BUILD_WITH_INSTALL_RPATH ON
)
