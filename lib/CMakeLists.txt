add_library(tinytensor_frontend INTERFACE)
target_link_libraries(tinytensor_frontend INTERFACE tinytensor_support)
target_include_directories(tinytensor_frontend INTERFACE
  ${CMAKE_CURRENT_BINARY_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if(TINYTENSOR_HAS_MLIR)
  add_library(tinytensor_ttdialect INTERFACE)
  target_link_libraries(tinytensor_ttdialect INTERFACE
    MLIRIR
    MLIRSupport
  )
  target_include_directories(tinytensor_ttdialect INTERFACE
    ${CMAKE_CURRENT_BINARY_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
  )

  add_library(tinytensor_compiler INTERFACE)
  target_link_libraries(tinytensor_compiler INTERFACE
    tinytensor_frontend
    tinytensor_ttdialect
    MLIRFuncDialect
    MLIRIR
    MLIRSupport
  )
  target_include_directories(tinytensor_compiler INTERFACE
    ${CMAKE_CURRENT_BINARY_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
  )
endif()
