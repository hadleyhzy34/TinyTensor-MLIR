#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  
  // Register all MLIR built-in dialects and passes
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  // Run the optimizer driver
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TinyTensor Optimizer Driver\n", registry));
}
