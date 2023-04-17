#include "multiphase_lbm.h"

//! Main function...
int main(int argc, char const* argv[]) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " commandFile" << std::endl;
    return 0;
  }

  MultiphaseLBM box;
  std::cout << "Opening file : " << argv[1] << std::endl;
  box.read_data(argv[1]);
  box.init();
  box.run_gpu();

  return 0;
}
