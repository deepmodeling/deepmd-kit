#include <iostream>
#include "tensorflow/core/public/version.h"
int main(int argc, char * argv[])
{
#if (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION>=9) || TF_MAJOR_VERSION > 2 
#include "tensorflow/core/util/version_info.h"
  std::cout << TF_CXX11_ABI_FLAG;
#else
  std::cout << tf_cxx11_abi_flag();
#endif
  return 0;
}
