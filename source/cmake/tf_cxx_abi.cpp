#include <iostream>
#include "tensorflow/core/public/version.h"
int main(int argc, char * argv[])
{
#if (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION>=9) || TF_MAJOR_VERSION > 2 
#ifdef _GLIBCXX_USE_CXX11_ABI
  std::cout << _GLIBCXX_USE_CXX11_ABI;
#else
  std::cout << 0;
#endif
#else
  std::cout << tf_cxx11_abi_flag();
#endif
  return 0;
}
