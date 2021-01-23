#include <iostream>
#include "tensorflow/core/public/version.h"
int main(int argc, char * argv[])
{
  std::cout << tf_cxx11_abi_flag();
  return 0;
}
