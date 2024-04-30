// SPDX-License-Identifier: LGPL-3.0-or-later
#include <torch/torch.h>

#include <iostream>

torch::Tensor enable_mpi() {
#ifdef USE_MPI
  return torch::ones({1}, torch::kBool);
#else
  return torch::zeros({1}, torch::kBool);
#endif
}

TORCH_LIBRARY(deepmd, m) { m.def("enable_mpi", enable_mpi); }
