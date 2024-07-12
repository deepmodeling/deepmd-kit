// SPDX-License-Identifier: LGPL-3.0-or-later
#include <stdio.h>
#include <stdlib.h>

#include "deepmd/c_api.h"

int main() {
  const char* model = "graph.pb";
  double coord[] = {1., 0., 0., 0., 0., 1.5, 1., 0., 3.};
  double cell[] = {10., 0., 0., 0., 10., 0., 0., 0., 10.};
  int atype[] = {1, 0, 1};
  // init C pointers with given memory
  double* e = malloc(sizeof(*e));
  double* f = malloc(sizeof(*f) * 9);  // natoms * 3
  double* v = malloc(sizeof(*v) * 9);
  double* ae = malloc(sizeof(*ae) * 9);   // natoms
  double* av = malloc(sizeof(*av) * 27);  // natoms * 9
  // DP model
  DP_DeepPot* dp = DP_NewDeepPot(model);
  DP_DeepPotCompute(dp, 3, coord, atype, cell, e, f, v, ae, av);
  // print results
  printf("energy: %f\n", *e);
  for (int ii = 0; ii < 9; ++ii) {
    printf("force[%d]: %f\n", ii, f[ii]);
  }
  for (int ii = 0; ii < 9; ++ii) {
    printf("force[%d]: %f\n", ii, v[ii]);
  }
  // free memory
  free(e);
  free(f);
  free(v);
  free(ae);
  free(av);
  DP_DeleteDeepPot(dp);
}
