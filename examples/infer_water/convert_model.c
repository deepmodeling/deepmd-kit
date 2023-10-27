// SPDX-License-Identifier: LGPL-3.0-or-later
#include "deepmd/c_api.h"

int main() {
  DP_ConvertPbtxtToPb("../../source/tests/infer/deeppot.pbtxt", "graph.pb");
  return 0;
}
