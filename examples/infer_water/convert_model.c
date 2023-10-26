// SPDX-License-Identifier: LGPL-3.0-or-later
#include <stdio.h>
#include <stdlib.h>

#include "deepmd/c_api.h"

int main() {
    DP_ConvertPbtxtToPb("../../../source/tests/infer/deeppot.pbtxt",
                                        "graph.pb");
    return 0;
}
