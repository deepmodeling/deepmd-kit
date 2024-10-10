# SPDX-License-Identifier: LGPL-3.0-or-later
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    TEST_DEVICE = "cpu"
else:
    TEST_DEVICE = "cuda"

# see https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/store-information-in-variables#default-environment-variables
CI = os.environ.get == "true"
