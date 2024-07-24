# SPDX-License-Identifier: LGPL-3.0-or-later
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    TEST_DEVICE = "cpu"
else:
    TEST_DEVICE = "cuda"
