# SPDX-License-Identifier: LGPL-3.0-or-later
import os

import torch

# maybe has more elegant way to get DEVICE
if os.environ.get("DEVICE") == "cpu" or torch.cuda.is_available() is False:
    TEST_DEVICE = "cpu"
else:
    TEST_DEVICE = "cuda"
