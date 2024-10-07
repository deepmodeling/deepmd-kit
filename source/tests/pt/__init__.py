# SPDX-License-Identifier: LGPL-3.0-or-later
import torch

from ..common import (
    TEST_DEVICE,
)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# testing purposes; device should always be set explicitly
torch.set_default_device("cuda:9999999")

# ensure GPU is used when testing GPU code
if TEST_DEVICE == "cuda":
    assert torch.cuda.is_available()
