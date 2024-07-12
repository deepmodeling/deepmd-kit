# SPDX-License-Identifier: LGPL-3.0-or-later
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# testing purposes; device should always be set explicitly
torch.set_default_device("cuda:9999999")
