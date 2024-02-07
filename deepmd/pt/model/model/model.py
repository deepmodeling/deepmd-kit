# SPDX-License-Identifier: LGPL-3.0-or-later
import torch


class BaseModel(torch.nn.Module):
    def __init__(self):
        """Construct a basic model for different tasks."""
        super().__init__()
