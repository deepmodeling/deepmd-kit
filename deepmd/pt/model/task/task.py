# SPDX-License-Identifier: LGPL-3.0-or-later
import torch


class TaskBaseMethod(torch.nn.Module):
    def __init__(self, **kwargs):
        """Construct a basic head for different tasks."""
        super().__init__()

    def forward(self, **kwargs):
        """Task Output."""
        raise NotImplementedError
