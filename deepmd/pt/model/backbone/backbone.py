# SPDX-License-Identifier: LGPL-3.0-or-later
import torch


class BackBone(torch.nn.Module):
    def __init__(self, **kwargs):
        """BackBone base method."""
        super().__init__()

    def forward(self, **kwargs):
        """Calculate backBone."""
        raise NotImplementedError
