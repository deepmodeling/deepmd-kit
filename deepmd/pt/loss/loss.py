# SPDX-License-Identifier: LGPL-3.0-or-later
import torch


class TaskLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        """Construct loss."""
        super().__init__()

    def forward(self, model_pred, label, natoms, learning_rate):
        """Return loss ."""
        raise NotImplementedError
