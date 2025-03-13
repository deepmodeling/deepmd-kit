# SPDX-License-Identifier: LGPL-3.0-or-later

import logging

import torch

from deepmd.pt.model.task.denoise import (
    DenoiseNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)

log = logging.getLogger(__name__)


class DPDenoiseAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        if not isinstance(fitting, DenoiseNet):
            raise TypeError(
                "fitting must be an instance of DenoiseNet for DPDenoiseAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def apply_out_stat(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ):
        # Scale values to appropriate magnitudes
        noise_type = self.fitting_net.get_noise_type()
        cell_std = self.fitting_net.get_cell_pert_fraction() / 1.732
        if noise_type == "gaussian":
            coord_std = self.fitting_net.get_coord_noise()
        elif noise_type == "uniform":
            coord_std = self.fitting_net.get_coord_noise() / 1.732
        else:
            raise RuntimeError(f"Unknown noise type {noise_type}")
        ret["strain_components"] = (
            ret["strain_components"] * cell_std
            if cell_std > 0
            else ret["strain_components"]
        )
        ret["updated_coord"] = (
            ret["updated_coord"] * coord_std if coord_std > 0 else ret["updated_coord"]
        )
        return ret
