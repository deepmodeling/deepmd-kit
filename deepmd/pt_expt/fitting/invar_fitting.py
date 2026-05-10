# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import numpy as np
import torch

from deepmd.dpmodel.fitting.invar_fitting import InvarFitting as InvarFittingDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.pt_expt.utils.env import (
    DEVICE,
)


@BaseFitting.register("invar")
@torch_module
class InvarFitting(InvarFittingDP):
    def share_params(
        self,
        base_class: Any,
        shared_level: int,
        model_prob: float = 1.0,
        protection: float = 1e-2,
        resume: bool = False,
    ) -> None:
        """Share parameters with base_class for multi-task training.

        Level 0: share all sub-modules and buffers except bias_atom_e
        and case_embd.  When not resuming, fparam/aparam statistics are
        merged using probability-weighted averaging (matching PT).
        """
        assert self.__class__ == base_class.__class__, (
            "Only fitting nets of the same type can share params!"
        )
        if shared_level == 0:
            # --- weighted fparam stat merging ---
            if self.numb_fparam > 0:
                if not resume:
                    base_stats = base_class.get_param_stats().get("fparam", [])
                    self_stats = self.get_param_stats().get("fparam", [])
                    if base_stats and self_stats:
                        assert len(base_stats) == self.numb_fparam
                        merged = [
                            base_stats[ii] + self_stats[ii] * model_prob
                            for ii in range(self.numb_fparam)
                        ]
                        fparam_avg = np.array(
                            [s.compute_avg() for s in merged], dtype=np.float64
                        )
                        fparam_std = np.array(
                            [s.compute_std(protection=protection) for s in merged],
                            dtype=np.float64,
                        )
                        fparam_inv_std = 1.0 / fparam_std
                        base_class.fparam_avg.copy_(
                            torch.tensor(
                                fparam_avg,
                                device=DEVICE,
                                dtype=base_class.fparam_avg.dtype,
                            )
                        )
                        base_class.fparam_inv_std.copy_(
                            torch.tensor(
                                fparam_inv_std,
                                device=DEVICE,
                                dtype=base_class.fparam_inv_std.dtype,
                            )
                        )
                        # update stored stats so chained share_params works
                        base_class._param_stats["fparam"] = merged
                self._buffers["fparam_avg"] = base_class._buffers["fparam_avg"]
                self._buffers["fparam_inv_std"] = base_class._buffers["fparam_inv_std"]

            # --- weighted aparam stat merging ---
            if self.numb_aparam > 0:
                if not resume:
                    base_stats = base_class.get_param_stats().get("aparam", [])
                    self_stats = self.get_param_stats().get("aparam", [])
                    if base_stats and self_stats:
                        assert len(base_stats) == self.numb_aparam
                        merged = [
                            base_stats[ii] + self_stats[ii] * model_prob
                            for ii in range(self.numb_aparam)
                        ]
                        aparam_avg = np.array(
                            [s.compute_avg() for s in merged], dtype=np.float64
                        )
                        aparam_std = np.array(
                            [s.compute_std(protection=protection) for s in merged],
                            dtype=np.float64,
                        )
                        aparam_inv_std = 1.0 / aparam_std
                        base_class.aparam_avg.copy_(
                            torch.tensor(
                                aparam_avg,
                                device=DEVICE,
                                dtype=base_class.aparam_avg.dtype,
                            )
                        )
                        base_class.aparam_inv_std.copy_(
                            torch.tensor(
                                aparam_inv_std,
                                device=DEVICE,
                                dtype=base_class.aparam_inv_std.dtype,
                            )
                        )
                        base_class._param_stats["aparam"] = merged
                self._buffers["aparam_avg"] = base_class._buffers["aparam_avg"]
                self._buffers["aparam_inv_std"] = base_class._buffers["aparam_inv_std"]

            # --- share modules and remaining buffers ---
            for item in list(self._modules):
                if item in ("bias_atom_e", "case_embd"):
                    continue
                self._modules[item] = base_class._modules[item]
            for item in list(self._buffers):
                if item in (
                    "bias_atom_e",
                    "case_embd",
                    "fparam_avg",
                    "fparam_inv_std",
                    "aparam_avg",
                    "aparam_inv_std",
                ):
                    continue
                self._buffers[item] = base_class._buffers[item]
        else:
            raise NotImplementedError
