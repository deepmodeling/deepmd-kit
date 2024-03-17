# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import os
import tempfile
from typing import (
    Dict,
    Optional,
)

import numpy as np
import torch

from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.pt.utils.stat import (
    compute_output_stats,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)

from .dp_model import (
    DPModel,
)

log = logging.getLogger(__name__)


class EnergyModel(DPModel):
    model_type = "ener"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        coord,
        atype,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            if self.do_grad_r("energy"):
                model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
            if self.do_grad_c("energy"):
                model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(
                        -3
                    )
            else:
                model_predict["force"] = model_ret["dforce"]
            if "mask" in model_ret:
                model_predict["mask"] = model_ret["mask"]
        else:
            model_predict = model_ret
            model_predict["updated_coord"] += coord
        return model_predict

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ):
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            if self.do_grad_r("energy"):
                model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
            if self.do_grad_c("energy"):
                model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["extended_virial"] = model_ret[
                        "energy_derv_c"
                    ].squeeze(-3)
            else:
                assert model_ret["dforce"] is not None
                model_predict["dforce"] = model_ret["dforce"]
        else:
            model_predict = model_ret
        return model_predict

    def change_out_bias(
        self, merged, origin_type_map, full_type_map, bias_shift="delta"
    ) -> None:
        """Change the energy bias according to the input data and the pretrained model.

        Parameters
        ----------
        merged : Union[Callable[[], List[dict]], List[dict]]
            - List[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], List[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        origin_type_map : List[str]
            The original type_map in dataset, they are targets to change the energy bias.
        full_type_map : List[str]
            The full type_map in pre-trained model
        bias_shift : str
            The mode for changing energy bias : ['delta', 'statistic']
            'delta' : perform predictions on energies of target dataset,
                    and do least sqaure on the errors to obtain the target shift as bias.
            'statistic' : directly use the statistic energy bias in the target dataset.
        """
        sorter = np.argsort(full_type_map)
        idx_type_map = sorter[
            np.searchsorted(full_type_map, origin_type_map, sorter=sorter)
        ]
        original_bias = self.get_fitting_net()["bias_atom_e"]
        if bias_shift == "delta":
            tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
            model = torch.jit.script(self)
            torch.jit.save(model, tmp_model.name)
            dp = DeepEval(tmp_model.name)
            os.unlink(tmp_model.name)
            delta_bias_e = compute_output_stats(
                merged,
                self.atomic_model.get_ntypes(),
                model=dp,
            )
            bias_atom_e = delta_bias_e + original_bias
        elif bias_shift == "statistic":
            bias_atom_e = compute_output_stats(
                merged,
                self.atomic_model.get_ntypes(),
            )
        else:
            raise RuntimeError("Unknown bias_shift mode: " + bias_shift)
        log.info(
            f"Change energy bias of {origin_type_map!s} "
            f"from {to_numpy_array(original_bias[idx_type_map]).reshape(-1)!s} "
            f"to {to_numpy_array(bias_atom_e[idx_type_map]).reshape(-1)!s}."
        )
        self.get_fitting_net()["bias_atom_e"] = bias_atom_e
