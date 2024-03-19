# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import logging
from typing import (
    Callable,
    List,
    Optional,
    Union,
)

import numpy as np
import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pt.model.task.ener import (
    InvarFitting,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)
from deepmd.utils.out_stat import (
    compute_stats_from_atomic,
    compute_stats_from_redu,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)


@Fitting.register("dos")
class DOSFittingNet(InvarFitting):
    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        numb_dos: int = 300,
        neuron: List[int] = [128, 128, 128],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        rcond: Optional[float] = None,
        bias_dos: Optional[torch.Tensor] = None,
        trainable: Union[bool, List[bool]] = True,
        seed: Optional[int] = None,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        exclude_types: List[int] = [],
        mixed_types: bool = True,
    ):
        if bias_dos is not None:
            self.bias_dos = bias_dos
        else:
            self.bias_dos = torch.zeros(
                (ntypes, numb_dos), dtype=dtype, device=env.DEVICE
            )
        super().__init__(
            var_name="dos",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=numb_dos,
            neuron=neuron,
            bias_atom_e=bias_dos,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            rcond=rcond,
            seed=seed,
            exclude_types=exclude_types,
            trainable=trainable,
        )

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.dim_out],
                    reduciable=True,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    def compute_output_stats(
        self,
        merged: Union[Callable[[], List[dict]], List[dict]],
        stat_file_path: Optional[DPPath] = None,
    ) -> None:
        """
        Compute the output statistics (e.g. dos bias) for the fitting net from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], List[dict]], List[dict]]
            - List[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], List[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        stat_file_path : Optional[DPPath]
            The path to the stat file.

        """
        if stat_file_path is not None:
            stat_file_path = stat_file_path / "bias_dos"
        if stat_file_path is not None and stat_file_path.is_file():
            bias_dos = stat_file_path.load_numpy()
        else:
            if callable(merged):
                # only get data for once
                sampled = merged()
            else:
                sampled = merged
            for sys in range(len(sampled)):
                nframs = sampled[sys]["atype"].shape[0]

                if "atom_dos" in sampled[sys]:
                    sys_atom_dos = compute_stats_from_atomic(
                        sampled[sys]["atom_dos"].numpy(force=True),
                        sampled[sys]["atype"].numpy(force=True),
                    )[0]
                else:
                    sys_type_count = np.zeros(
                        (nframs, self.ntypes), dtype=env.GLOBAL_NP_FLOAT_PRECISION
                    )
                    for itype in range(self.ntypes):
                        type_mask = sampled[sys]["atype"] == itype
                        sys_type_count[:, itype] = type_mask.sum(dim=1).numpy(
                            force=True
                        )
                    sys_bias_redu = sampled[sys]["dos"].numpy(force=True)

                    sys_atom_dos = compute_stats_from_redu(
                        sys_bias_redu, sys_type_count, rcond=self.rcond
                    )[0]
                if stat_file_path is not None:
                    stat_file_path.save_numpy(sys_atom_dos)
                self.bias_dos = torch.tensor(sys_atom_dos, device=env.DEVICE)

    @classmethod
    def deserialize(cls, data: dict) -> "DOSFittingNet":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        data.pop("var_name", None)
        data.pop("tot_ener_zero", None)
        data.pop("layer_name", None)
        data.pop("use_aparam_as_mask", None)
        data.pop("spin", None)
        data.pop("atom_ener", None)
        data["numb_dos"] = data.pop("dim_out")
        obj = super().deserialize(data)

        return obj

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        # dd = super(InvarFitting, self).serialize()
        dd = {
            **InvarFitting.serialize(self),
            "type": "dos",
            "dim_out": self.dim_out,
        }
        dd["@variables"]["bias_atom_e"] = to_numpy_array(self.bias_atom_e)

        return dd

    # make jit happy with torch 2.0.0
    exclude_types: List[int]
