# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Union,
)

import numpy as np

from deepmd.dpmodel.common import (
    DEFAULT_PRECISION,
)
from deepmd.dpmodel.fitting.invar_fitting import (
    InvarFitting,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.fitting.general_fitting import (
        GeneralFitting,
    )

from deepmd.utils.out_stat import (
    compute_stats_from_redu,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


@InvarFitting.register("ener")
class EnergyFittingNet(InvarFitting):
    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        neuron: list[int] = [120, 120, 120],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        rcond: Optional[float] = None,
        tot_ener_zero: bool = False,
        trainable: Optional[list[bool]] = None,
        atom_ener: Optional[list[float]] = None,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        layer_name: Optional[list[Optional[str]]] = None,
        use_aparam_as_mask: bool = False,
        spin: Any = None,
        mixed_types: bool = False,
        exclude_types: list[int] = [],
        type_map: Optional[list[str]] = None,
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        super().__init__(
            var_name="energy",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=1,
            neuron=neuron,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            rcond=rcond,
            tot_ener_zero=tot_ener_zero,
            trainable=trainable,
            atom_ener=atom_ener,
            activation_function=activation_function,
            precision=precision,
            layer_name=layer_name,
            use_aparam_as_mask=use_aparam_as_mask,
            spin=spin,
            mixed_types=mixed_types,
            exclude_types=exclude_types,
            type_map=type_map,
            seed=seed,
        )

    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 3, 1)
        data.pop("var_name")
        data.pop("dim_out")
        return super().deserialize(data)

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {
            **super().serialize(),
            "type": "ener",
        }

    def compute_output_stats(self, all_stat: dict, mixed_type: bool = False) -> None:
        """Compute the output statistics.

        Parameters
        ----------
        all_stat
            must have the following components:
            all_stat['energy'] of shape n_sys x n_batch x n_frame
            can be prepared by model.make_stat_input
        mixed_type
            Whether to perform the mixed_type mode.
            If True, the input data has the mixed_type format (see doc/model/train_se_atten.md),
            in which frames in a system may have different natoms_vec(s), with the same nloc.
        """
        self.bias_atom_e = self._compute_output_stats(
            all_stat, rcond=self.rcond, mixed_type=mixed_type
        )

    def _compute_output_stats(self, all_stat, rcond=1e-3, mixed_type=False):
        data = all_stat["energy"]
        # data[sys_idx][batch_idx][frame_idx]
        sys_ener = []
        for ss in range(len(data)):
            sys_data = []
            for ii in range(len(data[ss])):
                for jj in range(len(data[ss][ii])):
                    sys_data.append(data[ss][ii][jj])
            sys_data = np.concatenate(sys_data)
            sys_ener.append(np.average(sys_data))
        sys_ener = np.array(sys_ener)
        sys_tynatom = []
        if mixed_type:
            data = all_stat["real_natoms_vec"]
            nsys = len(data)
            for ss in range(len(data)):
                tmp_tynatom = []
                for ii in range(len(data[ss])):
                    for jj in range(len(data[ss][ii])):
                        tmp_tynatom.append(data[ss][ii][jj].astype(np.float64))
                tmp_tynatom = np.average(np.array(tmp_tynatom), axis=0)
                sys_tynatom.append(tmp_tynatom)
        else:
            data = all_stat["natoms_vec"]
            nsys = len(data)
            for ss in range(len(data)):
                sys_tynatom.append(data[ss][0].astype(np.float64))
        sys_tynatom = np.array(sys_tynatom)
        sys_tynatom = np.reshape(sys_tynatom, [nsys, -1])
        sys_tynatom = sys_tynatom[:, 2:]
        if len(self.atom_ener) > 0:
            # Atomic energies stats are incorrect if atomic energies are assigned.
            # In this situation, we directly use these assigned energies instead of computing stats.
            # This will make the loss decrease quickly
            assigned_atom_ener = np.array(
                [ee if ee is not None else np.nan for ee in self.atom_ener_v]
            )
        else:
            assigned_atom_ener = None
        energy_shift, _ = compute_stats_from_redu(
            sys_ener.reshape(-1, 1),
            sys_tynatom,
            assigned_bias=assigned_atom_ener,
            rcond=rcond,
        )
        return energy_shift.ravel()
