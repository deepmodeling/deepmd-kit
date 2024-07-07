# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    TYPE_CHECKING,
    List,
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

from deepmd.utils.version import (
    check_version_compatibility,
)


@InvarFitting.register("dos")
class DOSFittingNet(InvarFitting):
    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        numb_dos: int = 300,
        neuron: List[int] = [120, 120, 120],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        bias_dos: Optional[np.ndarray] = None,
        rcond: Optional[float] = None,
        trainable: Union[bool, List[bool]] = True,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = False,
        exclude_types: List[int] = [],
        type_map: Optional[List[str]] = None,
        seed: Optional[Union[int, List[int]]] = None,
    ):
        if bias_dos is not None:
            self.bias_dos = bias_dos
        else:
            self.bias_dos = np.zeros((ntypes, numb_dos), dtype=DEFAULT_PRECISION)
        super().__init__(
            var_name="dos",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=numb_dos,
            neuron=neuron,
            resnet_dt=resnet_dt,
            bias_atom=bias_dos,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            rcond=rcond,
            trainable=trainable,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            exclude_types=exclude_types,
            type_map=type_map,
            seed=seed,
        )

    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 2, 1)
        data["numb_dos"] = data.pop("dim_out")
        data.pop("tot_ener_zero", None)
        data.pop("var_name", None)
        data.pop("layer_name", None)
        data.pop("use_aparam_as_mask", None)
        data.pop("spin", None)
        data.pop("atom_ener", None)
        return super().deserialize(data)

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        dd = {
            **super().serialize(),
            "type": "dos",
        }
        dd["@variables"]["bias_atom_e"] = self.bias_atom_e

        return dd
