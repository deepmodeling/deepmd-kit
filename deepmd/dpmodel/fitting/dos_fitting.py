# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
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
        trainable: Optional[List[bool]] = None,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = False,
        exclude_types: List[int] = [],
        # not used
        seed: Optional[int] = None,
    ):
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
        )

    def __setitem__(self, key, value):
        if key in ["bias_dos"]:
            self.bias_dos = value
        else:
            super().__setitem__(key, value)

    def __getitem__(self, key):
        if key in ["bias_dos"]:
            return self.bias_atom_e
        else:
            return super().__getitem__(key)
        
    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("var_name")
        data.pop("dim_out")
        data.pop("tot_ener_zero")
        data.pop("layer_name")
        data.pop("use_aparam_as_mask")
        data.pop("spin")
        data.pop("atom_ener")
        return super().deserialize(data)

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        dd = {
            **super().serialize(),
            "type": "dos",
        }
        dd["@variables"]["bias_dos"] = dd["@variables"].pop("bias_atom_e")
        
        return dd