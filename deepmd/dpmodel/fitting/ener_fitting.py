# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Union,
)

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
