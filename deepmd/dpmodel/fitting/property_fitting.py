# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    Callable,
    Union,
    TYPE_CHECKING,
    Any,
    List,
    Optional,
)

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.utils.path import (
    DPPath,
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


@InvarFitting.register("property")
class PropertyFittingNet(InvarFitting):
    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        task_num: int = 1,
        neuron: List[int] = [128, 128, 128],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        rcond: Optional[float] = None,
        tot_ener_zero: bool = False,
        trainable: Optional[List[bool]] = None,
        atom_ener: Optional[List[float]] = None,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        layer_name: Optional[List[Optional[str]]] = None,
        use_aparam_as_mask: bool = False,
        spin: Any = None,
        mixed_types: bool = False,
        exclude_types: List[int] = [],
        # not used
        seed: Optional[int] = None,
    ):
        self.task_num = task_num
        super().__init__(
            var_name="property",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=task_num,
            neuron=neuron,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
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
        )

    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("var_name")
        data.pop("dim_out")
        return super().deserialize(data)

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {
            **super().serialize(),
            "type": "property",
            "task_num": self.task_num
        }

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
    ):
        """
        Compute the output statistics (e.g. energy bias) for the fitting net from packed data.

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
        pass

    # make jit happy with torch 2.0.0
    exclude_types: List[int]