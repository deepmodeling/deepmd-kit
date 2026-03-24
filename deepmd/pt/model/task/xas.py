# SPDX-License-Identifier: LGPL-3.0-or-later
import logging

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
from deepmd.utils.version import (
    check_version_compatibility,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)


@Fitting.register("xas")
class XASFittingNet(InvarFitting):
    """PyTorch fitting network for XAS spectra.

    Parameters
    ----------
    ntypes : int
        Number of atom types.
    dim_descrpt : int
        Dimension of the descriptor.
    numb_xas : int
        Number of XAS energy grid points.
    neuron : list[int]
        Hidden layer sizes.
    resnet_dt : bool
        Whether to use ResNet time step.
    numb_fparam : int
        Dimension of frame parameters (e.g. edge type encoding).
    numb_aparam : int
        Dimension of atomic parameters.
    dim_case_embd : int
        Dimension of case embedding.
    rcond : float or None
        Cutoff for small singular values in bias init.
    bias_xas : torch.Tensor or None
        Initial bias, shape (ntypes, numb_xas).
    trainable : bool or list[bool]
        Whether parameters are trainable.
    seed : int, list[int], or None
        Random seed.
    activation_function : str
        Activation function.
    precision : str
        Float precision.
    exclude_types : list[int]
        Atom types to exclude (set by XASAtomicModel automatically).
    mixed_types : bool
        Whether to use a shared network across types.
    type_map : list[str] or None
        Mapping from type index to element symbol.
    default_fparam : list or None
        Default frame parameter values.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        numb_xas: int = 500,
        neuron: list[int] = [128, 128, 128],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        rcond: float | None = None,
        bias_xas: torch.Tensor | None = None,
        trainable: bool | list[bool] = True,
        seed: int | list[int] | None = None,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        exclude_types: list[int] = [],
        mixed_types: bool = False,
        type_map: list[str] | None = None,
        default_fparam: list | None = None,
    ) -> None:
        if bias_xas is not None:
            self.bias_xas = bias_xas
        else:
            self.bias_xas = torch.zeros(
                (ntypes, numb_xas), dtype=dtype, device=env.DEVICE
            )
        super().__init__(
            var_name="xas",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=numb_xas,
            neuron=neuron,
            bias_atom_e=bias_xas,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            rcond=rcond,
            seed=seed,
            exclude_types=exclude_types,
            trainable=trainable,
            type_map=type_map,
            default_fparam=default_fparam,
        )

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.dim_out],
                    reducible=True,
                    intensive=True,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    @classmethod
    def deserialize(cls, data: dict) -> "XASFittingNet":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 4, 1)
        data.pop("@class", None)
        data.pop("var_name", None)
        data.pop("tot_ener_zero", None)
        data.pop("layer_name", None)
        data.pop("use_aparam_as_mask", None)
        data.pop("spin", None)
        data.pop("atom_ener", None)
        data["numb_xas"] = data.pop("dim_out")
        return super().deserialize(data)

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        dd = {
            **InvarFitting.serialize(self),
            "type": "xas",
            "dim_out": self.dim_out,
        }
        dd["@variables"]["bias_atom_e"] = to_numpy_array(self.bias_atom_e)
        return dd

    # make jit happy with torch 2.0.0
    exclude_types: list[int]
