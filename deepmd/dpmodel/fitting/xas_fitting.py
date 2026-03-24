# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    TYPE_CHECKING,
)

import numpy as np

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.common import (
    DEFAULT_PRECISION,
    to_numpy_array,
)
from deepmd.dpmodel.fitting.invar_fitting import (
    InvarFitting,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.fitting.general_fitting import (
        GeneralFitting,
    )

from deepmd.utils.version import (
    check_version_compatibility,
)


@InvarFitting.register("xas")
class XASFittingNet(InvarFitting):
    """Fitting network for X-ray Absorption Spectroscopy (XAS) spectra.

    Predicts per-atom XAS contributions in a relative energy (ΔE) space.
    The global XAS is the mean over all absorbing atoms, handled by the
    XAS model via ``intensive=True`` and type-selective masking.

    Parameters
    ----------
    ntypes : int
        Number of atom types.
    dim_descrpt : int
        Dimension of the descriptor.
    numb_xas : int
        Number of XAS energy grid points.
    neuron : list[int]
        Hidden layer sizes of the fitting network.
    resnet_dt : bool
        Whether to use residual network with time step.
    numb_fparam : int
        Dimension of frame parameters (e.g. edge type encoding).
    numb_aparam : int
        Dimension of atomic parameters.
    dim_case_embd : int
        Dimension of case embedding.
    bias_xas : Array or None
        Initial bias for XAS output, shape (ntypes, numb_xas).
    rcond : float or None
        Cutoff for small singular values.
    trainable : bool or list[bool]
        Whether the fitting parameters are trainable.
    activation_function : str
        Activation function for hidden layers.
    precision : str
        Precision for the fitting parameters.
    mixed_types : bool
        Whether to use a shared network for all atom types.
    exclude_types : list[int]
        Atom types to exclude from fitting (set automatically by XASAtomicModel).
    type_map : list[str] or None
        Mapping from type index to element symbol.
    seed : int, list[int], or None
        Random seed.
    default_fparam : list or None
        Default frame parameter values.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        numb_xas: int = 500,
        neuron: list[int] = [120, 120, 120],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        bias_xas: Array | None = None,
        rcond: float | None = None,
        trainable: bool | list[bool] = True,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = False,
        exclude_types: list[int] = [],
        type_map: list[str] | None = None,
        seed: int | list[int] | None = None,
        default_fparam: list | None = None,
    ) -> None:
        if bias_xas is not None:
            self.bias_xas = bias_xas
        else:
            self.bias_xas = np.zeros((ntypes, numb_xas), dtype=DEFAULT_PRECISION)
        super().__init__(
            var_name="xas",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=numb_xas,
            neuron=neuron,
            resnet_dt=resnet_dt,
            bias_atom=bias_xas,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            rcond=rcond,
            trainable=trainable,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            exclude_types=exclude_types,
            type_map=type_map,
            seed=seed,
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
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 4, 1)
        data["numb_xas"] = data.pop("dim_out")
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
            "type": "xas",
        }
        dd["@variables"]["bias_atom_e"] = to_numpy_array(self.bias_atom_e)
        return dd
