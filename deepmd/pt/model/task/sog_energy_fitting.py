# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Any,
    Optional,
    Union,
)

import numpy as np
import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)
from deepmd.pt.model.task.lr_fitting import (
    LRFittingNet,
)


SOG_DEFAULT_SHIFT = np.array([
    0.2750,
    0.1375,
    0.0688,
    0.0344,
    0.0172,
    0.0086,
    0.0043,
    0.0021,
    0.0011,
    0.0005,
    0.0003,
    0.0001,
], dtype=env.GLOBAL_NP_FLOAT_PRECISION)
SOG_DEFAULT_AMPLITUDE = np.array([
    2.8,
    5.7,
    11.4,
    22.7,
    45.5,
    91.0,
    182.0,
    364.0,
    728.0,
    1456.0,
    2912.0,
    5823.9,
], dtype=env.GLOBAL_NP_FLOAT_PRECISION)


@LRFittingNet.register("sog_energy")
@fitting_check_output
class SOGEnergyFittingNet(LRFittingNet):
    """Construct a SOG sr+lr interactions fitting net.

    Parameters
    ----------
    var_name : str
        The atomic property to fit.
    ntypes : int
        Element count.
    dim_descrpt : int
        Embedding width per atom.
    dim_out_sr : int
        The output dimension of the sr fitting net.
    dim_out_lr : int
        The output dimension of the lr fitting net.
    neuron_sr : list[int]
        Number of neurons in each hidden layers of the sr fitting net.
    neuron_lr : list[int]
        Number of neurons in each hidden layers of the lr fitting net.
    bias_atom_e : torch.Tensor, optional
        Average energy per atom for each element.
    resnet_dt : bool
        Using time-step in the ResNet construction.
    numb_fparam : int
        Number of frame parameters.
    numb_aparam : int
        Number of atomic parameters.
    dim_case_embd : int
        Dimension of case specific embedding.
    activation_function : str
        Activation function.
    precision : str
        Numerical precision.
    mixed_types : bool
        If true, use a uniform fitting net for all atom types, otherwise use
        different fitting nets for different atom types.
    rcond : float, optional
        The condition number for the regression of atomic energy.
    seed : int, optional
        Random seed.
    exclude_types: list[int]
        Atomic contributions of the excluded atom types are set zero.
    trainable : Union[list[bool], bool]
        If the parameters in the fitting net are trainable.
        Now this only supports setting all the parameters in the fitting net at one state.
        When in list[bool], the trainable will be True only if all the boolean parameters are True.
    remove_vaccum_contribution: list[bool], optional
        Remove vacuum contribution before the bias is added. The list assigned each
        type. For `mixed_types` provide `[True]`, otherwise it should be a list of the same
        length as `ntypes` signaling if or not removing the vacuum contribution for the atom types in the list.
    type_map: list[str], Optional
        A list of strings. Give the name to each type of atoms.
    use_aparam_as_mask: bool
        If True, the aparam will not be used in fitting net for embedding.
    default_fparam: list[float], optional
        The default frame parameter. If set, when `fparam.npy` files are not included in the data system,
        this value will be used as the default value for the frame parameter in the fitting net.
    """

    def __init__(
        self,
        var_name: str,
        ntypes: int,
        dim_descrpt: int,
        dim_out_sr: int,
        dim_out_lr: int,
        neuron_sr: list[int] = [128, 128, 128],
        neuron_lr: list[int] = [128, 128, 128],
        bias_atom_e: Optional[torch.Tensor] = None,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        rcond: Optional[float] = None,
        seed: Optional[Union[int, list[int]]] = None,
        exclude_types: list[int] = [],
        trainable: Union[bool, list[bool]] = True,
        remove_vaccum_contribution: Optional[list[bool]] = None,
        type_map: Optional[list[str]] = None,
        use_aparam_as_mask: bool = False,
        default_fparam: Optional[list[float]] = None,
        shift: Optional[Union[list[float], torch.Tensor]] = None,
        amplitude: Optional[Union[list[float], torch.Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            var_name=var_name,
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out_sr=dim_out_sr,
            dim_out_lr=dim_out_lr,
            neuron_sr=neuron_sr,
            neuron_lr=neuron_lr,
            bias_atom_e=bias_atom_e,
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
            remove_vaccum_contribution=remove_vaccum_contribution,
            type_map=type_map,
            use_aparam_as_mask=use_aparam_as_mask,
            default_fparam=default_fparam,
            **kwargs,
        )
        if isinstance(shift, (list, tuple)):
            shift = np.array(shift, dtype=env.GLOBAL_NP_FLOAT_PRECISION)
        if isinstance(amplitude, (list, tuple)):
            amplitude = np.array(amplitude, dtype=env.GLOBAL_NP_FLOAT_PRECISION)
        shift_tensor = to_torch_tensor(shift)
        amplitude_tensor = to_torch_tensor(amplitude)
        if shift_tensor is None:
            shift_tensor = to_torch_tensor(SOG_DEFAULT_SHIFT)
        if amplitude_tensor is None:
            amplitude_tensor = to_torch_tensor(SOG_DEFAULT_AMPLITUDE)
        # Register as trainable parameters so they are optimized with the fitting net.
        self.shift = torch.nn.Parameter(
            shift_tensor.to(dtype=dtype, device=device),
            requires_grad=bool(self.trainable),
        )
        self.amplitude = torch.nn.Parameter(
            amplitude_tensor.to(dtype=dtype, device=device),
            requires_grad=bool(self.trainable),
        )


    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    name="energy",
                    shape=[1],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                )
            ]
        )
    
    def serialize(self) -> dict:
        data = super().serialize()
        data["type"] = "sog_energy"
        data["@variables"]["shift"] = to_numpy_array(self.shift)
        data["@variables"]["amplitude"] = to_numpy_array(self.amplitude)
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "SOGEnergyFittingNet":
        data = data.copy()
        variables = data.get("@variables", {})
        obj = super().deserialize(data)
        shift_tensor = to_torch_tensor(variables.get("shift", None))
        amplitude_tensor = to_torch_tensor(variables.get("amplitude", None))
        # Backward compatibility: if serialized variables miss shift/amplitude,
        # keep defaults initialized in __init__.
        if shift_tensor is not None:
            obj.shift = torch.nn.Parameter(
                shift_tensor.to(dtype=dtype, device=device),
                requires_grad=bool(obj.trainable),
            )
        if amplitude_tensor is not None:
            obj.amplitude = torch.nn.Parameter(
                amplitude_tensor.to(dtype=dtype, device=device),
                requires_grad=bool(obj.trainable),
            )
        return obj

    def corr_head(
        self,
        lr_val: torch.Tensor,
        amplitude: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        # TODO:
        # Long-range correction energy calculation
        return torch.zeros_like(lr_val)

    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: Optional[torch.Tensor] = None,
        g2: Optional[torch.Tensor] = None,
        h2: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        out = self._forward_common(
            descriptor=descriptor,
            atype=atype,
            gr=gr,
            g2=g2,
            h2=h2,
            fparam=fparam,
            aparam=aparam,
        )
        short_energy = out["sr"]
        corr_energy = self.corr_head(out["lr"], self.amplitude, self.shift)
        result = {"energy": short_energy + corr_energy}
        if "middle_output" in out:
            result["middle_output"] = out["middle_output"]
        return result
    
    # make jit happy with torch 2.0.0
    exclude_types: list[int]