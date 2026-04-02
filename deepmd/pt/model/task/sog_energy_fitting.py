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


SOG_DEFAULT_AMPLITUDE = to_numpy_array(np.array([
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
]))
SOG_DEFAULT_SHIFT = to_numpy_array(np.array([
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
]))


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
    n_dl : int
        NUFFT long-range grid density control factor.
    remove_self_interaction : bool
        If True, remove self interaction term in long-range correction.
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
        n_dl: int = 1,
        remove_self_interaction: bool = False,
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
            shift = to_numpy_array(np.array(shift))
        if isinstance(amplitude, (list, tuple)):
            amplitude = to_numpy_array(np.array(amplitude))
        shift_tensor = to_torch_tensor(shift)
        amplitude_tensor = to_torch_tensor(amplitude)
        if shift_tensor is None:
            shift_tensor = to_torch_tensor(SOG_DEFAULT_SHIFT)
        if amplitude_tensor is None:
            amplitude_tensor = to_torch_tensor(SOG_DEFAULT_AMPLITUDE)

        shift_tensor = shift_tensor.to(dtype=dtype, device=device)
        amplitude_tensor = amplitude_tensor.to(dtype=dtype, device=device)
        pi_tensor = torch.tensor(torch.pi, dtype=dtype, device=device)
        sqr_pi_tensor = torch.sqrt(pi_tensor)
        shift_safe = torch.clamp(
            shift_tensor,
            min=torch.finfo(shift_tensor.dtype).eps,
        )
        wl_tensor = amplitude_tensor * (sqr_pi_tensor**3) * (shift_safe**3)
        sl_tensor = -torch.log(2.0 / shift_safe)

        self.n_dl = max(1, int(n_dl))
        self.wl = torch.nn.Parameter(
            wl_tensor,
            requires_grad=bool(self.trainable),
        )
        self.sl = torch.nn.Parameter(
            sl_tensor,
            requires_grad=bool(self.trainable),
        )
        self.remove_self_interaction = bool(remove_self_interaction)
        self._nufft_fallback_warned = False

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    name="energy",
                    shape=[1],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
                OutputVariableDef(
                    name="latent_charge",
                    shape=[self.dim_out_lr],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                )
            ]
        )

    @staticmethod
    def _wl_sl_to_shift_amplitude(
        wl_tensor: torch.Tensor,
        sl_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pi_tensor = torch.tensor(
            torch.pi,
            dtype=sl_tensor.dtype,
            device=sl_tensor.device,
        )
        sqr_pi_tensor = torch.sqrt(pi_tensor)
        shift_tensor = 2.0 * torch.exp(sl_tensor)
        amplitude_tensor = wl_tensor / ((sqr_pi_tensor**3) * (shift_tensor**3))
        return shift_tensor, amplitude_tensor

    @staticmethod
    def _shift_amplitude_to_wl_sl(
        shift_tensor: torch.Tensor,
        amplitude_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pi_tensor = torch.tensor(
            torch.pi,
            dtype=shift_tensor.dtype,
            device=shift_tensor.device,
        )
        sqr_pi_tensor = torch.sqrt(pi_tensor)
        shift_safe = torch.clamp(
            shift_tensor,
            min=torch.finfo(shift_tensor.dtype).eps,
        )
        wl_tensor = amplitude_tensor * (sqr_pi_tensor**3) * (shift_safe**3)
        sl_tensor = -torch.log(2.0 / shift_safe)
        return wl_tensor, sl_tensor

    def serialize(self) -> dict:
        data = super().serialize()
        data["type"] = "sog_energy"
        variables = data["@variables"]
        variables["wl"] = to_numpy_array(self.wl)
        variables["sl"] = to_numpy_array(self.sl)
        shift_tensor, amplitude_tensor = self._wl_sl_to_shift_amplitude(
            self.wl,
            self.sl,
        )
        variables["shift"] = to_numpy_array(shift_tensor)
        variables["amplitude"] = to_numpy_array(amplitude_tensor)
        data["n_dl"] = self.n_dl
        data["remove_self_interaction"] = bool(self.remove_self_interaction)
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "SOGEnergyFittingNet":
        data = data.copy()

        variables = data.get("@variables", {}).copy()

        wl_tensor = to_torch_tensor(variables.pop("wl", None))
        sl_tensor = to_torch_tensor(variables.pop("sl", None))
        shift_tensor = to_torch_tensor(variables.pop("shift", None))
        amplitude_tensor = to_torch_tensor(variables.pop("amplitude", None))
        data["@variables"] = variables

        obj = super().deserialize(data)

        with torch.no_grad():
            if wl_tensor is not None and sl_tensor is not None:
                obj.wl.copy_(wl_tensor.to(dtype=obj.wl.dtype, device=obj.wl.device))
                obj.sl.copy_(sl_tensor.to(dtype=obj.sl.dtype, device=obj.sl.device))
            elif shift_tensor is not None and amplitude_tensor is not None:
                wl_tensor, sl_tensor = cls._shift_amplitude_to_wl_sl(
                    shift_tensor.to(dtype=obj.wl.dtype, device=obj.wl.device),
                    amplitude_tensor.to(dtype=obj.wl.dtype, device=obj.wl.device),
                )
                obj.wl.copy_(wl_tensor)
                obj.sl.copy_(sl_tensor)
        return obj

    def _kernel_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.wl, self.sl

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
        result = {
            "energy": out["sr"],
            "latent_charge": out["lr"],
        }
        if "middle_output" in out:
            result["middle_output"] = out["middle_output"]
        return result
    
    # make jit happy with torch 2.0.0
    exclude_types: list[int]