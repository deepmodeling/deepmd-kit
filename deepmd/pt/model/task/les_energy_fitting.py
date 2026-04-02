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


LES_DEFAULT_SIGMA = to_numpy_array(np.array(2.8 / np.sqrt(2.0)))


@LRFittingNet.register("les_energy")
@fitting_check_output
class LESEnergyFittingNet(LRFittingNet):
    """Construct a LES sr+lr interactions fitting net.

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
        sigma: Optional[Union[float, list[float], torch.Tensor]] = None,
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
        if isinstance(sigma, (list, tuple)):
            sigma = sigma[0] if len(sigma) > 0 else None
        sigma_tensor = to_torch_tensor(sigma)
        if sigma_tensor is None:
            sigma_tensor = to_torch_tensor(LES_DEFAULT_SIGMA)
        sigma_tensor = sigma_tensor.to(dtype=dtype, device=device).reshape(1)
        sigma_tensor = torch.clamp(
            sigma_tensor,
            min=torch.finfo(sigma_tensor.dtype).eps,
        )

        self.n_dl = max(1, int(n_dl))
        self.sigma = torch.nn.Parameter(
            sigma_tensor,
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
    
    def serialize(self) -> dict:
        data = super().serialize()
        data["type"] = "les_energy"
        data["@variables"]["sigma"] = to_numpy_array(self.sigma)
        data["n_dl"] = self.n_dl
        data["remove_self_interaction"] = bool(self.remove_self_interaction)
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "LESEnergyFittingNet":
        data = data.copy()
        variables = data.get("@variables", {}).copy()

        sigma_tensor = to_torch_tensor(variables.pop("sigma", None))
        data["@variables"] = variables

        obj = super().deserialize(data)

        with torch.no_grad():
            if sigma_tensor is None:
                raise ValueError("LES fitting net deserialize requires `sigma` in @variables.")
            obj.sigma.copy_(
                sigma_tensor.to(dtype=obj.sigma.dtype, device=obj.sigma.device).reshape(1)
            )
        return obj

    def _kernel_params(self) -> tuple[torch.Tensor]:
        return (self.sigma,)

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