# SPDX-License-Identifier: LGPL-3.0-or-later
"""SeZM GLU energy fitting networks."""

from __future__ import (
    annotations,
)

import math
from typing import (
    Any,
    ClassVar,
)

import torch

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.network.mlp import (
    GLULayer,
    MLPLayer,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
    GeneralFitting,
)
from deepmd.pt.model.task.invar_fitting import (
    InvarFitting,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
    DEVICE,
    PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class CaseFiLMConditioner(torch.nn.Module):
    """
    Case-conditioned FiLM generator for SeZM fitting features.

    Parameters
    ----------
    dim_case_embd
        Case one-hot width.
    dim_descrpt
        Descriptor output width.
    target_dims
        Feature widths of all FiLM modulation targets.
    activation_function
        Activation used by the case MLP hidden layer.
    precision
        Numerical precision.
    seed
        Random seed.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        dim_case_embd: int,
        dim_descrpt: int,
        target_dims: list[int],
        activation_function: str,
        precision: str,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.dim_case_embd = int(dim_case_embd)
        self.dim_descrpt = int(dim_descrpt)
        self.target_dims = [int(dim) for dim in target_dims]
        self.activation_function = str(activation_function)
        self.precision = str(precision)
        self.prec = PRECISION_DICT[self.precision]
        self.code_dim = 4 * self.dim_descrpt
        hidden_dim = int(32 * math.ceil((4.0 * float(self.dim_case_embd)) / 32.0))

        self.case_layer1 = MLPLayer(
            self.dim_case_embd,
            hidden_dim,
            bias=False,
            use_timestep=False,
            activation_function=self.activation_function,
            resnet=False,
            precision=self.precision,
            seed=child_seed(seed, 0),
            trainable=trainable,
        )
        self.case_layer2 = MLPLayer(
            hidden_dim,
            self.code_dim,
            bias=False,
            use_timestep=False,
            activation_function=None,
            resnet=False,
            precision=self.precision,
            seed=child_seed(seed, 1),
            trainable=trainable,
        )
        self.projectors = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.zeros(
                        self.code_dim,
                        2 * target_dim,
                        dtype=self.prec,
                        device=DEVICE,
                    )
                )
                for target_dim in self.target_dims
            ]
        )
        strength_init = math.log(0.01)
        self.adam_case_film_scale_strength_log = torch.nn.Parameter(
            torch.full(
                (len(self.target_dims),),
                strength_init,
                dtype=self.prec,
                device=DEVICE,
            )
        )
        self.adam_case_film_shift_strength_log = torch.nn.Parameter(
            torch.full(
                (len(self.target_dims),),
                strength_init,
                dtype=self.prec,
                device=DEVICE,
            )
        )

        for param in self.parameters():
            param.requires_grad = trainable

    def encode(self, case_embd: torch.Tensor) -> torch.Tensor:
        """
        Encode a compact case one-hot vector.

        Parameters
        ----------
        case_embd
            Case one-hot vector with shape (K,) or (1, K).

        Returns
        -------
        torch.Tensor
            Case code with shape (1, 4*dim_descrpt).
        """
        code = case_embd.reshape(1, self.dim_case_embd)
        return self.case_layer2(self.case_layer1(code))

    def apply(
        self,
        xx: torch.Tensor,
        case_code: torch.Tensor,
        target_idx: int,
    ) -> torch.Tensor:
        """
        Apply one FiLM target to a feature tensor.

        Parameters
        ----------
        xx
            Feature tensor with shape (..., target_dim).
        case_code
            Encoded case tensor with shape (1, 4*dim_descrpt).
        target_idx
            Index of the target to modulate.

        Returns
        -------
        torch.Tensor
            Modulated feature tensor with the same shape as ``xx``.
        """
        film = torch.matmul(case_code, self.projectors[target_idx])
        gamma, beta = film.chunk(2, dim=-1)
        view_shape = [1 for _ in range(xx.ndim - 1)] + [xx.shape[-1]]
        gamma = gamma.reshape(view_shape)
        beta = beta.reshape(view_shape)
        scale_strength = torch.exp(self.adam_case_film_scale_strength_log[target_idx])
        shift_strength = torch.exp(self.adam_case_film_shift_strength_log[target_idx])
        return xx * (1.0 + scale_strength * torch.tanh(gamma)) + (
            shift_strength * torch.tanh(beta)
        )


class GLUFittingNet(torch.nn.Module):
    """
    GLU-based fitting network for SeZM.

    Parameters
    ----------
    in_dim
        Input dimension.
    out_dim
        Output dimension.
    neuron
        Hidden layer sizes. Empty list means direct linear projection.
    activation_function
        Activation function used for GLU gating.
    resnet_dt
        Reserved for compatibility; not used in GLU layers.
    precision
        Numerical precision.
    bias_out
        Whether the output layer uses bias.
    seed
        Random seed.
    trainable
        Whether parameters are trainable.
    descriptor_dim
        Descriptor feature width. Used by case FiLM to avoid modulating
        frame/atomic parameters.
    dim_case_embd
        Case one-hot width.
    case_film_embd
        Whether to use case FiLM instead of input concatenation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        neuron: list[int] | None = None,
        activation_function: str = "silu",
        resnet_dt: bool = False,
        precision: str = DEFAULT_PRECISION,
        bias_out: bool = False,
        seed: int | list[int] | None = None,
        trainable: bool | list[bool] = True,
        descriptor_dim: int | None = None,
        dim_case_embd: int = 0,
        case_film_embd: bool = False,
    ) -> None:
        super().__init__()
        if neuron is None:
            neuron = []
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.neuron = [int(nn_dim) for nn_dim in neuron]
        if isinstance(trainable, bool):
            self.trainable = [trainable] * (len(self.neuron) + 1)
        else:
            self.trainable = [bool(flag) for flag in trainable]
            if len(self.trainable) != len(self.neuron) + 1:
                raise ValueError(
                    "trainable must contain one flag per hidden layer plus "
                    "one flag for the output layer"
                )
        self.activation_function = activation_function
        self.resnet_dt = bool(resnet_dt)
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.bias_out = bool(bias_out)
        self.descriptor_dim = (
            self.in_dim if descriptor_dim is None else int(descriptor_dim)
        )
        self.dim_case_embd = int(dim_case_embd)
        self.case_film_embd = bool(case_film_embd and self.dim_case_embd > 0)

        # === Step 1. Build GLU hidden layers ===
        hidden_layers = []
        dim_in = self.in_dim
        for layer_idx, hidden_dim in enumerate(self.neuron):
            hidden_layers.append(
                GLULayer(
                    dim_in,
                    hidden_dim,
                    activation_function=self.activation_function,
                    precision=self.precision,
                    seed=child_seed(seed, layer_idx),
                    trainable=self.trainable[layer_idx],
                )
            )
            dim_in = hidden_dim
        self.hidden_layers = torch.nn.ModuleList(hidden_layers)

        # === Step 2. Build optional case FiLM conditioner ===
        if self.case_film_embd:
            self.case_film = CaseFiLMConditioner(
                dim_case_embd=self.dim_case_embd,
                dim_descrpt=self.descriptor_dim,
                target_dims=[self.descriptor_dim, *self.neuron],
                activation_function=self.activation_function,
                precision=self.precision,
                seed=child_seed(seed, len(self.neuron)),
                trainable=all(self.trainable),
            )
        else:
            self.case_film = None

        # === Step 3. Build output projection ===
        self.output_layer = MLPLayer(
            num_in=dim_in,
            num_out=self.out_dim,
            bias=self.bias_out,
            use_timestep=False,
            activation_function=None,
            resnet=False,
            precision=self.precision,
            seed=child_seed(seed, len(self.neuron) + int(self.case_film_embd)),
            trainable=self.trainable[-1],
        )

        # The layer constructors retain ``trainable`` as serialization
        # metadata but do not consistently apply it to newly created
        # ``Parameter`` objects. Reapply the policy at this owning module so a
        # deserialize round trip cannot make frozen layers optimizer-visible.
        for layer, layer_trainable in zip(
            self.hidden_layers, self.trainable[:-1], strict=True
        ):
            for param in layer.parameters():
                param.requires_grad = layer_trainable
        if self.case_film is not None:
            for param in self.case_film.parameters():
                param.requires_grad = all(self.trainable)
        for param in self.output_layer.parameters():
            param.requires_grad = self.trainable[-1]

    def _apply_input_film(
        self,
        xx: torch.Tensor,
        case_code: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FiLM only to the descriptor slice of the fitting input."""
        descrpt = self.case_film.apply(xx[..., : self.descriptor_dim], case_code, 0)
        if self.descriptor_dim == self.in_dim:
            return descrpt
        return torch.cat([descrpt, xx[..., self.descriptor_dim :]], dim=-1)

    def forward(
        self,
        xx: torch.Tensor,
        case_embd: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for the GLU fitting net.

        Parameters
        ----------
        xx
            Input tensor.
        case_embd
            Optional compact case one-hot vector with shape (K,).

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        if self.case_film_embd:
            case_code = self.case_film.encode(case_embd)
            xx = self._apply_input_film(xx, case_code)
            for layer_idx, layer in enumerate(self.hidden_layers):
                xx = layer(xx)
                xx = self.case_film.apply(xx, case_code, layer_idx + 1)
        else:
            for layer in self.hidden_layers:
                xx = layer(xx)
        return self.output_layer(xx)

    def call_until_last(
        self,
        xx: torch.Tensor,
        case_embd: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return activations before the output projection.

        Parameters
        ----------
        xx
            Input tensor.
        case_embd
            Optional compact case one-hot vector with shape (K,).

        Returns
        -------
        torch.Tensor
            Hidden activations, or input if no hidden layers exist.
        """
        if self.case_film_embd:
            case_code = self.case_film.encode(case_embd)
            xx = self._apply_input_film(xx, case_code)
            for layer_idx, layer in enumerate(self.hidden_layers):
                xx = layer(xx)
                xx = self.case_film.apply(xx, case_code, layer_idx + 1)
            return xx
        for layer in self.hidden_layers:
            xx = layer(xx)
        return xx

    def serialize(self) -> dict[str, Any]:
        """Serialize the network to a dict."""
        state = self.state_dict()
        return {
            "@class": "GLUFittingNet",
            "@version": 1,
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "neuron": self.neuron.copy(),
            "activation_function": self.activation_function,
            "resnet_dt": self.resnet_dt,
            "precision": self.precision,
            "bias_out": self.bias_out,
            "descriptor_dim": self.descriptor_dim,
            "dim_case_embd": self.dim_case_embd,
            "case_film_embd": self.case_film_embd,
            # Keep the per-layer freeze policy stable across backend round trips.
            "trainable": self.trainable.copy(),
            "@variables": {key: to_numpy_array(value) for key, value in state.items()},
        }

    @classmethod
    def deserialize(cls, data: dict) -> GLUFittingNet:
        """Deserialize the network from a dict."""
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        variables = data.pop("@variables", {})
        obj = cls(**data)
        state = {key: to_torch_tensor(value) for key, value in variables.items()}
        obj.load_state_dict(state)
        return obj


class SeZMNetworkCollection(torch.nn.Module):
    """
    Network collection for SeZM fitting networks.

    Parameters
    ----------
    ndim
        The number of type dimensions.
    ntypes
        Number of atom types.
    network_type
        The network type name. Only "sezm_fitting_network" is supported.
    networks
        The networks to initialize with.
    """

    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "sezm_fitting_network": GLUFittingNet,
    }

    def __init__(
        self,
        ndim: int,
        ntypes: int,
        network_type: str = "sezm_fitting_network",
        networks: list[GLUFittingNet | dict | None] | None = None,
    ) -> None:
        super().__init__()
        self.ndim = int(ndim)
        self.ntypes = int(ntypes)
        if network_type not in self.NETWORK_TYPE_MAP:
            raise ValueError(f"Unknown network_type: {network_type}")
        self.network_type = self.NETWORK_TYPE_MAP[network_type]
        if networks is None:
            networks = []

        total = self.ntypes**self.ndim
        self._networks: list[GLUFittingNet | None] = [None for _ in range(total)]
        for idx, network in enumerate(networks):
            self[idx] = network
        if any(net is None for net in self._networks):
            raise RuntimeError("SeZMNetworkCollection is incomplete.")
        self.networks = torch.nn.ModuleList(self._networks)

    def _convert_key(self, key: int | tuple | str) -> int:
        if isinstance(key, int):
            idx = key
        else:
            if isinstance(key, tuple):
                pass
            elif isinstance(key, str):
                key = tuple([int(tt) for tt in key.split("_")[1:]])
            else:
                raise TypeError(key)
            assert isinstance(key, tuple)
            assert len(key) == self.ndim
            idx = sum([tt * self.ntypes**ii for ii, tt in enumerate(key)])
        return idx

    def __getitem__(self, key: int | tuple | str) -> GLUFittingNet:
        idx = self._convert_key(key)
        nn = self._networks[idx]
        assert nn is not None
        return nn

    def __setitem__(self, key: int | tuple | str, value: GLUFittingNet | dict) -> None:
        if isinstance(value, self.network_type):
            network = value
        elif isinstance(value, dict):
            network = self.network_type.deserialize(value)
        else:
            raise TypeError(value)
        idx = self._convert_key(key)
        self._networks[idx] = network

    def serialize(self) -> dict[str, Any]:
        """Serialize the networks to a dict."""
        network_type_map_inv = {v: k for k, v in self.NETWORK_TYPE_MAP.items()}
        return {
            "@class": "NetworkCollection",
            "@version": 1,
            "ndim": self.ndim,
            "ntypes": self.ntypes,
            "network_type": network_type_map_inv[self.network_type],
            "networks": [
                nn.serialize() if nn is not None else None for nn in self._networks
            ],
        }

    @classmethod
    def deserialize(cls, data: dict) -> SeZMNetworkCollection:
        """Deserialize the networks from a dict."""
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        return cls(**data)


def _resolve_auto_neuron(
    neuron: list[int] | None,
    *,
    dim_descrpt: int,
    numb_fparam: int,
    numb_aparam: int,
    dim_case_embd: int,
    case_film_embd: bool,
    use_aparam_as_mask: bool,
) -> list[int]:
    """Resolve SeZM fitting hidden widths, using 0 as the auto-width marker."""
    resolved_neuron = [0] if neuron is None else [int(width) for width in neuron]
    if any(width < 0 for width in resolved_neuron):
        raise ValueError("`fitting_net.neuron` entries must be >= 0")
    if 0 not in resolved_neuron:
        return resolved_neuron
    case_dim = 0 if case_film_embd else int(dim_case_embd)
    dim_in = (
        int(dim_descrpt)
        + int(numb_fparam)
        + (0 if use_aparam_as_mask else int(numb_aparam))
        + case_dim
    )
    resolved_width = int(32 * math.ceil((8.0 * float(dim_in) / 3.0) / 32.0))
    return [resolved_width if width == 0 else width for width in resolved_neuron]


@Fitting.register("dpa4_ener")
@Fitting.register("sezm_ener")
class SeZMEnergyFittingNet(InvarFitting):
    """
    SeZM energy fitting with GLU hidden layers.

    This uses the same configuration keys as the standard energy fitting
    but replaces hidden MLP layers with GLU blocks.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        neuron: list[int] | None = None,
        bias_atom_e: torch.Tensor | None = None,
        resnet_dt: bool = False,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        case_film_embd: bool = False,
        activation_function: str = "silu",
        bias_out: bool = False,
        precision: str = "float32",
        mixed_types: bool = True,
        seed: int | list[int] | None = None,
        type_map: list[str] | None = None,
        default_fparam: list | None = None,
        **kwargs: Any,
    ) -> None:
        neuron = _resolve_auto_neuron(
            neuron,
            dim_descrpt=dim_descrpt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            case_film_embd=case_film_embd,
            use_aparam_as_mask=bool(kwargs.get("use_aparam_as_mask", False)),
        )
        super().__init__(
            "energy",
            ntypes,
            dim_descrpt,
            1,
            neuron=neuron,
            bias_atom_e=bias_atom_e,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            seed=seed,
            type_map=type_map,
            default_fparam=default_fparam,
            **kwargs,
        )
        self.bias_out = bool(bias_out)
        self.case_film_embd = bool(case_film_embd and self.dim_case_embd > 0)
        self._build_glu_fitting_layers()

    def _build_glu_fitting_layers(self) -> None:
        # === Step 1. Derive input/output dimensions ===
        case_dim = 0 if self.case_film_embd else self.dim_case_embd
        in_dim = (
            self.dim_descrpt
            + self.numb_fparam
            + (0 if self.use_aparam_as_mask else self.numb_aparam)
            + case_dim
        )
        net_dim_out = self._net_out_dim()
        n_networks = self.ntypes if not self.mixed_types else 1

        # === Step 2. Build GLU fitting networks ===
        self.filter_layers = SeZMNetworkCollection(
            1 if not self.mixed_types else 0,
            self.ntypes,
            network_type="sezm_fitting_network",
            networks=[
                GLUFittingNet(
                    in_dim,
                    net_dim_out,
                    self.neuron,
                    activation_function=self.activation_function,
                    resnet_dt=self.resnet_dt,
                    precision=self.precision,
                    bias_out=self.bias_out,
                    seed=child_seed(self.seed, idx),
                    trainable=self.trainable,
                    descriptor_dim=self.dim_descrpt,
                    dim_case_embd=self.dim_case_embd,
                    case_film_embd=self.case_film_embd,
                )
                for idx in range(n_networks)
            ],
        )
        for param in self.parameters():
            param.requires_grad = self.trainable

    def _forward_common(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: torch.Tensor | None = None,
        g2: torch.Tensor | None = None,
        h2: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        return_atomic_feature: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run the SeZM fitting path with optional case FiLM."""
        if not self.case_film_embd:
            return super()._forward_common(
                descriptor,
                atype,
                gr,
                g2,
                h2,
                fparam,
                aparam,
                return_atomic_feature=return_atomic_feature,
            )
        return self._forward_case_film(
            descriptor,
            atype,
            fparam,
            aparam,
            return_atomic_feature=return_atomic_feature,
        )

    def _forward_case_film(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        return_atomic_feature: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward path for SeZM case FiLM.

        Parameters
        ----------
        descriptor
            Descriptor tensor with shape (nf, nloc, dim_descrpt).
        atype
            Atom types with shape (nf, nloc).
        fparam
            Frame parameters with shape (nf, numb_fparam).
        aparam
            Atomic parameters with shape (nf, nloc, numb_aparam).
        return_atomic_feature
            When True, also return the last hidden activation under the
            ``atomic_feature`` key.

        Returns
        -------
        dict[str, torch.Tensor]
            Per-atom fitting outputs.
        """
        xx = descriptor.to(self.prec)
        nf, nloc, nd = xx.shape
        if self.numb_fparam > 0 and fparam is None:
            assert self.default_fparam_tensor is not None
            fparam = torch.tile(self.default_fparam_tensor.unsqueeze(0), [nf, 1])
        fparam = fparam.to(self.prec) if fparam is not None else None
        aparam = aparam.to(self.prec) if aparam is not None else None

        if self.remove_vaccum_contribution is not None:
            xx_zeros = torch.zeros_like(xx)
        else:
            xx_zeros = None
        net_dim_out = self._net_out_dim()

        if nd != self.dim_descrpt:
            raise ValueError(
                f"get an input descriptor of dim {nd},"
                f"which is not consistent with {self.dim_descrpt}."
            )

        if self.numb_fparam > 0:
            assert fparam is not None, "fparam should not be None"
            assert self.fparam_avg is not None
            assert self.fparam_inv_std is not None
            if fparam.numel() != nf * self.numb_fparam:
                raise ValueError(
                    f"input fparam: cannot reshape {list(fparam.shape)} "
                    f"into ({nf}, {self.numb_fparam})."
                )
            fparam = fparam.view([nf, self.numb_fparam])
            nb, _ = fparam.shape
            t_fparam_avg = self._extend_f_avg_std(self.fparam_avg, nb)
            t_fparam_inv_std = self._extend_f_avg_std(self.fparam_inv_std, nb)
            fparam = (fparam - t_fparam_avg) * t_fparam_inv_std
            fparam = torch.tile(fparam.reshape([nf, 1, -1]), [1, nloc, 1])
            xx = torch.cat([xx, fparam], dim=-1)
            if xx_zeros is not None:
                xx_zeros = torch.cat([xx_zeros, fparam], dim=-1)

        if self.numb_aparam > 0 and not self.use_aparam_as_mask:
            assert aparam is not None, "aparam should not be None"
            assert self.aparam_avg is not None
            assert self.aparam_inv_std is not None
            if aparam.numel() % (nf * self.numb_aparam) != 0:
                raise ValueError(
                    f"input aparam: cannot reshape {list(aparam.shape)} "
                    f"into ({nf}, nloc, {self.numb_aparam})."
                )
            aparam = aparam.view([nf, -1, self.numb_aparam])
            nb, nloc, _ = aparam.shape
            t_aparam_avg = self._extend_a_avg_std(self.aparam_avg, nb, nloc)
            t_aparam_inv_std = self._extend_a_avg_std(self.aparam_inv_std, nb, nloc)
            aparam = (aparam - t_aparam_avg) * t_aparam_inv_std
            xx = torch.cat([xx, aparam], dim=-1)
            if xx_zeros is not None:
                xx_zeros = torch.cat([xx_zeros, aparam], dim=-1)

        assert self.case_embd is not None
        outs = torch.zeros(
            (nf, nloc, net_dim_out),
            dtype=self.prec,
            device=descriptor.device,
        )
        results = {}

        fitting = self.filter_layers.networks[0]
        atom_property = fitting(xx, self.case_embd)
        if return_atomic_feature:
            results["atomic_feature"] = fitting.call_until_last(xx, self.case_embd)
        if xx_zeros is not None:
            atom_property -= fitting(xx_zeros, self.case_embd)
        outs = outs + atom_property + self.bias_atom_e[atype].to(self.prec)

        mask = self.emask(atype).to(torch.bool)
        outs = torch.where(mask[:, :, None], outs, 0.0)
        results.update({self.var_name: outs})
        return results

    @classmethod
    def deserialize(cls, data: dict) -> GeneralFitting:
        data = data.copy()
        variables = data.pop("@variables")
        nets = data.pop("nets")
        check_version_compatibility(data.pop("@version", 1), 4, 1)
        data.pop("var_name")
        data.pop("dim_out")
        obj = cls(**data)
        for kk in variables.keys():
            obj[kk] = to_torch_tensor(variables[kk])
        obj.filter_layers = SeZMNetworkCollection.deserialize(nets)
        return obj

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {
            **super().serialize(),
            "type": "sezm_ener",
            "case_film_embd": self.case_film_embd,
        }
