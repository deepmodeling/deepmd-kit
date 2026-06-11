# SPDX-License-Identifier: LGPL-3.0-or-later
"""SeZM (DPA4) GLU energy fitting network, dpmodel implementation.

Mirrors ``deepmd.pt.model.task.sezm_ener`` with the array-API ``call``
convention and the pt-state_dict-key serialization contract.
"""

import math
from typing import (
    Any,
    ClassVar,
)

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.utils.network import (
    NativeLayer,
    get_activation_fn,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .invar_fitting import (
    InvarFitting,
)


class GLUFittingNet(NativeOP):
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
        Descriptor feature width. Kept for serialization compatibility
        with the case-FiLM path (not implemented here).
    dim_case_embd
        Case one-hot width.
    case_film_embd
        Whether to use case FiLM instead of input concatenation.
        Not implemented in the dpmodel backend.
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
        if case_film_embd and int(dim_case_embd) > 0:
            raise NotImplementedError(
                "case_film_embd is not implemented in the dpmodel backend"
            )
        if neuron is None:
            neuron = []
        if isinstance(trainable, list):
            trainable = all(trainable)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.neuron = [int(nn_dim) for nn_dim in neuron]
        self.activation_function = activation_function
        self.resnet_dt = bool(resnet_dt)
        self.precision = precision
        self.bias_out = bool(bias_out)
        self.descriptor_dim = (
            self.in_dim if descriptor_dim is None else int(descriptor_dim)
        )
        self.dim_case_embd = int(dim_case_embd)
        self.case_film_embd = bool(case_film_embd and self.dim_case_embd > 0)

        # === Step 1. Build GLU hidden layers ===
        # Each hidden layer is a linear map to 2*hidden_dim, split into
        # value and gate halves: out = val * act(gate).
        hidden_layers = []
        dim_in = self.in_dim
        for layer_idx, hidden_dim in enumerate(self.neuron):
            hidden_layers.append(
                NativeLayer(
                    dim_in,
                    2 * hidden_dim,
                    bias=True,
                    use_timestep=False,
                    activation_function=None,
                    resnet=False,
                    precision=self.precision,
                    seed=child_seed(seed, layer_idx),
                    trainable=trainable,
                )
            )
            dim_in = hidden_dim
        self.hidden_layers = hidden_layers

        # === Step 2. Build output projection ===
        self.output_layer = NativeLayer(
            dim_in,
            self.out_dim,
            bias=self.bias_out,
            use_timestep=False,
            activation_function=None,
            resnet=False,
            precision=self.precision,
            seed=child_seed(seed, len(self.neuron) + int(self.case_film_embd)),
            trainable=trainable,
        )

    def call_until_last(self, xx: Array) -> Array:
        """Return activations before the output projection."""
        act = get_activation_fn(self.activation_function)
        for hidden_dim, layer in zip(self.neuron, self.hidden_layers, strict=True):
            yy = layer(xx)
            val, gate = yy[..., :hidden_dim], yy[..., hidden_dim:]
            xx = val * act(gate)
        return xx

    def call(self, xx: Array) -> Array:
        """Forward pass for the GLU fitting net."""
        return self.output_layer(self.call_until_last(xx))

    def serialize(self) -> dict[str, Any]:
        """Serialize the network to a dict (pt state_dict key contract)."""
        variables: dict[str, Any] = {}
        for layer_idx, layer in enumerate(self.hidden_layers):
            variables[f"hidden_layers.{layer_idx}.linear.matrix"] = layer.w
            variables[f"hidden_layers.{layer_idx}.linear.bias"] = layer.b
        variables["output_layer.matrix"] = self.output_layer.w
        if self.bias_out:
            variables["output_layer.bias"] = self.output_layer.b
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
            "@variables": variables,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "GLUFittingNet":
        """Deserialize the network from a dict."""
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        variables = data.pop("@variables", {})
        obj = cls(**data)
        for layer_idx, layer in enumerate(obj.hidden_layers):
            layer["matrix"] = variables[f"hidden_layers.{layer_idx}.linear.matrix"]
            layer["bias"] = variables[f"hidden_layers.{layer_idx}.linear.bias"]
        obj.output_layer["matrix"] = variables["output_layer.matrix"]
        if obj.bias_out:
            obj.output_layer["bias"] = variables["output_layer.bias"]
        return obj


class SeZMNetworkCollection:
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
        networks: list[Any] | None = None,
    ) -> None:
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
        self.networks = self._networks

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
            if len(key) != self.ndim:
                raise KeyError(
                    f"key {key} has length {len(key)}, expected ndim {self.ndim}"
                )
            if any(not (0 <= int(tt) < self.ntypes) for tt in key):
                raise KeyError(
                    f"key {key} contains type indices outside [0, {self.ntypes})"
                )
            idx = sum([tt * self.ntypes**ii for ii, tt in enumerate(key)])
        if not (0 <= idx < self.ntypes**self.ndim):
            raise KeyError(
                f"key {key} maps to index {idx}, outside [0, {self.ntypes**self.ndim})"
            )
        return idx

    def __getitem__(self, key: int | tuple | str) -> GLUFittingNet:
        idx = self._convert_key(key)
        nn = self._networks[idx]
        if nn is None:
            raise KeyError(f"network for key {key} is not set")
        return nn

    def __setitem__(self, key: int | tuple | str, value: Any) -> None:
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
    def deserialize(cls, data: dict) -> "SeZMNetworkCollection":
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


@InvarFitting.register("dpa4_ener")
@InvarFitting.register("sezm_ener")
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
        bias_atom_e: Array | None = None,
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
        if int(dim_case_embd) > 0:
            raise NotImplementedError(
                "dim_case_embd > 0 is not implemented in the dpmodel backend"
            )
        if case_film_embd:
            raise NotImplementedError(
                "case_film_embd is not implemented in the dpmodel backend"
            )
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
            bias_atom=bias_atom_e,
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
        self.seed = seed
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
        self.nets = SeZMNetworkCollection(
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

    @classmethod
    def deserialize(cls, data: dict) -> "SeZMEnergyFittingNet":
        data = data.copy()
        variables = data.pop("@variables")
        nets = data.pop("nets")
        check_version_compatibility(data.pop("@version", 1), 4, 1)
        data.pop("@class", None)
        data.pop("type", None)
        data.pop("var_name")
        data.pop("dim_out")
        obj = cls(**data)
        for kk in variables.keys():
            obj[kk] = variables[kk]
        obj.nets = SeZMNetworkCollection.deserialize(nets)
        return obj

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {
            **super().serialize(),
            "type": "sezm_ener",
            "case_film_embd": self.case_film_embd,
        }
