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

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    Array,
    xp_matmul,
)
from deepmd.dpmodel.common import (
    PRECISION_DICT,
    to_numpy_array,
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


class CaseFiLMConditioner(NativeOP):
    """Generate per-case scales and shifts for the fitting features."""

    PARAMETER_ATTRIBUTES: ClassVar[tuple[str, ...]] = (
        "adam_case_film_scale_strength_log",
        "adam_case_film_shift_strength_log",
    )
    PARAMETER_LIST_ATTRIBUTES: ClassVar[tuple[str, ...]] = ("projectors",)

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
        self.dim_case_embd = dim_case_embd
        self.dim_descrpt = dim_descrpt
        self.target_dims = list(target_dims)
        self.activation_function = activation_function
        self.precision = precision
        self.trainable = trainable
        self.code_dim = 4 * dim_descrpt
        hidden_dim = 32 * math.ceil(4 * dim_case_embd / 32)
        self.case_layer1 = NativeLayer(
            dim_case_embd,
            hidden_dim,
            bias=False,
            use_timestep=False,
            activation_function=activation_function,
            resnet=False,
            precision=precision,
            seed=child_seed(seed, 0),
            trainable=trainable,
        )
        self.case_layer2 = NativeLayer(
            hidden_dim,
            self.code_dim,
            bias=False,
            use_timestep=False,
            activation_function=None,
            resnet=False,
            precision=precision,
            seed=child_seed(seed, 1),
            trainable=trainable,
        )
        dtype = PRECISION_DICT[precision]
        self.projectors = [
            np.zeros((self.code_dim, 2 * target_dim), dtype=dtype)
            for target_dim in target_dims
        ]
        self.adam_case_film_scale_strength_log = np.full(
            len(target_dims), math.log(0.01), dtype=dtype
        )
        self.adam_case_film_shift_strength_log = np.full(
            len(target_dims), math.log(0.01), dtype=dtype
        )

    def encode(self, case_embd: Array) -> Array:
        """Encode one case vector with shape (K,) into shape (1, 4*dim_descrpt)."""
        xp = array_api_compat.array_namespace(case_embd)
        code = xp.reshape(case_embd, (1, self.dim_case_embd))
        return self.case_layer2(self.case_layer1(code))

    call = encode

    def apply(self, xx: Array, case_code: Array, target_idx: int) -> Array:
        """Modulate one feature target without changing its leading axes."""
        xp = array_api_compat.array_namespace(xx, case_code)
        film = xp_matmul(case_code, self.projectors[target_idx])
        dim = self.target_dims[target_idx]
        shape = (1,) * (xx.ndim - 1) + (dim,)
        gamma = xp.reshape(film[..., :dim], shape)
        beta = xp.reshape(film[..., dim:], shape)
        scale = xp.exp(self.adam_case_film_scale_strength_log[target_idx])
        shift = xp.exp(self.adam_case_film_shift_strength_log[target_idx])
        return xx * (1.0 + scale * xp.tanh(gamma)) + shift * xp.tanh(beta)

    def variables(self) -> dict[str, np.ndarray]:
        """Return weights under the PyTorch conditioner state-dict keys."""
        return {
            "case_layer1.matrix": to_numpy_array(self.case_layer1.w),
            "case_layer2.matrix": to_numpy_array(self.case_layer2.w),
            "adam_case_film_scale_strength_log": to_numpy_array(
                self.adam_case_film_scale_strength_log
            ),
            "adam_case_film_shift_strength_log": to_numpy_array(
                self.adam_case_film_shift_strength_log
            ),
            **{
                f"projectors.{i}": to_numpy_array(weight)
                for i, weight in enumerate(self.projectors)
            },
        }

    def set_variables(self, variables: dict[str, Array]) -> None:
        """Restore weights without altering the target configuration."""
        self.case_layer1["matrix"] = variables["case_layer1.matrix"]
        self.case_layer2["matrix"] = variables["case_layer2.matrix"]
        self.adam_case_film_scale_strength_log = variables[
            "adam_case_film_scale_strength_log"
        ]
        self.adam_case_film_shift_strength_log = variables[
            "adam_case_film_shift_strength_log"
        ]
        self.projectors = [
            variables[f"projectors.{index}"] for index in range(len(self.projectors))
        ]

    def serialize(self) -> dict[str, Any]:
        return {
            "dim_case_embd": self.dim_case_embd,
            "dim_descrpt": self.dim_descrpt,
            "target_dims": self.target_dims,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "seed": None,
            "trainable": self.trainable,
            "@variables": self.variables(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "CaseFiLMConditioner":
        data = data.copy()
        variables = data.pop("@variables")
        obj = cls(**data)
        obj.set_variables(variables)
        return obj


class GLUFittingNet(NativeOP):
    r"""
    GLU-based fitting network for SeZM.

    Each hidden layer splits its affine output into a value and a gate.  For
    input :math:`\mathbf x^{(l)}`, the update is

    .. math::
        [\mathbf v^{(l)},\mathbf g^{(l)}]
        = \mathbf x^{(l)}\mathbf W^{(l)}+\mathbf b^{(l)},
        \qquad
        \mathbf x^{(l+1)}=\mathbf v^{(l)}\odot
        \phi(\mathbf g^{(l)}),

    followed by the output projection

    .. math::
        \mathbf y=\mathbf x^{(L)}\mathbf W^{\mathrm{out}}
        +\mathbf b^{\mathrm{out}}.

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
        Whether the complete network is trainable. A list is combined with
        ``all`` to match the PyTorch SeZM fitting contract.
    descriptor_dim
        Descriptor feature width, excluding frame and atomic parameters.
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
        if neuron is None:
            neuron = []
        if isinstance(trainable, list):
            trainable = all(trainable)
        self.trainable = bool(trainable)
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

        # === Step 2. Build optional case FiLM conditioner ===
        self.case_film = (
            CaseFiLMConditioner(
                dim_case_embd=self.dim_case_embd,
                dim_descrpt=self.descriptor_dim,
                target_dims=[self.descriptor_dim, *self.neuron],
                activation_function=self.activation_function,
                precision=self.precision,
                seed=child_seed(seed, len(self.neuron)),
                trainable=trainable,
            )
            if self.case_film_embd
            else None
        )

        # === Step 3. Build output projection ===
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

    def call_until_last(self, xx: Array, case_embd: Array | None = None) -> Array:
        """Return activations before the output projection."""
        case_code = None
        if self.case_film_embd:
            assert self.case_film is not None and case_embd is not None
            xp = array_api_compat.array_namespace(xx)
            case_code = self.case_film.encode(case_embd)
            descriptor = self.case_film.apply(
                xx[..., : self.descriptor_dim], case_code, 0
            )
            xx = xp.concat([descriptor, xx[..., self.descriptor_dim :]], axis=-1)
        act = get_activation_fn(self.activation_function)
        for index, (hidden_dim, layer) in enumerate(
            zip(self.neuron, self.hidden_layers, strict=True)
        ):
            yy = layer(xx)
            val, gate = yy[..., :hidden_dim], yy[..., hidden_dim:]
            xx = val * act(gate)
            if case_code is not None:
                xx = self.case_film.apply(xx, case_code, index + 1)
        return xx

    def call_last(self, xx: Array) -> Array:
        """Project the activations of :meth:`call_until_last` to the output."""
        return self.output_layer(xx)

    def call(self, xx: Array, case_embd: Array | None = None) -> Array:
        """Forward pass for the GLU fitting net."""
        return self.call_last(self.call_until_last(xx, case_embd))

    def serialize(self) -> dict[str, Any]:
        """Serialize the network to a dict (pt state_dict key contract)."""
        variables: dict[str, Any] = {}
        for layer_idx, layer in enumerate(self.hidden_layers):
            variables[f"hidden_layers.{layer_idx}.linear.matrix"] = to_numpy_array(
                layer.w
            )
            variables[f"hidden_layers.{layer_idx}.linear.bias"] = to_numpy_array(
                layer.b
            )
        variables["output_layer.matrix"] = to_numpy_array(self.output_layer.w)
        if self.bias_out:
            variables["output_layer.bias"] = to_numpy_array(self.output_layer.b)
        if self.case_film is not None:
            variables.update(
                {
                    f"case_film.{key}": value
                    for key, value in self.case_film.variables().items()
                }
            )
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
            "trainable": self.trainable,
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
        if obj.case_film is not None:
            obj.case_film.set_variables(
                {
                    key.removeprefix("case_film."): value
                    for key, value in variables.items()
                    if key.startswith("case_film.")
                }
            )
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


def _with_default_trainable(
    networks: dict[str, Any],
    trainable: bool | list[bool],
) -> dict[str, Any]:
    """Inherit outer trainability for legacy GLU records without that field."""
    return {
        **networks,
        "networks": [
            {"trainable": trainable, **network} if network is not None else None
            for network in networks["networks"]
        ],
    }


def _set_trainable_metadata(module: Any, trainable: bool) -> None:
    """Align an existing module declaration with a whole-module runtime policy."""
    if hasattr(module, "trainable"):
        module.trainable = (
            [trainable] * len(module.trainable)
            if isinstance(module.trainable, list)
            else trainable
        )


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
    but replaces hidden MLP layers with GLU blocks. A ``trainable`` list is
    combined with ``all``, matching the PyTorch whole-network freeze contract.
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

    def _call_common(
        self,
        descriptor: Array,
        atype: Array,
        gr: Array | None = None,
        g2: Array | None = None,
        h2: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        vacuum_descriptor: Array | None = None,
        return_middle_output: bool = False,
    ) -> dict[str, Array]:
        if not self.case_film_embd:
            return super()._call_common(
                descriptor,
                atype,
                gr,
                g2,
                h2,
                fparam,
                aparam,
                vacuum_descriptor=vacuum_descriptor,
                return_middle_output=return_middle_output,
            )
        return self._call_case_film(
            descriptor,
            atype,
            fparam,
            aparam,
            vacuum_descriptor=vacuum_descriptor,
            return_middle_output=return_middle_output,
        )

    def _call_case_film(
        self,
        descriptor: Array,
        atype: Array,
        fparam: Array | None = None,
        aparam: Array | None = None,
        vacuum_descriptor: Array | None = None,
        return_middle_output: bool = False,
    ) -> dict[str, Array]:
        """Evaluate the case-modulated readout with common vacuum conditioning."""
        xp = array_api_compat.array_namespace(descriptor, atype)
        nf, nloc, nd = descriptor.shape
        if nd != self.dim_descrpt:
            raise ValueError(f"Expected descriptor width {self.dim_descrpt}, got {nd}.")
        cond = self.conditioning_columns(
            descriptor, fparam, aparam, include_case_embedding=False
        )
        xx = descriptor
        xx_zeros = (
            xp.zeros_like(descriptor)
            if self.remove_vaccum_contribution is not None
            else None
        )
        if cond is not None:
            xx = xp.concat([xx, cond], axis=-1)
            if xx_zeros is not None:
                xx_zeros = xp.concat([xx_zeros, cond], axis=-1)
        xx_vac = self.vacuum_input(
            vacuum_descriptor, atype, cond, include_case_embedding=False
        )
        fitting = self.nets[()]
        hidden = fitting.call_until_last(xx, self.case_embd)
        atom_property = fitting.call_last(hidden)
        results = {}
        if return_middle_output or self.eval_return_middle_output:
            results["middle_output"] = hidden
        if xx_zeros is not None:
            atom_property -= fitting(xx_zeros, self.case_embd)
        if xx_vac is not None:
            atom_property -= self.vacuum_output(fitting(xx_vac, self.case_embd), atype)
        bias = xp.asarray(
            self.bias_atom_e,
            dtype=atom_property.dtype,
            device=array_api_compat.device(atom_property),
        )
        outs = atom_property + xp.reshape(
            xp.take(bias, xp.reshape(atype, (-1,)), axis=0),
            (nf, nloc, self.dim_out),
        )
        mask = xp.astype(self.emask.build_type_exclude_mask(atype), xp.bool)
        results[self.var_name] = xp.where(mask[..., None], outs, xp.zeros_like(outs))
        return results

    def vacuum_property(self, vacuum_descriptor: Array | None) -> Array:
        if not self.case_film_embd:
            return super().vacuum_property(vacuum_descriptor)
        if vacuum_descriptor is None:
            vacuum_descriptor = self.vacuum_table
        if vacuum_descriptor is None:
            raise ValueError("vacuum_ref requires the descriptor of every atom type.")
        xp = array_api_compat.array_namespace(vacuum_descriptor)
        atype = xp.arange(
            self.ntypes,
            dtype=xp.int64,
            device=array_api_compat.device(vacuum_descriptor),
        )
        xx_vac = self.vacuum_input(
            vacuum_descriptor, atype, None, include_case_embedding=False
        )
        return self.nets[()](xx_vac, self.case_embd)

    def fold_case_embedding(self) -> None:
        """Absorb a deployed case into the GLU affine maps.

        A fixed FiLM target is an affine map of its features. For a following
        linear layer, ``(x * scale + shift) @ W + b`` equals
        ``x @ (scale[:, None] * W) + shift @ W + b``. Input FiLM modifies
        only descriptor columns, leaving frame and atomic parameters intact.
        """
        if not self.dim_case_embd:
            return
        if self.case_film_embd and self.eval_return_middle_output:
            return
        case = to_numpy_array(self.case_embd)
        n_networks = self.ntypes if not self.mixed_types else 1
        for index in range(n_networks):
            net = self.nets[index]
            layers = [*net.hidden_layers, net.output_layer]
            if self.case_film_embd:
                conditioner = net.case_film
                code = to_numpy_array(conditioner.encode(self.case_embd))
                scale_strength = np.exp(
                    to_numpy_array(conditioner.adam_case_film_scale_strength_log)
                )
                shift_strength = np.exp(
                    to_numpy_array(conditioner.adam_case_film_shift_strength_log)
                )
                for target_idx, layer in enumerate(layers):
                    width = conditioner.target_dims[target_idx]
                    film = code @ to_numpy_array(conditioner.projectors[target_idx])
                    scale = 1.0 + scale_strength[target_idx] * np.tanh(film[0, :width])
                    shift = shift_strength[target_idx] * np.tanh(film[0, width:])
                    weight = to_numpy_array(layer.w).copy()
                    bias = (
                        np.zeros(weight.shape[1], dtype=weight.dtype)
                        if layer.b is None
                        else to_numpy_array(layer.b).copy()
                    )
                    bias += shift @ weight[:width]
                    weight[:width] *= scale[:, None]
                    layer["matrix"], layer["bias"] = weight, bias
                net.case_film = None
                net.case_film_embd = False
                net.bias_out = True
            else:
                layer = layers[0]
                weight = to_numpy_array(layer.w)
                kept = weight.shape[0] - self.dim_case_embd
                bias = (
                    np.zeros(weight.shape[1], dtype=weight.dtype)
                    if layer.b is None
                    else to_numpy_array(layer.b)
                )
                layer["matrix"] = weight[:kept].copy()
                layer["bias"] = bias + case @ weight[kept:]
                net.in_dim = kept
                if not net.hidden_layers:
                    net.bias_out = True
            net.dim_case_embd = 0
        self.bias_out = all(self.nets[index].bias_out for index in range(n_networks))
        self.case_film_embd = False
        self.dim_case_embd = 0
        self.case_embd = None

    @classmethod
    def deserialize(cls, data: dict) -> "SeZMEnergyFittingNet":
        data = data.copy()
        variables = data.pop("@variables")
        nets = data.pop("nets")
        check_version_compatibility(data.pop("@version", 1), 5, 1)
        data.pop("@class", None)
        data.pop("type", None)
        data.pop("var_name")
        data.pop("dim_out")
        obj = cls(**data)
        for kk in variables.keys():
            obj[kk] = variables[kk]
        obj.nets = SeZMNetworkCollection.deserialize(
            _with_default_trainable(nets, obj.trainable)
        )
        return obj

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {
            **super().serialize(),
            "type": "sezm_ener",
            "case_film_embd": self.case_film_embd,
        }
