# SPDX-License-Identifier: LGPL-3.0-or-later
import os
from typing import (
    Any,
    ClassVar,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepmd.pt.utils import (
    env,
)

device = env.DEVICE

from deepmd.dpmodel.utils import (
    NativeLayer,
)
from deepmd.dpmodel.utils import NetworkCollection as DPNetworkCollection
from deepmd.dpmodel.utils import (
    make_embedding_network,
    make_fitting_network,
    make_multilayer_network,
)
from deepmd.pt.model.network.init import (
    kaiming_normal_,
    normal_,
    trunc_normal_,
    xavier_uniform_,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
    get_generator,
    to_numpy_array,
    to_torch_tensor,
)


def empty_t(shape: tuple[int, ...], precision: torch.dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=precision, device=device)


@torch.compiler.assume_constant_result
def _use_k1_compile_visible_linear(
    input_device: torch.device | None = None,
) -> bool:
    """Keep the SM80 linear topology stable for one compiled graph."""
    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}
    cute_enabled = os.environ.get("DP_NEO_CUTE_INFER", "").strip().lower()
    if cute_enabled not in truthy:
        return False
    thin_enabled = os.environ.get("DP_CUTE_K1_THIN_WRAPPER", "").strip().lower()
    if thin_enabled in falsy:
        return False
    if thin_enabled in truthy:
        return True
    if input_device is not None and input_device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        return tuple(torch.cuda.get_device_capability(input_device)) == (8, 0)
    except RuntimeError:
        return False


def _matmul_bias(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Avoid eager addmm's expanded-bias copy and expose the add to Inductor."""
    output = torch.matmul(value, weight)
    return output if bias is None else output + bias


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        xx: torch.Tensor,
    ) -> torch.Tensor:
        """The Identity operation layer."""
        return xx

    def serialize(self) -> dict:
        return {
            "@class": "Identity",
            "@version": 1,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Identity":
        return Identity()


class MLPLayer(nn.Module):
    def __init__(
        self,
        num_in: int,
        num_out: int,
        bias: bool = True,
        use_timestep: bool = False,
        activation_function: str | None = None,
        resnet: bool = False,
        bavg: float = 0.0,
        stddev: float = 1.0,
        precision: str = DEFAULT_PRECISION,
        init: str = "default",
        seed: int | list[int] | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.trainable = trainable
        self._deepmd_cute_compile_visible_linear = False
        # only use_timestep when skip connection is established.
        self.use_timestep = use_timestep and (
            num_out == num_in or num_out == num_in * 2
        )
        self.num_in = num_in
        self.num_out = num_out
        self.activate_name = activation_function
        self.activate = ActivationFn(self.activate_name)
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.matrix = nn.Parameter(data=empty_t((num_in, num_out), self.prec))
        random_generator = get_generator(seed)
        if bias:
            self.bias = nn.Parameter(
                data=empty_t([num_out], self.prec),
            )
        else:
            self.bias = None
        if self.use_timestep:
            self.idt = nn.Parameter(data=empty_t([num_out], self.prec))
        else:
            self.idt = None
        self.resnet = resnet
        if init == "default":
            self._default_normal_init(
                bavg=bavg, stddev=stddev, generator=random_generator
            )
        elif init == "trunc_normal":
            self._trunc_normal_init(1.0, generator=random_generator)
        elif init == "relu":
            self._trunc_normal_init(2.0, generator=random_generator)
        elif init == "glorot":
            self._glorot_uniform_init(generator=random_generator)
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "kaiming_normal":
            self._normal_init(generator=random_generator)
        elif init == "final":
            self._zero_init(False)
        else:
            raise ValueError(f"Unknown initialization method: {init}")

    def check_type_consistency(self) -> None:
        precision = self.precision

        def check_var(var: torch.Tensor | None) -> None:
            if var is not None:
                # assertion "float64" == "double" would fail
                assert PRECISION_DICT[var.dtype.name] is PRECISION_DICT[precision]

        check_var(self.matrix)
        check_var(self.bias)
        check_var(self.idt)

    def dim_in(self) -> int:
        return self.matrix.shape[0]

    def dim_out(self) -> int:
        return self.matrix.shape[1]

    def _default_normal_init(
        self,
        bavg: float = 0.0,
        stddev: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> None:
        normal_(
            self.matrix.data,
            std=stddev / np.sqrt(self.num_out + self.num_in),
            generator=generator,
        )
        if self.bias is not None:
            normal_(self.bias.data, mean=bavg, std=stddev, generator=generator)
        if self.idt is not None:
            normal_(self.idt.data, mean=0.1, std=0.001, generator=generator)

    def _trunc_normal_init(
        self, scale: float = 1.0, generator: torch.Generator | None = None
    ) -> None:
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.matrix.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        trunc_normal_(self.matrix, mean=0.0, std=std, generator=generator)

    def _glorot_uniform_init(self, generator: torch.Generator | None = None) -> None:
        xavier_uniform_(self.matrix, gain=1, generator=generator)

    def _zero_init(self, use_bias: bool = True) -> None:
        with torch.no_grad():
            self.matrix.fill_(0.0)
            if use_bias and self.bias is not None:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self, generator: torch.Generator | None = None) -> None:
        kaiming_normal_(self.matrix, nonlinearity="linear", generator=generator)

    def forward(
        self,
        xx: torch.Tensor,
    ) -> torch.Tensor:
        """One MLP layer used by DP model.

        Parameters
        ----------
        xx : torch.Tensor
            The input.

        Returns
        -------
        yy: torch.Tensor
            The output.
        """
        ori_prec = xx.dtype
        if not env.DP_DTYPE_PROMOTION_STRICT:
            xx = xx.to(self.prec)
        if torch.jit.is_scripting():
            yy = F.linear(xx, self.matrix.t(), self.bias)
        elif (
            not self.training
            and xx.dtype == torch.float32
            and self.matrix.dtype == torch.float32
            and (self.bias is None or self.bias.dtype == torch.float32)
            and not torch.is_autocast_enabled(xx.device.type)
            and self._deepmd_cute_compile_visible_linear
            and _use_k1_compile_visible_linear(xx.device)
        ):
            yy = _matmul_bias(xx, self.matrix, self.bias)
        else:
            yy = F.linear(xx, self.matrix.t(), self.bias)
        yy = self.activate(yy)
        yy = yy * self.idt if self.idt is not None else yy
        if self.resnet:
            if xx.shape[-1] == yy.shape[-1]:
                yy = yy + xx
            elif 2 * xx.shape[-1] == yy.shape[-1]:
                yy = yy + torch.concat([xx, xx], dim=-1)
            else:
                yy = yy
        if not env.DP_DTYPE_PROMOTION_STRICT:
            yy = yy.to(ori_prec)
        return yy

    def serialize(self) -> dict:
        """Serialize the layer to a dict.

        Returns
        -------
        dict
            The serialized layer.
        """
        nl = NativeLayer(
            self.matrix.shape[0],
            self.matrix.shape[1],
            bias=self.bias is not None,
            use_timestep=self.idt is not None,
            activation_function=self.activate_name,
            resnet=self.resnet,
            precision=self.precision,
            trainable=self.trainable,
        )
        nl.w, nl.b, nl.idt = (
            to_numpy_array(self.matrix),
            to_numpy_array(self.bias),
            to_numpy_array(self.idt),
        )
        return nl.serialize()

    @classmethod
    def deserialize(cls, data: dict) -> "MLPLayer":
        """Deserialize the layer from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        nl = NativeLayer.deserialize(data)
        obj = cls(
            nl["matrix"].shape[0],
            nl["matrix"].shape[1],
            bias=nl["bias"] is not None,
            use_timestep=nl["idt"] is not None,
            activation_function=nl["activation_function"],
            resnet=nl["resnet"],
            precision=nl["precision"],
            trainable=nl["trainable"],
        )
        prec = PRECISION_DICT[obj.precision]

        def check_load_param(ss: str) -> nn.Parameter | None:
            return (
                nn.Parameter(data=to_torch_tensor(nl[ss]))
                if nl[ss] is not None
                else None
            )

        obj.matrix = check_load_param("matrix")
        obj.bias = check_load_param("bias")
        obj.idt = check_load_param("idt")
        return obj


def enable_neo_cute_compile_visible_linears(module: nn.Module) -> None:
    """Select the alternate eval linear topology only inside one Neo model."""
    for child in module.modules():
        if isinstance(child, MLPLayer):
            child._deepmd_cute_compile_visible_linear = True


MLP_ = make_multilayer_network(MLPLayer, nn.Module)


class MLP(MLP_):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layers = torch.nn.ModuleList(self.layers)

    forward = MLP_.call


EmbeddingNet = make_embedding_network(MLP, MLPLayer)

FittingNet = make_fitting_network(EmbeddingNet, MLP, MLPLayer)


class GLULayer(nn.Module):
    """
    A GLU block for MLPs: Linear -> split -> value * act(gate).

    Parameters
    ----------
    num_in
        Input dimension.
    num_out
        Output dimension.
    activation_function
        Activation function applied to the gate branch.
    precision
        Numerical precision.
    bias
        Whether to use bias in the linear layer.
    seed
        Random seed for weight initialization.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        num_in: int,
        num_out: int,
        activation_function: str,
        precision: str,
        seed: int | list[int] | None,
        trainable: bool,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_in = int(num_in)
        self.num_out = int(num_out)
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]

        self.linear = MLPLayer(
            num_in=self.num_in,
            num_out=2 * self.num_out,
            bias=bias,
            use_timestep=False,
            activation_function=None,
            resnet=False,
            precision=self.precision,
            seed=seed,
            trainable=trainable,
        )
        self.activation = ActivationFn(self.activation_function)

    def forward(self, xx: torch.Tensor) -> torch.Tensor:
        """
        Apply GLU transformation.

        Parameters
        ----------
        xx
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        yy = self.linear(xx)
        val, gate = yy.chunk(2, dim=-1)
        return val * self.activation(gate)


class NetworkCollection(DPNetworkCollection, nn.Module):
    """PyTorch implementation of NetworkCollection."""

    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "network": MLP,
        "embedding_network": EmbeddingNet,
        "fitting_network": FittingNet,
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # init both two base classes
        DPNetworkCollection.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)
        self.networks = self._networks = torch.nn.ModuleList(self._networks)
