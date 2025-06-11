# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

from typing import (
    ClassVar,
)

import numpy as np
import paddle
import paddle.nn as nn

from deepmd.pd.utils import (
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
from deepmd.pd.model.network.init import (
    PaddleGenerator,
    kaiming_normal_,
    normal_,
    trunc_normal_,
    xavier_uniform_,
)
from deepmd.pd.utils.env import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)
from deepmd.pd.utils.utils import (
    ActivationFn,
    get_generator,
    to_numpy_array,
    to_paddle_tensor,
)


def empty_t(shape, precision):
    return paddle.empty(shape, dtype=precision).to(device=device)


class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        xx: paddle.Tensor,
    ) -> paddle.Tensor:
        """The Identity operation layer."""
        return xx

    def serialize(self) -> dict:
        return {
            "@class": "Identity",
            "@version": 1,
        }

    @classmethod
    def deserialize(cls, data: dict) -> Identity:
        return Identity()


class MLPLayer(nn.Layer):
    def __init__(
        self,
        num_in,
        num_out,
        bias: bool = True,
        use_timestep: bool = False,
        activation_function: str | None = None,
        resnet: bool = False,
        bavg: float = 0.0,
        stddev: float = 1.0,
        precision: str = DEFAULT_PRECISION,
        init: str = "default",
        seed: int | list[int] | None = None,
    ):
        super().__init__()
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
        self.matrix = self.create_parameter(
            (num_in, num_out),
            dtype=self.prec,
            default_initializer=nn.initializer.Assign(
                empty_t([num_in, num_out], self.prec)
            ),
        )
        random_generator = get_generator(seed)
        if bias:
            self.bias = self.create_parameter(
                [num_out],
                dtype=self.prec,
                default_initializer=nn.initializer.Assign(
                    empty_t([num_out], self.prec)
                ),
            )
        else:
            self.bias = None
        if self.use_timestep:
            self.idt = self.create_parameter(
                [num_out],
                dtype=self.prec,
                default_initializer=nn.initializer.Assign(
                    empty_t([num_out], self.prec)
                ),
            )
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

    def check_type_consistency(self):
        precision = self.precision

        def check_var(var):
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
        generator: PaddleGenerator | None = None,
    ):
        normal_(
            self.matrix.data,
            std=stddev / np.sqrt(self.num_out + self.num_in),
            generator=generator,
        )
        if self.bias is not None:
            normal_(self.bias.data, mean=bavg, std=stddev, generator=generator)
        if self.idt is not None:
            normal_(self.idt.data, mean=0.1, std=0.001, generator=generator)

    def _trunc_normal_init(self, scale=1.0, generator: PaddleGenerator | None = None):
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.matrix.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        trunc_normal_(self.matrix, mean=0.0, std=std, generator=generator)

    def _glorot_uniform_init(self, generator: PaddleGenerator | None = None):
        xavier_uniform_(self.matrix, gain=1, generator=generator)

    def _zero_init(self, use_bias=True):
        with paddle.no_grad():
            self.matrix.fill_(0.0)
            if use_bias and self.bias is not None:
                with paddle.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self, generator: PaddleGenerator | None = None):
        kaiming_normal_(self.matrix, nonlinearity="linear", generator=generator)

    def forward(
        self,
        xx: paddle.Tensor,
    ) -> paddle.Tensor:
        """One MLP layer used by DP model.

        Parameters
        ----------
        xx : paddle.Tensor
            The input.

        Returns
        -------
        yy: paddle.Tensor
            The output.
        """
        ori_prec = xx.dtype
        xx = xx.astype(self.prec)
        yy = (
            paddle.matmul(xx, self.matrix) + self.bias
            if self.bias is not None
            else paddle.matmul(xx, self.matrix)
        )
        yy = self.activate(yy).clone()
        yy = yy * self.idt if self.idt is not None else yy
        if self.resnet:
            if xx.shape[-1] == yy.shape[-1]:
                yy += xx
            elif 2 * xx.shape[-1] == yy.shape[-1]:
                yy += paddle.concat([xx, xx], axis=-1)
            # else:
            #     yy = yy
        yy = yy.astype(ori_prec)
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
        )
        nl.w, nl.b, nl.idt = (
            to_numpy_array(self.matrix),
            to_numpy_array(self.bias),
            to_numpy_array(self.idt),
        )
        return nl.serialize()

    @classmethod
    def deserialize(cls, data: dict) -> MLPLayer:
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
        )
        prec = PRECISION_DICT[obj.precision]

        def check_load_param(ss):
            if nl[ss] is not None:
                tensor = to_paddle_tensor(nl[ss])
                return paddle.create_parameter(
                    tensor.shape,
                    dtype=tensor.dtype,
                    default_initializer=nn.initializer.Assign(tensor),
                )
            return None

        obj.matrix = check_load_param("matrix")
        obj.bias = check_load_param("bias")
        obj.idt = check_load_param("idt")
        return obj


MLP_ = make_multilayer_network(MLPLayer, nn.Layer)


class MLP(MLP_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = paddle.nn.LayerList(self.layers)

    forward = MLP_.call


EmbeddingNet = make_embedding_network(MLP, MLPLayer)

FittingNet = make_fitting_network(EmbeddingNet, MLP, MLPLayer)


class NetworkCollection(DPNetworkCollection, nn.Layer):
    """Paddle implementation of NetworkCollection."""

    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "network": MLP,
        "embedding_network": EmbeddingNet,
        "fitting_network": FittingNet,
    }

    def __init__(self, *args, **kwargs):
        # init both two base classes
        DPNetworkCollection.__init__(self, *args, **kwargs)
        nn.Layer.__init__(self)
        self.networks = self._networks = paddle.nn.LayerList(self._networks)
