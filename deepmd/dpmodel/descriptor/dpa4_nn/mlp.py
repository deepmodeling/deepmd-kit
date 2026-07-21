# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bias-free SwiGLU multilayer perceptrons for DPA4-family descriptors."""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.network import (
    NativeLayer,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .activation import (
    SwiGLU,
)


def resolve_swiglu_hidden_width(width: int, multiple: int = 8) -> int:
    r"""Return the parameter-matched SwiGLU hidden width.

    The post-gate width is :math:`8d/3`, rounded up to ``multiple``. The
    corresponding hidden affine map produces twice this width for the value
    and gate branches.

    Parameters
    ----------
    width
        Input and output model width.
    multiple
        Alignment multiple for the post-gate hidden width.

    Returns
    -------
    int
        Aligned post-gate hidden width.
    """
    if width <= 0:
        raise ValueError(f"`width` must be positive, got {width}")
    if multiple <= 0:
        raise ValueError(f"`multiple` must be positive, got {multiple}")
    numerator = 8 * int(width)
    denominator = 3 * int(multiple)
    return int(multiple) * ((numerator + denominator - 1) // denominator)


class SwiGLUMLP(NativeOP):
    """Apply bias-free SwiGLU hidden layers and a linear output projection.

    For hidden width ``H``, each hidden affine map produces ``2H`` channels.
    :class:`SwiGLU` splits them into equal gate and value branches and returns
    ``SiLU(gate) * value`` with width ``H``. The final layer is linear.

    Parameters
    ----------
    mlp_layers
        Layer widths including input, hidden, and output dimensions.
    output_scale
        Fixed multiplier applied to the final output.
    precision
        Parameter precision.
    trainable
        Whether the linear weights are trainable.
    seed
        Random seed.
    """

    def __init__(
        self,
        mlp_layers: list[int],
        *,
        output_scale: float = 1.0,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        if len(mlp_layers) < 2:
            raise ValueError("`mlp_layers` must contain input and output widths")
        if any(width <= 0 for width in mlp_layers):
            raise ValueError(f"`mlp_layers` must be positive, got {mlp_layers}")
        self.mlp_layers = [int(width) for width in mlp_layers]
        self.output_scale = float(output_scale)
        self.precision = str(precision)
        self.trainable = bool(trainable)

        layers = []
        for index, (width_in, width_out) in enumerate(
            zip(self.mlp_layers[:-1], self.mlp_layers[1:], strict=True)
        ):
            is_output = index == len(self.mlp_layers) - 2
            layers.append(
                NativeLayer(
                    width_in,
                    width_out if is_output else 2 * width_out,
                    bias=False,
                    precision=self.precision,
                    seed=child_seed(seed, index),
                    trainable=self.trainable,
                )
            )
        self.layers = layers
        self.activation = SwiGLU()

    def call(self, inputs: Any) -> Any:
        """Evaluate the SwiGLU MLP.

        Parameters
        ----------
        inputs
            Input with shape ``(..., mlp_layers[0])``.

        Returns
        -------
        Any
            Output with shape ``(..., mlp_layers[-1])``.
        """
        return self.call_output(self.call_hidden(inputs))

    def call_hidden(self, inputs: Any) -> Any:
        """Evaluate every hidden layer and return the latent state.

        The latent state is exposed separately so that several output heads
        can branch off one trunk evaluation.

        Parameters
        ----------
        inputs
            Input with shape ``(..., mlp_layers[0])``.

        Returns
        -------
        Any
            Activated latent state with shape ``(..., mlp_layers[-2])``.
        """
        output = inputs
        for layer in self.layers[:-1]:
            output = self.activation(layer(output))
        return output

    def call_output(self, hidden: Any) -> Any:
        """Apply the final scaled linear projection to a latent state.

        Parameters
        ----------
        hidden
            Latent state with shape ``(..., mlp_layers[-2])``, as returned by
            :meth:`call_hidden`.

        Returns
        -------
        Any
            Output with shape ``(..., mlp_layers[-1])``.
        """
        return self.layers[-1](hidden) * self.output_scale

    def serialize(self) -> dict[str, Any]:
        """Serialize the MLP configuration and linear weights."""
        return {
            "@class": "SwiGLUMLP",
            "@version": 1,
            "mlp_layers": self.mlp_layers.copy(),
            "output_scale": self.output_scale,
            "precision": np.dtype(PRECISION_DICT[self.precision]).name,
            "trainable": self.trainable,
            "@variables": {
                f"{index}.matrix": to_numpy_array(layer.w)
                for index, layer in enumerate(self.layers)
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SwiGLUMLP:
        """Deserialize a :class:`SwiGLUMLP`."""
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        if data.pop("@class") != "SwiGLUMLP":
            raise ValueError("Invalid serialized class for SwiGLUMLP")
        variables = data.pop("@variables")
        obj = cls(**data)
        dtype = PRECISION_DICT[obj.precision]
        for key, value in variables.items():
            index, _, name = key.partition(".")
            if name != "matrix":
                raise ValueError(f"Invalid SwiGLUMLP variable {key!r}")
            obj.layers[int(index)].w = np.asarray(value, dtype=dtype)
        return obj
