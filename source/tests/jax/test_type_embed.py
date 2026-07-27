# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.jax.env import (
    jnp,
    nnx,
)
from deepmd.jax.utils.type_embed import (
    TypeEmbedNet,
)


class TestJAXTypeEmbedNet(unittest.TestCase):
    def test_call_supports_nnx_jit_and_grad_tracing(self) -> None:
        type_embedding = TypeEmbedNet(
            ntypes=2,
            neuron=[4],
            padding=True,
            activation_function="Linear",
            precision="float32",
            seed=1,
        )

        @nnx.jit
        def forward(model):
            return model.call()

        @nnx.jit
        def grad(model):
            def loss(model):
                return jnp.sum(model.call())

            return nnx.grad(loss)(model)

        out = forward(type_embedding)
        self.assertEqual(tuple(out.shape), (3, 4))
        self.assertIsNotNone(grad(type_embedding))
        self.assertFalse(np.any(np.isnan(np.asarray(out))))


if __name__ == "__main__":
    unittest.main()
