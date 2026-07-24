# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
from jax import (
    tree_util,
)

from deepmd.jax.env import (
    jnp,
    nnx,
)
from deepmd.jax.utils.type_embed import (
    TypeEmbedNet,
)


class TestJAXTypeEmbedNet(unittest.TestCase):
    def test_call_supports_nnx_jit_and_grad_tracing(self) -> None:
        @nnx.jit
        def forward(model):
            return model.call()

        @nnx.jit
        def grad(model):
            def loss(model):
                return jnp.sum(model.call())

            return nnx.grad(loss)(model)

        for use_econf_tebd in (False, True):
            with self.subTest(use_econf_tebd=use_econf_tebd):
                type_embedding = TypeEmbedNet(
                    ntypes=2,
                    neuron=[4],
                    padding=True,
                    activation_function="Linear",
                    precision="float32",
                    seed=1,
                    use_econf_tebd=use_econf_tebd,
                    type_map=["O", "H"] if use_econf_tebd else None,
                )

                out = forward(type_embedding)
                grad_state = grad(type_embedding)
                grad_leaves = [
                    np.asarray(leaf) for leaf in tree_util.tree_leaves(grad_state)
                ]
                self.assertEqual(tuple(out.shape), (3, 4))
                self.assertFalse(np.any(np.isnan(np.asarray(out))))
                self.assertTrue(grad_leaves)
                self.assertTrue(all(np.all(np.isfinite(leaf)) for leaf in grad_leaves))
                self.assertTrue(any(np.any(np.abs(leaf) > 0.0) for leaf in grad_leaves))


if __name__ == "__main__":
    unittest.main()
