# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import array_api_compat
import numpy as np
from jax import (
    tree_util,
)

from deepmd.jax.env import (
    jnp,
    nnx,
)
from deepmd.jax.fitting.fitting import (
    EnergyFittingNet,
)


class TestJAXFitting(unittest.TestCase):
    def test_runtime_buffers_support_nnx_jit_and_grad_tracing(self) -> None:
        """Portable fitting buffers must remain usable inside JAX tracing."""
        fitting = EnergyFittingNet(
            ntypes=2,
            dim_descrpt=3,
            neuron=[5, 5],
            numb_fparam=2,
            numb_aparam=1,
            dim_case_embd=2,
            default_fparam=[0.5, -0.25],
            precision="float32",
            mixed_types=True,
            seed=20260724,
        )
        # Non-default values ensure every portable buffer contributes to the
        # traced result instead of being optimized away as an all-zero no-op.
        buffer_values = {
            "bias_atom_e": np.array([[1.5], [-0.75]], dtype=np.float32),
            "fparam_avg": np.array([0.1, -0.2], dtype=np.float32),
            "fparam_inv_std": np.array([2.0, 0.5], dtype=np.float32),
            "aparam_avg": np.array([0.25], dtype=np.float32),
            "aparam_inv_std": np.array([4.0], dtype=np.float32),
            "case_embd": np.array([0.3, -0.6], dtype=np.float32),
            "default_fparam_tensor": np.array([0.5, -0.25], dtype=np.float32),
        }
        for name, value in buffer_values.items():
            fitting[name] = value

        descriptor = jnp.asarray(
            [[[0.2, -0.1, 0.4], [0.5, 0.3, -0.2], [-0.4, 0.7, 0.1]]],
            dtype=jnp.float32,
        )
        atype = jnp.asarray([[0, 1, 0]], dtype=jnp.int32)
        aparam = jnp.asarray([[[0.5], [0.0], [1.0]]], dtype=jnp.float32)

        @nnx.jit
        def forward(model, descriptor, atype, aparam):
            return model(descriptor, atype, aparam=aparam)["energy"]

        @nnx.jit
        def grad(model, descriptor, atype, aparam):
            def loss(model):
                return jnp.sum(model(descriptor, atype, aparam=aparam)["energy"])

            return nnx.grad(loss)(model)

        buffer_snapshots = {
            name: (
                fitting[name],
                type(fitting[name]),
                array_api_compat.array_namespace(fitting[name]),
                np.asarray(fitting[name]).copy(),
            )
            for name in buffer_values
        }

        out = forward(fitting, descriptor, atype, aparam)
        grad_state = grad(fitting, descriptor, atype, aparam)
        grad_leaves = [np.asarray(leaf) for leaf in tree_util.tree_leaves(grad_state)]
        self.assertEqual(tuple(out.shape), (1, 3, 1))
        self.assertFalse(np.any(np.isnan(np.asarray(out))))
        self.assertTrue(grad_leaves)
        self.assertTrue(all(np.all(np.isfinite(leaf)) for leaf in grad_leaves))

        # Tracing may read portable buffers, but it must not replace their NNX
        # variables, change their array namespace, or mutate their values.
        for name, (
            original,
            original_type,
            namespace,
            values,
        ) in buffer_snapshots.items():
            current = fitting[name]
            self.assertIs(current, original)
            self.assertIs(type(current), original_type)
            self.assertIs(array_api_compat.array_namespace(current), namespace)
            np.testing.assert_array_equal(np.asarray(current), values)

        neutral_values = {
            "bias_atom_e": np.zeros((2, 1), dtype=np.float32),
            "fparam_avg": np.zeros(2, dtype=np.float32),
            "fparam_inv_std": np.ones(2, dtype=np.float32),
            "aparam_avg": np.zeros(1, dtype=np.float32),
            "aparam_inv_std": np.ones(1, dtype=np.float32),
            "case_embd": np.zeros(2, dtype=np.float32),
            "default_fparam_tensor": np.zeros(2, dtype=np.float32),
        }
        baseline = np.asarray(out)
        for name, neutral in neutral_values.items():
            with self.subTest(buffer=name):
                fitting[name] = neutral
                changed = np.asarray(forward(fitting, descriptor, atype, aparam))
                self.assertFalse(np.allclose(changed, baseline))
                fitting[name] = buffer_values[name]


if __name__ == "__main__":
    unittest.main()
