# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

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
        fitting["bias_atom_e"] = np.array([[1.5], [-0.75]], dtype=np.float32)
        fitting["fparam_avg"] = np.array([0.1, -0.2], dtype=np.float32)
        fitting["fparam_inv_std"] = np.array([2.0, 0.5], dtype=np.float32)
        fitting["aparam_avg"] = np.array([0.25], dtype=np.float32)
        fitting["aparam_inv_std"] = np.array([4.0], dtype=np.float32)
        fitting["case_embd"] = np.array([0.3, -0.6], dtype=np.float32)

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

        out = forward(fitting, descriptor, atype, aparam)
        self.assertEqual(tuple(out.shape), (1, 3, 1))
        self.assertFalse(np.any(np.isnan(np.asarray(out))))
        self.assertIsNotNone(grad(fitting, descriptor, atype, aparam))


if __name__ == "__main__":
    unittest.main()
