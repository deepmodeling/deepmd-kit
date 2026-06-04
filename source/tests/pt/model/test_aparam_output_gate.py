# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch

from deepmd.pt.model.task.invar_fitting import (
    InvarFitting,
)
from deepmd.pt.utils import (
    env,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE


class TestAparamOutputGate(unittest.TestCase):
    def test_zero_aparam_zeros_output(self) -> None:
        nf, nloc, dim_descrpt = 1, 2, 8
        sigma = 2.0
        fitting = InvarFitting(
            "energy",
            ntypes=1,
            dim_descrpt=dim_descrpt,
            dim_out=1,
            neuron=[4, 4],
            numb_aparam=1,
            mixed_types=True,
            use_aparam_output_gate=True,
            aparam_gate_norm=1.0,
            aparam_gate_clamp=True,
        ).to(device)
        fitting.aparam_inv_std.copy_(torch.tensor([1.0 / sigma], dtype=dtype))

        descriptor = torch.randn(nf, nloc, dim_descrpt, dtype=dtype, device=device)
        atype = torch.zeros(nf, nloc, dtype=torch.int64, device=device)
        aparam_zero = torch.zeros(nf, nloc, 1, dtype=dtype, device=device)
        aparam_sigma = torch.full(
            (nf, nloc, 1), sigma, dtype=dtype, device=device
        )

        out_zero = fitting(descriptor, atype, aparam=aparam_zero)["energy"]
        out_sigma = fitting(descriptor, atype, aparam=aparam_sigma)["energy"]

        self.assertTrue(torch.allclose(out_zero, torch.zeros_like(out_zero)))
        self.assertGreater(out_sigma.abs().max().item(), 0.0)

    def test_gate_matches_formula(self) -> None:
        nf, nloc, dim_descrpt = 1, 1, 4
        sigma = 3.0
        norm = 2.0
        fitting = InvarFitting(
            "energy",
            ntypes=1,
            dim_descrpt=dim_descrpt,
            dim_out=1,
            neuron=[4],
            numb_aparam=1,
            mixed_types=True,
            use_aparam_output_gate=True,
            aparam_gate_norm=norm,
            aparam_gate_clamp=False,
        ).to(device)
        fitting.aparam_inv_std.copy_(torch.tensor([1.0 / sigma], dtype=dtype))

        descriptor = torch.randn(nf, nloc, dim_descrpt, dtype=dtype, device=device)
        atype = torch.zeros(nf, nloc, dtype=torch.int64, device=device)
        a_val = 1.5
        aparam = torch.full((nf, nloc, 1), a_val, dtype=dtype, device=device)

        fitting_gate = fitting._compute_aparam_output_gate(aparam)
        expected = (a_val * a_val) / (sigma * sigma * norm)
        self.assertTrue(torch.allclose(fitting_gate, torch.tensor(expected, dtype=dtype)))

    def test_serialize_roundtrip(self) -> None:
        fitting = InvarFitting(
            "energy",
            ntypes=1,
            dim_descrpt=4,
            dim_out=1,
            neuron=[4],
            numb_aparam=1,
            use_aparam_output_gate=True,
            aparam_gate_norm=1.5,
            aparam_gate_clamp=False,
        )
        restored = InvarFitting.deserialize(fitting.serialize())
        self.assertTrue(restored.use_aparam_output_gate)
        self.assertEqual(restored.aparam_gate_norm, 1.5)
        self.assertFalse(restored.aparam_gate_clamp)


if __name__ == "__main__":
    unittest.main()
