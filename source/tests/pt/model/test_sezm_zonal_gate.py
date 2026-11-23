# SPDX-License-Identifier: LGPL-3.0-or-later
"""Behavior tests for the fused-zonal-scatter level binding.

Every case here runs the real ``forward`` and asks which composition it
selected -- the PyTorch ``index_add_`` reference, or the fused CUDA
operator -- by spying on the operator entry point. With the cumulative
inference levels off, or on a build whose
``torch.ops.deepmd.dpa4_zonal_scatter`` is absent, the reference
composition must run. With the Triton level (or the CUDA level-1 suite)
up and the operator present, the same forward must call the fused
operator. Outputs are checked against an independently composed
reference in both regimes.
"""

import os
import unittest
from unittest import mock

import torch

from deepmd.pt.utils.env import (
    DEVICE,
)

_LMAX = 2
_D = (_LMAX + 1) ** 2
_CHANNELS = 64
_N_NODES = 5
_N_EDGES = 24


def _seeded_embedding(device: str) -> torch.nn.Module:
    from deepmd.pt.model.descriptor.sezm_nn.embedding import (
        GeometricInitialEmbedding,
    )

    module = GeometricInitialEmbedding(
        lmax=_LMAX,
        channels=_CHANNELS,
        dtype=torch.float32,
    ).to(device)
    module.eval()
    return module


def _forward_inputs(device: str, seed: int = 20261123):
    """Build the smallest edge cache one real forward call needs.

    ``zonal_coupling`` is passed explicitly so the forward does not need
    ``Dt_full``; the remaining cache fields are the ones the scatter and
    the degree normalization read.
    """
    from deepmd.pt.model.descriptor.sezm_nn.edge_cache import (
        EdgeFeatureCache,
    )

    gen = torch.Generator(device="cpu").manual_seed(seed)
    src = torch.randint(
        0, _N_NODES, (_N_EDGES,), generator=gen, device="cpu", dtype=torch.int64
    )
    dst = torch.randint(
        0, _N_NODES, (_N_EDGES,), generator=gen, device="cpu", dtype=torch.int64
    )
    radial = torch.randn(
        _N_EDGES, _LMAX, _CHANNELS, generator=gen, device="cpu", dtype=torch.float32
    )
    zonal = torch.randn(
        _N_EDGES, _D - 1, generator=gen, device="cpu", dtype=torch.float32
    )
    edge_vec = torch.randn(
        _N_EDGES, 3, generator=gen, device="cpu", dtype=torch.float32
    )
    deg = torch.randint(
        1, 4, (_N_NODES,), generator=gen, device="cpu", dtype=torch.int64
    ).to(torch.float32)
    cache = EdgeFeatureCache(
        src=src.to(device),
        dst=dst.to(device),
        edge_type_feat=torch.empty(0, device=device, dtype=torch.float32),
        edge_vec=edge_vec.to(device),
        edge_rbf=torch.empty(0, device=device, dtype=torch.float32),
        edge_env=torch.empty(0, device=device, dtype=torch.float32),
        deg=deg.to(device),
        inv_sqrt_deg=deg.rsqrt().view(_N_NODES, 1, 1).to(device),
        # An empty store: the fused path builds the destination CSR on
        # first use through ``cached_edge_csr``.
        csr_cache={},
    )
    return cache, radial.to(device), zonal.to(device)


def _reference_output(module, cache, radial, zonal):
    """The eager composition: broadcast the radial rows, scatter, normalize.

    The packed-row indices come from the module (they define the layout);
    the arithmetic below is written out independently of the module's own
    reference branch.
    """
    rows = module.non_scalar_row_index
    slots = module.radial_slot_index_for_row
    message = zonal.unsqueeze(-1) * radial.index_select(1, slots)
    out = torch.zeros(
        _N_NODES,
        module.ebed_dim,
        module.channels,
        device=radial.device,
        dtype=radial.dtype,
    )
    scattered = out.new_zeros(_N_NODES, rows.numel(), module.channels)
    scattered.index_add_(0, cache.dst, message)
    out[:, rows, :] = scattered
    out.mul_(cache.inv_sqrt_deg)
    return out


def _spy_on_fused_operator():
    from deepmd.pt_expt.kernels.cuda.dpa4 import (
        zonal_scatter as zonal_module,
    )

    return mock.patch.object(
        zonal_module, "zonal_scatter", wraps=zonal_module.zonal_scatter
    )


def _run_forward(module, cache, radial, zonal):
    return module(
        n_nodes=_N_NODES,
        edge_cache=cache,
        radial_feat=radial,
        zonal_coupling=zonal,
    )


class TestSeZMZonalLevelGate(unittest.TestCase):
    """The forward picks the fused operator from the levels and the build."""

    def setUp(self):
        self._saved = {
            k: os.environ.get(k) for k in ("DP_CUDA_INFER", "DP_TRITON_INFER")
        }
        os.environ.pop("DP_CUDA_INFER", None)
        os.environ.pop("DP_TRITON_INFER", None)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_default_levels_off_run_the_pytorch_fallback(self):
        module = _seeded_embedding(DEVICE)
        cache, radial, zonal = _forward_inputs(DEVICE)
        with _spy_on_fused_operator() as spy:
            out = _run_forward(module, cache, radial, zonal)
        self.assertEqual(spy.call_count, 0)
        torch.testing.assert_close(out, _reference_output(module, cache, radial, zonal))

    @unittest.skipUnless(
        torch.cuda.is_available(), "fused selection needs CUDA operands"
    )
    def test_triton_level_selects_the_fused_operator(self):
        from deepmd.pt_expt.kernels.cuda.dpa4.zonal_scatter import (
            op_available,
        )

        if not op_available():
            self.skipTest("compiled zonal scatter operator unavailable")
        os.environ["DP_TRITON_INFER"] = "1"
        os.environ["DP_CUDA_INFER"] = "0"
        module = _seeded_embedding("cuda")
        cache, radial, zonal = _forward_inputs("cuda")
        self.assertTrue(
            module._cuda_scatter,
            "the Triton level must bind the operator on a build that has it",
        )
        with _spy_on_fused_operator() as spy:
            out = _run_forward(module, cache, radial, zonal)
        self.assertGreaterEqual(
            spy.call_count, 1, "the forward must call the fused operator"
        )
        torch.testing.assert_close(out, _reference_output(module, cache, radial, zonal))

    @unittest.skipUnless(
        torch.cuda.is_available(), "fused selection needs CUDA operands"
    )
    def test_cuda_suite_selects_the_fused_operator(self):
        from deepmd.pt_expt.kernels.cuda.dpa4.zonal_scatter import (
            op_available,
        )

        if not op_available():
            self.skipTest("compiled zonal scatter operator unavailable")
        os.environ["DP_CUDA_INFER"] = "1"
        module = _seeded_embedding("cuda")
        cache, radial, zonal = _forward_inputs("cuda")
        self.assertTrue(module._cuda_scatter)
        with _spy_on_fused_operator() as spy:
            out = _run_forward(module, cache, radial, zonal)
        self.assertGreaterEqual(spy.call_count, 1)
        torch.testing.assert_close(out, _reference_output(module, cache, radial, zonal))

    def test_missing_operator_runs_the_pytorch_fallback(self):
        # A build without the compiled operator: the availability probe
        # reports False at construction time, whatever the levels say.
        from deepmd.pt_expt.kernels.cuda.dpa4 import (
            zonal_scatter as zonal_module,
        )

        os.environ["DP_TRITON_INFER"] = "1"
        os.environ["DP_CUDA_INFER"] = "0"
        with mock.patch.object(zonal_module, "op_available", return_value=False):
            module = _seeded_embedding(DEVICE)
        self.assertFalse(module._cuda_scatter)
        cache, radial, zonal = _forward_inputs(DEVICE)
        with _spy_on_fused_operator() as spy:
            out = _run_forward(module, cache, radial, zonal)
        self.assertEqual(
            spy.call_count, 0, "a build without the operator must take the fallback"
        )
        torch.testing.assert_close(out, _reference_output(module, cache, radial, zonal))


if __name__ == "__main__":
    unittest.main()
