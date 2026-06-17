# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for wiring the SO3/S2 grid nets into the DPA4 FFN + SO2Convolution.

These exercise the previously-guarded grid paths:

- ``ffn_so3_grid=True`` -> ``EquivariantFFN`` builds an ``SO3GridNet`` (self
  mode) for the equivariant nonlinearity.
- ``node_wise_so3``/``message_node_so3`` -> ``SO2Convolution`` builds a
  cross-mode ``SO3GridNet`` (SO3 wins over S2 when both are set).
- ``node_wise_s2``/``message_node_s2`` -> cross-mode ``S2GridNet``.

The flagship config (``examples/water/dpa4/input.json``) uses ``lmax=3,
mmax=1`` with both ``ffn_so3_grid`` and ``message_node_so3`` on, so the
SO3 grid path is exercised with a truncated ``mmax``.
"""

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa4 import (
    DescrptDPA4,
)


def build_neighbor_list_np(coord, rcut, nnei):
    """Build a padded, distance-sorted gas-phase (local) neighbor list."""
    nf, nloc, _ = coord.shape
    nlist = -np.ones((nf, nloc, nnei), dtype=np.int64)
    for f in range(nf):
        dist = np.linalg.norm(coord[f][:, None, :] - coord[f][None, :, :], axis=-1)
        for i in range(nloc):
            neighbors = [
                (dist[i, j], j) for j in range(nloc) if j != i and dist[i, j] < rcut
            ]
            neighbors.sort()
            for slot, (_, j) in enumerate(neighbors[:nnei]):
                nlist[f, i, slot] = j
    return nlist


def make_inputs(seed=5, nf=2, nloc=6, rcut=6.0, nnei=8, ntypes=2):
    rng = np.random.default_rng(seed)
    coord = rng.uniform(0.0, 4.0, size=(nf, nloc, 3))
    atype = rng.integers(0, ntypes, size=(nf, nloc))
    nlist = build_neighbor_list_np(coord, rcut, nnei)
    return coord, atype, nlist


# Small flagship-shaped config: keep lmax=3, mmax=1 so the truncated-mmax
# SO3 grid path is exercised; shrink channels/sel/blocks for speed.
def _base_kwargs(**overrides):
    kwargs = {
        "ntypes": 2,
        "sel": 8,
        "rcut": 6.0,
        "channels": 16,
        "n_radial": 8,
        "lmax": 3,
        "mmax": 1,
        "n_blocks": 2,
        "so2_layers": 2,
        "n_focus": 2,
        "radial_so2_mode": "degree_channel",
        "radial_so2_rank": 1,
        "n_atten_head": 1,
        "grid_mlp": [False, False, False],
        "grid_branch": [1, 1, 1],
        "lebedev_quadrature": True,
        "precision": "float64",
        "seed": 42,
    }
    kwargs.update(overrides)
    return kwargs


def make_descriptor(**overrides) -> DescrptDPA4:
    return DescrptDPA4(**_base_kwargs(**overrides))


class TestGridWiringConstructsAndRuns:
    @pytest.mark.parametrize(
        "flags",
        [
            {"ffn_so3_grid": True},  # SO3 self-mode FFN grid only
            {"message_node_so3": True},  # SO3 cross-mode message-node grid only
            {"ffn_so3_grid": True, "message_node_so3": True},  # both
            {"node_wise_so3": True},  # SO3 cross-mode node-wise grid only
        ],
    )
    def test_descriptor_constructs_and_runs(self, flags) -> None:
        dd = make_descriptor(**flags)
        coord, atype, nlist = make_inputs()
        nf, nloc = atype.shape
        out = dd.call(coord.reshape(nf, -1), atype, nlist, mapping=None)
        assert out[0].shape == (nf, nloc, dd.get_dim_out())
        assert out[1:] == (None, None, None, None)
        assert np.isfinite(np.asarray(out[0])).all()

    def test_serialize_roundtrip(self) -> None:
        dd = make_descriptor(ffn_so3_grid=True, message_node_so3=True)
        data = dd.serialize()
        dd2 = DescrptDPA4.deserialize(data)
        coord, atype, nlist = make_inputs()
        nf = atype.shape[0]
        out1 = np.asarray(dd.call(coord.reshape(nf, -1), atype, nlist)[0])
        out2 = np.asarray(dd2.call(coord.reshape(nf, -1), atype, nlist)[0])
        np.testing.assert_array_equal(out1, out2)


class TestS2CrossPath:
    @pytest.mark.parametrize(
        "flags",
        [
            {"node_wise_s2": True},  # S2 cross-mode node-wise grid (m-major)
            {"message_node_s2": True},  # S2 cross-mode message-node grid (packed)
            {"node_wise_s2": True, "message_node_s2": True},  # both S2 paths
        ],
    )
    def test_s2_cross_constructs_and_runs(self, flags) -> None:
        dd = make_descriptor(**flags)
        coord, atype, nlist = make_inputs()
        nf, nloc = atype.shape
        out = dd.call(coord.reshape(nf, -1), atype, nlist, mapping=None)
        assert out[0].shape == (nf, nloc, dd.get_dim_out())
        assert np.isfinite(np.asarray(out[0])).all()

    def test_so3_wins_over_s2(self) -> None:
        # When both s2 and so3 are set for a path, the SO3 net must be built.
        from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import (
            SO3GridNet,
        )

        dd = make_descriptor(message_node_s2=True, message_node_so3=True)
        block = dd.blocks[0]
        assert isinstance(block.so2_conv.message_node_grid_product, SO3GridNet)


class TestDescriptorParityVsPt:
    """Weight-copy parity: build pt ``DescrptSeZM``, copy into dpmodel via
    ``DescrptDPA4.deserialize(pt.serialize())``, compare descriptor outputs.
    """

    def _build_pair(self, perturb_seed=2130, **overrides):
        import torch

        from deepmd.pt.model.descriptor.sezm import (
            DescrptSeZM,
        )
        from deepmd.pt.utils import env as pt_env

        kwargs = _base_kwargs(**overrides)
        pt_mod = DescrptSeZM(**kwargs).double().eval()
        rng = np.random.default_rng(perturb_seed)
        with torch.no_grad():
            for p in pt_mod.parameters():
                p += torch.from_numpy(0.05 * rng.normal(size=tuple(p.shape))).to(
                    pt_env.DEVICE
                )
        dp_mod = DescrptDPA4.deserialize(pt_mod.serialize())
        return pt_mod, dp_mod

    def _assert_parity(self, pt_mod, dp_mod) -> None:
        import torch

        from deepmd.pt.utils import env as pt_env

        # CPU: pt fp64 == numpy fp64 to ~1 ulp -> rtol 1e-10; CUDA index_add_
        # atomics are nondeterministic -> still 1e-10 is well below any logic bug.
        coord, atype, nlist = make_inputs()
        nf, nloc = atype.shape
        out_dp = dp_mod.call(coord.reshape(nf, -1), atype, nlist, mapping=None)
        out_pt = pt_mod(
            torch.from_numpy(coord).to(pt_env.DEVICE),
            torch.from_numpy(atype).to(pt_env.DEVICE),
            torch.from_numpy(nlist).to(pt_env.DEVICE),
            mapping=None,
        )
        assert tuple(out_dp[0].shape) == tuple(out_pt[0].shape)
        np.testing.assert_allclose(
            np.asarray(out_dp[0]),
            out_pt[0].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_parity_ffn_so3_grid(self) -> None:
        pt_mod, dp_mod = self._build_pair(ffn_so3_grid=True)
        self._assert_parity(pt_mod, dp_mod)

    def test_parity_message_node_so3(self) -> None:
        pt_mod, dp_mod = self._build_pair(message_node_so3=True)
        self._assert_parity(pt_mod, dp_mod)

    def test_parity_both_so3(self) -> None:
        pt_mod, dp_mod = self._build_pair(ffn_so3_grid=True, message_node_so3=True)
        self._assert_parity(pt_mod, dp_mod)

    def test_parity_node_wise_s2(self) -> None:
        pt_mod, dp_mod = self._build_pair(node_wise_s2=True)
        self._assert_parity(pt_mod, dp_mod)
