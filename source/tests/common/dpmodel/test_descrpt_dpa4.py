# SPDX-License-Identifier: LGPL-3.0-or-later
"""Torch-free unit tests for the dpmodel DPA4 (SeZM) descriptor."""

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa4 import (
    DescrptDPA4,
)


def build_neighbor_list_np(coord, rcut, nnei):
    """Build a padded, distance-sorted gas-phase neighbor list.

    Parameters
    ----------
    coord
        Coordinates with shape (nf, nloc, 3); no PBC.
    rcut
        Cutoff radius.
    nnei
        Number of neighbor slots; pads with -1.

    Returns
    -------
    np.ndarray
        Neighbor list with shape (nf, nloc, nnei).
    """
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


def make_descriptor(**overrides) -> DescrptDPA4:
    kwargs = {
        "ntypes": 3,
        "sel": 8,
        "rcut": 4.0,
        "channels": 16,
        "n_radial": 8,
        "lmax": 3,
        "mmax": 1,
        "n_blocks": 2,
        "grid_branch": [1, 1, 1],
        "s2_activation": [False, True],
        "random_gamma": False,
        "exclude_types": [(0, 0)],
        "precision": "float64",
        "seed": 42,
    }
    kwargs.update(overrides)
    return DescrptDPA4(**kwargs)


def make_inputs(seed=5, nf=2, nloc=6, rcut=4.0, nnei=8, ntypes=3):
    rng = np.random.default_rng(seed)
    coord = rng.uniform(0.0, 3.5, size=(nf, nloc, 3))
    atype = rng.integers(0, ntypes, size=(nf, nloc))
    nlist = build_neighbor_list_np(coord, rcut, nnei)
    return coord, atype, nlist


class TestDescrptDPA4:
    def test_shapes_and_interface(self) -> None:
        dd = make_descriptor()
        coord, atype, nlist = make_inputs()
        nf, nloc = atype.shape
        out = dd.call(coord.reshape(nf, -1), atype, nlist, mapping=None)
        assert out[0].shape == (nf, nloc, dd.get_dim_out())
        assert out[1:] == (None, None, None, None)
        assert np.isfinite(np.asarray(out[0])).all()
        # standard descriptor surface
        assert dd.get_rcut() == 4.0
        assert dd.get_rcut_smth() == 4.0
        assert dd.get_sel() == [8]
        assert dd.get_nsel() == 8
        assert dd.get_ntypes() == 3
        assert dd.get_type_map() == []
        assert dd.get_dim_out() == 16
        assert dd.get_dim_emb() == 16
        assert dd.mixed_types() is True
        assert dd.has_message_passing() is True
        assert dd.need_sorted_nlist_for_lower() is False
        assert dd.get_env_protection() == dd.eps

    def test_message_passing_semantics(self) -> None:
        # SeZM always resolves ghost neighbours on the lower path, so it always
        # reports message passing. The GRAPH lower implements the cross-rank
        # exchange via a real per-layer border_op, so every SeZM descriptor
        # (bridged or not) reports across_ranks True; its DENSE lower has no
        # comm_dict implementation (the dense adapter raises on it), so
        # dense_lower_supports_comm() is False and the freeze machinery
        # skips the dead dense with-comm artifact. Whether multi-rank is
        # POSSIBLE at all is supports_edge_parallel (see
        # test_capability_split_needs_vs_supports).
        dd = make_descriptor()
        assert dd.has_message_passing() is True
        assert dd.has_message_passing_across_ranks() is True
        assert dd.dense_lower_supports_comm() is False
        dd_bridge = make_descriptor(inner_clamp_r_inner=0.5, inner_clamp_r_outer=1.0)
        assert dd_bridge.has_message_passing() is True
        assert dd_bridge.has_message_passing_across_ranks() is True

    def test_capability_split_needs_vs_supports(self) -> None:
        """has_message_passing_across_ranks = NEEDS exchange (always True for
        SeZM); supports_edge_parallel = CAN run multi-rank (True for bridged
        models too since the SFPG cross-rank completion -- issue #5906).
        """
        dd_plain = make_descriptor()
        dd_bridged = make_descriptor(inner_clamp_r_inner=0.5, inner_clamp_r_outer=1.0)
        assert dd_plain.has_message_passing_across_ranks() is True
        assert dd_bridged.has_message_passing_across_ranks() is True
        assert dd_plain.supports_edge_parallel() is True
        assert dd_bridged.supports_edge_parallel() is True

    def test_gate_partial_exchange_dpmodel_raises(self) -> None:
        """The dpmodel backend is the single-process reference; comm on a
        bridged model must raise, never silently compute a partial gate.
        """
        dd = make_descriptor(inner_clamp_r_inner=0.5, inner_clamp_r_outer=1.0)
        with pytest.raises(NotImplementedError, match="dpmodel"):
            dd._gate_partial_exchange(np.zeros((4, 2)), {"nlocal": 2})

    def test_edge_src_gate_identity_exchange_is_noop(self) -> None:
        """The hook seam: an identity exchange reproduces the no-hook gate
        bit-exactly (pins the pack/unpack layout [log_eta, zero_count]).
        """
        from deepmd.dpmodel.descriptor.dpa4_nn.edge_cache import (
            compute_edge_src_gate,
        )

        dd = make_descriptor(inner_clamp_r_inner=0.5, inner_clamp_r_outer=1.0)
        sw = dd.bridging_switch
        # 4 nodes, 5 edges; one edge inside r_inner (w=0, hard-freezes its
        # src node), the rest spread across the transition zone and beyond.
        el = np.array([[0.3], [0.7], [0.9], [1.5], [0.75]], dtype=np.float64)
        src = np.array([0, 0, 1, 2, 3], dtype=np.int64)
        gate_ref = compute_edge_src_gate(
            edge_len=el, src=src, n_nodes=4, bridging_switch=sw
        )
        gate_hook = compute_edge_src_gate(
            edge_len=el,
            src=src,
            n_nodes=4,
            bridging_switch=sw,
            node_partial_exchange=lambda p: p,
        )
        np.testing.assert_array_equal(gate_ref, gate_hook)
        # geometry sanity: node 0 is hard-frozen, node 3 is inside the
        # transition zone (0 < gate < 1) -- the test data actually
        # exercises both gate branches.
        assert gate_ref[0, 0] == 0.0
        assert 0.0 < gate_ref[4, 0] < 1.0

    def test_serialize_roundtrip_exact(self) -> None:
        dd = make_descriptor()
        data = dd.serialize()
        assert data["type"] == "SeZM"
        dd2 = DescrptDPA4.deserialize(data)
        coord, atype, nlist = make_inputs()
        nf = atype.shape[0]
        out1 = np.asarray(dd.call(coord.reshape(nf, -1), atype, nlist)[0])
        out2 = np.asarray(dd2.call(coord.reshape(nf, -1), atype, nlist)[0])
        np.testing.assert_array_equal(out1, out2)

    def test_random_gamma_inference_deterministic(self) -> None:
        """The dpmodel backend never applies the random local-Z roll, even when configured.

        ``random_gamma`` is a training-only augmentation gated by the
        ``_in_training_mode`` runtime hook; dpmodel has no training mode, so
        the hook is ``False`` and two calls of a ``random_gamma=True``
        descriptor are bit-identical (the pt_expt twin's train-mode
        behavior is pinned in
        ``source/tests/pt_expt/descriptor/test_dpa4.py::test_random_gamma_train_eval_gate``).
        """
        dd = make_descriptor(random_gamma=True)
        assert dd._in_training_mode() is False
        coord, atype, nlist = make_inputs()
        nf = atype.shape[0]
        out1 = np.asarray(dd.call(coord.reshape(nf, -1), atype, nlist)[0])
        out2 = np.asarray(dd.call(coord.reshape(nf, -1), atype, nlist)[0])
        np.testing.assert_array_equal(out1, out2)

    def test_permutation_equivariance(self) -> None:
        dd = make_descriptor()
        coord, atype, nlist = make_inputs()
        nf, nloc = atype.shape
        out = np.asarray(dd.call(coord.reshape(nf, -1), atype, nlist)[0])
        rng = np.random.default_rng(11)
        perm = rng.permutation(nloc)
        inv = np.argsort(perm)
        coord2 = coord[:, perm, :]
        atype2 = atype[:, perm]
        nlist_p = nlist[:, perm, :]
        nlist2 = np.where(nlist_p >= 0, inv[np.where(nlist_p >= 0, nlist_p, 0)], -1)
        out2 = np.asarray(dd.call(coord2.reshape(nf, -1), atype2, nlist2)[0])
        np.testing.assert_allclose(out2, out[:, perm, :], rtol=1e-10, atol=1e-12)

    def test_rotation_invariance(self) -> None:
        dd = make_descriptor()
        coord, atype, nlist = make_inputs()
        nf = atype.shape[0]
        out = np.asarray(dd.call(coord.reshape(nf, -1), atype, nlist)[0])
        # a random proper rotation (QR with det fix)
        rng = np.random.default_rng(13)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        coord_rot = coord @ q.T  # distances (and the nlist) are unchanged
        out_rot = np.asarray(dd.call(coord_rot.reshape(nf, -1), atype, nlist)[0])
        np.testing.assert_allclose(out_rot, out, rtol=1e-10, atol=1e-12)

    def test_masked_edge_inertness(self) -> None:
        # an extra all-(-1) neighbor column must not change the descriptor
        dd = make_descriptor()
        coord, atype, nlist = make_inputs()
        nf, nloc = atype.shape
        out = np.asarray(dd.call(coord.reshape(nf, -1), atype, nlist)[0])
        pad = -np.ones((nf, nloc, 1), dtype=nlist.dtype)
        nlist2 = np.concatenate([nlist, pad], axis=-1)
        out2 = np.asarray(dd.call(coord.reshape(nf, -1), atype, nlist2)[0])
        np.testing.assert_allclose(out2, out, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize(
        "overrides",
        [
            # Charge/spin condition embedding; default_chg_spin lets forward run
            # without an explicit charge_spin input.
            pytest.param(
                {"add_chg_spin_ebd": True, "default_chg_spin": [0.5, -0.5]},
                id="add_chg_spin_ebd",
            ),
            pytest.param({"so2_attn_res": "independent"}, id="so2_attn_res"),
            pytest.param({"full_attn_res": "dependent"}, id="full_attn_res"),
            pytest.param({"layer_scale": True}, id="layer_scale"),
            pytest.param(
                {"lebedev_quadrature": [False, False]}, id="lebedev_quadrature_off"
            ),
            pytest.param({"atten_v_proj": True}, id="atten_v_proj"),
            pytest.param({"node_wise_so3": True}, id="node_wise_so3"),
            pytest.param({"message_node_so3": True}, id="message_node_so3"),
            pytest.param({"ffn_so3_grid": True}, id="ffn_so3_grid"),
        ],
    )
    def test_supported_feature_roundtrip(self, overrides) -> None:
        # Each flag enables a feature the migration now implements. Verify the
        # key steps: forward is finite with the right shape, and the descriptor
        # survives a serialize -> deserialize round-trip bit-exactly.
        dd = make_descriptor(**overrides)
        coord, atype, nlist = make_inputs()
        nf, nloc = atype.shape
        out1 = np.asarray(dd.call(coord.reshape(nf, -1), atype, nlist)[0])
        assert out1.shape == (nf, nloc, dd.get_dim_out())
        assert np.isfinite(out1).all()
        dd2 = DescrptDPA4.deserialize(dd.serialize())
        out2 = np.asarray(dd2.call(coord.reshape(nf, -1), atype, nlist)[0])
        np.testing.assert_array_equal(out1, out2)

    @pytest.mark.parametrize(
        "use_amp",
        [
            True,  # the constructor default; must not be clobbered either
            False,  # the value that was silently lost, re-enabling autocast
        ],
    )
    def test_use_amp_survives_roundtrip(self, use_amp) -> None:
        """``use_amp`` must round-trip through serialize/deserialize.

        It was absent from the serialized config, so any backend that rebuilds
        the descriptor from that dict (pt_expt does) silently reset a
        configured ``use_amp: false`` to the True default and kept training
        under bfloat16 autocast.  The forward-output round-trip test cannot
        catch this: dpmodel never autocasts, so the outputs match either way --
        only the attribute itself pins the contract.
        """
        dd = make_descriptor(use_amp=use_amp)
        assert dd.use_amp is use_amp
        assert dd.serialize()["config"]["use_amp"] is use_amp
        dd2 = DescrptDPA4.deserialize(dd.serialize())
        assert dd2.use_amp is use_amp

    def test_value_errors(self) -> None:
        with pytest.raises(ValueError):  # kmax must be <= lmax
            make_descriptor(kmax=4, lmax=3)
        with pytest.raises(ValueError):  # m_schedule entries must be <= l_schedule
            make_descriptor(l_schedule=[2, 2], m_schedule=[3, 1])
        with pytest.raises(ValueError):  # l_schedule must be non-increasing
            make_descriptor(l_schedule=[2, 3])
        with pytest.raises(ValueError):  # sandwich_norm must have length 4
            make_descriptor(sandwich_norm=[True, False])
        with pytest.raises(ValueError):  # env_exp must have length 2
            make_descriptor(env_exp=[7])
        with pytest.raises(ValueError):  # attn res mode token
            make_descriptor(full_attn_res="depth")
        with pytest.raises(ValueError):  # wrong class tag
            DescrptDPA4.deserialize({"@class": "NotDescriptor", "type": "SeZM"})
        with pytest.raises(ValueError):  # wrong type tag
            DescrptDPA4.deserialize({"@class": "Descriptor", "type": "se_e2_a"})
