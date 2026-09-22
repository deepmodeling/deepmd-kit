# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the dpmodel DPA4 (SeZM) descriptor.

The suite runs on the dpmodel backend alone except for the source-gate
cross-backend check, whose pt import lives inside the test function because
ruff TID253 bans module-level ``deepmd.pt`` imports under ``source/tests``.
"""

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


# The absolute scale gives every element the unit radius, so each pair's length
# scale is 1 Å and the two fractions are the window radii in Å directly.
BRIDGING_WINDOW = {
    "inner_clamp_f_inner": 0.5,
    "inner_clamp_f_outer": 1.0,
    "inner_clamp_scale": "absolute",
}

# A hand-built graph of four nodes and six edges for the source gates, read
# against ``BRIDGING_WINDOW``: the absolute scale gives every pair the unit
# length scale, so an edge length is already its reduced distance. Node 0 emits
# one frozen edge (inside the inner radius, amplitude exactly zero) and one open
# one, node 1 two open edges, node 2 a single edge beyond the outer radius
# (amplitude exactly one) and node 3 a single open edge.
GATE_EDGE_LEN = np.array([[0.3], [0.7], [0.7], [0.9], [1.5], [0.6]], dtype=np.float64)
GATE_SRC = np.array([0, 0, 1, 1, 2, 3], dtype=np.int64)
GATE_N_NODES = 4


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
        # rot_mat, g2, h2, sw, and the source gate of an unbridged descriptor
        assert out[1:] == (None, None, None, None, None)
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
        dd_bridge = make_descriptor(**BRIDGING_WINDOW)
        assert dd_bridge.has_message_passing() is True
        assert dd_bridge.has_message_passing_across_ranks() is True

    def test_capability_split_needs_vs_supports(self) -> None:
        """has_message_passing_across_ranks = NEEDS exchange (always True for
        SeZM); supports_edge_parallel = CAN run multi-rank (True for bridged
        models too since the SFPG cross-rank completion -- issue #5906).
        """
        dd_plain = make_descriptor()
        dd_bridged = make_descriptor(**BRIDGING_WINDOW)
        assert dd_plain.has_message_passing_across_ranks() is True
        assert dd_bridged.has_message_passing_across_ranks() is True
        assert dd_plain.supports_edge_parallel() is True
        assert dd_bridged.supports_edge_parallel() is True

    def test_gate_partial_exchange_dpmodel_raises(self) -> None:
        """The dpmodel backend is the single-process reference; comm on a
        bridged model must raise, never silently compute a partial gate.
        """
        dd = make_descriptor(**BRIDGING_WINDOW)
        with pytest.raises(NotImplementedError, match="dpmodel"):
            dd._gate_partial_exchange(np.zeros((4, 2)), {"nlocal": 2})

    def test_source_gates_are_the_full_and_leave_one_out_products(self) -> None:
        """The node gate multiplies every pair of a node, the edge gate all but one.

        A pair inside the frozen zone contributes an amplitude of exactly
        zero. It mutes its source node and every OTHER edge that node emits,
        while the frozen edge itself keeps the product over the remaining
        pairs of its source, so the two atoms of a frozen pair go on seeing
        each other at the clamped distance.
        """
        from deepmd.dpmodel.descriptor.dpa4_nn.edge_cache import (
            compute_source_gates,
        )

        switch = make_descriptor(**BRIDGING_WINDOW).bridging_switch
        contact = np.ones_like(GATE_EDGE_LEN)
        w = np.asarray(switch.call(GATE_EDGE_LEN, contact))[:, 0]
        # the data spans all three regimes of the switch: frozen, transition,
        # and fully open
        assert w[0] == 0.0
        assert all(0.0 < w[i] < 1.0 for i in (1, 2, 3, 5))
        assert w[4] == 1.0

        node_gate, edge_gate = compute_source_gates(
            edge_len=GATE_EDGE_LEN,
            edge_contact=contact,
            src=GATE_SRC,
            n_nodes=GATE_N_NODES,
            bridging_switch=switch,
        )
        assert node_gate.shape == (GATE_N_NODES,)
        assert edge_gate.shape == (GATE_EDGE_LEN.shape[0], 1)

        # node 0 owns the frozen pair, so its full product is exactly zero
        assert node_gate[0] == 0.0
        np.testing.assert_allclose(node_gate[1], w[2] * w[3], rtol=1e-14, atol=0.0)
        np.testing.assert_allclose(node_gate[2], w[4], rtol=1e-14, atol=0.0)
        np.testing.assert_allclose(node_gate[3], w[5], rtol=1e-14, atol=0.0)

        gate = np.asarray(edge_gate)[:, 0]
        # node 0: the frozen edge keeps its source's other pair, and that
        # other edge is muted by the frozen one
        np.testing.assert_allclose(gate[0], w[1], rtol=1e-14, atol=0.0)
        assert gate[1] == 0.0
        # node 1: each of the two open edges keeps the other one
        np.testing.assert_allclose(gate[2], w[3], rtol=1e-14, atol=0.0)
        np.testing.assert_allclose(gate[3], w[2], rtol=1e-14, atol=0.0)
        # nodes 2 and 3 emit one edge each: the leave-one-out product is empty
        np.testing.assert_array_equal(gate[4:], np.ones(2))

    def test_source_gates_match_the_pt_backend(self) -> None:
        """Both backends compute the two gates through the same reduction.

        The log-sum decomposition is written the same way on either side, so
        on the same float64 input the two implementations agree bit for bit.
        """
        import torch

        from deepmd.dpmodel.descriptor.dpa4_nn.edge_cache import (
            compute_source_gates,
        )
        from deepmd.pt.model.descriptor.sezm_nn.edge_cache import (
            compute_source_gates as compute_source_gates_pt,
        )
        from deepmd.pt.model.descriptor.sezm_nn.radial import (
            BridgingSwitch as BridgingSwitchPT,
        )

        switch = make_descriptor(**BRIDGING_WINDOW).bridging_switch
        contact = np.ones_like(GATE_EDGE_LEN)
        node_gate, edge_gate = compute_source_gates(
            edge_len=GATE_EDGE_LEN,
            edge_contact=contact,
            src=GATE_SRC,
            n_nodes=GATE_N_NODES,
            bridging_switch=switch,
        )

        switch_pt = BridgingSwitchPT(switch.f_inner, switch.f_outer).to("cpu")
        node_gate_pt, edge_gate_pt = compute_source_gates_pt(
            edge_len=torch.from_numpy(GATE_EDGE_LEN),
            edge_contact=torch.from_numpy(contact),
            src=torch.from_numpy(GATE_SRC),
            n_nodes=GATE_N_NODES,
            bridging_switch=switch_pt,
        )
        np.testing.assert_array_equal(np.asarray(node_gate), node_gate_pt.numpy())
        np.testing.assert_array_equal(np.asarray(edge_gate), edge_gate_pt.numpy())

    def test_source_gate_identity_exchange_is_noop(self) -> None:
        """The hook seam: an identity exchange reproduces the no-hook gates
        bit-exactly (pins the pack/unpack layout [log_eta, zero_count]).
        """
        from deepmd.dpmodel.descriptor.dpa4_nn.edge_cache import (
            compute_source_gates,
        )

        switch = make_descriptor(**BRIDGING_WINDOW).bridging_switch
        contact = np.ones_like(GATE_EDGE_LEN)
        kwargs = {
            "edge_len": GATE_EDGE_LEN,
            "edge_contact": contact,
            "src": GATE_SRC,
            "n_nodes": GATE_N_NODES,
            "bridging_switch": switch,
        }
        node_ref, edge_ref = compute_source_gates(**kwargs)
        node_hook, edge_hook = compute_source_gates(
            **kwargs, node_partial_exchange=lambda p: p
        )
        np.testing.assert_array_equal(node_ref, node_hook)
        np.testing.assert_array_equal(edge_ref, edge_hook)

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
                {"add_chg_spin_ebd": True, "default_chg_spin": [-1.0, 3.0]},
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

    def test_use_amp_stays_out_of_the_portable_record(self) -> None:
        """``use_amp`` is a runtime/training policy, not model state.

        The portable serialization must not carry it (a ``use_amp: true``
        record would e.g. be rejected by the JAX deserializer); a fresh
        deserialize falls back to the constructor default. Construction-time
        survival is pinned at the pt_expt assembly boundary instead
        (``test_get_model_dpa4.py``).
        """
        dd = make_descriptor(use_amp=False)
        assert dd.use_amp is False
        assert "use_amp" not in dd.serialize()["config"]

    def test_legacy_spin_gate_is_squared_on_deserialize(self) -> None:
        """Version 1.2 stores the env-seed spin gate after the quadratic form.

        The stored scalar used to multiply the spin coordinate channel
        BEFORE that form, so an amplitude ``a`` contributed ``a**2 * D_spin``
        and squaring reproduces the stored function exactly. ``@version`` is
        the source of truth, and a migrated payload is retagged so a second
        load leaves the gate alone.
        """
        dd = make_descriptor(use_spin=[True, False, False])
        data = dd.serialize()
        assert data["@version"] == DescrptDPA4.LATEST_VERSION == 1.3
        data["@version"] = 1.1
        data["@variables"]["env_seed_embedding.spin_scale"] = np.full(
            (1,), 3.0, dtype=np.float64
        )

        migrated = DescrptDPA4.deserialize(data)
        np.testing.assert_allclose(migrated.env_seed_embedding.spin_scale, 9.0)
        assert migrated.version == 1.2

        reloaded = DescrptDPA4.deserialize(migrated.serialize())
        np.testing.assert_allclose(reloaded.env_seed_embedding.spin_scale, 9.0)
        assert reloaded.version == 1.2

    def test_legacy_spin_free_routes_are_zeroed_on_deserialize(self) -> None:
        data = make_descriptor(use_spin=[False, False, False]).serialize()
        data["@version"] = 1.1
        dormant_keys = (
            "spin_embedding.mag_layer2.matrix",
            "spin_embedding.adam_spin_vec_weight",
            "spin_embedding.adam_spin_nbr_weight",
            "env_seed_embedding.spin_scale",
        )
        for key in dormant_keys:
            data["@variables"][key] = np.full_like(data["@variables"][key], 3.0)
        mag_layer1_key = "spin_embedding.mag_layer1.matrix"
        data["@variables"][mag_layer1_key] = np.full_like(
            data["@variables"][mag_layer1_key], 5.0
        )

        migrated = DescrptDPA4.deserialize(data)

        variables = migrated.serialize()["@variables"]
        for key in dormant_keys:
            np.testing.assert_array_equal(variables[key], np.zeros_like(variables[key]))
        np.testing.assert_array_equal(
            variables[mag_layer1_key], np.full_like(variables[mag_layer1_key], 5.0)
        )
        assert migrated.version == 1.2

    def test_bridged_records_before_the_window_are_refused(self) -> None:
        """A bridged record below 1.3 was trained under other window mechanics.

        Its radii still translate (``migrate_inner_clamp_keys``), but no
        rewrite of the stored variables expresses the clamp frozen at the
        inner radius or the full-product gate it was trained with, so the
        record is refused instead of being read under the current function.
        An unbridged record of the same version loads.
        """
        data = make_descriptor(**BRIDGING_WINDOW).serialize()
        data["@version"] = 1.2
        config = data["config"]
        config["inner_clamp_r_inner"] = config.pop("inner_clamp_f_inner")
        config["inner_clamp_r_outer"] = config.pop("inner_clamp_f_outer")
        config.pop("inner_clamp_scale")
        with pytest.raises(ValueError, match="Retrain"):
            DescrptDPA4.deserialize(data)

        plain = make_descriptor().serialize()
        plain["@version"] = 1.2
        assert DescrptDPA4.deserialize(plain).version == 1.2

    def test_pre_spin_versions_keep_their_own_tag(self) -> None:
        """Version 1.0 predates the spin route and the 1.1 forward math.

        Promoting it would silently switch ``deg_norm_floor`` and the radial
        envelope, so a payload below 1.1 is left exactly as it was written.
        """
        data = make_descriptor().serialize()
        data["@version"] = 1.0
        assert DescrptDPA4.deserialize(data).version == 1.0

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
