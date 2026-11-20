# SPDX-License-Identifier: LGPL-3.0-or-later
"""The contact clamp of a bridged model: profile, contact record, edges and backends."""

from pathlib import (
    Path,
)

import numpy as np
import pytest

from deepmd.dpmodel.utils.contact_clamp import (
    PADDING_EDGE,
    ContactClamp,
)
from deepmd.utils.element_radii import (
    covalent_radii_from_type_map,
)

TYPE_MAP = ["O", "H", "Si"]


def _edge(clamp: ContactClamp, center: int, neighbor: int) -> float:
    return float(np.asarray(clamp.edges)[center, neighbor])


def _read(clamp: ContactClamp, distance: np.ndarray, center: int, neighbor: int):
    types = np.full(distance.shape[0], center, dtype=np.int64)
    return np.asarray(
        clamp.call(distance, types, np.full_like(types, neighbor))
    ).reshape(-1)


def test_clamped_geometry_separates_read_radius_from_direction() -> None:
    from deepmd.dpmodel.descriptor.dpa4_nn.edge_cache import (
        clamped_edge_geometry,
    )

    clamp = ContactClamp(["O"], inner=0.75)
    vec = np.zeros((4, 3))
    vec[:, 0] = [0.0, 0.1, 0.2, 2.0]
    types = np.zeros(4, dtype=np.int64)
    mapped, radius = clamped_edge_geometry(
        vec, lambda r: clamp.call(r, types, types), eps=0.1
    )
    np.testing.assert_array_equal(mapped[0], np.zeros(3))
    np.testing.assert_array_equal(mapped[1], mapped[2])
    np.testing.assert_array_equal(radius[:3], np.full((3, 1), radius[1, 0]))
    np.testing.assert_allclose(radius[1, 0], np.hypot(mapped[1, 0], 0.1))
    np.testing.assert_array_equal(mapped[-1], vec[-1])


def test_initial_edges_are_the_covalent_bond_lengths() -> None:
    clamp = ContactClamp(TYPE_MAP)
    radius = covalent_radii_from_type_map(TYPE_MAP)
    edges = np.asarray(clamp.edges)
    assert edges.shape == (4, 4)
    np.testing.assert_allclose(edges[:3, :3], radius[:, None] + radius[None, :])
    np.testing.assert_array_equal(edges[3, :], PADDING_EDGE)
    np.testing.assert_array_equal(edges[:, 3], PADDING_EDGE)


@pytest.mark.parametrize("inner", [0.0, 1.0, -0.5, 1.5])
def test_inner_fraction_must_lie_in_the_open_unit_interval(inner) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        ContactClamp(TYPE_MAP, inner=inner)


def test_profile_is_identity_above_and_frozen_below() -> None:
    clamp = ContactClamp(TYPE_MAP, inner=0.75)
    edge = _edge(clamp, 0, 1)
    above = edge * np.array([1.0, 1.01, 1.5, 3.0])
    np.testing.assert_array_equal(_read(clamp, above, 0, 1), above)
    below = edge * np.array([0.0, 0.1, 0.5, 0.75])
    np.testing.assert_allclose(
        _read(clamp, below, 0, 1), 0.5 * (1.0 + 0.75) * edge, rtol=0.0, atol=1e-15
    )


def test_profile_is_non_decreasing_and_never_steeper_than_the_identity() -> None:
    clamp = ContactClamp(TYPE_MAP, inner=0.6)
    edge = _edge(clamp, 2, 2)
    distance = np.linspace(0.0, 1.5 * edge, 4001)
    read = _read(clamp, distance, 2, 2)
    slope = np.diff(read) / np.diff(distance)
    assert slope.min() >= 0.0
    assert slope.max() <= 1.0 + 1e-12
    # Below the edge the learned part reads a longer separation than the true
    # one; at and above it, the true one.
    assert np.all(read[distance < edge] >= distance[distance < edge])
    np.testing.assert_array_equal(read[distance >= edge], distance[distance >= edge])


def test_profile_is_c4_at_both_ends_of_the_ramp() -> None:
    """Four derivatives of the read separation match across the edge and the freeze."""
    torch = pytest.importorskip("torch")
    clamp = ContactClamp(TYPE_MAP, inner=0.7)
    edge = _edge(clamp, 0, 0)

    def derivatives(r: float) -> list[float]:
        x = torch.tensor([r], dtype=torch.float64, device="cpu", requires_grad=True)
        types = torch.zeros(1, dtype=torch.int64, device="cpu")
        values = [clamp.call(x, types, types).sum()]
        for _ in range(4):
            values.append(
                torch.autograd.grad(values[-1], x, create_graph=True)[0].sum()
            )
        return [float(v.detach()) for v in values]

    # The n-th derivative of the ramp scales as width ** (1 - n), and its
    # fourth derivative leaves either joint linearly, so the two sides are
    # probed 1e-12 apart to resolve the continuity of every order.
    width = 0.3 * edge
    for point, outside in ((edge, [edge, 1.0, 0.0, 0.0, 0.0]), (0.7 * edge, None)):
        left = derivatives(point * (1.0 - 1e-12))
        right = derivatives(point * (1.0 + 1e-12))
        for order in range(5):
            tolerance = 1e-6 * width ** (1 - order)
            assert abs(left[order] - right[order]) <= tolerance, (point, order)
        if outside is not None:
            np.testing.assert_allclose(right, outside, atol=1e-6)
    frozen = derivatives(0.5 * edge)
    np.testing.assert_allclose(frozen[1:], 0.0, atol=0.0)


def _observe(clamp: ContactClamp, distance, center, neighbor, frame, mask=None):
    distance = np.asarray(distance, dtype=np.float64)
    if mask is None:
        mask = np.ones(distance.shape[0], dtype=bool)
    clamp.observe(
        distance,
        np.asarray(center),
        np.asarray(neighbor),
        np.asarray(mask),
        np.asarray(frame),
    )


def test_one_frame_support_is_a_running_minimum_that_ignores_padding() -> None:
    clamp = ContactClamp(TYPE_MAP, support=1)
    start = np.asarray(clamp.edges).copy()
    oh = 0.4 * _edge(clamp, 0, 1)
    # The second edge is masked and the fourth involves the padding type, so
    # neither may move an edge; the third lies beyond its covalent start.
    _observe(
        clamp,
        [oh, 0.2, 5.0, 0.1],
        [0, 1, 2, -1],
        [1, 1, 2, 0],
        [0, 0, 0, 0],
        mask=[True, False, True, True],
    )
    edges = np.asarray(clamp.edges)
    assert edges[0, 1] == oh and edges[1, 0] == oh
    assert edges[1, 1] == start[1, 1]
    assert edges[2, 2] == start[2, 2]
    np.testing.assert_array_equal(edges[3, :], start[3, :])
    np.testing.assert_array_equal(edges[:, 3], start[:, 3])
    # A longer contact of a later batch changes nothing.
    _observe(clamp, [2 * oh], [1], [0], [0])
    assert _edge(clamp, 0, 1) == oh


def test_the_edge_follows_the_support_th_distinct_frame() -> None:
    clamp = ContactClamp(TYPE_MAP, support=3)
    start = _edge(clamp, 0, 1)
    # One frame with two O-H contacts, its shortest one counting for the frame.
    _observe(clamp, [0.50, 0.58], [0, 0], [1, 1], [0, 0])
    assert _edge(clamp, 0, 1) == start
    # The same frame seen again in a later epoch counts once, also when its
    # separation is recomputed with a rounding difference.
    _observe(clamp, [0.50], [1], [0], [0])
    _observe(clamp, [0.50 + 1e-9], [1], [0], [0])
    assert _edge(clamp, 0, 1) == start
    # Two more frames complete the support; the edge is the third contact.
    _observe(clamp, [0.70, 0.60], [0, 1], [1, 0], [0, 1])
    assert _edge(clamp, 0, 1) == 0.70 == _edge(clamp, 1, 0)
    # A frame with a contact between the first ones lowers it to the new third.
    _observe(clamp, [0.55], [0], [1], [0])
    assert _edge(clamp, 0, 1) == 0.60
    np.testing.assert_array_equal(np.asarray(clamp.contacts)[0, 1], [0.50, 0.55, 0.60])


def test_the_clamp_is_the_identity_at_and_above_every_edge() -> None:
    rng = np.random.default_rng(3)
    for support in (1, 3):
        clamp = ContactClamp(TYPE_MAP, support=support)
        distance = rng.uniform(0.3, 3.0, 64)
        center = rng.integers(0, 3, 64)
        neighbor = rng.integers(0, 3, 64)
        _observe(clamp, distance, center, neighbor, np.arange(64))
        read = np.asarray(clamp.call(distance, center, neighbor))
        edge = np.asarray(clamp.edges)[center, neighbor]
        above = distance >= edge
        np.testing.assert_array_equal(read[above], distance[above])
        if support == 1:
            assert above.all()


def test_observe_edges_reads_types_and_frames_from_the_graph() -> None:
    clamp = ContactClamp(TYPE_MAP, support=2)
    # Two frames of an O and a Si atom, each with both directed edges.
    edge_vec = np.array(
        [[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.4, 0.0, 0.0], [0.4, 0, 0]]
    )
    edge_index = np.array([[0, 1, 2, 3], [1, 0, 3, 2]])
    atype = np.array([0, 2, 0, 2])
    clamp.observe_edges(
        edge_vec, edge_index, atype, np.ones(4, dtype=bool), np.array([0, 0, 1, 1])
    )
    assert _edge(clamp, 0, 2) == pytest.approx(0.4)
    assert _edge(clamp, 2, 0) == pytest.approx(0.4)


def test_merged_records_equal_one_record_of_all_batches() -> None:
    rng = np.random.default_rng(5)
    batches = [
        (rng.uniform(0.3, 2.0, 40), rng.integers(0, 3, 40), rng.integers(0, 3, 40))
        for _ in range(3)
    ]
    whole = ContactClamp(TYPE_MAP, support=3)
    ranks = [ContactClamp(TYPE_MAP, support=3) for _ in batches]
    for index, (distance, center, neighbor) in enumerate(batches):
        frame = index * 100 + rng.integers(0, 10, 40)
        _observe(whole, distance, center, neighbor, frame)
        _observe(ranks[index], distance, center, neighbor, frame)
    merged = whole.merged(np.stack([np.asarray(rank.contacts) for rank in ranks]))
    np.testing.assert_array_equal(merged, np.asarray(whole.contacts))


def test_replaying_a_contact_leaves_the_record_unchanged() -> None:
    # The second contact repeats the first (6e-7 apart), the third does not
    # (1.2e-6 from the first).
    contacts = [0.5 + 3e-7, 0.5 + 9e-7, 0.5 + 1.5e-6, 0.6]
    clamp = ContactClamp(TYPE_MAP, support=3)
    for frame, separation in enumerate(contacts):
        _observe(clamp, [separation], [0], [1], [frame])
    record = np.array(clamp.contacts, copy=True)
    edge = _edge(clamp, 0, 1)
    assert edge == contacts[3]
    for replay, separation in enumerate(contacts):
        _observe(clamp, [separation], [0], [1], [10 + replay])
        np.testing.assert_array_equal(np.asarray(clamp.contacts), record)
        assert _edge(clamp, 0, 1) == edge


def test_the_record_does_not_depend_on_the_order_or_split_of_the_contacts() -> None:
    values = np.concatenate([0.5 + 5e-7 * np.arange(8), [0.7, 0.9]])
    zeros = np.zeros(10, dtype=np.int64)
    whole = ContactClamp(TYPE_MAP, support=3)
    _observe(whole, values, zeros, zeros + 1, np.arange(10))
    for seed in range(4):
        order = np.random.default_rng(seed).permutation(10)
        split = ContactClamp(TYPE_MAP, support=3)
        for chunk in np.array_split(order, 3):
            _observe(split, values[chunk], zeros[chunk], zeros[chunk] + 1, chunk)
        np.testing.assert_array_equal(
            np.asarray(split.contacts), np.asarray(whole.contacts)
        )


def test_serialize_roundtrip_keeps_the_record_and_the_edges() -> None:
    clamp = ContactClamp(TYPE_MAP, inner=0.7, support=2)
    _observe(clamp, [0.5, 0.6], [0, 0], [1, 1], [0, 1])
    restored = ContactClamp.deserialize(clamp.serialize())
    assert restored.inner == 0.7
    assert restored.support == 2
    assert restored.type_map == TYPE_MAP
    np.testing.assert_array_equal(np.asarray(restored.edges), np.asarray(clamp.edges))
    np.testing.assert_array_equal(
        np.asarray(restored.contacts), np.asarray(clamp.contacts)
    )


@pytest.mark.parametrize("support", [0, -1])
def test_support_must_be_positive(support) -> None:
    with pytest.raises(ValueError, match="support"):
        ContactClamp(TYPE_MAP, support=support)


def _check_synchronized_contacts(rank: int, store: str) -> None:
    import torch
    import torch.distributed as dist

    from deepmd.pt_expt.utils.contact_clamp import ContactClamp as TorchClamp
    from deepmd.pt_expt.utils.contact_clamp import (
        synchronize_contact_edges,
    )

    module = torch.nn.Module()
    module.clamp = TorchClamp(TYPE_MAP, inner=0.75, support=2).to("cpu")
    values = torch.tensor(
        ([0.5, 0.7], [0.6, 0.8])[rank], dtype=torch.float64, device="cpu"
    )
    atom_type = torch.zeros_like(values, dtype=torch.int64)
    module.clamp.observe(
        values, atom_type, atom_type + 1, values > 0, torch.arange(2, device="cpu")
    )
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=2
    )
    try:
        synchronize_contact_edges(module)
        torch.testing.assert_close(
            module.clamp.contacts[0, 1], values.new_tensor([0.5, 0.6])
        )
        assert module.clamp.edges[0, 1].item() == pytest.approx(0.6)
    finally:
        dist.destroy_process_group()


class TestTorchClamp:
    """The pt / pt_expt clamp: the dpmodel clamp on torch buffers, in place."""

    @pytest.fixture(autouse=True)
    def _require_torch(self) -> None:
        pytest.importorskip("torch")

    @staticmethod
    def _clamp(support: int = 3):
        from deepmd.pt_expt.utils.contact_clamp import ContactClamp as ContactClampPT

        return ContactClampPT(TYPE_MAP, inner=0.75, support=support).to("cpu")

    @pytest.mark.parametrize("training", [False, True])
    def test_matches_the_dpmodel_clamp(self, training: bool) -> None:
        import torch

        reference = ContactClamp(TYPE_MAP, inner=0.75)
        clamp = self._clamp().train(training)
        rng = np.random.default_rng(1)
        distance = rng.uniform(0.0, 3.0, (32, 1))
        center = rng.integers(-1, 3, 32)
        neighbor = rng.integers(0, 3, 32)
        expected = np.asarray(reference.call(distance, center, neighbor))
        actual = clamp.call(
            torch.from_numpy(distance),
            torch.from_numpy(center),
            torch.from_numpy(neighbor),
        )
        np.testing.assert_array_equal(actual.numpy(), expected)

    def test_observe_matches_the_dpmodel_record_and_updates_buffers_in_place(
        self,
    ) -> None:
        import torch

        rng = np.random.default_rng(2)
        distance = rng.uniform(0.3, 2.0, 50)
        center = rng.integers(-1, 3, 50)
        neighbor = rng.integers(0, 3, 50)
        mask = rng.uniform(size=50) > 0.2
        frame = rng.integers(0, 6, 50)
        reference = ContactClamp(TYPE_MAP, support=3)
        reference.observe(distance, center, neighbor, mask, frame)
        clamp = self._clamp()
        buffers = dict(clamp.named_buffers())
        assert {"edges", "contacts"} <= set(buffers)
        clamp.observe(
            *(torch.from_numpy(a) for a in (distance, center, neighbor, mask, frame))
        )
        assert clamp.edges is buffers["edges"]
        assert clamp.contacts is buffers["contacts"]
        np.testing.assert_array_equal(clamp.contacts.numpy(), reference.contacts)
        np.testing.assert_array_equal(clamp.edges.numpy(), reference.edges)

    def test_state_without_a_clamp_keeps_the_empty_record(self) -> None:
        clamp = self._clamp()
        start = clamp.edges.clone()
        clamp.load_state_dict({}, strict=True)
        np.testing.assert_array_equal(clamp.edges.numpy(), start.numpy())
        assert np.isinf(clamp.contacts.numpy()).all()

    @pytest.mark.parametrize(("written", "read"), [(3, 1), (1, 3), (2, 5)])
    def test_state_of_another_support_keeps_its_smallest_contacts(
        self, written: int, read: int
    ) -> None:
        rng = np.random.default_rng(4)
        arrays = (
            rng.uniform(0.3, 1.0, 40),
            rng.integers(0, 3, 40),
            rng.integers(0, 3, 40),
            np.ones(40, dtype=bool),
            np.arange(40),
        )
        import torch

        source = self._clamp(written)
        source.observe(*(torch.from_numpy(a) for a in arrays))
        target = self._clamp(read)
        target.load_state_dict(source.state_dict(), strict=True)
        reference = ContactClamp(TYPE_MAP, support=read)
        reference.observe(*arrays)
        kept = min(written, read)
        np.testing.assert_array_equal(
            target.contacts.numpy()[..., :kept], reference.contacts[..., :kept]
        )
        assert np.isinf(target.contacts.numpy()[..., kept:]).all()
        np.testing.assert_array_equal(
            target.edges.numpy(), np.asarray(target.edges_of(target.contacts))
        )

    def test_synchronization_without_a_process_group_is_a_no_op(self) -> None:
        import torch

        from deepmd.pt_expt.utils.contact_clamp import (
            synchronize_contact_edges,
        )

        module = torch.nn.Module()
        module.clamp = self._clamp()
        start = module.clamp.edges.clone()
        synchronize_contact_edges(module)
        np.testing.assert_array_equal(module.clamp.edges.numpy(), start.numpy())

    def test_ranks_adopt_the_same_contact_record(self, tmp_path: Path) -> None:
        import torch.distributed as dist
        import torch.multiprocessing as mp

        if not dist.is_available() or not dist.is_gloo_available():
            pytest.skip("Gloo process groups are unavailable")
        mp.spawn(
            _check_synchronized_contacts, args=(str(tmp_path / "store"),), nprocs=2
        )

    def test_a_coincident_contact_keeps_the_read_gradient_finite(self) -> None:
        import torch

        clamp = self._clamp(support=1)
        types = torch.zeros(1, dtype=torch.int64, device="cpu")
        clamp.observe(
            torch.zeros(1, dtype=torch.float64, device="cpu"),
            types,
            types,
            torch.ones(1, dtype=torch.bool, device="cpu"),
            torch.zeros(1, dtype=torch.int64, device="cpu"),
        )
        distance = torch.full(
            (1,), 0.8, dtype=torch.float64, requires_grad=True, device="cpu"
        )
        clamp.call(distance, types, types).sum().backward()
        assert torch.isfinite(distance.grad).all()
