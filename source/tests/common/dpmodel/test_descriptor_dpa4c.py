# SPDX-License-Identifier: LGPL-3.0-or-later

import dataclasses
from typing import (
    Any,
)
from unittest import (
    mock,
)

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa4c import (
    DescrptDPA4C,
)
from deepmd.dpmodel.descriptor.dpa4c_nn import (
    build_angular_basis,
    build_bispectrum_layout,
    enumerate_degree_triples,
    packed_l2_to_stf,
)
from deepmd.dpmodel.utils import (
    neighbor_graph,
)
from deepmd.dpmodel.utils.lebedev import (
    load_lebedev_rule,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    graph_from_dense_quartet,
)
from deepmd.dpmodel.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.dpmodel.utils.update_sel import (
    UpdateSel,
)

COORD = np.array(
    [
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.2, -0.1],
            [-0.4, 0.9, 0.3],
            [0.2, -0.5, 1.2],
            [-0.7, -0.3, -0.8],
        ]
    ],
    dtype=np.float64,
)
ATYPE = np.array([[0, 1, 0, 1, 0]], dtype=np.int64)


def dense_inputs(
    descriptor: DescrptDPA4C,
    coord: np.ndarray = COORD,
    atype: np.ndarray = ATYPE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a complete bounded neighbor list for a small test system."""
    return extend_input_and_build_neighbor_list(
        coord,
        atype,
        descriptor.get_rcut(),
        [8],
        mixed_types=True,
        box=None,
    )


def evaluate(
    descriptor: DescrptDPA4C,
    coord: np.ndarray = COORD,
    atype: np.ndarray = ATYPE,
) -> np.ndarray:
    """Evaluate a descriptor through the dense compatibility interface."""
    coord_ext, atype_ext, mapping, nlist = dense_inputs(
        descriptor,
        coord,
        atype,
    )
    return descriptor(
        coord_ext,
        atype_ext,
        nlist,
        mapping=mapping,
    )[0]


def build_graph(
    descriptor: DescrptDPA4C,
    coord: np.ndarray = COORD,
    atype: np.ndarray = ATYPE,
) -> tuple[Any, np.ndarray]:
    """Build the flat neighbor graph consumed by the graph-native equations."""
    coord_ext, atype_ext, mapping, nlist = dense_inputs(descriptor, coord, atype)
    return graph_from_dense_quartet(coord_ext, atype_ext, nlist, mapping)


def edge_features(
    descriptor: DescrptDPA4C,
    graph: Any,
    atype_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the masked edge amplitudes, harmonics, and cutoff envelope."""
    return descriptor.build_edge_features(
        graph,
        atype_local,
        *descriptor.pair_film.call(descriptor.type_embedding.call()),
    )


def moment_blocks(
    descriptor: DescrptDPA4C,
    moments: np.ndarray,
) -> list[np.ndarray]:
    """Split flat moments and apply the fixed degree-one/two alignment."""
    readout = descriptor.readout
    blocks = []
    for degree, width in enumerate(descriptor.degree_channels):
        start, end = readout.degree_offsets[degree : degree + 2]
        blocks.append(
            moments[:, start:end].reshape(
                moments.shape[0],
                2 * degree + 1,
                width,
            )
        )
    for degree, projection in enumerate(readout.channel_alignment, start=1):
        blocks[degree] = projection.call(blocks[degree])
    return blocks


def projected_blocks(
    descriptor: DescrptDPA4C,
    moments: np.ndarray,
) -> list[np.ndarray]:
    """Build effective low-rank blocks from aligned moments."""
    blocks = moment_blocks(descriptor, moments)
    return [
        block if projection is None else projection.call(block)
        for projection, block in zip(
            descriptor.readout.probe_projections,
            blocks[1:],
            strict=True,
        )
    ]


class TestDPA4C:
    def setup_method(self) -> None:
        self.descriptor = DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            channels=8,
            lmax=2,
            n_radial=4,
            precision="float64",
            seed=17,
        )

    def test_single_reduction_dense_graph_parity(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = 0
        segment_sum = neighbor_graph.segment_sum

        def count_segment_sum(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            return segment_sum(*args, **kwargs)

        monkeypatch.setattr(neighbor_graph, "segment_sum", count_segment_sum)
        coord_ext, atype_ext, mapping, nlist = dense_inputs(self.descriptor)
        dense = self.descriptor(
            coord_ext,
            atype_ext,
            nlist,
            mapping=mapping,
        )[0]
        assert calls == 1

        graph, atype_local = graph_from_dense_quartet(
            coord_ext,
            atype_ext,
            nlist,
            mapping,
        )
        calls = 0
        graph_output, rotation = self.descriptor.call_graph(graph, atype_local)
        assert calls == 1
        assert rotation is None
        np.testing.assert_allclose(
            graph_output.reshape(dense.shape),
            dense,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_moments_match_explicit_reference(self) -> None:
        descriptor = DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            channels=32,
            lmax=2,
            n_radial=4,
            radial_modes=2,
            precision="float64",
            seed=17,
        )
        graph, atype_local = build_graph(descriptor)
        amplitude, basis, envelope = edge_features(descriptor, graph, atype_local)
        dst = graph.edge_index[1]
        n_total = atype_local.shape[0]
        moments, divisors = descriptor.aggregate_moments(
            amplitude,
            basis,
            envelope,
            dst,
            n_total,
        )
        scalar = moments[:, : descriptor.channels]
        angular = moments[:, descriptor.channels :]
        scalar_mass = np.zeros(n_total, dtype=amplitude.dtype)
        angular_mass = np.zeros(n_total, dtype=amplitude.dtype)
        np.add.at(scalar_mass, dst, envelope**2)
        np.add.at(angular_mass, dst, envelope**4)
        floor = descriptor._DEGREE_NORM_FLOOR
        expected_scalar_normalizer = 1.0 / np.sqrt(scalar_mass + floor)
        expected_angular_normalizer = 1.0 / np.sqrt(angular_mass + floor)

        # The reduction also returns the two divisors, which are exactly the
        # reciprocals of the normalizers applied to the moments.
        np.testing.assert_allclose(
            divisors,
            np.stack(
                [1.0 / expected_scalar_normalizer, 1.0 / expected_angular_normalizer],
                axis=-1,
            ),
            atol=2e-13,
            rtol=2e-13,
        )
        reduced_scalar = np.zeros(
            (n_total, descriptor.channels),
            dtype=amplitude.dtype,
        )
        np.add.at(reduced_scalar, dst, amplitude[:, : descriptor.channels])
        np.testing.assert_allclose(
            scalar,
            reduced_scalar * expected_scalar_normalizer[:, None],
            atol=2e-13,
            rtol=2e-13,
        )

        # Every non-scalar degree carries one additional envelope factor and
        # the matched normalizer derived from the fourth envelope power.
        expected_blocks = []
        for degree in (1, 2):
            channels = list(range(descriptor.degree_channels[degree]))
            edge_value = (
                amplitude[:, channels][:, None, :]
                * basis[:, degree**2 : (degree + 1) ** 2, None]
                * envelope[:, None, None]
            )
            reduced = np.zeros(
                (n_total, 2 * degree + 1, len(channels)),
                dtype=edge_value.dtype,
            )
            np.add.at(reduced, dst, edge_value)
            reduced *= expected_angular_normalizer[:, None, None]
            expected_blocks.append(reduced.reshape(n_total, -1))
        np.testing.assert_allclose(
            angular,
            np.concatenate(expected_blocks, axis=1),
            atol=2e-13,
            rtol=2e-13,
        )

    def test_lmax4_o3_translation_and_permutation_invariance(self) -> None:
        descriptor = DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            channels=16,
            lmax=4,
            n_radial=4,
            precision="float64",
            seed=19,
        )
        rotation = np.array(
            [
                [-2.0 / 3.0, 2.0 / 15.0, 11.0 / 15.0],
                [2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0],
                [1.0 / 3.0, 14.0 / 15.0, 2.0 / 15.0],
            ],
            dtype=np.float64,
        )

        reference = evaluate(descriptor)
        rotated = evaluate(descriptor, COORD @ rotation.T)
        reflected_coord = COORD.copy()
        reflected_coord[..., 0] *= -1.0
        reflected = evaluate(descriptor, reflected_coord)
        translated = evaluate(
            descriptor,
            COORD + np.array([1.7, -0.8, 2.1]),
        )
        permutation = np.array([2, 4, 0, 3, 1])
        permuted = evaluate(
            descriptor,
            COORD[:, permutation],
            ATYPE[:, permutation],
        )

        np.testing.assert_allclose(rotated, reference, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(reflected, reference, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(translated, reference, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(
            permuted,
            reference[:, permutation],
            atol=2e-12,
            rtol=2e-12,
        )

    def test_serialization_roundtrip_preserves_fixed_structure(self) -> None:
        self.descriptor.compute_input_stats([{"coord": COORD, "atype": ATYPE}])
        reference = evaluate(self.descriptor)

        # The two neighborhood masses are standardized rather than merely
        # rescaled: their information lies on an offset far larger than any
        # other invariant carries, so only these coordinates take a mean.
        geometry_end = self.descriptor.get_dim_out() - self.descriptor.channels
        mass = slice(geometry_end - 2, geometry_end)
        assert np.all(self.descriptor.mean[mass] > 0.0)
        np.testing.assert_array_equal(np.delete(self.descriptor.mean, mass), 0.0)

        data = self.descriptor.serialize()
        assert data["radial_embedding"]["mlp_layers"] == [4, 24, 8]
        assert data["channels"] == 8
        assert data["lmax"] == 2
        assert "bispectrum_ranks" not in data
        assert data["readout"]["channels"] == 8
        assert data["readout"]["lmax"] == 2
        assert "bispectrum_ranks" not in data["readout"]

        restored = DescrptDPA4C.deserialize(data)
        result = evaluate(restored)
        np.testing.assert_array_equal(result, reference)
        np.testing.assert_array_equal(restored.mean, self.descriptor.mean)
        np.testing.assert_array_equal(restored.stddev, self.descriptor.stddev)

        default_data = DescrptDPA4C(rcut=3.0, ntypes=1, use_amp=True).serialize()
        assert default_data["channels"] == 32
        assert default_data["lmax"] == 2
        assert default_data["radial_modes"] == 0
        # Mixed precision is an execution policy supplied at load time, so a
        # checkpoint must not carry it.
        assert "use_amp" not in default_data

    @pytest.mark.parametrize(
        "divergence",
        [
            {"rcut": 4.0},
            {"ntypes": 3},
            {"channels": 16},
            {"lmax": 3},
            {"basis_type": "gaussian"},
            {"n_radial": 8},
            {"radial_modes": 2},
            {"use_amp": True},
            {"trainable": False},
            {"type_map": ["H", "O"]},
            {"precision": "float32"},
        ],
    )
    def test_sharing_rejects_incompatible_structures(
        self,
        divergence: dict[str, Any],
    ) -> None:
        """Every field that shapes a shared module must block sharing.

        Sharing binds the type table, radial basis, radial network, mode head,
        pair cache, and readout of the base descriptor into the replica. A
        divergence that the signature fails to catch is silent rather than
        loud: gathers may run out of bounds, a replica may inherit frozen
        weights while still reporting itself trainable, or two branches may
        read the same type table under different element orders.
        """
        config = {
            "rcut": 3.0,
            "ntypes": 2,
            "channels": 8,
            "lmax": 2,
            "basis_type": "bessel",
            "n_radial": 4,
            "radial_modes": 1,
            "trainable": True,
            "type_map": ["O", "H"],
            "precision": "float64",
        }
        base = DescrptDPA4C(**config, seed=0)
        DescrptDPA4C(**config, seed=1).share_params(base, 0)
        with pytest.raises(ValueError, match="identical structural parameters"):
            DescrptDPA4C(**{**config, **divergence}, seed=1).share_params(base, 0)

    def test_shared_replica_matches_the_base_descriptor(self) -> None:
        config = {
            "rcut": 3.0,
            "ntypes": 2,
            "channels": 8,
            "lmax": 2,
            "n_radial": 4,
            "radial_modes": 1,
            "precision": "float64",
        }
        base = DescrptDPA4C(**config, seed=0)
        replica = DescrptDPA4C(**config, seed=1)
        assert not np.allclose(evaluate(replica), evaluate(base))
        replica.share_params(base, 0)
        np.testing.assert_array_equal(evaluate(replica), evaluate(base))

    def test_aligned_grams_match_explicit_contractions(self) -> None:
        descriptor = DescrptDPA4C(
            rcut=3.0,
            ntypes=1,
            channels=16,
            lmax=4,
            precision="float64",
            seed=29,
        )
        rng = np.random.default_rng(31)
        moments = rng.normal(size=(5, descriptor.readout.degree_offsets[-1]))
        output = descriptor.readout.call(moments)
        blocks = moment_blocks(descriptor, moments)

        cursor = descriptor.channels
        for block, width in zip(
            blocks[1:],
            descriptor.degree_channels[1:],
            strict=True,
        ):
            gram = np.transpose(block, (0, 2, 1)) @ block
            row, column = np.triu_indices(width)
            expected = gram[:, row, column]
            expected *= np.where(row == column, 1.0, np.sqrt(2.0))[None, :]
            actual = output[:, cursor : cursor + row.size]
            np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)
            # The off-diagonal scale exists to make the half-vectorization
            # norm preserving, so assert that property rather than the
            # constant the implementation happens to use.
            np.testing.assert_allclose(
                np.sum(actual**2, axis=1),
                np.sum(gram**2, axis=(1, 2)),
                atol=1e-12,
                rtol=1e-12,
            )
            cursor += row.size
        assert cursor == descriptor.channels + descriptor.readout.gram_index.size

    def test_bispectrum_and_quartic_match_explicit_references(
        self,
    ) -> None:
        descriptor = DescrptDPA4C(
            rcut=3.0,
            ntypes=1,
            channels=32,
            lmax=4,
            precision="float64",
            seed=37,
        )
        rng = np.random.default_rng(37)
        moments = rng.normal(size=(3, descriptor.readout.degree_offsets[-1]))
        output = descriptor.readout.call(moments)
        projected = projected_blocks(descriptor, moments)

        expected_parts = []
        for triple_index, degrees in enumerate(descriptor.readout.degree_triples):
            coupling_start, coupling_end = descriptor.readout.coupling_offsets[
                triple_index : triple_index + 2
            ]
            coupling = descriptor.readout.bispectrum_coupling[
                coupling_start:coupling_end
            ].reshape(*(2 * degree + 1 for degree in degrees))
            full = np.einsum(
                "ijk,nia,njb,nkc->nabc",
                coupling,
                projected[degrees[0] - 1],
                projected[degrees[1] - 1],
                projected[degrees[2] - 1],
                optimize=True,
            ).reshape(3, -1)
            probe_start, probe_end = descriptor.readout.probe_offsets[
                triple_index : triple_index + 2
            ]
            reduced = (
                full[
                    :,
                    descriptor.readout.probe_index[probe_start:probe_end],
                ]
                * descriptor.readout.probe_scale[None, probe_start:probe_end]
            )
            # Equal-degree axes emit one representative per orbit. The
            # multiplicity scale exists so that dropping the rest preserves
            # the norm of the full symmetric tensor.
            np.testing.assert_allclose(
                np.sum(reduced**2, axis=1),
                np.sum(full**2, axis=1),
                atol=1e-12,
                rtol=1e-12,
            )
            expected_parts.append(reduced)
        expected_bispectrum = np.concatenate(expected_parts, axis=1)
        bispectrum_start = descriptor.channels + descriptor.readout.gram_index.size
        bispectrum_end = bispectrum_start + expected_bispectrum.shape[1]
        np.testing.assert_allclose(
            output[:, bispectrum_start:bispectrum_end],
            expected_bispectrum,
            atol=2e-13,
            rtol=2e-13,
        )

        vectors = np.transpose(projected[0], (0, 2, 1))
        tensors = packed_l2_to_stf(np.transpose(projected[1], (0, 2, 1)))
        tensor_vector = np.matmul(
            tensors[:, :, None, :, :],
            vectors[:, None, :, :, None],
        )[..., 0]
        expected_quartic = np.sum(
            tensor_vector * tensor_vector,
            axis=-1,
        ).reshape(3, -1)
        np.testing.assert_allclose(
            output[:, bispectrum_end:],
            expected_quartic,
            atol=2e-13,
            rtol=2e-13,
        )

    def test_jax_lmax4_matches_numpy(self) -> None:
        pytest.importorskip("jax")
        from deepmd.jax.env import (
            jax,
            jnp,
        )

        descriptor = DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            channels=16,
            lmax=4,
            n_radial=4,
            radial_modes=2,
            precision="float64",
            seed=43,
        )
        graph, atype_local = build_graph(descriptor)
        reference, _ = descriptor.call_graph(graph, atype_local)
        graph_jax = dataclasses.replace(
            graph,
            n_node=jnp.asarray(graph.n_node),
            edge_index=jnp.asarray(graph.edge_index),
            edge_vec=jnp.asarray(graph.edge_vec),
            edge_mask=jnp.asarray(graph.edge_mask),
        )
        atype_jax = jnp.asarray(atype_local)

        def evaluate_jax(edge_vec: object) -> object:
            current_graph = dataclasses.replace(graph_jax, edge_vec=edge_vec)
            return descriptor.call_graph(current_graph, atype_jax)[0]

        np.testing.assert_allclose(
            np.asarray(jax.jit(evaluate_jax)(graph_jax.edge_vec)),
            reference,
            atol=3e-10,
            rtol=3e-10,
        )

    def test_addition_theorem_and_parity_through_degree_four(self) -> None:
        rng = np.random.default_rng(41)
        left = rng.normal(size=(32, 3))
        right = rng.normal(size=(32, 3))
        left /= np.linalg.norm(left, axis=-1, keepdims=True)
        right /= np.linalg.norm(right, axis=-1, keepdims=True)
        left_basis = build_angular_basis(left, 4)
        right_basis = build_angular_basis(right, 4)
        reflected_basis = build_angular_basis(-left, 4)
        cosine = np.sum(left * right, axis=-1)
        legendre = (
            np.ones_like(cosine),
            cosine,
            0.5 * (3.0 * cosine**2 - 1.0),
            0.5 * (5.0 * cosine**3 - 3.0 * cosine),
            0.125 * (35.0 * cosine**4 - 30.0 * cosine**2 + 3.0),
        )

        for degree in range(5):
            start, end = degree**2, (degree + 1) ** 2
            np.testing.assert_allclose(
                np.sum(
                    left_basis[:, start:end] * right_basis[:, start:end],
                    axis=1,
                ),
                legendre[degree],
                atol=2e-15,
                rtol=2e-15,
            )
            np.testing.assert_allclose(
                reflected_basis[:, start:end],
                (-1) ** degree * left_basis[:, start:end],
                atol=2e-15,
                rtol=2e-15,
            )

    def test_allowed_degree_triples_and_gaunt_couplings(self) -> None:
        expected_triples = (
            (1, 1, 2),
            (1, 2, 3),
            (1, 3, 4),
            (2, 2, 2),
            (2, 2, 4),
            (2, 3, 3),
            (2, 4, 4),
            (3, 3, 4),
            (4, 4, 4),
        )
        assert enumerate_degree_triples(4) == expected_triples

        layout = build_bispectrum_layout(4, [1, 1, 1, 1])
        points, weights = load_lebedev_rule(17)
        basis = build_angular_basis(points, 4)
        for triple_index, degrees in enumerate(expected_triples):
            start, end = layout.coupling_offsets[triple_index : triple_index + 2]
            coupling = layout.coupling[start:end].reshape(
                *(2 * degree + 1 for degree in degrees)
            )
            degree_1, degree_2, degree_3 = degrees
            reference = np.einsum(
                "n,ni,nj,nk->ijk",
                weights,
                basis[:, degree_1**2 : (degree_1 + 1) ** 2],
                basis[:, degree_2**2 : (degree_2 + 1) ** 2],
                basis[:, degree_3**2 : (degree_3 + 1) ** 2],
                optimize=True,
            )
            reference /= np.linalg.norm(reference)
            first = np.flatnonzero(np.abs(reference) > 1.0e-14)[0]
            if reference.flat[first] < 0.0:
                reference = -reference
            np.testing.assert_allclose(coupling, reference, atol=2e-14, rtol=2e-14)
            np.testing.assert_allclose(np.linalg.norm(coupling), 1.0, atol=1e-15)


@pytest.mark.parametrize(
    ("config", "error"),
    [
        ({"rcut": 0.0}, ValueError),
        ({"ntypes": 0}, ValueError),
        ({"n_radial": 0}, ValueError),
        ({"channels": 15}, ValueError),
        ({"channels": 256}, ValueError),
        ({"channels": 32.0}, TypeError),
        ({"lmax": 1}, ValueError),
        ({"lmax": 5}, ValueError),
        ({"lmax": 2.0}, TypeError),
        # `bool` is an `int` subclass, so the guards reject it explicitly.
        ({"channels": True}, TypeError),
        ({"radial_modes": True}, ValueError),
        ({"radial_modes": -1}, ValueError),
        ({"spin": {}}, NotImplementedError),
    ],
)
def test_configuration_boundaries(
    config: dict[str, Any],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        DescrptDPA4C(**{"rcut": 3.0, "ntypes": 1, **config})


@pytest.mark.parametrize("neighbors", [8, 4096])
def test_neighbor_statistics_report_distance_without_deriving_a_capacity(
    neighbors: int,
) -> None:
    """Statistics must never reject an environment or introduce a ``sel``.

    The descriptor is graph-native, so an environment denser than the dense
    adapter bound is still valid; only the minimum neighbor distance is read.
    """
    config = {"type": "dpa4c", "rcut": 6.0, "channels": 32, "lmax": 2}
    with mock.patch.object(
        UpdateSel,
        "get_nbor_stat",
        return_value=(0.8, [neighbors]),
    ):
        updated, min_nbor_dist = DescrptDPA4C.update_sel(None, ["O", "H"], config)

    assert updated == config
    assert "sel" not in updated
    assert min_nbor_dist == 0.8


def _bispectrum_dimension(lmax: int, ranks: list[int]) -> int:
    """Return the independent bispectrum width from rank combinatorics."""
    dimension = 0
    for degree_1, degree_2, degree_3 in enumerate_degree_triples(lmax):
        rank_1 = ranks[degree_1 - 1]
        rank_2 = ranks[degree_2 - 1]
        rank_3 = ranks[degree_3 - 1]
        if degree_1 == degree_3:
            dimension += rank_1 * (rank_1 + 1) * (rank_1 + 2) // 6
        elif degree_1 == degree_2:
            dimension += rank_1 * (rank_1 + 1) * rank_3 // 2
        elif degree_2 == degree_3:
            dimension += rank_1 * rank_2 * (rank_2 + 1) // 2
        else:
            dimension += rank_1 * rank_2 * rank_3
    return dimension


@pytest.mark.parametrize(
    ("channels", "base_degree_channels", "base_ranks"),
    [
        (8, [8, 4, 4], [4, 2]),
        (16, [16, 4, 4], [4, 2]),
        (32, [32, 8, 4], [4, 2]),
        (64, [64, 8, 4], [4, 2]),
        (128, [128, 16, 8], [8, 2]),
    ],
)
def test_automatic_profiles_and_output_dimensions(
    channels: int,
    base_degree_channels: list[int],
    base_ranks: list[int],
) -> None:
    for lmax in (2, 3, 4):
        descriptor = DescrptDPA4C(
            rcut=3.0,
            ntypes=1,
            channels=channels,
            lmax=lmax,
            radial_modes=3,
        )
        degree_channels = base_degree_channels + [1] * (lmax - 2)
        ranks = base_ranks + [1] * (lmax - 2)
        assert descriptor.degree_channels == degree_channels
        assert descriptor.bispectrum_ranks == ranks

        # The radial function class never enters the descriptor layout. The
        # trailing pair is the two neighborhood masses.
        expected_dim = (
            2 * channels
            + sum(width * (width + 1) // 2 for width in degree_channels[1:])
            + _bispectrum_dimension(lmax, ranks)
            + ranks[0] * ranks[1]
            + 2
        )
        assert descriptor.get_dim_out() == expected_dim
