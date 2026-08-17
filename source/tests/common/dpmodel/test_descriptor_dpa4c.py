# SPDX-License-Identifier: LGPL-3.0-or-later

import dataclasses
import math
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
from deepmd.dpmodel.descriptor.dpa4c_nn.spin import (
    SpinChannels,
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
        descriptor.pair_film.pair_latent(descriptor.type_embedding.call()),
    )[:3]


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
            None,
            None,
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


# === Native spin ===

SPIN_COORD = np.array(
    [
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.3, -0.2],
            [-0.5, 1.2, 0.4],
            [0.3, -0.7, 1.5],
            [-0.9, -0.4, -1.0],
            [1.1, -1.2, 0.6],
        ]
    ],
    dtype=np.float64,
)
SPIN_ATYPE = np.array([[0, 1, 0, 1, 0, 1]], dtype=np.int64)


def make_spin_descriptor(**overrides: Any) -> DescrptDPA4C:
    """Build a descriptor whose first atom type carries a magnetic moment.

    The branch gate is opened, since a fresh descriptor starts spin-free by
    design and the tests below are about the branch behind the gate. The gate
    itself is covered by :class:`TestDPA4CSpinGate`.
    """
    config: dict[str, Any] = {
        "rcut": 3.0,
        "ntypes": 2,
        "channels": 8,
        "lmax": 2,
        "n_radial": 4,
        "precision": "float64",
        "seed": 23,
        "use_spin": [True, False],
    }
    config.update(overrides)
    descriptor = DescrptDPA4C(**config)
    if descriptor.spin is not None:
        descriptor.spin.spin_gate[...] = 1.0
    return descriptor


def spin_reference_terms(
    descriptor: DescrptDPA4C,
    graph: Any,
    atype: np.ndarray,
    spin: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate the two-body spin sums the readout is meant to reproduce.

    The reference reconstructs the edge stage from the descriptor's own
    modules rather than from the moments, so a layout error cannot cancel
    between the two sides of the comparison.
    """
    masked = descriptor.spin.conditioned_spin(spin, atype)
    source, destination = graph.edge_index[0], graph.edge_index[1]
    edge_vec = graph.edge_vec
    distance = np.sqrt(
        np.sum(edge_vec * edge_vec, axis=-1, keepdims=True) + descriptor._EPS**2
    )
    direction = edge_vec / distance
    center, neighbor = atype[destination], atype[source]
    real = (center < descriptor.ntypes) & (neighbor < descriptor.ntypes)
    envelope = descriptor.evaluate_cutoff_envelope(distance)[:, 0] * (
        graph.edge_mask & real
    )
    radial = descriptor.radial_embedding.call(descriptor.radial_basis.call(distance))
    scale, shift, _, spin_scale, spin_shift = descriptor.pair_film.call(
        descriptor.type_embedding.call()
    )
    pair = center * (descriptor.ntypes + 1) + neighbor
    channels = descriptor.spin_channels
    weight = (radial[:, :channels] * spin_scale[pair] + spin_shift[pair]) * (
        envelope * envelope
    )[:, None]
    amplitude = (radial * scale[pair] + shift[pair]) * envelope[:, None]

    center_spin = masked[destination]
    neighbor_spin = masked[source]
    dot = np.sum(center_spin * neighbor_spin, axis=-1)
    projection = np.sum(center_spin * direction, axis=-1)
    neighbor_projection = np.sum(neighbor_spin * direction, axis=-1)
    center_norm = np.sum(center_spin * center_spin, axis=-1)
    neighbor_norm = np.sum(neighbor_spin * neighbor_spin, axis=-1)

    nodes = atype.shape[0]

    def reduce(values: np.ndarray) -> np.ndarray:
        out = np.zeros((nodes, values.shape[1]), dtype=values.dtype)
        np.add.at(out, destination, values)
        return out

    return {
        # Heisenberg exchange, one sum per spin channel.
        "heisenberg": reduce(weight * dot[:, None]),
        # Symmetric anisotropic two-ion exchange, one sum per spin channel.
        "two_ion_anisotropy": reduce(
            weight * (projection * neighbor_projection)[:, None]
        ),
        # Biquadratic exchange from the leading spin channel only, which is
        # the one the neighbour quadrupole family reads.
        "biquadratic": reduce(
            weight[:, :1]
            * (0.5 * (3.0 * dot * dot - center_norm * neighbor_norm))[:, None]
        ),
        # Single-ion anisotropy against every geometric degree-two channel.
        "anisotropy": reduce(
            amplitude[:, : descriptor.degree_channels[2]]
            * envelope[:, None]
            * (0.5 * (3.0 * projection * projection - center_norm))[:, None]
        ),
    }


def vector_gram_selectors(descriptor: DescrptDPA4C) -> dict[str, np.ndarray]:
    """Return boolean column selectors of the joint degree-one spin Gram.

    The block holds the on-site moment, the ``spin_channels`` isotropic
    neighbour channels and the ``spin_channels`` bond-projected ones, in that
    order, so the entries of each physical interaction are addressed by their
    channel coordinates rather than by a hard-coded offset.
    """
    channels = descriptor.spin_channels
    row, column = np.triu_indices(descriptor.spin.vector_width)
    return {
        "heisenberg": (row == 0) & (column >= 1) & (column <= channels),
        "two_ion_anisotropy": (row == 0) & (column > channels),
        "bond": (row > channels) | (column > channels),
    }


class TestDPA4CSpinGate:
    """The scalar gate that carries the whole spin branch.

    Activating a magnetic type on a spin-free pretraining releases weights
    that never received a gradient, so the branch must start at the zero
    function instead. No weight inside the branch can do that: the families
    reach the fitting network by several routes and at two spin orders, and a
    factor on the conditioned moment would enter the invariants quadratically
    and leave zero a stationary point.
    """

    def setup_method(self) -> None:
        self.descriptor = make_spin_descriptor()
        rng = np.random.default_rng(5)
        self.spin = rng.normal(size=(SPIN_ATYPE.size, 3))
        self.graph = neighbor_graph.build_neighbor_graph(
            SPIN_COORD,
            SPIN_ATYPE,
            None,
            self.descriptor.get_rcut(),
        )
        self.atype = SPIN_ATYPE.reshape(-1)

    def evaluate(self, spin: np.ndarray | None) -> np.ndarray:
        return self.descriptor.call_graph(self.graph, self.atype, spin=spin)[0]

    def test_fresh_descriptor_starts_spin_free(self) -> None:
        """A freshly built descriptor closes the gate."""
        fresh = DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            channels=8,
            lmax=2,
            n_radial=4,
            precision="float64",
            seed=23,
            use_spin=[True, False],
        )
        np.testing.assert_array_equal(fresh.spin.spin_gate, 0.0)

    def spin_block(self, output: np.ndarray) -> np.ndarray:
        start = self.descriptor.readout.get_dim_out()
        return output[:, start : start + self.descriptor.spin.get_dim_out()]

    def test_closed_gate_erases_the_whole_branch(self) -> None:
        """A closed gate zeroes every family, for arbitrary moments.

        Zero moments are not the spin-free reference here: the magnetic
        effective coordination weighs the per-type mask rather than the
        moment, so it survives ``s = 0`` and naming a type magnetic moves the
        descriptor on its own. Only the gate removes that term as well.
        """
        assert np.any(self.spin_block(self.evaluate(np.zeros_like(self.spin))))
        self.descriptor.spin.spin_gate[...] = 0.0
        for moments in (self.spin, np.zeros_like(self.spin)):
            np.testing.assert_array_equal(self.spin_block(self.evaluate(moments)), 0.0)

    def test_invariants_are_linear_in_the_gate(self) -> None:
        """Linearity is what leaves the closed gate a nonzero gradient.

        A factor applied to the conditioned moment instead would reach the
        degree-one Grams squared and the quadrupole Grams to the fourth
        power, so its derivative would vanish with the gate itself.
        """
        unit = self.spin_block(self.evaluate(self.spin))
        for gate in (0.25, -1.5, 3.0):
            self.descriptor.spin.spin_gate[...] = gate
            np.testing.assert_allclose(
                self.spin_block(self.evaluate(self.spin)),
                gate * unit,
                rtol=1e-12,
                atol=1e-14,
            )

    def test_serialization_carries_the_gate(self) -> None:
        """The gate is trained state and therefore round-trips."""
        self.descriptor.spin.spin_gate[...] = 0.42
        payload = self.descriptor.spin.serialize()
        np.testing.assert_allclose(
            SpinChannels.deserialize(payload).spin_gate, 0.42, rtol=1e-12
        )


class TestDPA4CSpin:
    def setup_method(self) -> None:
        self.descriptor = make_spin_descriptor()
        rng = np.random.default_rng(5)
        self.spin = rng.normal(size=(SPIN_ATYPE.size, 3))
        self.graph = neighbor_graph.build_neighbor_graph(
            SPIN_COORD,
            SPIN_ATYPE,
            None,
            self.descriptor.get_rcut(),
        )
        self.atype = SPIN_ATYPE.reshape(-1)

    def evaluate(self, spin: np.ndarray | None) -> np.ndarray:
        return self.descriptor.call_graph(self.graph, self.atype, spin=spin)[0]

    def spin_block(self, descriptor_output: np.ndarray) -> np.ndarray:
        start = self.descriptor.readout.get_dim_out()
        return descriptor_output[:, start : start + self.descriptor.spin.get_dim_out()]

    def test_axial_o3_invariance_including_reflections(self) -> None:
        # Spin is an axial vector, so an improper transformation rotates it
        # and flips its sign. Because every emitted coordinate has even spin
        # order the descriptor is additionally invariant under the polar
        # convention, which this test also pins.
        rng = np.random.default_rng(11)
        reference = self.evaluate(self.spin)
        for determinant in (1.0, -1.0):
            orthogonal, triangular = np.linalg.qr(rng.normal(size=(3, 3)))
            orthogonal = orthogonal * np.sign(np.diag(triangular))
            if np.linalg.det(orthogonal) * determinant < 0.0:
                orthogonal = -orthogonal
            rotated = neighbor_graph.build_neighbor_graph(
                SPIN_COORD @ orthogonal.T,
                SPIN_ATYPE,
                None,
                self.descriptor.get_rcut(),
            )
            for spin in (
                (self.spin @ orthogonal.T) * np.linalg.det(orthogonal),
                self.spin @ orthogonal.T,
            ):
                output = self.descriptor.call_graph(rotated, self.atype, spin=spin)[0]
                np.testing.assert_allclose(output, reference, atol=1e-12)

    def test_time_reversal_is_exact(self) -> None:
        # Every spin family has even spin order, so a global moment flip is a
        # bitwise symmetry rather than an approximate one.
        np.testing.assert_array_equal(
            self.evaluate(self.spin),
            self.evaluate(-self.spin),
        )

    def test_non_magnetic_types_are_ignored_bitwise(self) -> None:
        # The per-type gate is multiplicative, so a non-magnetic atom has no
        # spin degree of freedom at any derivative order rather than merely a
        # vanishing one.
        polluted = self.spin.copy()
        polluted[self.atype == 1] += 7.0
        np.testing.assert_array_equal(
            self.evaluate(self.spin),
            self.evaluate(polluted),
        )

    def test_spin_coordinates_vanish_with_the_moments(self) -> None:
        block = self.spin_block(self.evaluate(np.zeros_like(self.spin)))
        # Every family except the trailing magnetic coordination reads the
        # spin value and is therefore exactly zero without moments.
        channels = self.descriptor.spin_channels
        np.testing.assert_array_equal(
            block[:, :-channels],
            np.zeros_like(block[:, :-channels]),
        )
        assert np.any(block[:, -channels:] != 0.0)

    def test_a_missing_moment_is_rejected(self) -> None:
        # A vanishing moment and an absent one are different states: the
        # first is a demagnetized configuration, the second is a missing
        # input whose magnetic force would be silently zero.
        with pytest.raises(ValueError, match="requires a per-node magnetic"):
            self.evaluate(None)

    def moment_divisor(self, spin: np.ndarray) -> np.ndarray:
        """Return the neighborhood divisor the spin families are scaled by."""
        masked = self.descriptor.spin.conditioned_spin(spin, self.atype)
        return self.descriptor.aggregate_moments(
            *self.descriptor.build_edge_features(
                self.graph,
                self.atype,
                self.descriptor.pair_film.pair_latent(
                    self.descriptor.type_embedding.call()
                ),
                None,
                masked,
            ),
            self.descriptor.spin.onsite_payload(masked, self.atype),
            self.graph.edge_index[1],
            self.atype.shape[0],
        )[1][:, 1]

    def test_two_body_terms_are_exactly_representable(self) -> None:
        # The emitted invariants are not merely correlated with the physical
        # sums: each is that sum times a known constant.
        spin = self.descriptor.spin
        reference = spin_reference_terms(
            self.descriptor,
            self.graph,
            self.atype,
            self.spin,
        )
        block = self.spin_block(self.evaluate(self.spin))
        divisor = self.moment_divisor(self.spin)
        vector_weight = spin.adam_spin_vector_weight[self.atype]
        quadrupole_weight = spin.adam_spin_quadrupole_weight[self.atype]

        vector_gram = block[:, : spin.vector_gram_index.shape[0]]
        selector = vector_gram_selectors(self.descriptor)
        np.testing.assert_allclose(
            vector_gram[:, selector["heisenberg"]],
            math.sqrt(2.0)
            * (vector_weight / divisor)[:, None]
            * reference["heisenberg"],
            atol=1e-12,
        )

        offset = spin.vector_gram_index.shape[0]
        quadrupole_gram = block[
            :, offset : offset + spin.quadrupole_gram_index.shape[0]
        ]
        # The on-site self-term is not emitted, so the on-site x neighbour
        # entry leads the quadrupole block.
        np.testing.assert_allclose(
            quadrupole_gram[:, 0:1],
            math.sqrt(2.0)
            * (quadrupole_weight / divisor)[:, None]
            * reference["biquadratic"],
            atol=1e-12,
        )

        offset += spin.quadrupole_gram_index.shape[0]
        degree_two = self.descriptor.degree_channels[2]
        cross = block[:, offset : offset + spin.quadrupole_width * degree_two].reshape(
            -1, spin.quadrupole_width, degree_two
        )
        np.testing.assert_allclose(
            cross[:, 0, :],
            (quadrupole_weight / divisor)[:, None] * reference["anisotropy"],
            atol=1e-12,
        )

    def test_two_ion_anisotropy_is_exactly_representable(self) -> None:
        # The pseudo-dipolar sum sum_j K(r_ij) (s_i.u_ij)(s_j.u_ij) is the
        # interaction the bond-projected family exists for. It is emitted as
        # an exact single sum, one entry per spin channel, so a linear
        # readout spans an arbitrary K(r) inside the channel amplitudes.
        spin = self.descriptor.spin
        reference = spin_reference_terms(
            self.descriptor,
            self.graph,
            self.atype,
            self.spin,
        )
        block = self.spin_block(self.evaluate(self.spin))
        divisor = self.moment_divisor(self.spin)
        vector_weight = spin.adam_spin_vector_weight[self.atype]

        vector_gram = block[:, : spin.vector_gram_index.shape[0]]
        selector = vector_gram_selectors(self.descriptor)
        np.testing.assert_allclose(
            vector_gram[:, selector["two_ion_anisotropy"]],
            math.sqrt(2.0)
            * (vector_weight / divisor)[:, None]
            * reference["two_ion_anisotropy"],
            atol=1e-12,
        )
        # The two families are genuinely different observables and not one
        # rescaled copy of the other.
        assert not np.allclose(
            reference["two_ion_anisotropy"],
            reference["heisenberg"],
        )

    def test_output_width_matches_the_spin_layout(self) -> None:
        spin = self.descriptor.spin
        channels = self.descriptor.spin_channels
        # The degree-one block carries the on-site moment plus the isotropic
        # and bond-projected neighbour channels. The quadrupole Gram omits its
        # on-site self-term, which the identity |B_2(s)|^2 = |s|^4 makes a
        # function of the vector self-term.
        width = 1 + 2 * channels
        assert spin.vector_width == width
        expected = (
            width * (width + 1) // 2
            + 2
            + 2 * self.descriptor.degree_channels[2]
            + 2 * channels
        )
        assert spin.get_dim_out() == expected
        assert self.evaluate(self.spin).shape[1] == self.descriptor.get_dim_out()
        assert (
            self.descriptor.get_dim_out()
            == make_spin_descriptor(use_spin=None).get_dim_out() + expected
        )

    def test_serialization_roundtrip_preserves_spin(self) -> None:
        self.descriptor.spin.set_spin_reference(np.array([1.7, 1.0, 1.0]))
        restored = DescrptDPA4C.deserialize(self.descriptor.serialize())
        assert restored.use_spin == [True, False]
        assert restored.supports_native_spin()
        np.testing.assert_array_equal(
            restored.spin.spin_reference,
            self.descriptor.spin.spin_reference,
        )
        np.testing.assert_allclose(
            restored.call_graph(self.graph, self.atype, spin=self.spin)[0],
            self.evaluate(self.spin),
            atol=1e-14,
        )

    def test_sharing_rejects_a_different_spin_configuration(self) -> None:
        with pytest.raises(ValueError, match="identical structural"):
            self.descriptor.share_params(make_spin_descriptor(use_spin=None), 0)


#: Vertices of a regular tetrahedron. Every component has the same magnitude,
#: so the four squared norms agree bitwise and the four neighbours below share
#: one radial amplitude exactly rather than to rounding.
TETRAHEDRON = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ],
    dtype=np.float64,
)


def test_permuting_equidistant_moments_moves_the_descriptor() -> None:
    """Relabelling equidistant neighbours must reach the output.

    Four identical neighbours sit at one distance from the centre, so every
    radial spin family sees the same amplitude on every bond and is blind to
    which moment sits on which bond. The physical pseudo-dipolar sum
    ``sum_j (s_i.u_ij)(s_j.u_ij)`` is not blind to it, and the bond-projected
    family is what carries that dependence into the readout.
    """
    descriptor = make_spin_descriptor(ntypes=1, use_spin=[True])
    direction = TETRAHEDRON / np.linalg.norm(TETRAHEDRON, axis=-1, keepdims=True)
    coord = np.concatenate([np.zeros((1, 3)), 1.5 * direction])[None]
    atype = np.zeros((1, 5), dtype=np.int64)
    graph = neighbor_graph.build_neighbor_graph(
        coord,
        atype,
        None,
        descriptor.get_rcut(),
    )
    flat_atype = atype.reshape(-1)

    spin = np.random.default_rng(3).normal(size=(5, 3))
    relabelled = spin.copy()
    relabelled[1:] = spin[[2, 3, 4, 1]]

    def pseudo_dipolar(moments: np.ndarray) -> float:
        return float(
            np.sum(
                (moments[0] @ direction.T)
                * np.einsum("jd,jd->j", moments[1:], direction)
            )
        )

    assert abs(pseudo_dipolar(spin) - pseudo_dipolar(relabelled)) > 1.0e-2

    reference = descriptor.call_graph(graph, flat_atype, spin=spin)[0]
    permuted = descriptor.call_graph(graph, flat_atype, spin=relabelled)[0]

    # The geometry is untouched, so the whole geometric block is unchanged.
    geometry = descriptor.readout.get_dim_out()
    np.testing.assert_allclose(
        permuted[:, :geometry],
        reference[:, :geometry],
        atol=1e-13,
    )

    # Inside the spin block only the entries that touch the bond-projected
    # channels may move: every other family reads the moments through a
    # permutation-symmetric sum over the equidistant shell.
    spin_width = descriptor.spin.get_dim_out()
    before = reference[0, geometry : geometry + spin_width]
    after = permuted[0, geometry : geometry + spin_width]
    bond = np.zeros(spin_width, dtype=bool)
    gram_width = descriptor.spin.vector_gram_index.shape[0]
    bond[:gram_width] = vector_gram_selectors(descriptor)["bond"]
    np.testing.assert_allclose(after[~bond], before[~bond], atol=1e-13)
    assert np.abs(after[bond] - before[bond]).max() > 1.0e-2


def test_periodic_cell_spin_matches_its_supercell() -> None:
    """A periodic magnetic cell agrees with its own doubled cell.

    The graph builder folds every periodic image onto its local owner through
    ``src = mapping[neighbor]``, so a cell narrower than the cutoff produces
    edges whose source is the centre itself and edges that reach one owner
    several times. Both are exercised here and by nothing else in this file.
    """
    descriptor = make_spin_descriptor()
    cell = 2.9
    coord = np.array([[[0.0, 0.0, 0.0], [1.45, 1.45, 0.2]]])
    atype = np.array([[0, 1]], dtype=np.int64)
    box = np.array([[cell, 0.0, 0.0, 0.0, cell, 0.0, 0.0, 0.0, cell]])
    spin = np.random.default_rng(13).normal(size=(2, 3))

    graph = neighbor_graph.build_neighbor_graph(
        coord,
        atype,
        box,
        descriptor.get_rcut(),
    )
    source, destination = graph.edge_index[0], graph.edge_index[1]
    assert np.any((source == destination) & graph.edge_mask)

    super_box = box.copy()
    super_box[0, 0] = 2.0 * cell
    super_graph = neighbor_graph.build_neighbor_graph(
        np.concatenate([coord, coord + np.array([cell, 0.0, 0.0])], axis=1),
        np.concatenate([atype, atype], axis=1),
        super_box,
        descriptor.get_rcut(),
    )

    reference = descriptor.call_graph(graph, atype.reshape(-1), spin=spin)[0]
    doubled = descriptor.call_graph(
        super_graph,
        np.concatenate([atype, atype], axis=1).reshape(-1),
        spin=np.concatenate([spin, spin], axis=0),
    )[0]
    np.testing.assert_allclose(
        doubled,
        np.concatenate([reference, reference], axis=0),
        atol=1e-11,
    )


@pytest.mark.parametrize("channels", [8, 32, 128])
def test_ordered_spin_tables_start_at_a_usable_scale(channels: int) -> None:
    """Both spin tables must start near the scale of the geometric ones.

    The geometric heads are structurally anchored, so they start at a
    root-mean-square of one and one quarter respectively. The spin heads may
    not be anchored on a constant, because an exchange amplitude is signed,
    and the descriptor calibration freezes a preconditioner at whatever scale
    it measures. A spin table emerging from the bias-free trunk alone would
    fix that preconditioner orders of magnitude below the block it belongs to.
    """
    descriptor = make_spin_descriptor(
        channels=channels,
        ntypes=4,
        n_radial=12,
        use_spin=[True, True, False, False],
    )
    scale, shift, _mixing, spin_scale, spin_shift = descriptor.pair_film.call(
        descriptor.type_embedding.call()
    )
    assert 0.9 <= float(np.sqrt(np.mean(np.square(scale)))) <= 1.1
    assert float(np.sqrt(np.mean(np.square(shift)))) > 0.1
    for table in (spin_scale, spin_shift):
        assert 0.3 <= float(np.sqrt(np.mean(np.square(table)))) <= 0.7
        # The anchor fixes the magnitude of every channel, so the scale holds
        # entry by entry rather than only in the aggregate.
        assert float(np.abs(table).min()) >= 0.3
        assert float(np.abs(table).max()) <= 0.7


def test_ordered_spin_tables_do_not_anchor_their_sign() -> None:
    """The exchange amplitude of an ordered pair may take either sign.

    The geometric scale is anchored on the constant one and is therefore
    strictly positive by construction. The spin tables carry no such bias: a
    ferromagnetic and an antiferromagnetic pair are equally reachable from the
    initialization, so both signs occur across seeds.
    """
    signs: set[float] = set()
    for seed in range(8):
        descriptor = make_spin_descriptor(seed=seed)
        _scale, _shift, _mixing, spin_scale, spin_shift = descriptor.pair_film.call(
            descriptor.type_embedding.call()
        )
        signs.update(np.sign(spin_scale).reshape(-1).tolist())
        signs.update(np.sign(spin_shift).reshape(-1).tolist())
    assert signs == {-1.0, 1.0}


#: Per-type moment scales of the calibration corpus. The second magnetic type
#: is rare and carries a much larger moment than the first.
CORPUS_MOMENT_SCALE = (1.0, 4.0, 0.0)


def magnetic_corpus(seed: int = 47) -> tuple[list[dict], np.ndarray]:
    """Build a magnetic calibration corpus and its per-type moment scales.

    Returns
    -------
    corpus
        One sampled system carrying ``coord``, ``atype``, ``box`` and
        ``spin``, in the packing ``compute_input_stats`` consumes.
    reference
        Independently computed per-type root-mean-square moment with shape
        ``(ntypes + 1,)``.
    """
    rng = np.random.default_rng(seed)
    nframes, natoms, cell = 8, 12, 9.0
    # One rare magnetic atom of the second type per frame; the rest alternate
    # between the abundant magnetic type and the non-magnetic one.
    atype = np.tile(np.array([0, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 1]), (nframes, 1))
    coord = rng.uniform(0.0, cell, size=(nframes, natoms, 3))
    box = np.tile(np.diag([cell, cell, cell]).reshape(1, 9), (nframes, 1))
    scale = np.take(np.asarray(CORPUS_MOMENT_SCALE), atype)[..., None]
    spin = rng.normal(size=(nframes, natoms, 3)) * scale

    reference = np.ones(len(CORPUS_MOMENT_SCALE) + 1, dtype=np.float64)
    magnitude = np.sum(np.square(spin), axis=-1)
    for kind in range(len(CORPUS_MOMENT_SCALE)):
        selected = magnitude[atype == kind]
        if np.any(selected > 0.0):
            reference[kind] = np.sqrt(np.mean(selected))
    corpus = [{"coord": coord, "atype": atype, "box": box, "spin": spin}]
    return corpus, reference


def calibrated_descriptor(corpus: list[dict]) -> DescrptDPA4C:
    """Return a three-type magnetic descriptor calibrated on ``corpus``."""
    descriptor = make_spin_descriptor(ntypes=3, use_spin=[True, True, False])
    descriptor.compute_input_stats(corpus)
    return descriptor


def test_calibration_measures_the_per_type_reference_moment() -> None:
    corpus, reference = magnetic_corpus()
    descriptor = calibrated_descriptor(corpus)
    np.testing.assert_allclose(descriptor.spin.spin_reference, reference, rtol=1e-12)
    # The rare species keeps its own scale rather than the population one, so
    # the conditioned moments of the two magnetic types land on one scale.
    assert reference[1] / reference[0] > 3.0
    # A type observed only with a vanishing moment, and the padding row, keep
    # the unit reference that leaves the raw spin untouched.
    assert descriptor.spin.spin_reference[2] == 1.0
    assert descriptor.spin.spin_reference[3] == 1.0


def test_calibration_conditions_every_spin_coordinate() -> None:
    corpus, _reference = magnetic_corpus()
    descriptor = calibrated_descriptor(corpus)
    start = descriptor.readout.get_dim_out()
    stddev = descriptor.stddev[start : start + descriptor.spin.get_dim_out()]
    # Every spin coordinate activates on a magnetic corpus, so none of them
    # falls back to the identity preconditioner, and none is driven to the
    # extreme gain an unanchored spin table used to produce.
    assert np.all(stddev > 0.0)
    assert not np.any(stddev == 1.0)
    assert float(stddev.max() / stddev.min()) < 1.0e4


def test_calibration_accepts_either_moment_key() -> None:
    corpus, _reference = magnetic_corpus()
    renamed, _reference = magnetic_corpus()
    renamed[0]["model_spin"] = renamed[0].pop("spin")
    under_spin = calibrated_descriptor(corpus)
    under_model_spin = calibrated_descriptor(renamed)
    np.testing.assert_array_equal(
        under_model_spin.spin.spin_reference,
        under_spin.spin.spin_reference,
    )
    np.testing.assert_array_equal(under_model_spin.stddev, under_spin.stddev)


def test_calibration_rejects_a_corpus_without_moments() -> None:
    # A key mismatch would otherwise leave a unit reference magnitude and an
    # identity preconditioner on most spin coordinates, with no error.
    corpus, _reference = magnetic_corpus()
    corpus[0].pop("spin")
    with pytest.raises(ValueError, match="requires a per-node magnetic"):
        calibrated_descriptor(corpus)


#: Frame conditions exercised by the charge-state tests, as
#: ``[charge, multiplicity]`` pairs.
NEUTRAL_SINGLET = np.array([[0.0, 1.0]])
CATION_TRIPLET = np.array([[2.0, 3.0]])


def make_charge_descriptor(
    *,
    seed: int = 17,
    activate: bool = True,
    **kwargs: Any,
) -> DescrptDPA4C:
    """Return a charge-conditioned descriptor.

    Parameters
    ----------
    seed
        Parameter-initialization seed.
    activate
        Whether to replace the zero-initialized condition output head with
        deterministic weights. Every property that depends on the condition
        actually reaching the descriptor needs an active head; leaving it at
        its initialization is itself the subject of one test.
    **kwargs
        Overrides forwarded to the descriptor constructor.

    Returns
    -------
    DescrptDPA4C
        Charge-conditioned descriptor.
    """
    kwargs.setdefault("default_chg_spin", [0.0, 1.0])
    descriptor = DescrptDPA4C(
        rcut=3.0,
        ntypes=2,
        channels=8,
        lmax=2,
        n_radial=4,
        precision="float64",
        seed=seed,
        add_chg_spin_ebd=True,
        **kwargs,
    )
    if activate:
        head = descriptor.charge_spin_embedding.network.layers[-1]
        head.w = np.random.default_rng(seed).normal(0.0, 0.5, size=head.w.shape)
    return descriptor


def evaluate_conditioned(
    descriptor: DescrptDPA4C,
    charge_spin: np.ndarray | None,
    coord: np.ndarray = COORD,
    atype: np.ndarray = ATYPE,
) -> np.ndarray:
    """Evaluate a charge-conditioned descriptor on the graph interface."""
    graph, atype_local = build_graph(descriptor, coord, atype)
    return descriptor.call_graph(graph, atype_local, charge_spin=charge_spin)[0]


def charge_route_heads(descriptor: DescrptDPA4C) -> tuple[np.ndarray, np.ndarray]:
    """Split the condition output head into its two routes.

    Returns
    -------
    type_route
        Head restricted to the centre type-embedding columns.
    pair_route
        Head restricted to the ordered pair encoder columns.
    """
    weight = descriptor.charge_spin_embedding.network.layers[-1].w
    type_route, pair_route = weight.copy(), weight.copy()
    type_route[:, descriptor.channels :] = 0.0
    pair_route[:, : descriptor.channels] = 0.0
    return type_route, pair_route


def test_an_unconditioned_descriptor_declares_no_frame_condition() -> None:
    descriptor = DescrptDPA4C(rcut=3.0, ntypes=2, channels=8, lmax=2, n_radial=4)
    assert descriptor.charge_spin_embedding is None
    assert not descriptor.supports_charge_spin()
    assert descriptor.get_dim_chg_spin() == 0
    assert not descriptor.has_default_chg_spin()


def test_an_untrained_condition_head_reproduces_the_plain_descriptor() -> None:
    """The condition output projection starts at zero.

    An untrained descriptor is therefore independent of the charge state for
    every value of it, so the fixed output calibration measured once before
    training carries no random condition offset.
    """
    plain = DescrptDPA4C(
        rcut=3.0,
        ntypes=2,
        channels=8,
        lmax=2,
        n_radial=4,
        precision="float64",
        seed=17,
    )
    conditioned = make_charge_descriptor(activate=False)
    reference = evaluate(plain).reshape(-1, plain.get_dim_out())
    for condition in (NEUTRAL_SINGLET, CATION_TRIPLET):
        np.testing.assert_array_equal(
            evaluate_conditioned(conditioned, condition),
            reference,
        )


def test_each_condition_route_reaches_the_descriptor() -> None:
    """Both injection points must be live.

    The centre type route alone would be indistinguishable from handing the
    condition to the fitting network as a frame parameter. Only the ordered
    pair route changes how a given geometry maps to the degree-wise moments,
    so the two are asserted separately rather than through their sum.
    """
    descriptor = make_charge_descriptor()
    head = descriptor.charge_spin_embedding.network.layers[-1]
    for route in charge_route_heads(descriptor):
        head.w = route
        assert not np.allclose(
            evaluate_conditioned(descriptor, NEUTRAL_SINGLET),
            evaluate_conditioned(descriptor, CATION_TRIPLET),
        )


def test_frames_carry_independent_conditions() -> None:
    """A batched evaluation must agree with per-frame evaluations.

    Each frame occupies one contiguous block of the flat node axis and an
    edge inherits the frame of the centre it reduces onto, so a batch of
    mixed charge states is only correct if that map is exact.
    """
    descriptor = make_charge_descriptor()
    shifted = COORD + 0.05
    batched = evaluate_conditioned(
        descriptor,
        np.concatenate([NEUTRAL_SINGLET, CATION_TRIPLET], axis=0),
        np.concatenate([COORD, shifted], axis=0),
        np.concatenate([ATYPE, ATYPE], axis=0),
    )
    np.testing.assert_allclose(
        batched,
        np.concatenate(
            [
                evaluate_conditioned(descriptor, NEUTRAL_SINGLET, COORD),
                evaluate_conditioned(descriptor, CATION_TRIPLET, shifted),
            ],
            axis=0,
        ),
        atol=1e-12,
    )


def test_a_frame_condition_preserves_rotation_invariance() -> None:
    # The condition is a pair of scalars, so it may not disturb the O(3)
    # invariance the readout establishes.
    descriptor = make_charge_descriptor()
    angle = 0.7
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(
        evaluate_conditioned(descriptor, CATION_TRIPLET, COORD @ rotation.T),
        evaluate_conditioned(descriptor, CATION_TRIPLET),
        atol=1e-12,
    )


def test_a_missing_condition_falls_back_to_the_configured_default() -> None:
    descriptor = make_charge_descriptor()
    np.testing.assert_array_equal(
        evaluate_conditioned(descriptor, None),
        evaluate_conditioned(descriptor, NEUTRAL_SINGLET),
    )


def test_a_missing_condition_without_a_default_is_an_error() -> None:
    descriptor = make_charge_descriptor(default_chg_spin=None)
    with pytest.raises(ValueError, match="requires a frame `charge_spin`"):
        evaluate_conditioned(descriptor, None)


def test_the_dense_interface_conditions_on_the_frame_state() -> None:
    # The dense adapter is the reference path of the common descriptor ABI;
    # dropping the condition there would be silent.
    descriptor = make_charge_descriptor()
    coord_ext, atype_ext, mapping, nlist = dense_inputs(descriptor)
    dense = descriptor(
        coord_ext,
        atype_ext,
        nlist,
        mapping=mapping,
        charge_spin=CATION_TRIPLET,
    )[0]
    np.testing.assert_allclose(
        dense.reshape(-1, descriptor.get_dim_out()),
        evaluate_conditioned(descriptor, CATION_TRIPLET),
        atol=1e-12,
    )


def test_serialization_preserves_the_conditioned_descriptor() -> None:
    descriptor = make_charge_descriptor()
    restored = DescrptDPA4C.deserialize(descriptor.serialize())
    assert restored.get_default_chg_spin() == descriptor.get_default_chg_spin()
    np.testing.assert_allclose(
        evaluate_conditioned(restored, CATION_TRIPLET),
        evaluate_conditioned(descriptor, CATION_TRIPLET),
        atol=1e-14,
    )


def test_the_frozen_pair_cache_reproduces_the_per_edge_route() -> None:
    """Compression folds the condition into the ordered pair cache.

    Training evaluates the conditioning heads on the edge axis, because the
    product of the frame and ordered-pair axes exceeds the edge count for the
    molecular systems a charge state describes. Compression evaluates the
    same heads once over the finite pair table. The compressed artifact is
    only valid because the two agree exactly.
    """
    descriptor = make_charge_descriptor()
    type_embedding = descriptor.type_embedding.call()
    _type_shift, pair_hidden_bias = descriptor.charge_spin_embedding.call(
        CATION_TRIPLET
    )
    folded = descriptor.pair_film.call(type_embedding, hidden_bias=pair_hidden_bias[0])
    pre_activation, base_shift = descriptor.pair_film.pair_latent(type_embedding)
    pair_index = np.arange(pre_activation.shape[0])
    per_edge = descriptor.build_pair_conditioning(
        (pre_activation, base_shift),
        pair_index,
        np.broadcast_to(
            pair_hidden_bias, (pair_index.size, pair_hidden_bias.shape[-1])
        ),
    )
    for cache, edge in zip(folded, per_edge, strict=True):
        if cache is None:
            assert edge is None
        else:
            np.testing.assert_allclose(edge, cache, atol=1e-14)


def test_the_padding_type_keeps_its_zero_centre_features() -> None:
    """The condition shifts only the real rows of the centre type table.

    Compressed inference conditions a frozen table whose padding row stays
    zero, so shifting that row on the portable path would break the parity
    between the two.
    """
    descriptor = make_charge_descriptor()
    type_embedding = descriptor.type_embedding.call()
    atype = np.array([0, 1, descriptor.ntypes], dtype=np.int64)
    type_shift, _pair_hidden_bias = descriptor.charge_spin_embedding.call(
        CATION_TRIPLET
    )
    features = descriptor.build_center_type_features(
        type_embedding,
        atype,
        np.broadcast_to(type_shift, (atype.size, descriptor.channels)),
    )
    np.testing.assert_array_equal(features[2], np.zeros(descriptor.channels))
    assert not np.allclose(features[0], type_embedding[0])


def charge_corpus(charge_spin: np.ndarray | None) -> list[dict]:
    """Build a two-type calibration corpus carrying one frame condition."""
    rng = np.random.default_rng(3)
    nframes, natoms, cell = 6, 8, 9.0
    system = {
        "coord": rng.uniform(0.0, cell, size=(nframes, natoms, 3)),
        "atype": np.tile(np.array([0, 1, 0, 1, 0, 1, 0, 1]), (nframes, 1)),
        "box": np.tile(np.diag([cell, cell, cell]).reshape(1, 9), (nframes, 1)),
    }
    if charge_spin is not None:
        system["charge_spin"] = charge_spin
    return [system]


@pytest.mark.parametrize(
    "charge_spin",
    [
        np.tile(np.array([[-1.0, 2.0]]), (6, 1)),
        np.array([[-1.0, 2.0]]),
        np.array([-1.0, 2.0]),
    ],
    ids=["per-frame", "single-pair-2d", "single-pair-1d"],
)
def test_the_calibration_accepts_every_shape_evaluation_accepts(
    charge_spin: np.ndarray,
) -> None:
    """A system may state one condition for all of its frames.

    Evaluation broadcasts a single pair over the frame axis, so a calibration
    that required one row per frame would reject a corpus the trained model
    then runs on without complaint.
    """
    descriptor = make_charge_descriptor()
    descriptor.compute_input_stats(charge_corpus(charge_spin))
    frames = descriptor._calibration_frames(charge_corpus(charge_spin)[0])
    for frame in frames:
        np.testing.assert_array_equal(frame["charge_spin"], np.array([[-1.0, 2.0]]))


def test_the_calibration_samples_the_corpus_charge_states() -> None:
    """The preconditioner must be measured over the sampled charge states.

    It is frozen once and has to hold for every state the corpus contains, so
    a calibration that read one state, or none, would fix it on the wrong
    scale.
    """
    descriptor = make_charge_descriptor()
    rng = np.random.default_rng(3)
    nframes, natoms, cell = 6, 8, 9.0
    corpus = [
        {
            "coord": rng.uniform(0.0, cell, size=(nframes, natoms, 3)),
            "atype": np.tile(np.array([0, 1, 0, 1, 0, 1, 0, 1]), (nframes, 1)),
            "box": np.tile(np.diag([cell, cell, cell]).reshape(1, 9), (nframes, 1)),
            "charge_spin": np.tile(np.array([[-1.0, 2.0]]), (nframes, 1)),
        }
    ]
    frames = descriptor._calibration_frames(corpus[0])
    np.testing.assert_array_equal(frames[0]["charge_spin"], np.array([[-1.0, 2.0]]))

    descriptor.compute_input_stats(corpus)
    default_state = make_charge_descriptor()
    default_state.compute_input_stats(
        [{key: value for key, value in corpus[0].items() if key != "charge_spin"}]
    )
    assert not np.allclose(descriptor.stddev, default_state.stddev)


def test_dense_lower_escape_hatch_is_refused() -> None:
    """DPA4C overrides ``uses_graph_lower`` and owes the paired hatch.

    The base contract requires the postcondition ``uses_graph_lower() is
    False`` after the hatch. DPA4C has no dense form to reach: it carries
    every neighbour within the cutoff and reports an unreachable ``get_sel``,
    so it refuses instead of inheriting a no-op that reports success while
    leaving the graph lower in place.
    """
    descriptor = DescrptDPA4C(
        rcut=4.0, ntypes=2, channels=8, lmax=2, n_radial=4, seed=1
    )
    assert descriptor.uses_graph_lower()
    with pytest.raises(NotImplementedError, match="no dense lower"):
        descriptor.disable_graph_lower()
    assert descriptor.uses_graph_lower()


def test_virtual_atom_spin_scheme_is_refused() -> None:
    """DPA4C is a DPA4-family descriptor, so deepspin must be rejected.

    ``SpinModel`` pulls the dense escape hatch its descriptors do not have,
    and the family's constraint lives in the model factory, so a new member
    has to be registered there rather than rely on a downstream failure.
    """
    from deepmd.dpmodel.model.model import (
        get_spin_model,
    )

    with pytest.raises(NotImplementedError, match="deepspin"):
        get_spin_model(
            {
                "type_map": ["Ni", "O"],
                "spin": {"use_spin": [True, False]},
                "descriptor": {
                    "type": "dpa4c",
                    "rcut": 4.0,
                    "channels": 8,
                    "lmax": 2,
                    "n_radial": 4,
                    "seed": 1,
                },
                "fitting_net": {"neuron": [8, 8], "seed": 1},
            }
        )
