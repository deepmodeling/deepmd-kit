# SPDX-License-Identifier: LGPL-3.0-or-later
"""Uni-Mol pretraining heads on a DPA backbone."""

import unittest

import numpy as np

from deepmd.dpmodel.atomic_model import (
    DPUniMolDPAAtomicModel,
)
from deepmd.dpmodel.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.dpmodel.fitting.unimol_dpa_heads import (
    EquivariantCoordHead,
    PairDistanceHead,
    l1_to_cartesian,
)
from deepmd.dpmodel.fitting.unimol_dpa_pretrain import (
    UniMolDPAPretrainFitting,
)
from deepmd.dpmodel.loss.unimol import (
    UniMolLoss,
)

SMALL = {"nloc": 6, "nnei": 5, "ntypes": 2}


def _rotation(rng):
    """A random proper rotation."""
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


class TestEquivariantCoordHead(unittest.TestCase):
    """The head that replaces Uni-Mol's pair-channel coordinate update."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)
        nloc, nnei = SMALL["nloc"], SMALL["nnei"]
        # float64 throughout: the property under test is exact, and the default
        # single precision would only show it to about 1e-6.
        self.desc = DescrptDPA4(
            rcut=6.0,
            rcut_smth=5.0,
            sel=nnei,
            ntypes=SMALL["ntypes"],
            seed=0,
            precision="float64",
        )
        self.head = EquivariantCoordHead(
            lmax=self.desc.node_readout_lmax,
            channels=self.desc.channels,
            precision="float64",
            seed=1,
        )
        self.coord = self.rng.normal(size=(1, nloc, 3)) * 1.5
        self.atype = self.rng.integers(0, SMALL["ntypes"], size=(1, nloc))
        self.nlist = np.stack(
            [
                np.stack(
                    [
                        np.array([j for j in range(nloc) if j != i][:nnei])
                        for i in range(nloc)
                    ]
                )
            ]
        )
        self.mapping = np.tile(np.arange(nloc), (1, 1))

    def _predict(self, coord):
        _, latent = self.desc.call_with_latent(
            coord, self.atype, self.nlist, mapping=self.mapping
        )
        return self.head(latent)

    def test_one_vector_per_atom(self) -> None:
        out = self._predict(self.coord)
        self.assertEqual(out.shape, (SMALL["nloc"], 3))
        self.assertTrue(bool(np.all(np.isfinite(out))))

    def test_rotating_the_molecule_rotates_the_prediction(self) -> None:
        """The point of reading l=1 rather than three arbitrary numbers.

        A coordinate update has to be equivariant; if it were not, the model
        could not denoise a rotated copy of a structure it had already learned.
        """
        base = self._predict(self.coord)
        for trial in range(5):
            rot = _rotation(self.rng)
            with self.subTest(rotation=trial):
                turned = self._predict(self.coord @ rot.T)
                np.testing.assert_allclose(turned, base @ rot.T, rtol=1e-9, atol=1e-9)

    def test_translating_the_molecule_changes_nothing(self) -> None:
        base = self._predict(self.coord)
        shifted = self._predict(self.coord + self.rng.normal(size=3))
        np.testing.assert_allclose(shifted, base, rtol=1e-9, atol=1e-9)

    def test_reading_the_rows_as_xyz_would_not_be_equivariant(self) -> None:
        """Guards the decode itself, not just the head around it.

        SeZM packs l=1 in its own basis, so taking the rows as (x, y, z) gives
        something that does not rotate. This pins the mapping that does.
        """
        rows = self.rng.normal(size=(4, 3))
        np.testing.assert_allclose(
            l1_to_cartesian(rows),
            np.stack([-rows[:, 2], rows[:, 0], rows[:, 1]], axis=-1),
        )

    def test_a_backbone_without_l1_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least degree 1"):
            EquivariantCoordHead(lmax=0, channels=8)

    def test_serialize_round_trip(self) -> None:
        revived = EquivariantCoordHead.deserialize(self.head.serialize())
        _, latent = self.desc.call_with_latent(
            self.coord, self.atype, self.nlist, mapping=self.mapping
        )
        np.testing.assert_allclose(revived(latent), self.head(latent))


if __name__ == "__main__":
    unittest.main()


class TestPairDistanceHead(unittest.TestCase):
    """The head that replaces Uni-Mol's pair-channel distance prediction."""

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.nf, self.nloc, self.nnei, self.dim = 2, 5, 3, 8
        self.node = rng.normal(size=(self.nf, self.nloc, self.dim))
        self.nlist = np.full((self.nf, self.nloc, self.nnei), -1, dtype=np.int64)
        for f in range(self.nf):
            for i in range(self.nloc):
                others = [j for j in range(self.nloc) if j != i][:2]
                self.nlist[f, i, : len(others)] = others

    def _head(self, coverage):
        return PairDistanceHead(
            dim_descrpt=self.dim, hidden=6, coverage=coverage, seed=0
        )

    def test_a_distance_is_symmetric(self) -> None:
        """Not approximately: the pair feature is symmetric in the endpoints."""
        for coverage in PairDistanceHead.COVERAGES:
            with self.subTest(coverage=coverage):
                dist, _ = self._head(coverage)(self.node, self.nlist)
                np.testing.assert_array_equal(dist, np.transpose(dist, (0, 2, 1)))

    def test_the_diagonal_is_never_covered(self) -> None:
        for coverage in PairDistanceHead.COVERAGES:
            with self.subTest(coverage=coverage):
                _, mask = self._head(coverage)(self.node, self.nlist)
                np.testing.assert_array_equal(
                    np.einsum("fii->fi", mask), np.zeros((self.nf, self.nloc))
                )

    def test_coverage_selects_pairs_and_nothing_else(self) -> None:
        """The option changes which pairs the objective sees, not the model.

        Uni-Mol's distance term covers every pair; a neighbour list covers a
        subset. Both heads predict the same numbers from the same weights, so
        the choice is purely one of coverage.
        """
        near, whole = self._head("neighbour"), self._head("all_pairs")
        d_near, m_near = near(self.node, self.nlist)
        d_whole, m_whole = whole(self.node, self.nlist)

        np.testing.assert_allclose(d_near, d_whole)
        self.assertTrue(bool(np.all(m_near <= m_whole)))
        # all_pairs is every off-diagonal pair; neighbour is strictly fewer here
        self.assertEqual(int(m_whole.sum()), self.nf * self.nloc * (self.nloc - 1))
        self.assertLess(int(m_near.sum()), int(m_whole.sum()))
        # and it is exactly the neighbour list, padding excluded
        expected = np.zeros_like(m_near)
        for f in range(self.nf):
            for i in range(self.nloc):
                for j in self.nlist[f, i]:
                    if j >= 0:
                        expected[f, i, j] = 1.0
        np.testing.assert_array_equal(m_near, expected)

    def test_padding_in_the_neighbour_list_covers_nothing(self) -> None:
        """A padded entry is negative and would wrap if used as an index."""
        empty = np.full_like(self.nlist, -1)
        _, mask = self._head("neighbour")(self.node, empty)
        np.testing.assert_array_equal(mask, np.zeros_like(mask))

    def test_an_unknown_coverage_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown coverage"):
            PairDistanceHead(dim_descrpt=self.dim, coverage="everything")

    def test_serialize_round_trip(self) -> None:
        head = self._head("neighbour")
        revived = PairDistanceHead.deserialize(head.serialize())
        a, ma = head(self.node, self.nlist)
        b, mb = revived(self.node, self.nlist)
        np.testing.assert_allclose(a, b)
        np.testing.assert_array_equal(ma, mb)


class TestUniMolDPAPretrainFitting(unittest.TestCase):
    """The three heads assembled on a real DPA4 backbone."""

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.nloc, nnei, ntypes = 6, 5, 3
        self.max_atoms = 10
        self.desc = DescrptDPA4(
            rcut=6.0,
            rcut_smth=5.0,
            sel=nnei,
            ntypes=ntypes,
            seed=0,
            precision="float64",
        )
        self.fitting = UniMolDPAPretrainFitting(
            ntypes=ntypes,
            dim_descrpt=self.desc.channels,
            node_readout_lmax=self.desc.node_readout_lmax,
            max_atoms=self.max_atoms,
            dist_hidden=16,
            precision="float64",
            seed=1,
        )
        self.coord = rng.normal(size=(1, self.nloc, 3)) * 1.5
        self.atype = rng.integers(0, ntypes, size=(1, self.nloc))
        self.nlist = np.stack(
            [
                np.stack(
                    [
                        np.array([j for j in range(self.nloc) if j != i][:nnei])
                        for i in range(self.nloc)
                    ]
                )
            ]
        )

    def _run(self):
        node, latent = self.desc.call_with_latent(
            self.coord,
            self.atype,
            self.nlist,
            mapping=np.tile(np.arange(self.nloc), (1, 1)),
        )
        return self.fitting.call_atoms(node, latent, self.nlist)

    def test_every_output_is_declared(self) -> None:
        """The output machinery indexes the definition for every key returned.

        A key the definition does not declare raises KeyError on the way out of
        the model, so returning one is not a harmless extra.
        """
        out = self._run()
        self.assertEqual(set(out), set(self.fitting.output_def().var_defs))

    def test_shapes(self) -> None:
        out = self._run()
        self.assertEqual(out["token_logits"].shape, (1, self.nloc, 31))
        self.assertEqual(out["coord_update"].shape, (1, self.nloc, 3))
        self.assertEqual(out["pair_dist"].shape, (1, self.nloc, self.max_atoms))
        self.assertEqual(out["pair_mask"].shape, (1, self.nloc, self.max_atoms))
        for name, value in out.items():
            with self.subTest(output=name):
                self.assertTrue(bool(np.all(np.isfinite(value))))

    def test_columns_past_the_frame_are_not_covered(self) -> None:
        """The declared width is fixed; a shorter frame pads the rest."""
        out = self._run()
        np.testing.assert_array_equal(
            out["pair_mask"][:, :, self.nloc :],
            np.zeros((1, self.nloc, self.max_atoms - self.nloc)),
        )

    def test_a_frame_wider_than_max_atoms_is_refused(self) -> None:
        narrow = UniMolDPAPretrainFitting(
            ntypes=3,
            dim_descrpt=self.desc.channels,
            node_readout_lmax=self.desc.node_readout_lmax,
            max_atoms=self.nloc - 1,
            dist_hidden=16,
            precision="float64",
            seed=1,
        )
        node, latent = self.desc.call_with_latent(
            self.coord,
            self.atype,
            self.nlist,
            mapping=np.tile(np.arange(self.nloc), (1, 1)),
        )
        with self.assertRaisesRegex(ValueError, "exceeds max_atoms"):
            narrow.call_atoms(node, latent, self.nlist)

    def test_the_five_tuple_path_is_refused(self) -> None:
        """These heads need the equivariant state and the neighbour list."""
        with self.assertRaisesRegex(NotImplementedError, "call_atoms"):
            self.fitting.call(np.zeros((1, 1, 1)), np.zeros((1, 1), dtype=np.int64))

    def test_serialize_round_trip(self) -> None:
        revived = UniMolDPAPretrainFitting.deserialize(self.fitting.serialize())
        node, latent = self.desc.call_with_latent(
            self.coord,
            self.atype,
            self.nlist,
            mapping=np.tile(np.arange(self.nloc), (1, 1)),
        )
        a = self.fitting.call_atoms(node, latent, self.nlist)
        b = revived.call_atoms(node, latent, self.nlist)
        for key in a:
            with self.subTest(output=key):
                np.testing.assert_allclose(a[key], b[key])


class TestUniMolDPAAtomicModel(unittest.TestCase):
    """The objective wired onto a DPA backbone, through the model machinery."""

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.nloc, nnei, ntypes = 6, 5, 3
        self.type_map = ["C", "N", "[MASK]"]
        self.desc = DescrptDPA4(
            rcut=6.0,
            rcut_smth=5.0,
            sel=nnei,
            ntypes=ntypes,
            seed=0,
            precision="float64",
        )
        self.fitting = UniMolDPAPretrainFitting(
            ntypes=ntypes,
            dim_descrpt=self.desc.channels,
            node_readout_lmax=self.desc.node_readout_lmax,
            max_atoms=self.nloc,
            dist_hidden=16,
            precision="float64",
            seed=1,
            type_map=self.type_map,
        )
        self.model = DPUniMolDPAAtomicModel(self.desc, self.fitting, self.type_map)
        self.coord = rng.normal(size=(1, self.nloc, 3)) * 1.5
        self.atype = rng.integers(0, ntypes, size=(1, self.nloc))
        self.nlist = np.stack(
            [
                np.stack(
                    [
                        np.array([j for j in range(self.nloc) if j != i][:nnei])
                        for i in range(self.nloc)
                    ]
                )
            ]
        )
        self.mapping = np.tile(np.arange(self.nloc), (1, 1))

    def _predict(self):
        return self.model.forward_common_atomic(
            self.coord, self.atype, self.nlist, mapping=self.mapping
        )

    def test_the_heads_reach_the_model_output(self) -> None:
        out = self._predict()
        for name in ("token_logits", "coord_update", "pair_dist", "pair_mask"):
            with self.subTest(output=name):
                self.assertIn(name, out)
        # the base class supplies the real-atom mask the objective needs
        self.assertIn("mask", out)

    def test_a_backbone_without_the_equivariant_seam_is_refused(self) -> None:
        """The coordinate head reads a state most descriptors do not expose."""
        from deepmd.dpmodel.descriptor.se_e2_a import (
            DescrptSeA,
        )

        plain = DescrptSeA(rcut=6.0, rcut_smth=5.0, sel=[10, 10, 10])
        with self.assertRaisesRegex(TypeError, "call_with_latent"):
            DPUniMolDPAAtomicModel(plain, self.fitting, self.type_map)

    def test_the_objective_scores_it(self) -> None:
        """The point of the whole branch: this output feeds Uni-Mol's loss.

        The backbone has no virtual tokens, and Uni-Mol's two norm regularisers
        constrain quantities belonging to its own transformer, so they carry no
        weight here.
        """
        rng = np.random.default_rng(1)
        pred = self._predict()
        token_target = np.zeros((1, self.nloc), dtype=np.int64)
        token_target[0, [1, 3]] = [5, 6]
        labels = {
            "unimol_token_target": token_target,
            "unimol_coord_target": rng.normal(size=(1, self.nloc, 3)),
        }
        loss = UniMolLoss(
            x_norm_loss=0.0, delta_pair_repr_norm_loss=0.0, virtual_tokens=False
        )
        total, terms = loss.call(1.0, 0, pred, labels)
        self.assertEqual(sorted(terms), ["coord_loss", "dist_loss", "token_loss"])
        for name, value in terms.items():
            with self.subTest(term=name):
                self.assertTrue(bool(np.isfinite(float(value))))
        self.assertTrue(bool(np.isfinite(float(total))))


class TestUniMolDPAExample(unittest.TestCase):
    """The shipped example must stay valid as the arguments evolve.

    Checked here rather than in the shared example test, because that one also
    requires the referenced dataset to exist in the repository, and this example
    points at data the user converts from upstream.
    """

    def test_example_configuration_is_valid(self) -> None:
        import json
        from pathlib import (
            Path,
        )

        from deepmd.utils.argcheck import (
            normalize,
        )

        path = (
            Path(__file__).parents[4]
            / "examples"
            / "unimol"
            / "dpa_pretrain"
            / "input.json"
        )
        self.assertTrue(path.is_file(), path)
        config = normalize(json.loads(path.read_text()), multi_task=False)

        fitting = config["model"]["fitting_net"]
        self.assertEqual(fitting["type"], "unimol_dpa_pretrain")
        self.assertEqual(fitting["dist_coverage"], "neighbour")
        self.assertIn("[MASK]", config["model"]["type_map"])
        # a DPA backbone wraps nothing around a molecule, and Uni-Mol's two norm
        # regularisers constrain its own transformer rather than this one
        self.assertFalse(config["loss"]["virtual_tokens"])
        self.assertEqual(config["loss"]["x_norm_loss"], 0.0)
        self.assertEqual(config["loss"]["delta_pair_repr_norm_loss"], 0.0)
