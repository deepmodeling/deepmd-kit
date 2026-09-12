# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity tests for the Uni-Mol v1 port.

Every expected value in ``unimol_v1_golden.npz`` was produced by running
upstream Uni-Mol (commit 90f52c4) unmodified on CPU, over molecules 0 to 3 of
the example data shipped with that repository, at seed 1 and epoch 1. The
generator is ``golden/dump_golden.py`` in the development task directory; it
needs the upstream sources on ``PYTHONPATH`` and is not run by the test suite.

Tolerances are not arbitrary. Upstream makes three fp32 choices that deepmd
does not have to make, and they set the floor:

* the Gaussian basis is evaluated in fp32,
* the distance matrix is precomputed in fp32 by its data pipeline,
* ``log_softmax`` and both norm regularisers are evaluated in fp32.

With those matched, agreement is at the fp32 rounding level, around 1e-7
relative. The encoder itself, fed upstream's own attention bias, agrees to fp64
rounding, which is what the strictest test below asserts.
"""

import os
import unittest

import numpy as np

from deepmd.dpmodel.descriptor.unimol import (
    UNIMOL_ELEMENTS,
    DescrptUniMol,
    unimol_vocabulary,
)
from deepmd.dpmodel.descriptor.unimol_nn import (
    DistanceHead,
    GaussianLayer,
    MaskLMHead,
    NonLinearHead,
    TransformerEncoderWithPair,
    coord_update,
)
from deepmd.dpmodel.fitting.unimol_pretrain import (
    UniMolPretrainFitting,
)
from deepmd.dpmodel.loss.unimol import (
    UniMolLoss,
)
from deepmd.dpmodel.utils.unimol_transform import (
    mask_points,
    unimol_frame_transform,
)

GOLDEN = os.path.join(os.path.dirname(__file__), "unimol_v1_golden.npz")
SEED, EPOCH, INDICES = 1, 1, [0, 1]
SMALL = {"layers": 2, "dim": 32, "ffn": 64, "heads": 4, "k": 128, "vocab": 31}


def _set_linear(layer, weights, prefix, weight_key=None, bias_key=None):
    layer.w = np.ascontiguousarray(weights[weight_key or prefix + ".weight"].T)
    key = bias_key or prefix + ".bias"
    if key in weights and layer.b is not None:
        layer.b = weights[key].copy()


def _set_layer_norm(layer, weights, prefix):
    layer.w = weights[prefix + ".weight"].copy()
    layer.b = weights[prefix + ".bias"].copy()


class UniMolGoldenMixin:
    @classmethod
    def setUpClass(cls) -> None:
        cls.golden = np.load(GOLDEN)
        cls.weights = {
            k[len("small_weights/") :]: v
            for k, v in cls.golden.items()
            if k.startswith("small_weights/")
        }
        cls.n_real = (cls.golden["input/src_tokens"] != 0).sum(axis=1) - 2

    def build_encoder(self):
        enc = TransformerEncoderWithPair(
            encoder_layers=SMALL["layers"],
            embed_dim=SMALL["dim"],
            ffn_embed_dim=SMALL["ffn"],
            attention_heads=SMALL["heads"],
            activation_function="gelu_erf",
        )
        w = self.weights
        _set_layer_norm(enc.emb_layer_norm, w, "encoder.emb_layer_norm")
        _set_layer_norm(enc.final_layer_norm, w, "encoder.final_layer_norm")
        _set_layer_norm(enc.final_head_layer_norm, w, "encoder.final_head_layer_norm")
        for i, layer in enumerate(enc.layers):
            p = f"encoder.layers.{i}."
            _set_linear(layer.self_attn.in_proj, w, p + "self_attn.in_proj")
            _set_linear(layer.self_attn.out_proj, w, p + "self_attn.out_proj")
            _set_layer_norm(layer.self_attn_layer_norm, w, p + "self_attn_layer_norm")
            _set_linear(layer.fc1, w, p + "fc1")
            _set_linear(layer.fc2, w, p + "fc2")
            _set_layer_norm(layer.final_layer_norm, w, p + "final_layer_norm")
        return enc

    def build_descriptor(self, **overrides):
        kwargs = {
            "type_map": [*UNIMOL_ELEMENTS, "[MASK]"],
            "encoder_layers": SMALL["layers"],
            "encoder_embed_dim": SMALL["dim"],
            "encoder_ffn_embed_dim": SMALL["ffn"],
            "encoder_attention_heads": SMALL["heads"],
            "max_atoms": int(self.n_real.max()) + 1,
            "activation_function": "gelu_erf",
            "virtual_token_position": "origin",
        }
        kwargs.update(overrides)
        desc = DescrptUniMol(**kwargs)
        w = self.weights
        desc.embed_tokens.w = w["embed_tokens.weight"].copy()
        for key in ("means", "stds", "mul", "bias"):
            getattr(desc.gbf, key).w = w[f"gbf.{key}.weight"].copy()
        _set_linear(desc.gbf_proj.linear1, w, "gbf_proj.linear1")
        _set_linear(desc.gbf_proj.linear2, w, "gbf_proj.linear2")
        enc = self.build_encoder()
        desc.encoder = enc
        return desc

    def deepmd_inputs(self):
        """Rebuild the deepmd-side inputs for the molecules in the golden batch."""
        vocab = unimol_vocabulary()
        type_map = [*UNIMOL_ELEMENTS, "[MASK]"]
        type_of = {sym: i for i, sym in enumerate(type_map)}
        src_tokens = self.golden["input/src_tokens"]
        src_coord = self.golden["input/src_coord"].astype(np.float64)
        nf, nloc = len(self.n_real), int(self.n_real.max())
        coord = np.zeros((nf, nloc, 3))
        atype = np.zeros((nf, nloc), dtype=np.int64)
        nlist = np.full((nf, nloc, nloc - 1), -1, dtype=np.int64)
        for f in range(nf):
            n = int(self.n_real[f])
            coord[f, :n] = src_coord[f, 1 : n + 1]
            atype[f, :n] = [type_of[vocab[t]] for t in src_tokens[f, 1 : n + 1]]
            for i in range(n):
                others = [j for j in range(n) if j != i]
                nlist[f, i, : len(others)] = others
        return coord, atype, nlist


class TestUniMolTransform(UniMolGoldenMixin, unittest.TestCase):
    """The data-side corruption must reproduce upstream's random stream."""

    def test_frame_transform_matches_upstream(self) -> None:
        vocab = {sym: i for i, sym in enumerate(unimol_vocabulary())}
        specials = [vocab[s] for s in ("[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]")]
        for row, index in enumerate(INDICES):
            atoms = self.golden[f"raw/{index}/atoms"]
            conformers = self.golden[f"raw/{index}/conformers"]
            got = unimol_frame_transform(
                atoms,
                conformers,
                vocab=vocab,
                num_types=len(vocab),
                special_indices=specials,
                pad_idx=vocab["[PAD]"],
                bos_idx=vocab["[CLS]"],
                eos_idx=vocab["[SEP]"],
                mask_idx=vocab["[MASK]"],
                unk_idx=vocab["[UNK]"],
                seed=SEED,
                epoch=EPOCH,
                index=index,
                max_atoms=256,
                mask_prob=0.15,
                leave_unmasked_prob=0.05,
                random_token_prob=0.05,
                noise_type="uniform",
                noise=1.0,
            )
            n = int(self.n_real[row]) + 2
            with self.subTest(molecule=index):
                # Tokens, targets and both coordinate arrays are bitwise equal,
                # which is what proves the random stream itself is reproduced.
                np.testing.assert_array_equal(
                    got["src_tokens"], self.golden["input/src_tokens"][row, :n]
                )
                np.testing.assert_array_equal(
                    got["tokens_target"], self.golden["target/tokens_target"][row, :n]
                )
                np.testing.assert_array_equal(
                    got["src_edge_type"],
                    self.golden["input/src_edge_type"][row, :n, :n],
                )
                np.testing.assert_array_equal(
                    got["src_coord"], self.golden["input/src_coord"][row, :n]
                )
                np.testing.assert_array_equal(
                    got["coord_target"], self.golden["target/coord_target"][row, :n]
                )
                # scipy's distance_matrix and a sqrt of summed squares differ in
                # the last fp32 place.
                np.testing.assert_allclose(
                    got["src_distance"],
                    self.golden["input/src_distance"][row, :n, :n],
                    atol=1e-5,
                )

    def test_masking_statistics(self) -> None:
        """About 15% of atoms are selected, and only replaced ones are moved.

        The ported corruption is run here rather than read off the fixture, so
        a change in the port can fail this test.
        """
        vocab = {sym: i for i, sym in enumerate(unimol_vocabulary())}
        specials = [vocab[s] for s in ("[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]")]
        rng = np.random.default_rng(0)
        selected_counts = []
        for index in range(24):
            size = 40
            tokens = rng.integers(4, 30, size=size)
            coords = rng.normal(size=(size, 3)).astype(np.float32)
            out = mask_points(
                tokens,
                coords,
                num_types=len(vocab),
                special_indices=specials,
                pad_idx=0,
                mask_idx=vocab["[MASK]"],
                seed=1,
                epoch=1,
                index=index,
            )
            picked = out["targets"] != 0
            selected_counts.append(int(picked.sum()))
            moved = np.abs(out["coordinates"] - coords).max(axis=-1) > 0
            # Only selected atoms move, and an atom left with its own element
            # is still predicted.
            self.assertTrue(bool(np.all(~moved | picked)))
            np.testing.assert_array_equal(
                out["targets"][picked], np.asarray(tokens)[picked]
            )
        mean_fraction = float(np.mean(selected_counts)) / 40
        self.assertGreater(mean_fraction, 0.10)
        self.assertLess(mean_fraction, 0.20)

        for row, index in enumerate(INDICES):
            n = int(self.n_real[row])
            targets = self.golden["target/tokens_target"][row, 1 : n + 1]
            selected = int((targets != 0).sum())
            self.assertGreater(selected, 0)
            self.assertLessEqual(selected, max(1, int(0.5 * n)))
            noisy = self.golden["input/src_coord"][row, 1 : n + 1]
            clean = self.golden["target/coord_target"][row, 1 : n + 1]
            moved = np.abs(noisy - clean).max(axis=-1) > 0
            # Every moved atom is a selected atom; the unchanged 5% are not moved.
            self.assertTrue(bool(np.all(~moved | (targets != 0))))
            self.assertLessEqual(int(moved.sum()), selected)


class TestUniMolEncoder(UniMolGoldenMixin, unittest.TestCase):
    """The backbone, isolated from upstream's fp32 choices."""

    def test_encoder_matches_upstream_in_fp64(self) -> None:
        enc = self.build_encoder()
        emb = self.golden["small_fp64/enc_in/emb"].astype(np.float64)
        bias = self.golden["small_fp64/enc_in/attn_bias"].astype(np.float64)
        pad = self.golden["small_fp64/enc_in/padding_mask"].astype(np.float64)
        x, pair, delta, x_norm, delta_norm = enc(emb, bias.copy(), pad)
        # Fed upstream's own bias, everything is fp64 end to end.
        np.testing.assert_allclose(
            x, self.golden["small_fp64/enc_out/x"], rtol=1e-11, atol=1e-11
        )
        finite = np.isfinite(self.golden["small_fp64/enc_out/pair_rep"])
        np.testing.assert_allclose(
            np.asarray(pair)[finite],
            self.golden["small_fp64/enc_out/pair_rep"][finite],
            rtol=1e-11,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            delta,
            self.golden["small_fp64/enc_out/delta_pair_rep"],
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            float(x_norm),
            float(self.golden["small_fp64/enc_out/x_norm"]),
            rtol=1e-6,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            float(delta_norm),
            float(self.golden["small_fp64/enc_out/delta_pair_norm"]),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_gaussian_basis_and_projection(self) -> None:
        gbf = GaussianLayer(SMALL["k"], SMALL["vocab"] ** 2)
        for key in ("means", "stds", "mul", "bias"):
            getattr(gbf, key).w = self.weights[f"gbf.{key}.weight"].copy()
        proj = NonLinearHead(SMALL["k"], SMALL["heads"], "gelu_erf", hidden=SMALL["k"])
        _set_linear(proj.linear1, self.weights, "gbf_proj.linear1")
        _set_linear(proj.linear2, self.weights, "gbf_proj.linear2")
        dist = self.golden["input/src_distance"].astype(np.float64)
        bias = proj(gbf(dist, self.golden["input/src_edge_type"]))
        nt = dist.shape[1]
        bias = np.ascontiguousarray(np.transpose(bias, (0, 3, 1, 2))).reshape(
            -1, nt, nt
        )
        # The basis is evaluated in fp32 upstream, so this is one fp32 ulp.
        np.testing.assert_allclose(
            bias, self.golden["small_fp64/enc_in/attn_bias"], rtol=1e-6, atol=1e-6
        )


class TestUniMolNormRegularisers(unittest.TestCase):
    """The hinge itself, which the golden values cannot constrain.

    On the small random model both regularisers sit at exactly zero, because
    the node norms happen to fall inside the tolerance. Comparing against zero
    says nothing about the formula, so it is checked directly here.
    """

    def test_hinge_is_zero_inside_the_tolerance_and_grows_outside(self) -> None:
        from deepmd.dpmodel.descriptor.unimol_nn.encoder import (
            norm_loss,
        )

        dim = 16
        root = dim**0.5
        unit = np.ones((1, 1, dim)) / root  # norm 1
        # Upstream's tolerance is 1: a norm within 1 of sqrt(dim) costs nothing.
        inside = unit * root
        np.testing.assert_allclose(norm_loss(inside), 0.0, atol=1e-6)
        np.testing.assert_allclose(norm_loss(unit * (root + 0.5)), 0.0, atol=1e-6)
        # Beyond it the cost is the excess, either side.
        np.testing.assert_allclose(norm_loss(unit * (root + 3.0)), 2.0, atol=1e-5)
        np.testing.assert_allclose(norm_loss(unit * (root - 3.0)), 2.0, atol=1e-5)

    def test_masked_mean_ignores_padding_and_survives_an_empty_row(self) -> None:
        from deepmd.dpmodel.descriptor.unimol_nn.encoder import (
            masked_mean,
        )

        value = np.array([[1.0, 2.0, 99.0], [4.0, 99.0, 99.0]])
        mask = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        # Per row: 1.5 and 4.0, then the mean over rows.
        np.testing.assert_allclose(float(masked_mean(mask, value)), 2.75, atol=1e-12)
        # An all-padding row returns zero rather than dividing by zero, which
        # is what upstream's epsilon in the denominator is for.
        empty = masked_mean(np.zeros((1, 3)), np.ones((1, 3)))
        self.assertTrue(np.isfinite(float(empty)))
        np.testing.assert_allclose(float(empty), 0.0, atol=1e-6)


class TestUniMolHeads(UniMolGoldenMixin, unittest.TestCase):
    """The three pretraining heads."""

    def test_heads_match_upstream(self) -> None:
        w = self.golden
        lm = MaskLMHead(SMALL["dim"], SMALL["vocab"], "gelu_erf")
        _set_linear(lm.dense, self.weights, "lm_head.dense")
        _set_layer_norm(lm.layer_norm, self.weights, "lm_head.layer_norm")
        _set_linear(
            lm.out_proj,
            self.weights,
            None,
            weight_key="lm_head.weight",
            bias_key="lm_head.bias",
        )
        dist_head = DistanceHead(SMALL["heads"], "gelu_erf")
        _set_linear(dist_head.dense, self.weights, "dist_head.dense")
        _set_layer_norm(dist_head.layer_norm, self.weights, "dist_head.layer_norm")
        _set_linear(dist_head.out_proj, self.weights, "dist_head.out_proj")
        p2c = NonLinearHead(SMALL["heads"], 1, "gelu_erf", hidden=SMALL["heads"])
        _set_linear(p2c.linear1, self.weights, "pair2coord_proj.linear1")
        _set_linear(p2c.linear2, self.weights, "pair2coord_proj.linear2")

        x = w["small_fp64/enc_out/x"].astype(np.float64)
        pair = w["small_fp64/enc_out/pair_rep"].astype(np.float64)
        pair = np.where(np.isneginf(pair), 0.0, pair)
        np.testing.assert_allclose(
            lm(x, w["input/masked_tokens"]),
            w["small_fp64/model/logits"],
            rtol=1e-11,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            dist_head(pair),
            w["small_fp64/model/encoder_distance"],
            rtol=1e-11,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            coord_update(
                w["input/src_coord"].astype(np.float64),
                w["small_fp64/enc_out/delta_pair_rep"].astype(np.float64),
                w["small_fp64/enc_in/padding_mask"].astype(np.float64),
                p2c,
            ),
            w["small_fp64/model/encoder_coord"],
            rtol=1e-11,
            atol=1e-11,
        )


class TestUniMolDescriptor(UniMolGoldenMixin, unittest.TestCase):
    """The descriptor, driven through deepmd-shaped inputs."""

    def test_forward_matches_upstream(self) -> None:
        desc = self.build_descriptor()
        coord, atype, nlist = self.deepmd_inputs()
        out = desc.forward_tokens(coord, atype, nlist)
        ref = self.golden["small_fp64/enc_out/x"].astype(np.float64)
        for f in range(len(self.n_real)):
            n = int(self.n_real[f])
            with self.subTest(molecule=f):
                np.testing.assert_allclose(
                    out["node_ebd"][f, 1 : n + 1],
                    ref[f, 1 : n + 1],
                    rtol=1e-6,
                    atol=1e-6,
                )
                np.testing.assert_array_equal(
                    out["tokens"][f, : n + 2],
                    self.golden["input/src_tokens"][f, : n + 2],
                )

    def test_call_drops_the_virtual_tokens(self) -> None:
        desc = self.build_descriptor()
        coord, atype, nlist = self.deepmd_inputs()
        node, rot, g2, h2, sw = desc.call(coord, atype, nlist)
        self.assertEqual(node.shape, (coord.shape[0], coord.shape[1], SMALL["dim"]))
        # The values must be the token-level ones with BOS and EOS removed,
        # not merely an array of the right shape.
        tokens = desc.forward_tokens(coord, atype, nlist)["node_ebd"]
        np.testing.assert_allclose(node, tokens[:, 1 : coord.shape[1] + 1, :], atol=0)
        self.assertGreater(float(np.abs(np.asarray(node)).max()), 0.0)
        self.assertIsNone(rot)
        self.assertIsNone(g2)
        self.assertIsNone(h2)
        self.assertIsNone(sw)

    def test_rejects_periodic_and_tiny_frames(self) -> None:
        desc = self.build_descriptor()
        coord, atype, nlist = self.deepmd_inputs()
        with self.assertRaisesRegex(ValueError, "every atom to be local"):
            desc.forward_tokens(np.concatenate([coord, coord], axis=1), atype, nlist)
        lonely = np.full_like(nlist[:, :1, :], -1)
        with self.assertRaisesRegex(ValueError, "two real atoms"):
            desc.forward_tokens(coord[:, :1], atype[:, :1], lonely)

    def test_serialize_round_trip(self) -> None:
        desc = self.build_descriptor()
        clone = DescrptUniMol.deserialize(desc.serialize())
        coord, atype, nlist = self.deepmd_inputs()
        np.testing.assert_allclose(
            clone.forward_tokens(coord, atype, nlist)["node_ebd"],
            desc.forward_tokens(coord, atype, nlist)["node_ebd"],
            rtol=1e-14,
            atol=1e-14,
        )


class TestUniMolLoss(UniMolGoldenMixin, unittest.TestCase):
    """The five-term objective."""

    def _labels(self, nloc, ncol):
        nf = len(self.n_real)
        labels = {
            "unimol_token_target": np.zeros((nf, nloc), dtype=np.int64),
            "unimol_coord_target": np.zeros((nf, nloc, 3)),
            "unimol_dist_target": np.zeros((nf, nloc, ncol)),
            "unimol_token_mask": np.zeros((nf, ncol), dtype=np.int64),
        }
        for f in range(nf):
            n = int(self.n_real[f])
            labels["unimol_token_target"][f, :n] = self.golden["target/tokens_target"][
                f, 1 : n + 1
            ]
            labels["unimol_coord_target"][f, :n] = self.golden["target/coord_target"][
                f, 1 : n + 1
            ]
            labels["unimol_dist_target"][f, :n, : n + 2] = self.golden[
                "target/distance_target"
            ][f, 1 : n + 1, : n + 2]
            labels["unimol_token_mask"][f, : n + 2] = 1
        return labels

    def test_terms_match_upstream(self) -> None:
        desc = self.build_descriptor()
        fitting = UniMolPretrainFitting(
            ntypes=len(desc.get_type_map()),
            dim_descrpt=SMALL["dim"],
            n_token=SMALL["vocab"],
            attention_heads=SMALL["heads"],
            max_atoms=desc.max_atoms,
            activation_function="gelu_erf",
        )
        w = self.weights
        _set_linear(fitting.lm_head.dense, w, "lm_head.dense")
        _set_layer_norm(fitting.lm_head.layer_norm, w, "lm_head.layer_norm")
        _set_linear(
            fitting.lm_head.out_proj,
            w,
            None,
            weight_key="lm_head.weight",
            bias_key="lm_head.bias",
        )
        _set_linear(fitting.pair2coord_proj.linear1, w, "pair2coord_proj.linear1")
        _set_linear(fitting.pair2coord_proj.linear2, w, "pair2coord_proj.linear2")
        _set_linear(fitting.dist_head.dense, w, "dist_head.dense")
        _set_layer_norm(fitting.dist_head.layer_norm, w, "dist_head.layer_norm")
        _set_linear(fitting.dist_head.out_proj, w, "dist_head.out_proj")

        coord, atype, nlist = self.deepmd_inputs()
        pred = fitting.call_tokens(desc.forward_tokens(coord, atype, nlist))
        labels = self._labels(coord.shape[1], fitting.max_atoms + 2)
        total, more = UniMolLoss().call(1.0, 0, pred, labels)

        expected = {
            "token_loss": "loss_token",
            "coord_loss": "loss_coord",
            "dist_loss": "loss_dist",
            "x_norm_loss": "loss_x_norm",
            "delta_pair_norm_loss": "loss_delta_pair_norm",
        }
        for mine, ref in expected.items():
            with self.subTest(term=mine):
                np.testing.assert_allclose(
                    float(more[mine]),
                    float(self.golden[f"small_fp64/loss/{ref}"]),
                    rtol=1e-5,
                    atol=1e-8,
                )
        np.testing.assert_allclose(
            float(total),
            float(self.golden["small_fp64/loss/loss_total"]),
            rtol=1e-5,
            atol=1e-8,
        )

    def test_derived_labels_match_explicit_ones(self) -> None:
        """Storing the distance target would cost O(natoms^2) per frame.

        The loss derives it, and the token column mask, from the clean
        coordinates and the real-atom mask instead. Both routes must agree.
        """
        nf, nloc = len(self.n_real), int(self.n_real.max())
        ncol = nloc + 2
        rng = np.random.default_rng(0)
        labels = self._labels(nloc, ncol)
        mask = np.zeros((nf, nloc), dtype=np.int64)
        for f in range(nf):
            mask[f, : int(self.n_real[f])] = 1
        pred = {
            "token_logits": rng.normal(size=(nf, nloc, SMALL["vocab"])),
            "coord_update": rng.normal(size=(nf, nloc, 3)),
            "pair_dist": rng.normal(size=(nf, nloc, ncol)),
            "x_norm": np.zeros((nf, nloc, 1)),
            "delta_pair_norm": np.zeros((nf, nloc, 1)),
            "mask": mask,
        }
        explicit, _ = UniMolLoss().call(1.0, 0, pred, labels)
        derived, _ = UniMolLoss().call(
            1.0,
            0,
            pred,
            {
                k: v
                for k, v in labels.items()
                if k not in ("unimol_dist_target", "unimol_token_mask")
            },
        )
        # The explicit target is upstream's, stored in fp32; the derived one is
        # computed in the working precision, so they part company there.
        np.testing.assert_allclose(float(derived), float(explicit), rtol=1e-7)

    def test_serialize_round_trip(self) -> None:
        loss = UniMolLoss(masked_coord_loss=3.0, beta=0.5)
        clone = UniMolLoss.deserialize(loss.serialize())
        self.assertEqual(clone.masked_coord_loss, 3.0)
        self.assertEqual(clone.beta, 0.5)


class TestUniMolExample(unittest.TestCase):
    """The shipped example must stay valid as the arguments evolve.

    It is checked here rather than in the shared example test, because that one
    also requires the referenced dataset to exist in the repository, and this
    example points at data the user converts from upstream.
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
            / "pretrain"
            / "input.json"
        )
        self.assertTrue(path.is_file(), f"missing example: {path}")
        config = normalize(json.loads(path.read_text()))
        self.assertEqual(config["model"]["descriptor"]["type"], "unimol")
        self.assertEqual(config["model"]["fitting_net"]["type"], "unimol_pretrain")
        self.assertEqual(config["loss"]["type"], "unimol")
        # A corrupted atom is carried as this pseudo-element, so the map needs it.
        self.assertIn("[MASK]", config["model"]["type_map"])


if __name__ == "__main__":
    unittest.main()
