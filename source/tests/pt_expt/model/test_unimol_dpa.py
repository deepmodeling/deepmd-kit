# SPDX-License-Identifier: LGPL-3.0-or-later
"""Uni-Mol's objective on a DPA backbone, through the configured path."""

import copy
import json
import os
import unittest

import numpy as np
import torch

from deepmd.pt_expt.loss.unimol import (
    UniMolLoss,
)
from deepmd.pt_expt.model import (
    get_model,
)
from deepmd.utils.argcheck import (
    normalize,
)

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(
    HERE, "..", "..", "..", "..", "examples", "water", "dpa4", "input.json"
)


def _config(coverage="neighbour", max_atoms=8):
    """A small DPA4 backbone, built from the shipped example's own keys."""
    with open(EXAMPLE) as fh:
        base = json.load(fh)["model"]["descriptor"]
    desc = {k: v for k, v in base.items() if k != "_comment"}
    desc.update(
        {
            "type": "dpa4",
            "channels": 32,
            "n_blocks": 1,
            "precision": "float64",
            "use_amp": False,
        }
    )
    return {
        "model": {
            "type_map": ["C", "N", "O", "H", "[MASK]"],
            "descriptor": desc,
            "fitting_net": {
                "type": "unimol_dpa_pretrain",
                "n_token": 31,
                "max_atoms": max_atoms,
                "dist_hidden": 16,
                "dist_coverage": coverage,
                "precision": "float64",
            },
        },
        "learning_rate": {"type": "exp", "start_lr": 1e-4, "stop_lr": 1e-6},
        # the two norm regularisers constrain Uni-Mol's own transformer and have
        # no counterpart on a DPA backbone, which also wraps nothing around a
        # molecule
        "loss": {
            "type": "unimol",
            "x_norm_loss": 0.0,
            "delta_pair_repr_norm_loss": 0.0,
            "virtual_tokens": False,
        },
        "training": {
            "training_data": {"systems": ["x"]},
            "numb_steps": 1,
            "seed": 1,
        },
    }


class TestUniMolDPAConfigured(unittest.TestCase):
    def setUp(self) -> None:
        self.nloc = 6
        rng = np.random.default_rng(0)
        self.coord = rng.normal(size=(1, self.nloc, 3)) * 1.5
        self.atype = rng.integers(0, 4, size=(1, self.nloc))
        self.nlist = np.stack(
            [
                np.stack(
                    [
                        np.array([j for j in range(self.nloc) if j != i])
                        for i in range(self.nloc)
                    ]
                )
            ]
        )

    def _build(self, coverage="neighbour"):
        return get_model(normalize(_config(coverage))["model"])

    def _run(self, model):
        device = next(model.parameters()).device
        return model.forward_lower(
            torch.as_tensor(self.coord, device=device),
            torch.as_tensor(self.atype, device=device),
            torch.as_tensor(self.nlist, device=device),
        )

    def test_the_configured_path_builds_and_runs(self) -> None:
        model = self._build()
        self.assertEqual(type(model).__name__, "UniMolDPAPretrainModel")
        out = self._run(model)
        for name in ("token_logits", "coord_update", "pair_dist", "pair_mask"):
            with self.subTest(output=name):
                self.assertIn(name, out)
                self.assertTrue(bool(torch.isfinite(out[name]).all()))

    def test_a_partial_neighbour_list_covers_fewer_pairs(self) -> None:
        """Neighbour coverage is a subset, which is the whole trade-off."""
        sparse = np.full((1, self.nloc, 2), -1, dtype=np.int64)
        for i in range(self.nloc):
            sparse[0, i] = [(i + 1) % self.nloc, (i + 2) % self.nloc]
        full = self.nlist

        covered = {}
        for name, nlist in (("neighbour", sparse), ("all_pairs", full)):
            model = self._build("neighbour" if name == "neighbour" else "all_pairs")
            device = next(model.parameters()).device
            out = model.forward_lower(
                torch.as_tensor(self.coord, device=device),
                torch.as_tensor(self.atype, device=device),
                torch.as_tensor(nlist, device=device),
            )
            covered[name] = float(out["pair_mask"].sum().item())
        self.assertLess(covered["neighbour"], covered["all_pairs"])
        self.assertEqual(covered["neighbour"], 2 * self.nloc)

    def test_a_complete_neighbour_list_differs_only_by_the_diagonal(self) -> None:
        """The two settings differ only by what a neighbour list cannot hold.

        Given a list that already holds every other atom, the only pairs left
        over are the self-pairs -- which upstream scores and a neighbour list
        can never contain. That is the whole of the difference, which is what
        makes the option one of coverage rather than of model.
        """
        covered = {}
        for coverage in ("neighbour", "all_pairs"):
            with self.subTest(coverage=coverage):
                out = self._run(self._build(coverage))
                covered[coverage] = float(out["pair_mask"].sum().item())
        self.assertEqual(covered["all_pairs"] - covered["neighbour"], float(self.nloc))

    def test_the_default_is_the_neighbour_list(self) -> None:
        self.assertEqual(
            normalize(_config())["model"]["fitting_net"]["dist_coverage"], "neighbour"
        )

    def test_the_coordinate_head_is_trainable(self) -> None:
        """It projects with SO3Linear, whose weights are plain arrays.

        On this backend a plain array is registered as a buffer, and the
        optimizer is built from the model's parameters, so without promotion
        the head would train as a frozen random projection -- silently, because
        the coordinate term still falls: gradients reach the backbone through
        the frozen projection either way.
        """
        model = self._build()
        head = model.atomic_model.fitting_net.coord_head
        names = [n for n, _ in head.named_parameters()]
        self.assertIn("proj.weight", names)
        self.assertIsInstance(head.proj.weight, torch.nn.Parameter)
        self.assertTrue(head.proj.weight.requires_grad)
        # and the optimizer would actually see it
        self.assertTrue(any("coord_head" in n for n, _ in model.named_parameters()))

    def test_the_coordinate_head_actually_moves(self) -> None:
        """Owning a parameter is not the same as learning one."""
        model = self._build()
        head = model.atomic_model.fitting_net.coord_head
        before = head.proj.weight.detach().clone()

        rng = np.random.default_rng(1)
        device = next(model.parameters()).device
        token_target = torch.zeros((1, self.nloc), dtype=torch.int64, device=device)
        token_target[0, 1] = 5
        labels = {
            "unimol_token_target": token_target,
            "unimol_coord_target": torch.as_tensor(
                rng.normal(size=(1, self.nloc, 3)), device=device
            ),
        }
        params = copy.deepcopy(normalize(_config())["loss"])
        params.pop("type")
        loss = UniMolLoss(**params)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(5):
            opt.zero_grad()
            total, _ = loss.call(1.0, 0, self._run(model), labels)
            total.backward()
            opt.step()
        self.assertGreater(float((head.proj.weight.detach() - before).abs().max()), 0.0)

    def test_scoring_a_dpa_backbone_with_virtual_tokens_is_refused(self) -> None:
        """Getting this wrong shifts every distance label by one column.

        The two conventions differ in which columns of the pair axis are real.
        Scoring a backbone that wraps nothing as though it wrapped BOS and EOS
        selects the same number of entries, so it neither crashes nor changes
        shape -- every corrupted row simply trains against another atom's
        distances.
        """
        out = self._run(self._build())
        rng = np.random.default_rng(1)
        device = next(self._build().parameters()).device
        token_target = torch.zeros((1, self.nloc), dtype=torch.int64, device=device)
        token_target[0, 1] = 5
        labels = {
            "unimol_token_target": token_target,
            "unimol_coord_target": torch.as_tensor(
                rng.normal(size=(1, self.nloc, 3)), device=device
            ),
        }
        wrong = UniMolLoss(
            x_norm_loss=0.0, delta_pair_repr_norm_loss=0.0, virtual_tokens=True
        )
        with self.assertRaisesRegex(ValueError, "wraps no virtual tokens"):
            wrong.call(1.0, 0, out, labels)

    def test_a_periodic_frame_is_refused_on_the_path_evaluation_uses(self) -> None:
        """The guard has to be where every path passes, not only at the top.

        A box handed to ``forward`` is the easy case. Evaluation and export go
        through the lower path, where a cell has already become ghost atoms --
        which the coverage mask drops and the distance target has no
        minimum-image convention for. Guarding only the upper path would leave
        the case that matters unguarded.
        """
        model = self._build()
        device = next(model.parameters()).device
        nall = self.nloc + 2
        coord = np.concatenate([self.coord, self.coord[:, :2] + 12.0], axis=1)
        atype = np.concatenate([self.atype, self.atype[:, :2]], axis=1)
        with self.assertRaisesRegex(ValueError, "ghost atom"):
            model.forward_lower(
                torch.as_tensor(coord, device=device),
                torch.as_tensor(atype, device=device),
                torch.as_tensor(self.nlist, device=device),
            )
        self.assertEqual(nall, atype.shape[1])

    def test_the_graph_path_is_refused_rather_than_half_advertised(self) -> None:
        """The base class advertises a graph entry these heads do not have."""
        model = self._build()
        self.assertFalse(model.atomic_model.supports_graph_export())
        self.assertFalse(model.atomic_model.uses_graph_lower())

    def test_the_objective_scores_the_configured_model(self) -> None:
        model = self._build()
        out = self._run(model)
        rng = np.random.default_rng(1)
        device = next(model.parameters()).device
        token_target = torch.zeros((1, self.nloc), dtype=torch.int64, device=device)
        token_target[0, 1] = 5
        token_target[0, 3] = 6
        labels = {
            "unimol_token_target": token_target,
            "unimol_coord_target": torch.as_tensor(
                rng.normal(size=(1, self.nloc, 3)), device=device
            ),
        }
        params = copy.deepcopy(normalize(_config())["loss"])
        params.pop("type")
        total, terms = UniMolLoss(**params).call(1.0, 0, out, labels)
        self.assertEqual(sorted(terms), ["coord_loss", "dist_loss", "token_loss"])
        self.assertTrue(bool(torch.isfinite(torch.as_tensor(total))))

    def test_keeping_the_loss_defaults_on_this_backbone_is_refused(self) -> None:
        """The two norm weights default to Uni-Mol's 0.01, unasked for.

        A configuration that selects this fitting and leaves the loss block
        alone therefore weights two regularisers the backbone does not
        produce, which used to surface as ``KeyError: 'x_norm'`` on the first
        batch -- after the data had been read and the statistics computed.
        The trainer checks it while the model and the loss are wired together,
        so this goes through ``get_loss``, the function that does the wiring,
        rather than calling the check directly.
        """
        from deepmd.pt_expt.train.training import (
            get_loss,
        )

        cfg = normalize(
            _config() | {"loss": {"type": "unimol", "virtual_tokens": False}}
        )
        # the weights nobody wrote
        self.assertEqual(cfg["loss"]["x_norm_loss"], 0.01)
        self.assertEqual(cfg["loss"]["delta_pair_repr_norm_loss"], 0.01)
        model = get_model(copy.deepcopy(cfg["model"]))
        with self.assertRaises(ValueError) as caught:
            get_loss(
                copy.deepcopy(cfg["loss"]),
                1e-4,
                len(cfg["model"]["type_map"]),
                model,
            )
        message = str(caught.exception)
        for expected in (
            "x_norm",
            "delta_pair_norm",
            "x_norm_loss",
            "delta_pair_repr_norm_loss",
        ):
            with self.subTest(names=expected):
                self.assertIn(expected, message)

    def test_a_loss_built_by_hand_is_refused_the_same_way(self) -> None:
        """Not every caller goes through the trainer."""
        out = self._run(self._build())
        device = next(self._build().parameters()).device
        token_target = torch.zeros((1, self.nloc), dtype=torch.int64, device=device)
        token_target[0, 1] = 5
        labels = {
            "unimol_token_target": token_target,
            "unimol_coord_target": torch.zeros(
                (1, self.nloc, 3), dtype=torch.float64, device=device
            ),
        }
        with self.assertRaisesRegex(ValueError, "emits no x_norm"):
            UniMolLoss(virtual_tokens=False).call(1.0, 0, out, labels)

    def test_the_shipped_configuration_is_not_refused(self) -> None:
        """A guard that fires on the supported case is worse than none."""
        from deepmd.pt_expt.train.training import (
            get_loss,
        )

        with open(
            os.path.join(
                HERE,
                "..",
                "..",
                "..",
                "..",
                "examples",
                "unimol",
                "dpa_pretrain",
                "input.json",
            )
        ) as fh:
            shipped = {
                k: v for k, v in json.load(fh)["loss"].items() if not k.startswith("_")
            }
        cfg = normalize(_config() | {"loss": shipped})
        model = get_model(copy.deepcopy(cfg["model"]))
        loss = get_loss(
            copy.deepcopy(cfg["loss"]), 1e-4, len(cfg["model"]["type_map"]), model
        )
        self.assertEqual(loss.x_norm_loss, 0.0)


if __name__ == "__main__":
    unittest.main()
