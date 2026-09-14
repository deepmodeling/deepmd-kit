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

    def test_a_complete_neighbour_list_covers_what_all_pairs_does(self) -> None:
        """The two settings differ only by what the neighbour list leaves out.

        Given a neighbour list that already holds every pair, they agree --
        which is what makes the option purely one of coverage.
        """
        covered = {}
        for coverage in ("neighbour", "all_pairs"):
            with self.subTest(coverage=coverage):
                out = self._run(self._build(coverage))
                covered[coverage] = float(out["pair_mask"].sum().item())
        self.assertEqual(covered["neighbour"], covered["all_pairs"])

    def test_the_default_is_the_neighbour_list(self) -> None:
        self.assertEqual(
            normalize(_config())["model"]["fitting_net"]["dist_coverage"], "neighbour"
        )

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


if __name__ == "__main__":
    unittest.main()
