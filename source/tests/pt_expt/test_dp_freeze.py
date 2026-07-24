# SPDX-License-Identifier: LGPL-3.0-or-later
import argparse
import os
import shutil
import tempfile
import unittest
from copy import (
    deepcopy,
)

import torch

from deepmd.pt_expt.entrypoints.main import (
    freeze,
    main,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)

model_se_e2_a = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [46, 92, 4],
        "rcut_smth": 0.50,
        "rcut": 4.00,
        "neuron": [25, 50, 100],
        "resnet_dt": False,
        "axis_neuron": 16,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [24, 24, 24],
        "resnet_dt": True,
        "seed": 1,
    },
    "data_stat_nbatch": 20,
}

# dpa1 with attn_layer == 0 — the only graph-eligible model family today
# (mixed_types and uses_graph_lower()==True), used to exercise the
# ``freeze --lower-kind graph`` public-CLI path.
model_dpa1_graph = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "se_atten",
        "sel": 30,
        "rcut_smth": 2.0,
        "rcut": 6.0,
        "neuron": [2, 4, 8],
        "axis_neuron": 4,
        "attn": 5,
        "attn_layer": 0,
        "attn_dotr": True,
        "attn_mask": False,
        "activation_function": "tanh",
        "scaling_factor": 1.0,
        "normalize": True,
        "temperature": 1.0,
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [5, 5, 5],
        "resnet_dt": True,
        "seed": 1,
    },
}


class TestDPFreezePtExpt(unittest.TestCase):
    """Test dp freeze for the pt_expt backend."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp()

        # Build a model and save a fake checkpoint
        model_params = deepcopy(model_se_e2_a)
        model = get_model(model_params)
        wrapper = ModelWrapper(model, model_params=model_params)
        state_dict = wrapper.state_dict()
        cls.ckpt_file = os.path.join(cls.tmpdir, "model.pt")
        torch.save({"model": state_dict}, cls.ckpt_file)

        # Pre-freeze shared .pte and .pt2 files so individual tests don't
        # each pay the AOTInductor compilation cost (~82s per .pt2).
        cls.shared_pte = os.path.join(cls.tmpdir, "shared.pte")
        freeze(model=cls.ckpt_file, output=cls.shared_pte)
        cls.shared_pt2 = os.path.join(cls.tmpdir, "shared.pt2")
        freeze(model=cls.ckpt_file, output=cls.shared_pt2)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmpdir)

    def test_freeze_pte(self) -> None:
        """Freeze to .pte and verify the file is created."""
        output = os.path.join(self.tmpdir, "frozen_model.pte")
        freeze(model=self.ckpt_file, output=output)
        self.assertTrue(os.path.exists(output))

    def test_freeze_main_dispatcher(self) -> None:
        """Test main() CLI dispatcher with freeze command."""
        output_file = os.path.join(self.tmpdir, "frozen_via_main.pte")
        flags = argparse.Namespace(
            command="freeze",
            checkpoint_folder=self.ckpt_file,
            output=output_file,
            head=None,
            log_level=2,  # WARNING
            log_path=None,
        )
        main(flags)
        self.assertTrue(os.path.exists(output_file))

    def test_freeze_default_suffix(self) -> None:
        """Test that main() defaults output suffix to .pte."""
        output_file = os.path.join(self.tmpdir, "frozen_default_suffix.pth")
        flags = argparse.Namespace(
            command="freeze",
            checkpoint_folder=self.ckpt_file,
            output=output_file,
            head=None,
            log_level=2,  # WARNING
            log_path=None,
        )
        main(flags)
        expected = os.path.join(self.tmpdir, "frozen_default_suffix.pte")
        self.assertTrue(os.path.exists(expected))

    def test_freeze_output_suffix_by_lower_kind(self) -> None:
        """A suffix-less output defaults to .pt2 for lower_kind='graph' and
        .pte for nlist, while preserving an explicit .pte/.pt2 (iProzd
        review). The suffix follows the RESOLVED lower kind inside freeze()
        (native-spin models force 'graph', so the CLI cannot pick it before
        the model is built); the mapping is checked on the helper freeze()
        defers to. End-to-end application through main()/freeze() is covered
        by test_freeze_default_suffix (nlist) and
        test_native_spin_default_freeze_routes_to_graph in test_dpa4_export
        (graph).
        """
        from deepmd.pt_expt.entrypoints.main import (
            _default_output_path,
        )

        cases = [
            ("graph", "out_g", ".pt2"),  # graph, no suffix -> .pt2
            ("nlist", "out_n", ".pte"),  # nlist, no suffix -> .pte
            ("graph", "out_g_explicit.pte", ".pte"),  # explicit .pte kept
            ("nlist", "out_n_explicit.pt2", ".pt2"),  # explicit .pt2 kept
        ]
        for lower_kind, name, expected_suffix in cases:
            with self.subTest(lower_kind=lower_kind, name=name):
                resolved = _default_output_path(
                    os.path.join(self.tmpdir, name), lower_kind
                )
                self.assertTrue(resolved.endswith(expected_suffix))

    def test_freeze_main_passes_lower_kind_through(self) -> None:
        """main() forwards --lower-kind and the raw output path to freeze()
        (suffix defaulting is owned by freeze(), after native-spin
        resolution).
        """
        from unittest import (
            mock,
        )

        captured: dict = {}

        def _fake_freeze(model, output, head=None, lower_kind="nlist", **kw):
            captured["output"] = output
            captured["lower_kind"] = lower_kind

        raw_output = os.path.join(self.tmpdir, "out_passthrough")
        flags = argparse.Namespace(
            command="freeze",
            checkpoint_folder=self.ckpt_file,
            output=raw_output,
            head=None,
            lower_kind="graph",
            log_level=2,
            log_path=None,
        )
        with mock.patch("deepmd.pt_expt.entrypoints.main.freeze", _fake_freeze):
            main(flags)
        self.assertEqual(captured["output"], raw_output)
        self.assertEqual(captured["lower_kind"], "graph")

    def test_freeze_graph_rejects_ineligible(self) -> None:
        """``--lower-kind graph`` on a non-graph-eligible model (se_e2_a,
        mixed_types=False) fails fast rather than emitting a broken .pt2.
        """
        output = os.path.join(self.tmpdir, "frozen_graph_reject.pt2")
        with self.assertRaises(ValueError):
            freeze(model=self.ckpt_file, output=output, lower_kind="graph")

    def test_freeze_graph_dpa1(self) -> None:
        """``freeze --lower-kind graph`` on a graph-eligible dpa1(attn_layer=0)
        model produces a .pt2 whose metadata records the graph lower (the
        user-facing entry point to the C++ graph inference path).
        """
        import json
        import zipfile

        model_params = deepcopy(model_dpa1_graph)
        model = get_model(model_params)
        wrapper = ModelWrapper(model, model_params=model_params)
        ckpt = os.path.join(self.tmpdir, "dpa1_graph.pt")
        torch.save({"model": wrapper.state_dict()}, ckpt)

        output = os.path.join(self.tmpdir, "frozen_dpa1_graph.pt2")
        freeze(model=ckpt, output=output, lower_kind="graph")
        self.assertTrue(os.path.exists(output))

        # the .pt2 is a zip; metadata.json must record the graph lower
        with zipfile.ZipFile(output) as zf:
            meta_name = next(
                n for n in zf.namelist() if n.endswith("extra/metadata.json")
            )
            metadata = json.loads(zf.read(meta_name))
        self.assertEqual(metadata["lower_input_kind"], "graph")

    def test_freeze_pt2(self) -> None:
        """Freeze to .pt2 (AOTInductor) and verify the file is loadable."""
        self.assertTrue(os.path.exists(self.shared_pt2))

        # Verify the .pt2 can be loaded and evaluated via DeepPot
        import numpy as np

        from deepmd.infer import (
            DeepPot,
        )

        dp = DeepPot(self.shared_pt2)
        self.assertEqual(dp.get_type_map(), ["O", "H", "B"])
        rcut = dp.get_rcut()
        self.assertGreater(rcut, 0.0)

        # Quick smoke-test eval
        coord = np.array(
            [0.0, 0.0, 0.1, 1.0, 0.3, 0.2, 0.1, 1.9, 3.4],
            dtype=np.float64,
        )
        box = np.array([5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0], dtype=np.float64)
        atype = [0, 1, 2]
        e, f, v = dp.eval(coord, box, atype)
        self.assertEqual(e.shape, (1, 1))
        self.assertEqual(f.shape, (1, 3, 3))
        self.assertEqual(v.shape, (1, 9))

    def test_freeze_pt2_eval_consistency(self) -> None:
        """Verify .pte and .pt2 produce identical results."""
        import numpy as np

        from deepmd.infer import (
            DeepPot,
        )

        coord = np.array(
            [0.0, 0.0, 0.1, 1.0, 0.3, 0.2, 0.1, 1.9, 3.4],
            dtype=np.float64,
        )
        box = np.array([5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0], dtype=np.float64)
        atype = [0, 1, 2]

        dp_pte = DeepPot(self.shared_pte)
        dp_pt2 = DeepPot(self.shared_pt2)

        e_pte, f_pte, v_pte = dp_pte.eval(coord, box, atype)
        e_pt2, f_pt2, v_pt2 = dp_pt2.eval(coord, box, atype)

        np.testing.assert_allclose(e_pte, e_pt2, atol=1e-10)
        np.testing.assert_allclose(f_pte, f_pt2, atol=1e-10)
        np.testing.assert_allclose(v_pte, v_pt2, atol=1e-10)

    def test_freeze_pt2_nopbc_negative_coords(self) -> None:
        """Verify .pt2 NoPBC works with negative coordinates.

        Regression test: the C++ NoPBC path creates a fake box and must
        shift coordinates so atoms with negative values are inside [0, L).
        Compares .pt2 (C++ fake-box path) against .pte (Python no-ghost path)
        — these are independent NoPBC implementations so cross-comparison
        validates both.
        """
        import numpy as np

        from deepmd.infer import (
            DeepPot,
        )

        # Coordinates with negative values — no periodic box
        coord = np.array(
            [-1.0, -2.0, 0.5, 1.0, 0.3, -0.2, 0.1, -1.9, 3.4],
            dtype=np.float64,
        )
        atype = [0, 1, 2]

        dp_pte = DeepPot(self.shared_pte)
        dp_pt2 = DeepPot(self.shared_pt2)

        e_pte, f_pte, v_pte = dp_pte.eval(coord, None, atype)
        e_pt2, f_pt2, v_pt2 = dp_pt2.eval(coord, None, atype)

        np.testing.assert_allclose(e_pte, e_pt2, atol=1e-10)
        np.testing.assert_allclose(f_pte, f_pt2, atol=1e-10)
        np.testing.assert_allclose(v_pte, v_pt2, atol=1e-10)

    def test_nonspin_model_rejects_spin(self) -> None:
        """Non-spin model must raise ValueError when spin is provided."""
        import numpy as np

        from deepmd.infer import (
            DeepPot,
        )

        pt2_path = os.path.join(self.tmpdir, "nonspin_reject.pt2")
        freeze(model=self.ckpt_file, output=pt2_path)

        coord = np.array(
            [0.0, 0.0, 0.1, 1.0, 0.3, 0.2, 0.1, 1.9, 3.4],
            dtype=np.float64,
        )
        box = np.array([5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0], dtype=np.float64)
        atype = [0, 1, 2]
        spin = np.zeros(9, dtype=np.float64)

        dp = DeepPot(pt2_path)
        with self.assertRaises(ValueError):
            dp.eval(coord, box, atype, spin=spin)


if __name__ == "__main__":
    unittest.main()
