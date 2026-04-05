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

    def test_freeze_pt2(self) -> None:
        """Freeze to .pt2 (AOTInductor) and verify the file is loadable."""
        output = os.path.join(self.tmpdir, "frozen_model.pt2")
        freeze(model=self.ckpt_file, output=output)
        self.assertTrue(os.path.exists(output))

        # Verify the .pt2 can be loaded and evaluated via DeepPot
        import numpy as np

        from deepmd.infer import (
            DeepPot,
        )

        dp = DeepPot(output)
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

        pte_path = os.path.join(self.tmpdir, "consistency.pte")
        pt2_path = os.path.join(self.tmpdir, "consistency.pt2")
        freeze(model=self.ckpt_file, output=pte_path)
        freeze(model=self.ckpt_file, output=pt2_path)

        coord = np.array(
            [0.0, 0.0, 0.1, 1.0, 0.3, 0.2, 0.1, 1.9, 3.4],
            dtype=np.float64,
        )
        box = np.array([5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0], dtype=np.float64)
        atype = [0, 1, 2]

        dp_pte = DeepPot(pte_path)
        dp_pt2 = DeepPot(pt2_path)

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

        pte_path = os.path.join(self.tmpdir, "nopbc_neg.pte")
        pt2_path = os.path.join(self.tmpdir, "nopbc_neg.pt2")
        freeze(model=self.ckpt_file, output=pte_path)
        freeze(model=self.ckpt_file, output=pt2_path)

        # Coordinates with negative values — no periodic box
        coord = np.array(
            [-1.0, -2.0, 0.5, 1.0, 0.3, -0.2, 0.1, -1.9, 3.4],
            dtype=np.float64,
        )
        atype = [0, 1, 2]

        dp_pte = DeepPot(pte_path)
        dp_pt2 = DeepPot(pt2_path)

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
