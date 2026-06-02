# SPDX-License-Identifier: LGPL-3.0-or-later
import importlib.util
import json
import os
import unittest
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

import numpy as np
import torch

from deepmd.infer.deep_pot import DeepPot as DeepPotUni
from deepmd.pt.entrypoints.main import (
    freeze,
    get_trainer,
)
from deepmd.pt.infer.deep_eval import (
    DeepPot,
)


class TestDeepPot(unittest.TestCase):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.config["training"]["training_data"]["systems"] = [
            str(Path(__file__).parent / "water/data/single")
        ]
        self.config["training"]["validation_data"]["systems"] = [
            str(Path(__file__).parent / "water/data/single")
        ]
        self.input_json = "test_dp_test.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)

        trainer = get_trainer(deepcopy(self.config))
        trainer.run()

        with torch.device("cpu"):
            input_dict, label_dict, _ = trainer.get_data(is_train=False)
        trainer.wrapper(**input_dict, label=label_dict, cur_lr=1.0)
        self.model = "model.pt"

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f in ["lcurve.out", self.input_json]:
                os.remove(f)

    def test_dp_test(self) -> None:
        dp = DeepPot(str(self.model))
        cell = np.array(
            [
                5.122106549439247480e00,
                4.016537340154059388e-01,
                6.951654033828678081e-01,
                4.016537340154059388e-01,
                6.112136112297989143e00,
                8.178091365465004481e-01,
                6.951654033828678081e-01,
                8.178091365465004481e-01,
                6.159552512682983760e00,
            ]
        ).reshape(1, 3, 3)
        coord = np.array(
            [
                2.978060152121375648e00,
                3.588469695887098077e00,
                2.792459820604495491e00,
                3.895592322591093115e00,
                2.712091020667753760e00,
                1.366836847133650501e00,
                9.955616170888935690e-01,
                4.121324820711413039e00,
                1.817239061889086571e00,
                3.553661462345699906e00,
                5.313046969500791583e00,
                6.635182659098815883e00,
                6.088601018589653080e00,
                6.575011420004332585e00,
                6.825240650611076099e00,
            ]
        ).reshape(1, -1, 3)
        atype = np.array([0, 0, 0, 1, 1]).reshape(1, -1)

        ret = dp.eval(coord, cell, atype, atomic=True)
        e, f, v, ae, av = ret[0], ret[1], ret[2], ret[3], ret[4]
        self.assertEqual(e.shape, (1, 1))
        self.assertEqual(f.shape, (1, 5, 3))
        self.assertEqual(v.shape, (1, 9))
        self.assertEqual(ae.shape, (1, 5, 1))
        self.assertEqual(av.shape, (1, 5, 9))

        self.assertEqual(dp.get_type_map(), ["O", "H"])
        self.assertEqual(dp.get_ntypes(), 2)
        self.assertEqual(dp.get_dim_fparam(), 0)
        self.assertEqual(dp.get_dim_aparam(), 0)
        self.assertEqual(dp.deep_eval.model_type, DeepPot)

    def test_uni(self) -> None:
        dp = DeepPotUni("model.pt")
        self.assertIsInstance(dp, DeepPot)
        # its methods has been tested in test_dp_test

    def test_eval_typeebd(self) -> None:
        dp = DeepPot(str(self.model))
        eval_typeebd = dp.eval_typeebd()
        self.assertEqual(
            eval_typeebd.shape, (len(self.config["model"]["type_map"]) + 1, 8)
        )
        np.testing.assert_allclose(eval_typeebd[-1], np.zeros_like(eval_typeebd[-1]))

    # --- nlist_backend (vesin) option ---------------------------------------

    _cell = np.array(
        [
            5.122106549439247480e00,
            4.016537340154059388e-01,
            6.951654033828678081e-01,
            4.016537340154059388e-01,
            6.112136112297989143e00,
            8.178091365465004481e-01,
            6.951654033828678081e-01,
            8.178091365465004481e-01,
            6.159552512682983760e00,
        ]
    ).reshape(1, 3, 3)
    _coord = np.array(
        [
            2.978060152121375648e00,
            3.588469695887098077e00,
            2.792459820604495491e00,
            3.895592322591093115e00,
            2.712091020667753760e00,
            1.366836847133650501e00,
            9.955616170888935690e-01,
            4.121324820711413039e00,
            1.817239061889086571e00,
            3.553661462345699906e00,
            5.313046969500791583e00,
            6.635182659098815883e00,
            6.088601018589653080e00,
            6.575011420004332585e00,
            6.825240650611076099e00,
        ]
    ).reshape(1, -1, 3)
    _atype = np.array([0, 0, 0, 1, 1]).reshape(1, -1)

    def test_nlist_backend_default_is_vesin(self) -> None:
        dp = DeepPot(str(self.model))
        self.assertEqual(dp.deep_eval.nlist_backend, "vesin")

    def test_nlist_backend_native_disables_vesin(self) -> None:
        dp = DeepPot(str(self.model), nlist_backend="native")
        self.assertEqual(dp.deep_eval.nlist_backend, "native")
        self.assertFalse(dp.deep_eval._use_vesin)

    def test_nlist_backend_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            DeepPot(str(self.model), nlist_backend="bogus")

    def test_nlist_backend_vesin_unavailable_falls_back(self) -> None:
        import deepmd.pt.infer.deep_eval as deep_eval_mod

        original = deep_eval_mod.is_vesin_available
        deep_eval_mod.is_vesin_available = lambda: False
        try:
            dp = DeepPot(str(self.model), nlist_backend="vesin")
            self.assertFalse(dp.deep_eval._use_vesin)
        finally:
            deep_eval_mod.is_vesin_available = original

    def test_nlist_backend_spin_gates_off_vesin(self) -> None:
        dp = DeepPot(str(self.model), nlist_backend="vesin")
        dp.deep_eval._has_spin = True
        dp.deep_eval._setup_nlist_backend("vesin")
        self.assertFalse(dp.deep_eval._use_vesin)

    def test_nlist_backend_hessian_gates_off_vesin(self) -> None:
        dp = DeepPot(str(self.model), nlist_backend="vesin")
        dp.deep_eval._has_hessian = True
        dp.deep_eval._setup_nlist_backend("vesin")
        self.assertFalse(dp.deep_eval._use_vesin)

    @unittest.skipUnless(
        importlib.util.find_spec("vesin") is not None, "vesin not installed"
    )
    def test_nlist_backend_vesin_consistency(self) -> None:
        """Vesin O(N) nlist must match the native builder (PBC + non-PBC)."""
        dp_native = DeepPot(str(self.model), nlist_backend="native")
        dp_vesin = DeepPot(str(self.model), nlist_backend="vesin")
        self.assertFalse(dp_native.deep_eval._use_vesin)
        self.assertTrue(dp_vesin.deep_eval._use_vesin)

        for cell in (self._cell, None):
            e1, f1, v1, ae1, av1 = dp_native.eval(
                self._coord, cell, self._atype, atomic=True
            )
            e2, f2, v2, ae2, av2 = dp_vesin.eval(
                self._coord, cell, self._atype, atomic=True
            )
            np.testing.assert_allclose(e1, e2, rtol=1e-10, atol=1e-10, err_msg="energy")
            np.testing.assert_allclose(f1, f2, rtol=1e-10, atol=1e-10, err_msg="force")
            np.testing.assert_allclose(v1, v2, rtol=1e-10, atol=1e-10, err_msg="virial")
            np.testing.assert_allclose(
                ae1, ae2, rtol=1e-10, atol=1e-10, err_msg="atom_energy"
            )
            np.testing.assert_allclose(
                av1, av2, rtol=1e-10, atol=1e-10, err_msg="atom_virial"
            )

    @unittest.skipUnless(
        importlib.util.find_spec("vesin") is not None, "vesin not installed"
    )
    def test_nlist_backend_vesin_multiframe(self) -> None:
        """Vesin nlist with multiple frames must match native, for both
        auto_batch_size disabled and enabled (the latter with a small batch
        size forcing the AutoBatchSize loop to split frames into batches).
        """
        nframes = 5
        coord = np.tile(self._coord, (nframes, 1, 1))
        cell = np.tile(self._cell, (nframes, 1, 1))
        # auto_batch_size=False: single call; =2: forces 3 batches over 5 frames.
        for auto_batch_size in (False, 2):
            dp_native = DeepPot(
                str(self.model),
                nlist_backend="native",
                auto_batch_size=auto_batch_size,
            )
            dp_vesin = DeepPot(
                str(self.model),
                nlist_backend="vesin",
                auto_batch_size=auto_batch_size,
            )
            e1, f1, v1 = dp_native.eval(coord, cell, self._atype)
            e2, f2, v2 = dp_vesin.eval(coord, cell, self._atype)
            self.assertEqual(e2.shape, (nframes, 1))
            np.testing.assert_allclose(
                e1, e2, rtol=1e-10, atol=1e-10, err_msg=f"energy abs={auto_batch_size}"
            )
            np.testing.assert_allclose(
                f1, f2, rtol=1e-10, atol=1e-10, err_msg=f"force abs={auto_batch_size}"
            )
            np.testing.assert_allclose(
                v1, v2, rtol=1e-10, atol=1e-10, err_msg=f"virial abs={auto_batch_size}"
            )


class TestDeepPotFrozen(TestDeepPot):
    def setUp(self) -> None:
        super().setUp()
        frozen_model = "frozen_model.pth"
        freeze(
            model=self.model,
            output=frozen_model,
            head=None,
        )
        self.model = frozen_model

    # Note: this can not actually disable cuda device to be used
    # only can be used to test whether devices are mismatched
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.mock.patch("deepmd.pt.utils.env.DEVICE", torch.device("cpu"))
    @unittest.mock.patch("deepmd.pt.infer.deep_eval.DEVICE", torch.device("cpu"))
    def test_dp_test_cpu(self) -> None:
        self.test_dp_test()


# TestFparamAparamPT: moved to infer/test_models.py
