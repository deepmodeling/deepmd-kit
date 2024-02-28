# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import unittest
from argparse import (
    Namespace,
)
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

import numpy as np

from deepmd.pt.entrypoints.main import (
    freeze,
    get_trainer,
)
from deepmd.pt.infer.deep_eval import (
    DeepPot,
)


class TestInitFrzModel(unittest.TestCase):
    def setUp(self):
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            config = json.load(f)
        config["training"]["numb_steps"] = 1
        config["training"]["save_freq"] = 1
        config["learning_rate"]["start_lr"] = 1.0
        config["training"]["training_data"]["systems"] = [
            str(Path(__file__).parent / "water/data/single")
        ]
        config["training"]["validation_data"]["systems"] = [
            str(Path(__file__).parent / "water/data/single")
        ]

        self.models = []
        for imodel in range(2):
            if imodel == 1:
                config["training"]["numb_steps"] = 0
                trainer = get_trainer(deepcopy(config), init_frz_model=self.models[-1])
            else:
                trainer = get_trainer(deepcopy(config))
            trainer.run()

            frozen_model = f"frozen_model{imodel}.pth"
            ns = Namespace(
                model="model.pt",
                output=frozen_model,
                head=None,
            )
            freeze(ns)
            self.models.append(frozen_model)

    def test_dp_test(self):
        dp1 = DeepPot(str(self.models[0]))
        dp2 = DeepPot(str(self.models[1]))
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

        e1, f1, v1, ae1, av1 = dp1.eval(coord, cell, atype, atomic=True)
        e2, f2, v2, ae2, av2 = dp2.eval(coord, cell, atype, atomic=True)
        np.testing.assert_allclose(e1, e2, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(f1, f2, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(v1, v2, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ae1, ae2, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(av1, av2, rtol=1e-10, atol=1e-10)
