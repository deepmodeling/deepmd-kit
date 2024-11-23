# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import shutil
import tempfile
import unittest
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

from .common import (
    run_dp,
)


class TestInitFrzModel(unittest.TestCase):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            config = json.load(f)
        config["model"]["descriptor"]["smooth_type_embedding"] = True
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
        for imodel in range(3):
            frozen_model = f"frozen_model{imodel}.pth"
            if imodel == 0:
                temp_config = deepcopy(config)
                trainer = get_trainer(temp_config)
            elif imodel == 1:
                temp_config = deepcopy(config)
                temp_config["training"]["numb_steps"] = 0
                trainer = get_trainer(temp_config, init_frz_model=self.models[-1])
            else:
                empty_config = deepcopy(config)
                empty_config["model"]["descriptor"] = {}
                empty_config["model"]["fitting_net"] = {}
                empty_config["training"]["numb_steps"] = 0
                tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
                with open(tmp_input.name, "w") as f:
                    json.dump(empty_config, f, indent=4)
                run_dp(
                    f"dp --pt train {tmp_input.name} --init-frz-model {self.models[-1]} --use-pretrain-script --skip-neighbor-stat"
                )
                trainer = None

            if imodel in [0, 1]:
                trainer.run()
            freeze(
                model="model.pt",
                output=frozen_model,
                head=None,
            )
            self.models.append(frozen_model)

    def test_dp_test(self) -> None:
        dp1 = DeepPot(str(self.models[0]))
        dp2 = DeepPot(str(self.models[1]))
        dp3 = DeepPot(str(self.models[2]))
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

        ret1 = dp1.eval(coord, cell, atype, atomic=True)
        e1, f1, v1, ae1, av1 = ret1[0], ret1[1], ret1[2], ret1[3], ret1[4]
        ret2 = dp2.eval(coord, cell, atype, atomic=True)
        e2, f2, v2, ae2, av2 = ret2[0], ret2[1], ret2[2], ret2[3], ret2[4]
        ret3 = dp3.eval(coord, cell, atype, atomic=True)
        e3, f3, v3, ae3, av3 = ret3[0], ret3[1], ret3[2], ret3[3], ret3[4]
        np.testing.assert_allclose(e1, e2, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(e1, e3, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(f1, f2, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(f1, f3, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(v1, v2, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(v1, v3, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ae1, ae2, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ae1, ae3, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(av1, av2, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(av1, av3, rtol=1e-10, atol=1e-10)

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("frozen_model") and f.endswith(".pth"):
                os.remove(f)
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)
