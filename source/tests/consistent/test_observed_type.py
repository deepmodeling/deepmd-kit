# SPDX-License-Identifier: LGPL-3.0-or-later
"""Consistency test: observed_type should match across PT and dpmodel backends."""

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

from .common import (
    INSTALLED_PT,
)

if INSTALLED_PT:
    from deepmd.infer import (
        DeepPot,
    )
    from deepmd.pt.entrypoints.main import (
        get_trainer,
    )

    from ..pt.common import (
        run_dp,
    )
    from ..pt.model.test_permutation import (
        model_se_e2_a,
    )


@unittest.skipUnless(INSTALLED_PT, "PyTorch is not installed")
class TestObservedTypeConsistent(unittest.TestCase):
    """Train PT model, freeze to .pth and .dp, compare get_observed_types()."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.work_dir = tempfile.mkdtemp()
        cls.orig_dir = os.getcwd()
        os.chdir(cls.work_dir)

        input_json = str(
            Path(__file__).parent.parent / "pt" / "water" / "se_atten.json"
        )
        with open(input_json) as f:
            config = json.load(f)
        config["training"]["numb_steps"] = 1
        config["training"]["save_freq"] = 1
        data_file = [
            str(Path(__file__).parent.parent / "pt" / "water" / "data" / "single")
        ]
        config["training"]["training_data"]["systems"] = data_file
        config["training"]["validation_data"]["systems"] = data_file
        config["model"] = deepcopy(model_se_e2_a)
        config["model"]["type_map"] = ["O", "H", "Au"]

        # Train and freeze .pth
        trainer = get_trainer(deepcopy(config))
        trainer.run()
        run_dp("dp --pt freeze")

        # Convert .pth → .dp via serialize/deserialize hooks
        from deepmd.dpmodel.utils.serialization import (
            save_dp_model,
        )
        from deepmd.pt.utils.serialization import (
            serialize_from_file,
        )

        model_dict = serialize_from_file("frozen_model.pth")
        save_dp_model("frozen_model.dp", model_dict)

        # Load both
        cls.pt_model = DeepPot("frozen_model.pth")
        cls.dp_model = DeepPot("frozen_model.dp")

    @classmethod
    def tearDownClass(cls) -> None:
        os.chdir(cls.orig_dir)
        shutil.rmtree(cls.work_dir)

    def test_get_observed_types_consistent(self) -> None:
        pt_result = self.pt_model.deep_eval.get_observed_types()
        dp_result = self.dp_model.deep_eval.get_observed_types()
        self.assertEqual(pt_result, dp_result)
        # Training data only has O and H
        self.assertEqual(pt_result["observed_type"], ["H", "O"])

    def test_type_map_consistent(self) -> None:
        self.assertEqual(
            self.pt_model.deep_eval.get_type_map(),
            self.dp_model.deep_eval.get_type_map(),
        )


if __name__ == "__main__":
    unittest.main()
