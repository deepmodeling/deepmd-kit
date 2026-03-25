# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import shutil
import tempfile
import unittest
from copy import (
    deepcopy,
)

import numpy as np
import torch

from deepmd.main import (
    main,
)
from deepmd.pt_expt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt_expt.utils.env import (
    DEVICE,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)


def to_numpy(x):
    """Convert array-like (numpy or torch.Tensor) to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


EXAMPLE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "examples",
    "water",
)


def run_dp(cmd: str) -> int:
    """Run DP directly from the entry point."""
    cmds = cmd.split()
    if cmds[0] == "dp":
        cmds = cmds[1:]
    else:
        raise RuntimeError("The command is not dp")
    main(cmds)
    return 0


def _make_config(data_dir: str) -> dict:
    """Build a minimal config dict for change-bias tests."""
    return {
        "model": {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [6, 12],
                "rcut_smth": 0.50,
                "rcut": 3.00,
                "neuron": [8, 16],
                "resnet_dt": False,
                "axis_neuron": 4,
                "type_one_side": True,
                "seed": 1,
            },
            "fitting_net": {
                "neuron": [16, 16],
                "resnet_dt": True,
                "seed": 1,
            },
            "data_stat_nbatch": 1,
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 500,
            "start_lr": 0.001,
            "stop_lr": 3.51e-8,
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 0,
            "limit_pref_v": 0,
        },
        "training": {
            "training_data": {
                "systems": [os.path.join(data_dir, "data_0")],
                "batch_size": 1,
            },
            "validation_data": {
                "systems": [os.path.join(data_dir, "data_0")],
                "batch_size": 1,
                "numb_btch": 1,
            },
            "numb_steps": 1,
            "seed": 10,
            "disp_file": "lcurve.out",
            "disp_freq": 1,
            "save_freq": 1,
        },
    }


class TestChangeBias(unittest.TestCase):
    """Test dp change-bias for the pt_expt backend."""

    @classmethod
    def setUpClass(cls) -> None:
        from .conftest import (
            _pop_device_contexts,
        )

        _pop_device_contexts()

        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        cls.data_dir = data_dir
        cls.data_file = [os.path.join(data_dir, "data_0")]

        cls.tmpdir = tempfile.mkdtemp()
        cls.old_cwd = os.getcwd()
        os.chdir(cls.tmpdir)

        # Build & train 1-step model
        config = _make_config(data_dir)
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)
        config["training"]["save_ckpt"] = "model.ckpt"
        cls.config = config
        trainer = get_trainer(deepcopy(config))
        trainer.run()

        cls.model_path = os.path.join(cls.tmpdir, "model.ckpt.pt")

        # Record original bias
        cls.original_bias = to_numpy(trainer.wrapper.model.get_out_bias())

    @classmethod
    def tearDownClass(cls) -> None:
        os.chdir(cls.old_cwd)
        shutil.rmtree(cls.tmpdir)

    def _load_model_from_ckpt(self, ckpt_path: str):
        """Load a pt_expt model from a .pt checkpoint."""
        import torch

        state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model_state = state_dict["model"]
        model_params = model_state["_extra_state"]["model_params"]
        model = get_model(model_params)
        wrapper = ModelWrapper(model)
        wrapper.load_state_dict(model_state)
        return model

    def test_change_bias_with_data(self) -> None:
        output_path = os.path.join(self.tmpdir, "model_data_bias.pt")
        run_dp(
            f"dp --pt-expt change-bias {self.model_path} "
            f"-s {self.data_file[0]} -o {output_path}"
        )
        updated_model = self._load_model_from_ckpt(output_path)
        updated_bias = to_numpy(updated_model.get_out_bias())
        original_bias = np.array(self.original_bias)
        # Bias should have changed from the original
        self.assertFalse(
            np.allclose(original_bias, updated_bias),
            "Bias should have changed after change-bias with data",
        )

    def test_change_bias_with_data_sys_file(self) -> None:
        tmp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt", dir=self.tmpdir
        )
        with open(tmp_file.name, "w") as f:
            f.writelines([sys + "\n" for sys in self.data_file])

        output_path = os.path.join(self.tmpdir, "model_file_bias.pt")
        run_dp(
            f"dp --pt-expt change-bias {self.model_path} "
            f"-f {tmp_file.name} -o {output_path}"
        )
        updated_model = self._load_model_from_ckpt(output_path)
        updated_bias = to_numpy(updated_model.get_out_bias())
        original_bias = np.array(self.original_bias)
        # Bias should have changed from the original
        self.assertFalse(
            np.allclose(original_bias, updated_bias),
            "Bias should have changed after change-bias with data file",
        )

    def test_change_bias_with_user_defined(self) -> None:
        user_bias = [0.1, 3.2]
        output_path = os.path.join(self.tmpdir, "model_user_bias.pt")
        run_dp(
            f"dp --pt-expt change-bias {self.model_path} "
            f"-b {' '.join(str(v) for v in user_bias)} -o {output_path}"
        )
        updated_model = self._load_model_from_ckpt(output_path)
        updated_bias = to_numpy(updated_model.get_out_bias())
        expected_bias = np.array(user_bias).reshape(updated_bias.shape)
        np.testing.assert_allclose(updated_bias, expected_bias)

    def test_change_bias_frozen_pte(self) -> None:
        from deepmd.pt_expt.entrypoints.main import (
            freeze,
        )
        from deepmd.pt_expt.model.model import (
            BaseModel,
        )
        from deepmd.pt_expt.utils.serialization import (
            serialize_from_file,
        )

        # Freeze the checkpoint
        pte_path = os.path.join(self.tmpdir, "frozen.pte")
        freeze(model=self.model_path, output=pte_path)

        # Get original bias
        original_data = serialize_from_file(pte_path)
        original_model = BaseModel.deserialize(original_data["model"])
        original_bias = to_numpy(original_model.get_out_bias())

        # Run change-bias on the frozen model
        output_pte = os.path.join(self.tmpdir, "frozen_updated.pte")
        run_dp(
            f"dp --pt-expt change-bias {pte_path} "
            f"-s {self.data_file[0]} -o {output_pte}"
        )

        # Load updated model and verify bias changed
        updated_data = serialize_from_file(output_pte)
        updated_model = BaseModel.deserialize(updated_data["model"])
        updated_bias = to_numpy(updated_model.get_out_bias())

        # Bias should have changed
        self.assertFalse(
            np.allclose(original_bias, updated_bias),
            "Bias should have changed after change-bias on frozen model",
        )


class TestChangeBiasFittingStats(unittest.TestCase):
    """Test that model_change_out_bias recomputes fitting stats for set-by-statistic."""

    def _make_mock_model(self):
        from unittest.mock import (
            MagicMock,
        )

        from deepmd.dpmodel.model.dp_model import (
            DPModelCommon,
        )

        fitting_net = MagicMock()

        class FakeModel(DPModelCommon):
            def get_out_bias(self):
                return np.array([[0.0, 0.0]])

            def get_type_map(self):
                return ["O", "H"]

            def get_fitting_net(self):
                return fitting_net

            def change_out_bias(self, *args, **kwargs):
                pass

        return FakeModel(), fitting_net

    def test_compute_input_stats_called(self) -> None:
        from deepmd.pt_expt.train.training import (
            model_change_out_bias,
        )

        model, fitting_net = self._make_mock_model()
        sample_func = [{"energy": np.zeros((1, 1))}]

        model_change_out_bias(model, sample_func, _bias_adjust_mode="set-by-statistic")

        fitting_net.compute_input_stats.assert_called_once_with(sample_func)

    def test_compute_input_stats_not_called_for_change(self) -> None:
        from deepmd.pt_expt.train.training import (
            model_change_out_bias,
        )

        model, fitting_net = self._make_mock_model()
        sample_func = [{"energy": np.zeros((1, 1))}]

        model_change_out_bias(
            model, sample_func, _bias_adjust_mode="change-by-statistic"
        )

        fitting_net.compute_input_stats.assert_not_called()


if __name__ == "__main__":
    unittest.main()
