# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end tests for the local JAX training entrypoint."""

import functools
import json
import os
import shutil
import signal
import tempfile
import unittest
from collections.abc import (
    Callable,
)
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
    TypeVar,
    cast,
)
from unittest.mock import (
    patch,
)

from deepmd.jax.entrypoints.train import (
    train,
)
from deepmd.utils.compat import (
    convert_optimizer_v31_to_v32,
)

_F = TypeVar("_F", bound=Callable[..., Any])


def _training_timeout(seconds: int) -> Callable[[_F], _F]:
    """Limit real training tests on platforms that support SIGALRM."""

    def decorate(func: _F) -> _F:
        if not hasattr(signal, "SIGALRM"):
            return func

        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            def raise_timeout(signum: int, frame: Any) -> None:
                raise TimeoutError(f"training test exceeded {seconds} seconds")

            previous_handler = signal.signal(signal.SIGALRM, raise_timeout)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_handler)

        return cast("_F", wrapped)

    return decorate


TRAINING_TEST_TIMEOUT = _training_timeout(60)

MODEL_SE_E2_A = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [46, 92, 4],
        "rcut_smth": 0.50,
        "rcut": 4.00,
        "neuron": [25, 50, 100],
        "resnet_dt": False,
        "axis_neuron": 16,
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [24, 24, 24],
        "resnet_dt": True,
        "seed": 1,
    },
    "data_stat_nbatch": 20,
}


class TestJAXTraining(unittest.TestCase):
    """Regression tests for complete JAX training runs."""

    def setUp(self) -> None:
        """Create a temporary work directory with a one-step training input."""
        self.work_dir = Path(tempfile.mkdtemp())
        self.cwd = Path.cwd()
        os.chdir(self.work_dir)

        source_dir = Path(__file__).resolve().parents[1] / "pt" / "water"
        shutil.copytree(source_dir, self.work_dir / "water")
        data_file = [str(self.work_dir / "water" / "data" / "data_0")]

        with (self.work_dir / "water" / "se_atten.json").open() as f:
            self.config = json.load(f)
        self.config = convert_optimizer_v31_to_v32(self.config, warning=False)
        self.config["model"] = deepcopy(MODEL_SE_E2_A)
        self.config["model"]["data_stat_nbatch"] = 1
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["disp_freq"] = 1
        self.config["training"]["save_freq"] = 1
        self.config["training"]["save_ckpt"] = "model"

        self.input_file = self.work_dir / "input.json"
        with self.input_file.open("w") as f:
            json.dump(self.config, f)

    def tearDown(self) -> None:
        """Remove temporary training outputs."""
        os.chdir(self.cwd)
        shutil.rmtree(self.work_dir)

    @TRAINING_TEST_TIMEOUT
    @patch("deepmd.jax.entrypoints.train.SummaryPrinter.__call__")
    def test_train_entrypoint_runs_one_step_from_scratch(self, _summary) -> None:
        """Run local JAX training and check that expected artifacts are written."""
        train(
            INPUT=str(self.input_file),
            init_model=None,
            restart=None,
            output="out.json",
            init_frz_model=None,
            mpi_log="master",
            log_level=2,
            log_path=None,
        )

        self.assertTrue(Path("out.json").is_file())
        self.assertTrue(Path("lcurve.out").is_file())
        self.assertTrue(Path("checkpoint").is_file())
        self.assertTrue(Path("model-1.jax").is_dir())
        self.assertIn("1", Path("lcurve.out").read_text())
