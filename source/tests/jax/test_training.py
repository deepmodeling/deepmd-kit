# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end tests for the local JAX training entrypoint."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import (
    Path,
)
from unittest.mock import (
    patch,
)

from deepmd.jax.entrypoints.freeze import (
    freeze,
)
from deepmd.jax.entrypoints.main import (
    main,
)
from deepmd.utils.compat import (
    convert_optimizer_v31_to_v32,
)

MODEL_SE_E2_A = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [6, 12, 1],
        "rcut_smth": 0.50,
        "rcut": 4.00,
        "neuron": [2, 4, 8],
        "resnet_dt": False,
        "axis_neuron": 2,
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [4, 4, 4],
        "resnet_dt": True,
        "seed": 1,
    },
    "data_stat_nbatch": 1,
}


TRAINING_SCRIPT = """
from pathlib import Path
from unittest.mock import patch

from deepmd.main import main

with patch("deepmd.jax.entrypoints.train.SummaryPrinter.__call__"):
    main(["--jax", "train", "input.json", "--log-level", "2"])

for path in ["out.json", "lcurve.out", "checkpoint", "model-1.jax"]:
    if not Path(path).exists():
        raise FileNotFoundError(path)
"""


_LCURVE_STEP_RE = re.compile(r"^\s*(\d+)\b")


def _lcurve_steps(path: Path) -> set[int]:
    """Return integer step numbers written in an lcurve.out file."""
    steps: set[int] = set()
    for line in path.read_text().splitlines():
        match = _LCURVE_STEP_RE.match(line)
        if match:
            steps.add(int(match.group(1)))
    return steps


class TestJAXTraining(unittest.TestCase):
    """Regression tests for complete JAX training runs."""

    def setUp(self) -> None:
        """Create a temporary work directory with a one-step training input."""
        self.work_dir = Path(tempfile.mkdtemp())
        self.cwd = Path.cwd()
        os.chdir(self.work_dir)

        source_dir = Path(__file__).resolve().parents[1] / "pt" / "water"
        shutil.copytree(source_dir, self.work_dir / "water")
        data_file = [str(self.work_dir / "water" / "data" / "single")]

        with (self.work_dir / "water" / "se_atten.json").open() as f:
            self.config = json.load(f)
        self.config = convert_optimizer_v31_to_v32(self.config, warning=False)
        self.config["model"] = MODEL_SE_E2_A
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

    def test_train_entrypoint_runs_one_step_from_scratch(self) -> None:
        """Run local JAX training in a child process and check artifacts."""
        if os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ):
            # TODO: Re-enable this in GitHub CUDA CI once the hosted/self-hosted
            # runner JAX/PJRT abort is understood. The same test passes on a
            # local GPU, but the GitHub Actions CUDA job can terminate with
            # CUDA_ERROR_LAUNCH_FAILED while PJRT releases device buffers.
            self.skipTest(
                "JAX training is temporarily skipped on GitHub Actions CUDA runners"
            )

        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(TRAINING_SCRIPT)],
            cwd=self.work_dir,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )

        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn(1, _lcurve_steps(self.work_dir / "lcurve.out"))

    @patch("deepmd.jax.entrypoints.freeze.deserialize_to_file")
    @patch("deepmd.jax.entrypoints.freeze.serialize_from_file")
    def test_freeze_entrypoint_uses_checkpoint_pointer(
        self, serialize_from_file, deserialize_to_file
    ) -> None:
        """Freeze resolves the stable checkpoint pointer without Hessian options."""
        checkpoint_dir = self.work_dir / "ckpt"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "checkpoint").write_text("model-1.jax")
        serialize_from_file.return_value = {"model": {}, "model_def_script": {}}

        freeze(checkpoint_folder=str(checkpoint_dir), output="frozen_model")

        serialize_from_file.assert_called_once_with(str(checkpoint_dir / "model-1.jax"))
        deserialize_to_file.assert_called_once_with(
            "frozen_model.hlo", serialize_from_file.return_value
        )

    @patch("deepmd.jax.entrypoints.main.freeze")
    def test_main_dispatches_freeze(self, freeze_entrypoint) -> None:
        """JAX CLI main imports and dispatches the freeze command."""
        args = argparse.Namespace(
            command="freeze",
            log_level=2,
            log_path=None,
            checkpoint_folder=".",
            output="frozen_model",
        )

        main(args)

        freeze_entrypoint.assert_called_once()
