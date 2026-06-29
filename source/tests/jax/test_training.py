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
from types import (
    SimpleNamespace,
)
from unittest.mock import (
    patch,
)

from deepmd.dpmodel.output_def import (
    OutputVariableCategory,
)
from deepmd.jax.entrypoints.freeze import (
    freeze,
)
from deepmd.jax.entrypoints.main import (
    main,
)
from deepmd.jax.entrypoints.train import (
    update_sel,
)
from deepmd.jax.infer.deep_eval import (
    DeepEval,
)
from deepmd.jax.model.hlo import (
    HLO,
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

    @patch("deepmd.jax.entrypoints.train.get_data")
    @patch("deepmd.jax.utils.update_sel.UpdateSel.get_nbor_stat")
    def test_update_sel_uses_jax_neighbor_stat(self, get_nbor_stat, get_data) -> None:
        """JAX update_sel should calculate neighbor statistics instead of skipping."""
        get_nbor_stat.return_value = 0.5, [10, 20]
        jdata = {
            "model": {
                "type_map": ["O", "H"],
                "descriptor": {
                    "type": "se_e2_a",
                    "rcut": 6.0,
                    "sel": "auto",
                },
            },
            "training": {"training_data": {}},
        }

        updated, min_nbor_dist = update_sel(jdata)

        self.assertEqual(updated["model"]["descriptor"]["sel"], [12, 24])
        self.assertEqual(min_nbor_dist, 0.5)
        get_data.assert_called_once_with({}, 0, ["O", "H"], None)
        get_nbor_stat.assert_called_once()

    @patch("deepmd.jax.entrypoints.freeze.deserialize_to_file")
    @patch("deepmd.jax.entrypoints.freeze.serialize_from_file")
    def test_freeze_entrypoint_uses_checkpoint_pointer(
        self, serialize_from_file, deserialize_to_file
    ) -> None:
        """Freeze resolves the stable checkpoint pointer and forwards Hessian."""
        checkpoint_dir = self.work_dir / "ckpt"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "checkpoint").write_text("model-1.jax")
        serialize_from_file.return_value = {"model": {}, "model_def_script": {}}

        freeze(
            checkpoint_folder=str(checkpoint_dir), output="frozen_model", hessian=True
        )

        serialize_from_file.assert_called_once_with(str(checkpoint_dir / "model-1.jax"))
        deserialize_to_file.assert_called_once_with(
            "frozen_model.hlo", serialize_from_file.return_value, hessian=True
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
            hessian=False,
        )

        main(args)

        freeze_entrypoint.assert_called_once()

    def test_hlo_hessian_mode_updates_output_def(self) -> None:
        """HLO output definition should expose Hessian when requested."""
        hlo = object.__new__(HLO)
        hlo._model_output_type = ["energy"]
        hlo.model_def_script = json.dumps({"hessian_mode": True})

        output_def = hlo.model_output_def()

        self.assertTrue(output_def["energy"].r_hessian)
        self.assertIn("energy_derv_r_derv_r", output_def.keys())

    def test_deep_eval_requests_hessian_for_hessian_model(self) -> None:
        """Non-atomic JAX evaluation should request Hessian outputs."""
        hlo = object.__new__(HLO)
        hlo._model_output_type = ["energy"]
        hlo.model_def_script = json.dumps({"hessian_mode": True})
        deep_eval = object.__new__(DeepEval)
        deep_eval.output_def = hlo.model_output_def()
        deep_eval.dp = SimpleNamespace(
            get_model_def_script=lambda: json.dumps({"hessian_mode": True})
        )

        request_defs = deep_eval._get_request_defs(atomic=False)

        self.assertTrue(deep_eval.get_has_hessian())
        self.assertIn(
            OutputVariableCategory.DERV_R_DERV_R,
            {odef.category for odef in request_defs},
        )
