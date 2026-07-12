# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end PT checkpoint conversion coverage for TF2 DPA4."""

import os
import subprocess
import sys
from pathlib import (
    Path,
)

import pytest

import deepmd

if os.environ.get("DP_TEST_TF2_ONLY") != "1":
    pytest.skip(
        "TF2 tests require DP_TEST_TF2_ONLY=1",
        allow_module_level=True,
    )


_CONVERSION_SCRIPT = r"""
import copy
import os
import sys
from pathlib import Path

# Prefer the checked-out source while retaining the installed compiled library.
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if type(finder).__name__ != "ScikitBuildRedirectingFinder"
]
sys.path.insert(0, os.environ["DEEPMD_SOURCE_ROOT"])
import deepmd

compiled_package = os.environ.get("DEEPMD_COMPILED_PACKAGE")
if compiled_package and compiled_package not in deepmd.__path__:
    deepmd.__path__.append(compiled_package)

# Load PT before TensorFlow. Some mixed CUDA plugin builds are not safe when
# Triton is initialized after TensorFlow in the same process.
import torch
from deepmd.pt.model.model import get_model as get_pt_model
from deepmd.pt.train.wrapper import ModelWrapper
from deepmd.pt.utils.serialization import serialize_from_file as serialize_from_pt_file
from deepmd.tf2.env import tf
from deepmd.tf2.utils.serialization import deserialize_to_file as deserialize_to_tf2_file
from deepmd.utils.argcheck import model_args

output_dir = Path(sys.argv[1])
model_params = model_args().normalize_value(
    {
        "type": "dpa4",
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "dpa4",
            "sel": 4,
            "rcut": 4.0,
            "channels": 4,
            "n_radial": 4,
            "lmax": 1,
            "mmax": 1,
            "n_blocks": 1,
            "radial_so2_mode": "degree_channel",
            "precision": "float64",
            "seed": 1,
        },
        "fitting_net": {
            "type": "dpa4_ener",
            "neuron": [4],
            "precision": "float64",
            "seed": 1,
        },
    },
    trim_pattern="_.*",
)
pt_model = get_pt_model(copy.deepcopy(model_params)).to(torch.float64)
wrapper = ModelWrapper(pt_model, model_params=copy.deepcopy(model_params))
pt_path = output_dir / "dpa4.pt"
tf2_path = output_dir / "dpa4.savedmodeltf"
torch.save({"model": wrapper.state_dict()}, pt_path)

data = serialize_from_pt_file(str(pt_path))
assert data["model"]["type"] == "SeZM"
deserialize_to_tf2_file(str(tf2_path), data, jit_compile=False)
restored = tf.saved_model.load(str(tf2_path))
assert callable(restored.call_lower)
assert callable(restored.call_lower_atomic_virial)
"""


def test_pt_dpa4_checkpoint_converts_to_tf2_savedmodel(tmp_path) -> None:
    """The public ``.pt`` SeZM schema exports through the TF2 adapter."""
    source_root = Path(__file__).parents[3]
    compiled_package = next(
        (
            package_path
            for package_path in deepmd.__path__
            if (Path(package_path) / "lib").is_dir()
        ),
        "",
    )
    env = os.environ.copy()
    env["DEEPMD_SOURCE_ROOT"] = str(source_root)
    env["DEEPMD_COMPILED_PACKAGE"] = str(compiled_package)
    result = subprocess.run(
        [sys.executable, "-c", _CONVERSION_SCRIPT, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
