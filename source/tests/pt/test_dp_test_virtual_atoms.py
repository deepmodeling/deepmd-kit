# SPDX-License-Identifier: LGPL-3.0-or-later
"""Exercise virtual-atom metrics with a real checkpoint and mixed-type data."""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.infer.model_test import (
    build_tester,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.utils.data import (
    DeepmdData,
)


@pytest.mark.parametrize("padding_label", [0.0, np.nan])
def test_dp_test_virtual_atom_metrics(tmp_path, monkeypatch, padding_label):
    """Check checkpoint, disk-data and CLI metrics without counting padding."""
    params = {
        "type_map": ["H"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [4],
            "rcut": 3.0,
            "rcut_smth": 2.0,
            "neuron": [2, 4],
            "axis_neuron": 2,
            "precision": "float64",
            "seed": 1,
        },
        "fitting_net": {"neuron": [4], "precision": "float64", "seed": 1},
        "hessian_mode": True,
    }
    checkpoint = tmp_path / "model.pt"
    model = get_model(params)
    torch.save(ModelWrapper(model, model_params=params).state_dict(), checkpoint)
    dp = DeepEval(str(checkpoint), auto_batch_size=False)
    assert dp.has_hessian

    types = np.array([[0, 0, -1], [0, 0, 0], [-1, 0, 0]])
    coord = np.tile([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], (3, 1, 1))
    _, force, _, hessian = dp.eval(coord.reshape(3, -1), None, types, mixed_type=True)
    valid = np.repeat(types >= 0, 3, axis=1)
    pairs = valid[:, :, None] & valid[:, None, :]
    frame_error = np.arange(1, 4, dtype=float)
    force_ref = np.where(
        valid, force.reshape(3, -1) + frame_error[:, None], padding_label
    )
    hessian_ref = np.where(
        pairs, hessian.reshape(3, 9, 9) + frame_error[:, None, None], padding_label
    )

    system = tmp_path / "system"
    set_dir = system / "set.000"
    set_dir.mkdir(parents=True)
    (system / "type.raw").write_text("0\n0\n0\n")
    (system / "type_map.raw").write_text("H\n")
    (system / "nopbc").touch()
    np.save(set_dir / "coord.npy", coord.reshape(3, -1))
    np.save(set_dir / "real_atom_types.npy", types)
    np.save(set_dir / "force.npy", force_ref)
    np.save(set_dir / "hessian.npy", hessian_ref.reshape(3, -1))
    np.save(set_dir / "atom_pref.npy", np.tile(frame_error[:, None], (1, 3)))

    expected = {}
    for suffix, counts in (
        ("f", valid.sum(axis=1)),
        ("h", pairs.sum(axis=(1, 2))),
        ("fw", valid.sum(axis=1) * frame_error),
    ):
        expected["mae_" + suffix] = (
            np.average(frame_error, weights=counts),
            counts.sum(),
        )
        expected["rmse_" + suffix] = (
            np.sqrt(np.average(frame_error**2, weights=counts)),
            counts.sum(),
        )

    for chunk_atoms in (3, 100):
        monkeypatch.setenv("DP_TEST_CHUNK_ATOMS", str(chunk_atoms))
        data = DeepmdData(
            str(system),
            "set",
            shuffle_test=False,
            type_map=dp.get_type_map(),
            sort_atoms=False,
        )
        assert data.mixed_type
        details = str(tmp_path / f"details_{chunk_atoms}")
        errors = build_tester(dp, atomic=False).run(
            data, str(system), numb_test=3, detail_file=details
        )
        for key, value in expected.items():
            np.testing.assert_allclose(errors[key], value, rtol=1e-6)
        force_detail = np.loadtxt(details + ".f.out")
        hessian_detail = np.loadtxt(details + ".h.out")
        np.testing.assert_allclose(force_detail[:, :3], force_ref.reshape(-1, 3))
        np.testing.assert_allclose(hessian_detail[:, 0], hessian_ref.reshape(-1))

    # Run the actual command in a separate process, including disk loading,
    # checkpoint loading, chunk reduction and the final run-level report.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deepmd",
            "--pt",
            "test",
            "-m",
            str(checkpoint),
            "-s",
            str(system),
            "-n",
            "0",
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    output = result.stdout + result.stderr
    for label, key in (("Force  RMSE", "rmse_f"), ("Hessian RMSE", "rmse_h")):
        assert label in output
        assert f"{expected[key][0]:e}" in output
