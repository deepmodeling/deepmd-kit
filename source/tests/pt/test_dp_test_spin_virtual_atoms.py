# SPDX-License-Identifier: LGPL-3.0-or-later
"""Validate padded spin metrics through a real checkpoint, dataset and CLI."""

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
def test_dp_test_spin_virtual_atom_metrics(tmp_path, monkeypatch, padding_label):
    """Real and magnetic errors use valid scalar counts after chunk merging."""
    params = {
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [4, 4],
            "rcut": 3.0,
            "rcut_smth": 2.0,
            "neuron": [2, 4],
            "axis_neuron": 2,
            "precision": "float64",
            "seed": 1,
        },
        "fitting_net": {"neuron": [4], "precision": "float64", "seed": 1},
        "spin": {"use_spin": [True, False], "virtual_scale": [0.5]},
    }
    checkpoint = tmp_path / "spin_model.pt"
    model = get_model(params)
    torch.save(ModelWrapper(model, model_params=params).state_dict(), checkpoint)
    dp = DeepEval(str(checkpoint), auto_batch_size=False)
    assert dp.has_spin

    types = np.array([[0, 1, -1], [1, 0, 0], [-1, 1, 0]])
    coord = np.tile([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], (3, 1, 1))
    spin = np.tile([0.0, 0.0, 1.0], (3, 3, 1)) * (types == 0)[..., None]
    _, force, _, force_mag, mask_mag = dp.eval(
        coord.reshape(3, -1),
        None,
        types,
        mixed_type=True,
        spin=spin.reshape(3, -1),
    )
    valid = types >= 0
    magnetic = types == 0
    np.testing.assert_array_equal(mask_mag.reshape(3, 3), magnetic)
    frame_error = np.arange(1, 4, dtype=float)
    force_ref = np.where(
        valid[..., None],
        force.reshape(3, 3, 3) + frame_error[:, None, None],
        padding_label,
    )
    force_mag_ref = np.where(
        magnetic[..., None],
        force_mag.reshape(3, 3, 3) + 2 * frame_error[:, None, None],
        padding_label,
    )

    system = tmp_path / "spin_system"
    set_dir = system / "set.000"
    set_dir.mkdir(parents=True)
    (system / "type.raw").write_text("0\n0\n0\n")
    (system / "type_map.raw").write_text("A\nB\n")
    (system / "nopbc").touch()
    for name, value in (
        ("coord", coord.reshape(3, -1)),
        ("real_atom_types", types),
        ("spin", spin.reshape(3, -1)),
        ("force", force_ref.reshape(3, -1)),
        ("force_mag", force_mag_ref.reshape(3, -1)),
    ):
        np.save(set_dir / (name + ".npy"), value)

    expected = {}
    for suffix, selection, values in (
        ("fr", valid, frame_error),
        ("fm", magnetic, 2 * frame_error),
    ):
        counts = 3 * selection.sum(axis=1)
        expected["mae_" + suffix] = (np.average(values, weights=counts), counts.sum())
        expected["rmse_" + suffix] = (
            np.sqrt(np.average(values**2, weights=counts)),
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
        details = str(tmp_path / f"spin_details_{chunk_atoms}")
        errors = build_tester(dp, atomic=False).run(
            data, str(system), numb_test=3, detail_file=details
        )
        assert errors.keys() == expected.keys()
        for key, value in expected.items():
            np.testing.assert_allclose(errors[key], value, rtol=1e-6)
        real_detail = np.loadtxt(details + ".fr.out", ndmin=2)
        mag_detail = np.loadtxt(details + ".fm.out", ndmin=2)
        np.testing.assert_allclose(real_detail[:, :3], force_ref.reshape(-1, 3))
        np.testing.assert_allclose(real_detail[:, 3:], force.reshape(-1, 3))
        np.testing.assert_allclose(mag_detail[:, :3], force_mag_ref[magnetic])
        np.testing.assert_allclose(
            mag_detail[:, 3:], force_mag.reshape(3, 3, 3)[magnetic]
        )

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
    for label, key in (("Force atom RMSE", "rmse_fr"), ("Force spin RMSE", "rmse_fm")):
        assert label in output
        assert f"{expected[key][0]:e}" in output
