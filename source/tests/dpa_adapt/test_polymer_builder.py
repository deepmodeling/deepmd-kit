# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end test for PolymerBuilder (CSV -> grouped DeepMD npy).

Needs RDKit for SMILES->3D; skips cleanly otherwise.  No compiled deepmd
backend required.
"""

from __future__ import (
    annotations,
)

import numpy as np
import pytest

pytest.importorskip("rdkit")

from dpa_adapt.grouped._polymer import (
    PolymerBuilder,
)

_CSV = """\
reference;SMILES_start_group;SMILES_end_group;SMILES_repeating_unitA;molpercent_repeating_unitA;SMILES_repeating_unitB;molpercent_repeating_unitB;Mn;polymer_concentration_wpercent;additive1;additive1_concentration_molar;pH;cloud_point
r1;[C](C)(C)C#N;[C](C)(C)C#N;[CH2][CH](C(=O)NC(C)C);0.8;[CH2][CH](C(=O)O);0.2;11500;0.01;NaCl;0.01;7;32.1
r2;;;[CH2][CH](C(=O)NC(C)C);1.0;;;20000;0.02;;;7;45.0
"""


def _write_csv(tmp_path):
    path = tmp_path / "cp.csv"
    path.write_text(_CSV, encoding="utf-8")
    return path


def test_from_csv_writes_grouped_polymer_systems(tmp_path):
    csv = _write_csv(tmp_path)
    out = tmp_path / "data"
    result = PolymerBuilder.from_csv(csv, target="cloud_point", decimal=".").write(out)

    assert result["n_groups"] == 2
    F = result["fparam_dim"]
    # schema = [mn_log, conc, pH, salt:NaCl]
    assert F == 4

    # ---- first polymer: 2 units (0.8/0.2) + 2 ends ----
    sys0 = out / result["systems"][0]
    set0 = sys0 / "set.000"

    coord = np.load(set0 / "coord.npy")
    real = np.load(set0 / "real_atom_types.npy")
    pool = np.load(set0 / "pool_mask.npy")
    weight = np.load(set0 / "weight.npy")
    gid = np.load(set0 / "group_id.npy")
    fparam = np.load(set0 / "fparam.npy")
    label = np.load(set0 / "cloud_point.npy")

    nframes = 4  # unitA, unitB, start, end
    natoms = real.shape[1]
    assert coord.shape == (nframes, natoms * 3)
    assert real.shape == (nframes, natoms)
    assert pool.shape == (nframes, natoms)

    # one group per polymer -> constant group_id within the system
    assert gid.shape == (nframes,)
    assert len(set(gid.tolist())) == 1

    # weights: repeating units keep their mole fraction; ends share the rest
    assert weight[0] == pytest.approx(0.8)
    assert weight[1] == pytest.approx(0.2)
    assert weight[2] == weight[3]  # both ends share the same weight

    # mixed_type padding: shorter frames carry -1 virtual atoms, pool_mask == (real>=0)
    assert (real < 0).any()
    np.testing.assert_array_equal(pool, (real >= 0).astype(pool.dtype))

    # per-group fparam: constant across the group's frames, width F
    assert fparam.shape == (nframes, F)
    assert np.allclose(fparam, fparam[0])

    # label repeated across frames
    assert label.shape == (nframes, 1)
    assert np.allclose(label, 32.1)

    # type_map is the element union across all monomers
    tmap = (sys0 / "type_map.raw").read_text().split()
    assert set(tmap) >= {"C", "H", "O", "N"}


def test_valid_split_reuses_training_scaler(tmp_path):
    csv = _write_csv(tmp_path)
    train = PolymerBuilder.from_csv(csv, target="cloud_point", decimal=".")
    res_train = train.write(tmp_path / "train")

    # a second builder standardized with the training scaler
    valid = PolymerBuilder.from_csv(csv, target="cloud_point", decimal=".")
    res_valid = valid.write(tmp_path / "valid", scaler=res_train["scaler"])

    assert res_valid["fparam_dim"] == res_train["fparam_dim"]
    # same scaler -> identical standardized fparam for the same row
    f_train = np.load(
        tmp_path / "train" / res_train["systems"][0] / "set.000" / "fparam.npy"
    )
    f_valid = np.load(
        tmp_path / "valid" / res_valid["systems"][0] / "set.000" / "fparam.npy"
    )
    np.testing.assert_allclose(f_train, f_valid)


def test_add_api_direct(tmp_path):
    builder = PolymerBuilder(target="cloud_point")
    builder.add(
        units={"[CH2][CH](C(=O)NC(C)C)": 0.7, "[CH2][CH](C(=O)O)": 0.3},
        ends=["[C](C)(C)C#N"],
        mol_weight=12000,
        fparam={"pH": 7, "salts": {"KCl": 0.05}},
        target=28.0,
    )
    result = builder.write(tmp_path / "d")
    assert result["n_groups"] == 1
    assert result["fparam_dim"] == 3  # mn_log + pH + salt:KCl
