# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for LMDB dataset support in ``dp eval-desc`` and ``dp embed``."""

import os
from pathlib import (
    Path,
)

import dpdata
import h5py
import lmdb
import msgpack
import numpy as np
import pytest
import torch

from deepmd.entrypoints.embedding import (
    embedding,
)
from deepmd.entrypoints.eval_desc import (
    eval_desc,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)


def _encode_array(arr: np.ndarray) -> dict[str, object]:
    """Encode a NumPy array in the ``deepmd/lmdb`` msgpack format."""
    contiguous = np.ascontiguousarray(arr)
    return {
        "type": contiguous.dtype.str,
        "shape": tuple(contiguous.shape),
        "data": contiguous.tobytes(),
    }


def _make_se_e2_a_params(
    *,
    numb_fparam: int = 0,
    default_fparam: list[float] | None = None,
    numb_aparam: int = 0,
) -> dict:
    """Return a small float64 ``se_e2_a`` model config for fast CPU tests."""
    fitting: dict[str, object] = {
        "neuron": [8, 8],
        "precision": "float64",
        "seed": 7,
    }
    if numb_fparam > 0:
        fitting["numb_fparam"] = numb_fparam
    if default_fparam is not None:
        fitting["default_fparam"] = default_fparam
    if numb_aparam > 0:
        fitting["numb_aparam"] = numb_aparam
    return {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [8, 8],
            "rcut_smth": 0.5,
            "rcut": 4.0,
            "neuron": [4, 8],
            "axis_neuron": 4,
            "precision": "float64",
            "seed": 7,
        },
        "fitting_net": fitting,
    }


def _save_pt_checkpoint(params: dict, path: Path, seed: int = 123) -> str:
    """Create a deterministic PyTorch checkpoint with non-trivial weights."""
    model = get_model(params)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            if param.is_floating_point():
                noise = torch.randn(
                    param.shape,
                    generator=generator,
                    dtype=param.dtype,
                    device="cpu",
                )
                param.copy_((noise * 0.1).to(param.device))
    model.eval()
    wrapper = ModelWrapper(model, model_params=params)
    torch.save({"model": wrapper.state_dict()}, str(path))
    return str(path)


def _make_frame_arrays(
    natoms: int,
    seed: int,
    *,
    pbc: bool = True,
    reverse_types: bool = False,
) -> dict[str, np.ndarray]:
    """Generate one deterministic frame with unsorted atom types."""
    rng = np.random.default_rng(seed)
    # Deliberately interleave atom types so atom-order preservation is tested.
    # Model type_map is ["O", "H"]; when reverse_types is True the dataset uses
    # ["H", "O", "C"], so O is index 1 and H is index 0.
    base_types = np.array(
        [1 if i % 3 != 1 else 0 for i in range(natoms)], dtype=np.int32
    )
    atype = (1 - base_types) if reverse_types else base_types
    coords = rng.uniform(0.5, 4.5, size=(natoms, 3)).astype(np.float64)
    cell = (
        np.diag(np.array([8.0, 8.0, 8.0], dtype=np.float64))
        if pbc
        else np.zeros((3, 3), dtype=np.float64)
    )
    return {
        "coords": coords,
        "cells": cell,
        "atom_types": atype,
    }


def _write_npy_system(
    sys_dir: Path,
    frames: list[dict[str, np.ndarray]],
    type_map: list[str],
    *,
    pbc: bool = True,
    fparam: np.ndarray | None = None,
    aparam: np.ndarray | None = None,
) -> Path:
    """Write a standard ``deepmd/npy`` system matching ``frames``."""
    set_dir = sys_dir / "set.000"
    set_dir.mkdir(parents=True, exist_ok=True)
    atype0 = frames[0]["atom_types"]
    mixed = any(not np.array_equal(f["atom_types"], atype0) for f in frames)
    np.savetxt(sys_dir / "type.raw", atype0, fmt="%d")
    (sys_dir / "type_map.raw").write_text("\n".join(type_map) + "\n", encoding="utf-8")
    if not pbc:
        (sys_dir / "nopbc").write_text("", encoding="utf-8")

    nframes = len(frames)
    coords = np.stack([f["coords"].reshape(-1) for f in frames], axis=0)
    boxes = np.stack([f["cells"].reshape(9) for f in frames], axis=0)
    np.save(set_dir / "coord.npy", coords)
    np.save(set_dir / "box.npy", boxes)
    if mixed:
        real_types = np.stack([f["atom_types"] for f in frames], axis=0)
        np.save(set_dir / "real_atom_types.npy", real_types)
    if fparam is not None:
        np.save(set_dir / "fparam.npy", fparam.reshape(nframes, -1))
    if aparam is not None:
        np.save(set_dir / "aparam.npy", aparam.reshape(nframes, -1))
    return sys_dir


def _write_lmdb_system(
    lmdb_dir: Path,
    frames: list[dict[str, np.ndarray]],
    type_map: list[str],
    *,
    include_cells: bool = True,
    fparam: np.ndarray | None = None,
    aparam: np.ndarray | None = None,
) -> Path:
    """Write a ``deepmd/lmdb`` directory matching ``frames``."""
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    frame_nlocs = [int(f["atom_types"].shape[0]) for f in frames]
    env = lmdb.open(str(lmdb_dir), map_size=10 * 1024 * 1024)
    try:
        with env.begin(write=True) as txn:
            meta: dict[str, object] = {
                "nframes": len(frames),
                "frame_idx_fmt": "012d",
                "type_map": list(type_map),
                "frame_nlocs": frame_nlocs,
            }
            txn.put(b"__metadata__", msgpack.packb(meta, use_bin_type=True))
            for idx, frame in enumerate(frames):
                record: dict[str, object] = {
                    "atom_types": _encode_array(frame["atom_types"].astype(np.int64)),
                    "coords": _encode_array(frame["coords"].astype(np.float64)),
                }
                if include_cells:
                    record["cells"] = _encode_array(frame["cells"].astype(np.float64))
                if fparam is not None:
                    record["fparam"] = _encode_array(
                        np.asarray(fparam[idx], dtype=np.float64)
                    )
                if aparam is not None:
                    record["aparam"] = _encode_array(
                        np.asarray(aparam[idx], dtype=np.float64)
                    )
                txn.put(
                    format(idx, "012d").encode(),
                    msgpack.packb(record, use_bin_type=True),
                )
    finally:
        env.close()
    return lmdb_dir


@pytest.fixture(scope="module")
def plain_model_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Save a small plain ``se_e2_a`` PyTorch model checkpoint."""
    tmp_dir = tmp_path_factory.mktemp("lmdb_eval_plain_model")
    return _save_pt_checkpoint(_make_se_e2_a_params(), tmp_dir / "plain.pt", seed=11)


@pytest.fixture(scope="module")
def conditioned_model_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Save a small ``se_e2_a`` PyTorch model with fparam default and aparam."""
    tmp_dir = tmp_path_factory.mktemp("lmdb_eval_cond_model")
    params = _make_se_e2_a_params(numb_fparam=1, default_fparam=[0.25], numb_aparam=1)
    return _save_pt_checkpoint(params, tmp_dir / "cond.pt", seed=29)


def test_single_nloc_lmdb_matches_npy_eval_desc_and_embed(
    plain_model_path: str, tmp_path: Path
) -> None:
    """Single-nloc LMDB reproduces ``deepmd/npy`` under type remapping and unsorted atoms."""
    frames = [
        _make_frame_arrays(6, seed=1, reverse_types=True),
        _make_frame_arrays(6, seed=2, reverse_types=True),
        _make_frame_arrays(6, seed=3, reverse_types=True),
    ]
    # Include an unused third element in LMDB type_map to verify unused types
    # do not restrict evaluation, while deepmd/npy uses reversed ["H", "O"].
    npy_dir = _write_npy_system(tmp_path / "water_npy", frames, type_map=["H", "O"])
    lmdb_dir = _write_lmdb_system(
        tmp_path / "water.lmdb", frames, type_map=["H", "O", "C"]
    )

    desc_npy_out = tmp_path / "desc_npy"
    desc_lmdb_out = tmp_path / "desc_lmdb"
    eval_desc(
        model=plain_model_path,
        system=str(npy_dir),
        datafile=None,
        output=str(desc_npy_out),
        dtype="fp64",
    )
    eval_desc(
        model=plain_model_path,
        system=str(lmdb_dir),
        datafile=None,
        output=str(desc_lmdb_out),
        dtype="fp64",
    )

    lmdb_desc_file = desc_lmdb_out / "water.lmdb.npy"
    assert lmdb_desc_file.is_file()
    desc_npy = np.load(desc_npy_out / "water_npy.npy")
    desc_lmdb = np.load(lmdb_desc_file)
    assert desc_lmdb.shape == desc_npy.shape
    np.testing.assert_allclose(desc_lmdb, desc_npy, rtol=1e-12, atol=1e-12)

    emb_npy_out = tmp_path / "emb_npy.hdf5"
    emb_lmdb_out = tmp_path / "emb_lmdb.hdf5"
    embedding(
        model=plain_model_path,
        system=str(npy_dir),
        datafile=None,
        output=str(emb_npy_out),
        dtype="fp64",
    )
    embedding(
        model=plain_model_path,
        system=str(lmdb_dir),
        datafile=None,
        output=str(emb_lmdb_out),
        dtype="fp64",
    )

    with h5py.File(emb_npy_out, "r") as f_npy, h5py.File(emb_lmdb_out, "r") as f_lmdb:
        assert list(f_lmdb.keys()) == ["water.lmdb"]
        g_npy = f_npy["water_npy"]
        g_lmdb = f_lmdb["water.lmdb"]
        assert g_lmdb.attrs["nframes"] == len(frames)
        assert g_lmdb.attrs["system"] == str(lmdb_dir)
        np.testing.assert_array_equal(g_lmdb["atom_types"][:], g_npy["atom_types"][:])
        for key in ("descriptor", "atomic_feature", "structural_feature"):
            np.testing.assert_allclose(
                g_lmdb[key][:], g_npy[key][:], rtol=1e-12, atol=1e-12, err_msg=key
            )


def test_mixed_nloc_lmdb_deterministic_order_and_parity(
    plain_model_path: str, tmp_path: Path
) -> None:
    """Mixed-nloc LMDB splits deterministically by ascending nloc and storage order."""
    # Deliberately interleave nloc=9, 3, 6 frames inside one LMDB.
    f_n9_0 = _make_frame_arrays(9, seed=10)
    f_n3_0 = _make_frame_arrays(3, seed=11)
    f_n6_0 = _make_frame_arrays(6, seed=12)
    f_n3_1 = _make_frame_arrays(3, seed=13)
    f_n9_1 = _make_frame_arrays(9, seed=14)
    f_n6_1 = _make_frame_arrays(6, seed=15)
    interleaved = [f_n9_0, f_n3_0, f_n6_0, f_n3_1, f_n9_1, f_n6_1]

    grouped_frames = {
        3: [f_n3_0, f_n3_1],
        6: [f_n6_0, f_n6_1],
        9: [f_n9_0, f_n9_1],
    }

    npy_root = tmp_path / "npy_groups"
    for nloc, group_frames in grouped_frames.items():
        _write_npy_system(npy_root / f"sys_nloc{nloc}", group_frames, ["O", "H"])

    lmdb_dir = _write_lmdb_system(tmp_path / "mixed.lmdb", interleaved, ["O", "H"])

    # eval-desc parity across all nloc groups
    desc_npy_out = tmp_path / "desc_npy"
    desc_lmdb_out = tmp_path / "desc_lmdb"
    eval_desc(
        model=plain_model_path,
        system=str(npy_root),
        datafile=None,
        output=str(desc_npy_out),
        dtype="fp64",
    )
    eval_desc(
        model=plain_model_path,
        system=str(lmdb_dir),
        datafile=None,
        output=str(desc_lmdb_out),
        dtype="fp64",
    )

    produced_files = sorted(p.name for p in desc_lmdb_out.iterdir())
    assert produced_files == [
        "mixed.lmdb_nloc3.npy",
        "mixed.lmdb_nloc6.npy",
        "mixed.lmdb_nloc9.npy",
    ]
    for nloc in (3, 6, 9):
        actual = np.load(desc_lmdb_out / f"mixed.lmdb_nloc{nloc}.npy")
        expected = np.load(desc_npy_out / f"sys_nloc{nloc}.npy")
        assert actual.shape[0] == 2
        assert actual.shape[1] == nloc
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    # embed parity across all nloc groups
    emb_npy_out = tmp_path / "emb_npy.hdf5"
    emb_lmdb_out = tmp_path / "emb_lmdb.hdf5"
    embedding(
        model=plain_model_path,
        system=str(npy_root),
        datafile=None,
        output=str(emb_npy_out),
        dtype="fp64",
    )
    embedding(
        model=plain_model_path,
        system=str(lmdb_dir),
        datafile=None,
        output=str(emb_lmdb_out),
        dtype="fp64",
    )

    with h5py.File(emb_npy_out, "r") as f_npy, h5py.File(emb_lmdb_out, "r") as f_lmdb:
        assert set(f_lmdb.keys()) == {
            "mixed.lmdb_nloc3",
            "mixed.lmdb_nloc6",
            "mixed.lmdb_nloc9",
        }
        for nloc in (3, 6, 9):
            g_lmdb = f_lmdb[f"mixed.lmdb_nloc{nloc}"]
            g_npy = f_npy[f"sys_nloc{nloc}"]
            assert g_lmdb.attrs["nframes"] == 2
            assert g_lmdb.attrs["system"] == f"{lmdb_dir} [nloc={nloc}]"
            np.testing.assert_array_equal(
                g_lmdb["atom_types"][:], g_npy["atom_types"][:]
            )
            for key in ("descriptor", "atomic_feature", "structural_feature"):
                np.testing.assert_allclose(
                    g_lmdb[key][:],
                    g_npy[key][:],
                    rtol=1e-12,
                    atol=1e-12,
                    err_msg=f"nloc={nloc} {key}",
                )


def test_recursive_discovery_collisions_and_datafile(
    plain_model_path: str, tmp_path: Path
) -> None:
    """Parent discovery finds NPY and LMDB datasets, deduplicates names, and supports datafile."""
    root = tmp_path / "dataset_root"
    frames_a = [_make_frame_arrays(4, seed=21)]
    frames_b = [_make_frame_arrays(5, seed=22)]
    frames_c = [_make_frame_arrays(6, seed=23)]
    frames_d = [_make_frame_arrays(4, seed=24)]

    lmdb_a = _write_lmdb_system(root / "a" / "shared.lmdb", frames_a, ["O", "H"])
    lmdb_b = _write_lmdb_system(root / "b" / "shared.lmdb", frames_b, ["O", "H"])
    # LMDB directory without the .lmdb suffix (detected via data.mdb)
    lmdb_plain = _write_lmdb_system(root / "c" / "plain_lmdb", frames_c, ["O", "H"])
    _write_npy_system(root / "d" / "npy_sys", frames_d, ["O", "H"])

    desc_out = tmp_path / "desc_recursive"
    eval_desc(
        model=plain_model_path,
        system=str(root),
        datafile=None,
        output=str(desc_out),
        dtype="fp64",
    )
    assert {p.name for p in desc_out.iterdir()} == {
        "shared.lmdb.npy",
        "shared.lmdb_1.npy",
        "plain_lmdb.npy",
        "npy_sys.npy",
    }

    emb_out = tmp_path / "emb_recursive.hdf5"
    embedding(
        model=plain_model_path,
        system=str(root),
        datafile=None,
        output=str(emb_out),
        dtype="fp64",
    )
    with h5py.File(emb_out, "r") as h5f:
        assert set(h5f.keys()) == {
            "shared.lmdb",
            "shared.lmdb_1",
            "plain_lmdb",
            "npy_sys",
        }
        systems = {h5f[k].attrs["system"] for k in h5f}
        assert str(lmdb_a) in systems
        assert str(lmdb_b) in systems
        assert str(lmdb_plain) in systems

    # Datafile listing LMDB paths directly (including trailing separator)
    datafile = tmp_path / "systems.txt"
    datafile.write_text(f"{lmdb_a}{os.sep}\n\n{lmdb_plain}\n", encoding="utf-8")
    desc_df_out = tmp_path / "desc_datafile"
    eval_desc(
        model=plain_model_path,
        system=None,
        datafile=str(datafile),
        output=str(desc_df_out),
        dtype="fp64",
    )
    assert {p.name for p in desc_df_out.iterdir()} == {
        "shared.lmdb.npy",
        "plain_lmdb.npy",
    }

    emb_df_out = tmp_path / "emb_datafile.hdf5"
    embedding(
        model=plain_model_path,
        system=None,
        datafile=str(datafile),
        output=str(emb_df_out),
        dtype="fp64",
    )
    with h5py.File(emb_df_out, "r") as h5f:
        assert set(h5f.keys()) == {"shared.lmdb", "plain_lmdb"}


def test_nopbc_lmdb_matches_npy(plain_model_path: str, tmp_path: Path) -> None:
    """Non-periodic LMDB matches ``deepmd/npy`` with ``nopbc``."""
    frames = [
        _make_frame_arrays(5, seed=31, pbc=False),
        _make_frame_arrays(5, seed=32, pbc=False),
    ]
    npy_dir = _write_npy_system(tmp_path / "mol_npy", frames, ["O", "H"], pbc=False)
    lmdb_dir = _write_lmdb_system(
        tmp_path / "mol.lmdb", frames, ["O", "H"], include_cells=False
    )

    desc_npy_out = tmp_path / "desc_npy"
    desc_lmdb_out = tmp_path / "desc_lmdb"
    eval_desc(
        model=plain_model_path,
        system=str(npy_dir),
        datafile=None,
        output=str(desc_npy_out),
        dtype="fp64",
    )
    eval_desc(
        model=plain_model_path,
        system=str(lmdb_dir),
        datafile=None,
        output=str(desc_lmdb_out),
        dtype="fp64",
    )
    np.testing.assert_allclose(
        np.load(desc_lmdb_out / "mol.lmdb.npy"),
        np.load(desc_npy_out / "mol_npy.npy"),
        rtol=1e-12,
        atol=1e-12,
    )

    emb_npy_out = tmp_path / "emb_npy.hdf5"
    emb_lmdb_out = tmp_path / "emb_lmdb.hdf5"
    embedding(
        model=plain_model_path,
        system=str(npy_dir),
        datafile=None,
        output=str(emb_npy_out),
        dtype="fp64",
    )
    embedding(
        model=plain_model_path,
        system=str(lmdb_dir),
        datafile=None,
        output=str(emb_lmdb_out),
        dtype="fp64",
    )
    with h5py.File(emb_npy_out, "r") as f_npy, h5py.File(emb_lmdb_out, "r") as f_lmdb:
        for key in ("descriptor", "atomic_feature", "structural_feature"):
            np.testing.assert_allclose(
                f_lmdb["mol.lmdb"][key][:],
                f_npy["mol_npy"][key][:],
                rtol=1e-12,
                atol=1e-12,
                err_msg=key,
            )


def test_lmdb_conditioning_fparam_and_aparam_match_npy(
    conditioned_model_path: str, tmp_path: Path
) -> None:
    """LMDB supplies ``fparam`` / ``aparam`` or falls back to model defaults in ``dp embed``."""
    frames = [
        _make_frame_arrays(4, seed=41),
        _make_frame_arrays(4, seed=42),
    ]
    fparam = np.array([[0.8], [-0.4]], dtype=np.float64)
    aparam = np.linspace(-0.5, 0.5, 2 * 4, dtype=np.float64).reshape(2, 4, 1)

    npy_with = _write_npy_system(
        tmp_path / "with_fparam_npy", frames, ["O", "H"], fparam=fparam, aparam=aparam
    )
    lmdb_with = _write_lmdb_system(
        tmp_path / "with_fparam.lmdb", frames, ["O", "H"], fparam=fparam, aparam=aparam
    )
    npy_default = _write_npy_system(
        tmp_path / "default_fparam_npy", frames, ["O", "H"], aparam=aparam
    )
    lmdb_default = _write_lmdb_system(
        tmp_path / "default_fparam.lmdb", frames, ["O", "H"], aparam=aparam
    )

    emb_npy_out = tmp_path / "cond_npy.hdf5"
    emb_lmdb_out = tmp_path / "cond_lmdb.hdf5"
    df_npy = tmp_path / "npy.txt"
    df_lmdb = tmp_path / "lmdb.txt"
    df_npy.write_text(f"{npy_with}\n{npy_default}\n", encoding="utf-8")
    df_lmdb.write_text(f"{lmdb_with}\n{lmdb_default}\n", encoding="utf-8")

    embedding(
        model=conditioned_model_path,
        system=None,
        datafile=str(df_npy),
        output=str(emb_npy_out),
        dtype="fp64",
    )
    embedding(
        model=conditioned_model_path,
        system=None,
        datafile=str(df_lmdb),
        output=str(emb_lmdb_out),
        dtype="fp64",
    )

    with h5py.File(emb_npy_out, "r") as f_npy, h5py.File(emb_lmdb_out, "r") as f_lmdb:
        for npy_name, lmdb_name in (
            ("with_fparam_npy", "with_fparam.lmdb"),
            ("default_fparam_npy", "default_fparam.lmdb"),
        ):
            for key in ("descriptor", "atomic_feature", "structural_feature"):
                np.testing.assert_allclose(
                    f_lmdb[lmdb_name][key][:],
                    f_npy[npy_name][key][:],
                    rtol=1e-12,
                    atol=1e-12,
                    err_msg=f"{lmdb_name} {key}",
                )
        # Verify explicit fparam actually changes the fitting output relative to default.
        assert (
            np.abs(
                f_lmdb["with_fparam.lmdb"]["atomic_feature"][:]
                - f_lmdb["default_fparam.lmdb"]["atomic_feature"][:]
            ).max()
            > 1e-4
        )

    # Also verify eval_desc parity with explicit fparam/aparam and default fallback.
    desc_npy_out = tmp_path / "desc_cond_npy"
    desc_lmdb_out = tmp_path / "desc_cond_lmdb"
    eval_desc(
        model=conditioned_model_path,
        system=None,
        datafile=str(df_npy),
        output=str(desc_npy_out),
        dtype="fp64",
    )
    eval_desc(
        model=conditioned_model_path,
        system=None,
        datafile=str(df_lmdb),
        output=str(desc_lmdb_out),
        dtype="fp64",
    )
    for npy_name, lmdb_name in (
        ("with_fparam_npy", "with_fparam.lmdb"),
        ("default_fparam_npy", "default_fparam.lmdb"),
    ):
        np.testing.assert_allclose(
            np.load(desc_lmdb_out / f"{lmdb_name}.npy"),
            np.load(desc_npy_out / f"{npy_name}.npy"),
            rtol=1e-12,
            atol=1e-12,
        )


def test_dpdata_exported_lmdb_parity(plain_model_path: str, tmp_path: Path) -> None:
    """LMDB datasets exported by ``dpdata`` work end-to-end in ``eval-desc`` and ``embed``."""
    f1 = _make_frame_arrays(3, seed=51)
    f2 = _make_frame_arrays(3, seed=52)
    f3 = _make_frame_arrays(6, seed=53)

    s1 = dpdata.System(
        data={
            "atom_names": ["O", "H"],
            "atom_numbs": [
                int(np.sum(f1["atom_types"] == 0)),
                int(np.sum(f1["atom_types"] == 1)),
            ],
            "atom_types": f1["atom_types"],
            "coords": np.stack([f1["coords"], f2["coords"]], axis=0),
            "cells": np.stack([f1["cells"], f2["cells"]], axis=0),
            "orig": np.zeros(3, dtype=np.float64),
        }
    )
    s2 = dpdata.System(
        data={
            "atom_names": ["O", "H"],
            "atom_numbs": [
                int(np.sum(f3["atom_types"] == 0)),
                int(np.sum(f3["atom_types"] == 1)),
            ],
            "atom_types": f3["atom_types"],
            "coords": f3["coords"][None, :, :],
            "cells": f3["cells"][None, :, :],
            "orig": np.zeros(3, dtype=np.float64),
        }
    )

    npy_s1 = tmp_path / "dpdata_s1_npy"
    npy_s2 = tmp_path / "dpdata_s2_npy"
    s1.to("deepmd/npy", str(npy_s1))
    s2.to("deepmd/npy", str(npy_s2))

    ms_lmdb = tmp_path / "dpdata_multi.lmdb"
    dpdata.MultiSystems(s1, s2).to("deepmd/lmdb", str(ms_lmdb))

    desc_lmdb_out = tmp_path / "desc_dpdata_lmdb"
    desc_npy_out = tmp_path / "desc_dpdata_npy"
    eval_desc(
        model=plain_model_path,
        system=str(ms_lmdb),
        datafile=None,
        output=str(desc_lmdb_out),
        dtype="fp64",
    )
    df_npy = tmp_path / "dpdata_npy.txt"
    df_npy.write_text(f"{npy_s1}\n{npy_s2}\n", encoding="utf-8")
    eval_desc(
        model=plain_model_path,
        system=None,
        datafile=str(df_npy),
        output=str(desc_npy_out),
        dtype="fp64",
    )

    np.testing.assert_allclose(
        np.load(desc_lmdb_out / "dpdata_multi.lmdb_nloc3.npy"),
        np.load(desc_npy_out / "dpdata_s1_npy.npy"),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.load(desc_lmdb_out / "dpdata_multi.lmdb_nloc6.npy"),
        np.load(desc_npy_out / "dpdata_s2_npy.npy"),
        rtol=1e-12,
        atol=1e-12,
    )

    emb_lmdb_out = tmp_path / "emb_dpdata_lmdb.hdf5"
    emb_npy_out = tmp_path / "emb_dpdata_npy.hdf5"
    embedding(
        model=plain_model_path,
        system=str(ms_lmdb),
        datafile=None,
        output=str(emb_lmdb_out),
        dtype="fp64",
    )
    embedding(
        model=plain_model_path,
        system=None,
        datafile=str(df_npy),
        output=str(emb_npy_out),
        dtype="fp64",
    )
    with h5py.File(emb_npy_out, "r") as f_npy, h5py.File(emb_lmdb_out, "r") as f_lmdb:
        for lmdb_grp, npy_grp in (
            ("dpdata_multi.lmdb_nloc3", "dpdata_s1_npy"),
            ("dpdata_multi.lmdb_nloc6", "dpdata_s2_npy"),
        ):
            for key in ("descriptor", "atomic_feature", "structural_feature"):
                np.testing.assert_allclose(
                    f_lmdb[lmdb_grp][key][:],
                    f_npy[npy_grp][key][:],
                    rtol=1e-12,
                    atol=1e-12,
                    err_msg=f"{lmdb_grp} {key}",
                )


def test_empty_lmdb_does_not_crash(plain_model_path: str, tmp_path: Path) -> None:
    """An LMDB with zero frames is skipped cleanly without raising IndexError."""
    empty_lmdb = _write_lmdb_system(tmp_path / "empty.lmdb", [], ["O", "H"])

    desc_out = tmp_path / "desc_empty"
    eval_desc(
        model=plain_model_path,
        system=str(empty_lmdb),
        datafile=None,
        output=str(desc_out),
        dtype="fp64",
    )
    assert list(desc_out.iterdir()) == []

    emb_out = tmp_path / "emb_empty.hdf5"
    embedding(
        model=plain_model_path,
        system=str(empty_lmdb),
        datafile=None,
        output=str(emb_out),
        dtype="fp64",
    )
    with h5py.File(emb_out, "r") as h5f:
        assert list(h5f.keys()) == []


def test_mixed_nloc_lmdb_collisions_and_source_unmodified(
    plain_model_path: str, tmp_path: Path
) -> None:
    """Colliding mixed-nloc LMDB basenames deduplicate cleanly and never mutate inputs."""
    root = tmp_path / "dup_mixed"
    frames_a = [_make_frame_arrays(3, seed=61), _make_frame_arrays(5, seed=62)]
    frames_b = [_make_frame_arrays(3, seed=63), _make_frame_arrays(5, seed=64)]
    lmdb_a = _write_lmdb_system(root / "a" / "mix.lmdb", frames_a, ["O", "H"])
    lmdb_b = _write_lmdb_system(root / "b" / "mix.lmdb", frames_b, ["O", "H"])
    before_a = sorted(p.name for p in lmdb_a.iterdir())
    before_b = sorted(p.name for p in lmdb_b.iterdir())

    desc_out = tmp_path / "desc_dup_mixed"
    eval_desc(
        model=plain_model_path,
        system=str(root),
        datafile=None,
        output=str(desc_out),
        dtype="fp64",
    )
    assert {p.name for p in desc_out.iterdir()} == {
        "mix.lmdb_nloc3.npy",
        "mix.lmdb_nloc5.npy",
        "mix.lmdb_nloc3_1.npy",
        "mix.lmdb_nloc5_1.npy",
    }

    emb_out = tmp_path / "emb_dup_mixed.hdf5"
    embedding(
        model=plain_model_path,
        system=str(root),
        datafile=None,
        output=str(emb_out),
        dtype="fp64",
    )
    with h5py.File(emb_out, "r") as h5f:
        assert set(h5f.keys()) == {
            "mix.lmdb_nloc3",
            "mix.lmdb_nloc5",
            "mix.lmdb_nloc3_1",
            "mix.lmdb_nloc5_1",
        }
        systems = {h5f[k].attrs["system"] for k in h5f}
        assert systems == {
            f"{lmdb_a} [nloc=3]",
            f"{lmdb_a} [nloc=5]",
            f"{lmdb_b} [nloc=3]",
            f"{lmdb_b} [nloc=5]",
        }

    # Ensure no files were written inside the LMDB source directories.
    assert sorted(p.name for p in lmdb_a.iterdir()) == before_a
    assert sorted(p.name for p in lmdb_b.iterdir()) == before_b


def test_lmdb_charge_spin_pt_expt_matches_npy(tmp_path: Path) -> None:
    """LMDB ``charge_spin`` conditioning and default fallback match ``deepmd/npy`` on pt_expt."""
    import copy

    from deepmd.pt_expt.model.get_model import get_model as pt_expt_get_model
    from deepmd.pt_expt.train.wrapper import ModelWrapper as PtExptModelWrapper
    from deepmd.utils.argcheck import (
        normalize,
    )
    from deepmd.utils.compat import (
        update_deepmd_input,
    )

    config = {
        "model": {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "dpa4c",
                "rcut": 4.0,
                "channels": 8,
                "lmax": 2,
                "n_radial": 8,
                "precision": "float64",
                "seed": 17,
                "add_chg_spin_ebd": True,
                "default_chg_spin": [0.0, 1.0],
            },
            "fitting_net": {
                "neuron": [8],
                "precision": "float64",
                "seed": 17,
            },
        },
        "training": {"training_data": {"systems": ["dummy"]}, "numb_steps": 1},
        "loss": {"type": "ener"},
        "learning_rate": {"type": "exp", "start_lr": 1e-3},
    }
    model_params = normalize(update_deepmd_input(config, warning=False))["model"]
    model = pt_expt_get_model(copy.deepcopy(model_params)).eval()
    generator = torch.Generator(device="cpu").manual_seed(37)
    with torch.no_grad():
        for param in model.parameters():
            if param.is_floating_point():
                noise = torch.randn(
                    param.shape,
                    generator=generator,
                    dtype=param.dtype,
                    device="cpu",
                )
                param.add_((noise * 0.2).to(param.device))
    ckpt_path = str(tmp_path / "dpa4c_chg.pt")
    wrapper = PtExptModelWrapper(model, model_params=copy.deepcopy(model_params))
    torch.save({"model": wrapper.state_dict()}, ckpt_path)

    frames = [_make_frame_arrays(4, seed=71), _make_frame_arrays(4, seed=72)]
    charge_spin = np.array([[1.0, 2.0], [-1.0, 2.0]], dtype=np.float64)

    # Write NPY with and without charge_spin
    npy_with = _write_npy_system(tmp_path / "chg_npy", frames, ["O", "H"])
    np.save(npy_with / "set.000" / "charge_spin.npy", charge_spin)
    npy_default = _write_npy_system(tmp_path / "default_chg_npy", frames, ["O", "H"])

    # Write LMDB with and without charge_spin
    lmdb_with = tmp_path / "chg.lmdb"
    lmdb_with.mkdir()
    env = lmdb.open(str(lmdb_with), map_size=10 * 1024 * 1024)
    try:
        with env.begin(write=True) as txn:
            meta = {
                "nframes": len(frames),
                "frame_idx_fmt": "012d",
                "type_map": ["O", "H"],
                "frame_nlocs": [4, 4],
            }
            txn.put(b"__metadata__", msgpack.packb(meta, use_bin_type=True))
            for idx, frame in enumerate(frames):
                rec = {
                    "atom_types": _encode_array(frame["atom_types"].astype(np.int64)),
                    "coords": _encode_array(frame["coords"].astype(np.float64)),
                    "cells": _encode_array(frame["cells"].astype(np.float64)),
                    "charge_spin": _encode_array(charge_spin[idx]),
                }
                txn.put(
                    format(idx, "012d").encode(), msgpack.packb(rec, use_bin_type=True)
                )
    finally:
        env.close()
    lmdb_default = _write_lmdb_system(tmp_path / "default_chg.lmdb", frames, ["O", "H"])

    df_npy = tmp_path / "chg_npy.txt"
    df_lmdb = tmp_path / "chg_lmdb.txt"
    df_npy.write_text(f"{npy_with}\n{npy_default}\n", encoding="utf-8")
    df_lmdb.write_text(f"{lmdb_with}\n{lmdb_default}\n", encoding="utf-8")

    emb_npy_out = tmp_path / "chg_npy.hdf5"
    emb_lmdb_out = tmp_path / "chg_lmdb.hdf5"
    embedding(
        model=ckpt_path,
        system=None,
        datafile=str(df_npy),
        output=str(emb_npy_out),
        dtype="fp64",
    )
    embedding(
        model=ckpt_path,
        system=None,
        datafile=str(df_lmdb),
        output=str(emb_lmdb_out),
        dtype="fp64",
    )

    with h5py.File(emb_npy_out, "r") as f_npy, h5py.File(emb_lmdb_out, "r") as f_lmdb:
        for npy_name, lmdb_name in (
            ("chg_npy", "chg.lmdb"),
            ("default_chg_npy", "default_chg.lmdb"),
        ):
            for key in ("descriptor", "atomic_feature", "structural_feature"):
                np.testing.assert_allclose(
                    f_lmdb[lmdb_name][key][:],
                    f_npy[npy_name][key][:],
                    rtol=1e-12,
                    atol=1e-12,
                    err_msg=f"{lmdb_name} {key}",
                )
        assert (
            np.abs(
                f_lmdb["chg.lmdb"]["descriptor"][:]
                - f_lmdb["default_chg.lmdb"]["descriptor"][:]
            ).max()
            > 1e-4
        )
