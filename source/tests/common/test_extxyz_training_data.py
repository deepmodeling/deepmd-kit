# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for labeled extended-XYZ training and validation inputs."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import (
    Any,
)

import dpdata
import numpy as np
import pytest
from scipy.constants import (
    electron_volt,
)

from deepmd.utils.data_conversion import (
    expand_extxyz_cache,
    materialize_extxyz,
    normalize_extxyz_training_data,
)

_DEFAULT_FORCES = object()
_ENERGY_FORCE_LOSS = {
    "type": "ener",
    "start_pref_e": 1.0,
    "limit_pref_e": 1.0,
    "start_pref_f": 1.0,
    "limit_pref_f": 1.0,
    "start_pref_v": 0.0,
    "limit_pref_v": 0.0,
}


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPMD_EXTXYZ_CACHE", str(tmp_path / "extxyz-cache"))


def _frame(
    *,
    species: tuple[str, ...] = ("H", "O"),
    positions: np.ndarray | None = None,
    forces: Any = _DEFAULT_FORCES,
    energy: float | None = -1.25,
    cell: np.ndarray | None = None,
    pbc: bool = True,
    tensor_name: str | None = None,
    tensor: np.ndarray | None = None,
    energy_unit: str | None = None,
    force_unit: str | None = None,
    stress_unit: str | None = None,
) -> str:
    natoms = len(species)
    if positions is None:
        positions = np.arange(natoms * 3, dtype=float).reshape(natoms, 3) / 10.0
    if forces is _DEFAULT_FORCES:
        forces = -np.arange(1, natoms * 3 + 1, dtype=float).reshape(natoms, 3) / 10.0
    if cell is None:
        cell = np.diag([2.0, 3.0, 4.0])

    properties = "Properties=species:S:1:pos:R:3"
    if forces is not None:
        properties += ":forces:R:3"
    fields = [
        f'Lattice="{" ".join(str(value) for value in cell.reshape(-1))}"',
        properties,
        f'pbc="{"T T T" if pbc else "F F F"}"',
    ]
    if energy is not None:
        fields.append(f"energy={energy}")
    if tensor_name is not None and tensor is not None:
        values = " ".join(str(value) for value in np.asarray(tensor).reshape(-1))
        fields.append(f'{tensor_name}="{values}"')
    if energy_unit is not None:
        fields.append(f"energy_unit={energy_unit}")
    if force_unit is not None:
        fields.append(f"force_unit={force_unit}")
    if stress_unit is not None:
        fields.append(f"stress_unit={stress_unit}")

    atom_lines = []
    for index, name in enumerate(species):
        values = [*positions[index]]
        if forces is not None:
            values.extend(forces[index])
        atom_lines.append(f"{name} " + " ".join(str(value) for value in values))
    return f"{natoms}\n{' '.join(fields)}\n" + "\n".join(atom_lines) + "\n"


def _write(path: Path, *frames: str) -> Path:
    path.write_text("".join(frames), encoding="utf-8")
    return path


def _converted(source: Path) -> tuple[Path, dict[str, Any]]:
    cache, manifest = materialize_extxyz(source)
    systems = expand_extxyz_cache(cache)
    assert systems is not None
    assert len(systems) == 1
    return Path(systems[0]), manifest


def _load_system(path: Path) -> dpdata.LabeledSystem:
    return dpdata.LabeledSystem(str(path), fmt="deepmd/npy")


def _partial_config(
    training_systems: str | list[str],
    validation_systems: str | list[str] | None = None,
    *,
    loss: dict[str, Any] | None = None,
    batch_size: str | int | list[int] = "auto",
) -> dict[str, Any]:
    training: dict[str, Any] = {
        "training_data": {
            "systems": training_systems,
            "batch_size": batch_size,
        }
    }
    if validation_systems is not None:
        training["validation_data"] = {
            "systems": validation_systems,
            "batch_size": batch_size,
        }
    return {
        "loss": copy.deepcopy(loss or _ENERGY_FORCE_LOSS),
        "training": training,
    }


def _energies(paths: list[str]) -> list[float]:
    return [float(_load_system(Path(path)).data["energies"][0]) for path in paths]


def test_periodic_energy_force_cell_and_pbc_round_trip(tmp_path: Path) -> None:
    positions = np.array([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]])
    forces = np.array([[0.4, 0.5, 0.6], [-0.4, -0.5, -0.6]])
    cell = np.array([[2.0, 0.1, 0.2], [0.0, 3.0, 0.3], [0.0, 0.0, 4.0]])
    source = _write(
        tmp_path / "periodic.extxyz",
        _frame(positions=positions, forces=forces, energy=-3.5, cell=cell),
        _frame(positions=positions + 0.25, forces=forces * 2, energy=-2.5, cell=cell),
    )

    system_path, manifest = _converted(source)
    system = _load_system(system_path)

    np.testing.assert_allclose(system.data["energies"], [-3.5, -2.5])
    np.testing.assert_allclose(system.data["coords"], [positions, positions + 0.25])
    np.testing.assert_allclose(system.data["forces"], [forces, forces * 2])
    np.testing.assert_allclose(system.data["cells"], [cell, cell])
    assert not system.nopbc
    assert manifest["systems"][0]["labels"] == ["energy", "force"]


def test_nonperiodic_pbc_survives_conversion(tmp_path: Path) -> None:
    source = _write(tmp_path / "nonperiodic.extxyz", _frame(pbc=False))

    system_path, manifest = _converted(source)
    system = _load_system(system_path)

    assert system.nopbc
    assert manifest["systems"][0]["nopbc"]


def test_partially_periodic_pbc_is_rejected(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "partial-pbc.extxyz",
        _frame().replace('pbc="T T T"', 'pbc="T T F"'),
    )

    with pytest.raises(ValueError, match="partially periodic"):
        materialize_extxyz(source)


def test_explicit_virial_round_trip(tmp_path: Path) -> None:
    virial = np.arange(1.0, 10.0).reshape(3, 3)
    source = _write(
        tmp_path / "virial.xyz",
        _frame(tensor_name="virial", tensor=virial),
    )

    system_path, manifest = _converted(source)
    system = _load_system(system_path)

    np.testing.assert_allclose(system.data["virials"], [virial])
    assert manifest["systems"][0]["labels"] == ["energy", "force", "virial"]


@pytest.mark.parametrize(
    ("stress", "expected_matrix"),
    [
        (
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            np.array([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]]),
        ),
        (
            np.arange(1.0, 10.0),
            np.arange(1.0, 10.0).reshape(3, 3),
        ),
    ],
)
def test_ase_stress_to_virial_sign_units_and_ordering(
    tmp_path: Path, stress: np.ndarray, expected_matrix: np.ndarray
) -> None:
    cell = np.diag([2.0, 3.0, 4.0])
    source = _write(
        tmp_path / f"stress-{stress.size}.extxyz",
        _frame(
            cell=cell,
            tensor_name="stress",
            tensor=stress,
            stress_unit="GPa",
        ),
    )

    system_path, _ = _converted(source)
    virial = _load_system(system_path).data["virials"][0]
    gpa_to_ev_per_angstrom3 = 1e9 * 1e-30 / electron_volt
    expected = -abs(np.linalg.det(cell)) * expected_matrix * gpa_to_ev_per_angstrom3
    np.testing.assert_allclose(virial, expected)


def test_energy_and_force_units_are_converted(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "units.extxyz",
        _frame(
            energy=1.0,
            forces=np.ones((2, 3)),
            energy_unit="hartree",
            force_unit="hartree/bohr",
        ),
    )

    system_path, _ = _converted(source)
    system = _load_system(system_path)

    # Independent CODATA values in DeePMD's eV and eV/angstrom units.
    np.testing.assert_allclose(system.data["energies"], [27.211386245988], rtol=1e-10)
    np.testing.assert_allclose(system.data["forces"], 51.4220674763, rtol=1e-10)


def test_multiple_relative_and_absolute_inputs_preserve_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _write(tmp_path / "first.extxyz", _frame(energy=-1.0))
    second = _write(tmp_path / "second.xyz", _frame(energy=-2.0))
    validation = _write(tmp_path / "validation.extxyz", _frame(energy=-3.0))
    monkeypatch.chdir(tmp_path)
    original = _partial_config(
        [first.name, str(second.resolve())],
        validation.name,
    )
    snapshot = copy.deepcopy(original)

    normalized = normalize_extxyz_training_data(original)

    assert original == snapshot
    from deepmd.utils.data_system import (
        process_systems,
    )

    training_paths = process_systems(normalized["training"]["training_data"]["systems"])
    validation_paths = process_systems(
        normalized["training"]["validation_data"]["systems"]
    )
    assert _energies(training_paths) == [-1.0, -2.0]
    assert _energies(validation_paths) == [-3.0]


def test_existing_numpy_raw_hdf5_parent_lmdb_and_mixed_inputs_are_unchanged(
    tmp_path: Path,
) -> None:
    from deepmd.utils.data_system import (
        process_systems,
    )
    from deepmd.utils.path import (
        DPH5Path,
    )

    source = _write(tmp_path / "new.extxyz", _frame(energy=-2.0))
    generated, _ = _converted(source)
    existing = tmp_path / "existing"
    shutil.copytree(generated, existing)
    parent = tmp_path / "parent"
    parent.mkdir()
    nested = parent / "nested"
    shutil.copytree(generated, nested)
    _write(parent / "not-discovered.xyz", _frame(energy=-99.0))
    system = _load_system(generated)
    raw = tmp_path / "raw-system"
    system.to("deepmd/raw", str(raw))
    hdf5 = tmp_path / "system.hdf5"
    system.to("deepmd/hdf5", str(hdf5))

    assert process_systems(str(existing)) == [str(existing)]
    assert process_systems(str(raw)) == [str(raw)]
    assert process_systems(str(hdf5)) == [f"{hdf5}#/"]
    assert process_systems(str(parent)) == [str(nested)]
    assert process_systems(str(tmp_path / "dataset.lmdb")) == [
        str(tmp_path / "dataset.lmdb")
    ]
    mixed = process_systems([str(existing), str(source)])
    assert mixed[0] == str(existing)
    assert _energies(mixed) == [-2.0, -2.0]

    # DPH5Path caches read handles globally; release the synthetic fixture on
    # Windows so pytest can remove its temporary directory.
    DPH5Path._load_h5py(str(hdf5), "r").close()
    DPH5Path._load_h5py.cache_clear()
    DPH5Path._file_keys.cache_clear()


def test_rglob_patterns_do_not_discover_or_filter_explicit_extxyz(
    tmp_path: Path,
) -> None:
    from deepmd.utils.data_system import (
        process_systems,
    )

    source = _write(tmp_path / "explicit.extxyz", _frame(energy=-4.0))
    assert process_systems(str(tmp_path), patterns=["*.xyz", "*.extxyz"]) == []
    explicit = process_systems(str(source), patterns=["does-not-match"])
    assert _energies(explicit) == [-4.0]


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (_frame(energy=None), "energy"),
        (_frame(forces=None), "force"),
    ],
)
def test_missing_required_energy_or_forces_fails_clearly(
    tmp_path: Path, frame: str, message: str
) -> None:
    source = _write(tmp_path / f"missing-{message}.extxyz", frame)
    with pytest.raises(ValueError, match=message):
        materialize_extxyz(source)


def test_coordinate_only_xyz_fails_clearly(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "coordinates.xyz",
        "2\ncoordinate-only XYZ\nH 0 0 0\nO 1 1 1\n",
    )
    with pytest.raises(ValueError, match=r"species.*positions.*energy.*forces"):
        materialize_extxyz(source)


def test_missing_virial_fails_when_virial_loss_is_enabled(tmp_path: Path) -> None:
    source = _write(tmp_path / "no-virial.extxyz", _frame())
    loss = copy.deepcopy(_ENERGY_FORCE_LOSS)
    loss["start_pref_v"] = 1.0
    loss["limit_pref_v"] = 1.0

    with pytest.raises(ValueError, match="missing required virial"):
        normalize_extxyz_training_data(_partial_config(str(source), loss=loss))


def test_heterogeneous_frames_split_deterministically(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "heterogeneous.extxyz",
        _frame(species=("H", "H"), energy=-1.0),
        _frame(species=("H", "O", "H"), energy=-2.0),
        _frame(species=("H", "H"), energy=-3.0),
    )

    cache, manifest = materialize_extxyz(source)
    paths = expand_extxyz_cache(cache)
    assert paths is not None
    assert [entry["frames"] for entry in manifest["systems"]] == [2, 1]
    assert [entry["atom_numbs"] for entry in manifest["systems"]] == [[2], [2, 1]]
    assert _load_system(Path(paths[0])).data["energies"].tolist() == [-1.0, -3.0]
    assert _load_system(Path(paths[1])).data["energies"].tolist() == [-2.0]


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("batch_size", [1], "batch_size"),
        ("sys_probs", [1.0], "sys_probs"),
        ("auto_prob", "prob_sys_size;0:1:1.0", "auto_prob"),
    ],
)
def test_heterogeneous_expansion_rejects_ambiguous_per_system_settings(
    tmp_path: Path, setting: str, value: Any, message: str
) -> None:
    source = _write(
        tmp_path / f"heterogeneous-{setting}.extxyz",
        _frame(species=("H", "H")),
        _frame(species=("H", "O", "H")),
    )
    config = _partial_config(str(source))
    config["training"]["training_data"][setting] = value

    with pytest.raises(ValueError, match=message):
        normalize_extxyz_training_data(config)


def test_atom_layout_changes_do_not_corrupt_coordinates_or_forces(
    tmp_path: Path,
) -> None:
    first_positions = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    first_forces = np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    second_positions = np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    second_forces = np.array([[30.0, 0.0, 0.0], [40.0, 0.0, 0.0]])
    source = _write(
        tmp_path / "layouts.extxyz",
        _frame(
            species=("H", "O"),
            positions=first_positions,
            forces=first_forces,
            energy=-1.0,
        ),
        _frame(
            species=("O", "H"),
            positions=second_positions,
            forces=second_forces,
            energy=-2.0,
        ),
    )

    system_path, manifest = _converted(source)
    system = _load_system(system_path)
    assert manifest["systems"][0]["frames"] == 2
    for frame_index in range(2):
        coord_force_pairs = sorted(
            zip(
                system.data["coords"][frame_index, :, 0],
                system.data["forces"][frame_index, :, 0],
                strict=True,
            )
        )
        expected = [[(1.0, 10.0), (2.0, 20.0)], [(3.0, 30.0), (4.0, 40.0)]]
        assert coord_force_pairs == expected[frame_index]


def test_cache_reuse_and_content_invalidation(tmp_path: Path) -> None:
    source = _write(tmp_path / "cache.extxyz", _frame(energy=-1.0))
    first_cache, first_manifest = materialize_extxyz(source)
    reused_cache, reused_manifest = materialize_extxyz(source)
    assert reused_cache == first_cache
    assert reused_manifest == first_manifest

    _write(source, _frame(energy=-2.0))
    second_cache, _ = materialize_extxyz(source)
    assert second_cache != first_cache
    paths = expand_extxyz_cache(second_cache)
    assert paths is not None
    assert _energies(paths) == [-2.0]


def test_normalization_does_not_modify_parsed_input_or_json_file(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "source.extxyz", _frame())
    input_file = tmp_path / "input.json"
    config = _partial_config(str(source), str(source))
    input_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
    before = input_file.read_bytes()
    parsed = json.loads(input_file.read_text(encoding="utf-8"))
    snapshot = copy.deepcopy(parsed)

    normalized = normalize_extxyz_training_data(parsed)

    assert parsed == snapshot
    assert input_file.read_bytes() == before
    assert normalized is not parsed
    assert normalized["training"]["training_data"]["systems"] != str(source)


def test_schema_documents_extxyz_paths() -> None:
    from deepmd.utils.argcheck import (
        training_data_args,
        validation_data_args,
    )

    assert ".extxyz" in training_data_args()["systems"].doc
    assert "dpdata" in training_data_args()["systems"].doc
    assert ".extxyz" in validation_data_args()["systems"].doc


def test_normalize_and_initialize_training_and_validation_data(
    tmp_path: Path,
) -> None:
    from deepmd.dpmodel.loss.ener import (
        EnergyLoss,
    )
    from deepmd.utils.argcheck import (
        normalize,
    )
    from deepmd.utils.data_system import (
        get_data,
        process_systems,
    )

    stress = np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6])
    virial = np.arange(1.0, 10.0).reshape(3, 3)
    training_source = _write(
        tmp_path / "train.extxyz",
        _frame(energy=-1.0, tensor_name="stress", tensor=stress),
    )
    validation_source = _write(
        tmp_path / "validation.extxyz",
        _frame(energy=-2.0, tensor_name="virial", tensor=virial),
    )
    repository_root = Path(__file__).resolve().parents[3]
    config = json.loads(
        (repository_root / "examples" / "water" / "se_e2_a" / "input.json").read_text(
            encoding="utf-8"
        )
    )
    config["training"]["training_data"] = {
        "systems": [str(training_source)],
        "batch_size": 1,
    }
    config["training"]["validation_data"] = {
        "systems": [str(validation_source)],
        "batch_size": 1,
        "numb_btch": 1,
    }
    config["loss"]["start_pref_v"] = 1.0
    config["loss"]["limit_pref_v"] = 1.0
    original = copy.deepcopy(config)

    normalized = normalize(config)

    assert config == original
    train_params = normalized["training"]["training_data"]
    validation_params = normalized["training"]["validation_data"]
    train_paths = process_systems(train_params["systems"])
    validation_paths = process_systems(validation_params["systems"])
    # Neighbor-statistics and loader construction both call process_systems;
    # repeated expansion must resolve to exactly the same cache paths.
    assert process_systems(train_params["systems"]) == train_paths
    assert process_systems(validation_params["systems"]) == validation_paths

    train_data = get_data(train_params, 6.0, ["O", "H"], None)
    validation_data = get_data(validation_params, 6.0, ["O", "H"], None)
    assert train_data.system_dirs == train_paths
    assert validation_data.system_dirs == validation_paths

    loss = EnergyLoss(
        starter_learning_rate=1.0,
        start_pref_e=1.0,
        limit_pref_e=1.0,
        start_pref_f=1.0,
        limit_pref_f=1.0,
        start_pref_v=1.0,
        limit_pref_v=1.0,
    )
    for data, expected_energy in (
        (train_data, -1.0),
        (validation_data, -2.0),
    ):
        data.add_data_requirements(loss.label_requirement)
        batch = data.get_batch(0)
        np.testing.assert_allclose(batch["energy"].reshape(-1), [expected_energy])
        assert np.all(batch["find_energy"] == 1.0)
        assert np.all(batch["find_force"] == 1.0)
        assert np.all(batch["find_virial"] == 1.0)
