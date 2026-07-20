# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
)
from unittest.mock import (
    Mock,
)

import h5py
import numpy as np
import pytest

from deepmd.dpmodel.model.model import (
    get_model,
)
from deepmd.dpmodel.utils.stat import (
    compute_output_stats,
)
from deepmd.utils.path import (
    DPH5Path,
    DPPath,
)
from deepmd.utils.stat_file import (
    StatFileSpec,
    load_paired_items,
    load_required_items,
    open_stat_file,
    replace_paired_items,
    run_stat_on_chief,
    stat_file_specs_by_task,
)

from .stat_file import (
    assert_energy_stat_cache_round_trip,
    energy_model_params,
    energy_stat_sample,
)


@pytest.mark.parametrize("mode", ["read", "invalid"])
def test_stat_file_spec_rejects_invalid_disabled_mode(mode: str) -> None:
    with pytest.raises(ValueError):
        StatFileSpec(None, mode)  # type: ignore[arg-type]


def test_stat_file_spec_rejects_empty_path() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        StatFileSpec("  ")


def test_disabled_stat_file_yields_none() -> None:
    with open_stat_file(StatFileSpec(None)) as path:
        assert path is None


def test_hdf5_handle_is_scoped(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "stat.hdf5"
    with open_stat_file(StatFileSpec(str(target))) as path:
        assert path is not None
        value = path / "value"
        value.save_numpy(np.array([1.0]))
        assert value.load_numpy().tolist() == [1.0]

    with pytest.raises(RuntimeError, match="closed"):
        value.load_numpy()
    with h5py.File(target, "r+") as file:
        assert file["value"][:].tolist() == [1.0]


def test_hdf5_handle_closes_after_exception(tmp_path: Path) -> None:
    target = tmp_path / "stat.hdf5"
    path = None
    with pytest.raises(RuntimeError, match="statistics failed"):
        with open_stat_file(StatFileSpec(str(target))) as path:
            assert path is not None
            (path / "value").save_numpy(np.array([1.0]))
            raise RuntimeError("statistics failed")

    assert path is not None
    with pytest.raises(RuntimeError, match="closed"):
        (path / "value").load_numpy()


def test_complete_update_cache_remains_byte_identical(tmp_path: Path) -> None:
    target = tmp_path / "stat.hdf5"
    with h5py.File(target, "w") as file:
        file.create_dataset("value", data=[1.0])
    original = target.read_bytes()

    with open_stat_file(StatFileSpec(str(target))) as path:
        assert path is not None
        assert (path / "value").load_numpy().tolist() == [1.0]

    assert target.read_bytes() == original


def test_existing_update_cache_promotes_on_first_write(tmp_path: Path) -> None:
    target = tmp_path / "stat.hdf5"
    with h5py.File(target, "w") as file:
        file.create_dataset("existing", data=[1.0])

    with open_stat_file(StatFileSpec(str(target))) as path:
        assert path is not None
        (path / "added").save_numpy(np.array([2.0]))

    with h5py.File(target, "r") as file:
        assert file["existing"][:].tolist() == [1.0]
        assert file["added"][:].tolist() == [2.0]


def test_dpmodel_energy_cache_round_trip_uses_fitting_outputs_only(
    tmp_path: Path,
) -> None:
    assert_energy_stat_cache_round_trip(
        lambda: get_model(energy_model_params()),
        tmp_path / "stat.hdf5",
    )


def test_read_mode_rejects_writes(tmp_path: Path) -> None:
    target = tmp_path / "stat.hdf5"
    with h5py.File(target, "w"):
        pass

    with open_stat_file(StatFileSpec(str(target), "read")) as path:
        assert path is not None
        with pytest.raises(ValueError, match="read-only"):
            (path / "value").save_numpy(np.array([1.0]))
        with pytest.raises(ValueError, match="read-only"):
            path.mkdir(exist_ok=True)


def test_directory_cache_uses_same_scope_interface(tmp_path: Path) -> None:
    target = tmp_path / "stat"
    with open_stat_file(StatFileSpec(str(target))) as path:
        assert path is not None
        path.mkdir(parents=True, exist_ok=True)
        (path / "value").save_numpy(np.array([1.0]))

    with open_stat_file(StatFileSpec(str(target), "read")) as path:
        assert path is not None
        assert (path / "value").load_numpy().tolist() == [1.0]


def test_multi_task_cache_paths_must_be_distinct(tmp_path: Path) -> None:
    target = str(tmp_path / "stat.hdf5")
    specs = {
        "task/one": StatFileSpec(target),
        "task_two": StatFileSpec(str(tmp_path / "." / "stat.hdf5")),
    }

    with pytest.raises(ValueError, match="distinct statistics-cache path"):
        stat_file_specs_by_task(specs, ["task/one", "task_two"])


def test_stat_file_specs_are_normalized_by_task() -> None:
    disabled = stat_file_specs_by_task(None, ["one", "two"])
    assert disabled == {"one": StatFileSpec(None), "two": StatFileSpec(None)}

    configured = {
        "one": StatFileSpec("one.hdf5"),
        "two": StatFileSpec("two.hdf5", "read"),
    }
    assert stat_file_specs_by_task(configured, ["one", "two"]) == configured
    with pytest.raises(TypeError, match="Multi-task"):
        stat_file_specs_by_task(StatFileSpec("shared.hdf5"), ["one", "two"])


def test_read_mode_reports_all_missing_required_items(tmp_path: Path) -> None:
    target = tmp_path / "stat.hdf5"
    with h5py.File(target, "w") as file:
        file.create_dataset("present", data=[1.0])

    with open_stat_file(StatFileSpec(str(target), "read")) as path:
        assert path is not None
        with pytest.raises(FileNotFoundError) as error:
            load_required_items(path, ["missing_one", "present", "missing_two"])

    message = str(error.value)
    assert "'missing_one'" in message
    assert "'missing_two'" in message


def test_update_mode_recomputes_partial_output_group(tmp_path: Path) -> None:
    target = tmp_path / "stat.hdf5"
    sampled = energy_stat_sample()
    sampled[0]["property"] = np.array([[1.0], [3.0]], dtype=np.float64)
    sampled[0]["find_property"] = np.float32(1.0)

    with open_stat_file(StatFileSpec(str(target))) as path:
        assert path is not None
        (path / "bias_atom_energy").save_numpy(np.full((2, 1), 100.0))
        (path / "bias_atom_property").save_numpy(np.full((2, 1), 100.0))
        (path / "std_atom_energy").save_numpy(np.full((2, 1), 100.0))
        sampler = Mock(return_value=sampled)

        bias, std = compute_output_stats(
            sampler,
            ntypes=2,
            keys=["energy", "property"],
            stat_file_path=path,
        )

        sampler.assert_called_once_with()
        assert set(bias) == {"energy", "property"}
        assert set(std) == {"energy", "property"}

    with h5py.File(target, "r") as file:
        assert set(file) == {
            "bias_atom_energy",
            "bias_atom_property",
            "std_atom_energy",
            "std_atom_property",
        }
        assert not np.all(file["bias_atom_energy"][:] == 100.0)


def test_update_mode_replaces_orphaned_output_pair(tmp_path: Path) -> None:
    target = tmp_path / "stat.hdf5"
    with h5py.File(target, "w") as file:
        file.create_dataset("bias_atom_property", data=np.zeros((2, 1)))

    sampler = Mock(return_value=energy_stat_sample())
    with open_stat_file(StatFileSpec(str(target))) as path:
        assert path is not None
        bias, std = compute_output_stats(
            sampler,
            ntypes=2,
            keys=["energy", "property"],
            stat_file_path=path,
        )

    sampler.assert_called_once_with()
    assert set(bias) == {"energy"}
    assert set(std) == {"energy"}
    with h5py.File(target, "r") as file:
        assert set(file) == {"bias_atom_energy", "std_atom_energy"}
    original = target.read_bytes()

    sampler.reset_mock(side_effect=True)
    sampler.side_effect = AssertionError(
        "A complete update cache must not sample data."
    )
    with open_stat_file(StatFileSpec(str(target))) as path:
        assert path is not None
        bias, std = compute_output_stats(
            sampler,
            ntypes=2,
            keys=["energy", "property"],
            stat_file_path=path,
        )

    sampler.assert_not_called()
    assert set(bias) == {"energy"}
    assert set(std) == {"energy"}
    assert target.read_bytes() == original


def test_interrupted_pair_replacement_requires_recovery(tmp_path: Path) -> None:
    target = tmp_path / "stat.hdf5"
    pairs = [("bias_atom_energy", "std_atom_energy")]
    invalid_items = {
        "bias_atom_energy": np.zeros((2, 1)),
        "std_atom_energy": np.array([object()], dtype=object),
    }

    with open_stat_file(StatFileSpec(str(target))) as path:
        assert path is not None
        with pytest.raises(TypeError):
            replace_paired_items(path, pairs, invalid_items)
        assert load_paired_items(path, pairs) is None

    with open_stat_file(StatFileSpec(str(target), "read")) as path:
        assert path is not None
        with pytest.raises(FileNotFoundError, match=r"incomplete.*transaction"):
            load_paired_items(path, pairs)

    expected = {
        "bias_atom_energy": np.zeros((2, 1)),
        "std_atom_energy": np.ones((2, 1)),
    }
    with open_stat_file(StatFileSpec(str(target))) as path:
        assert path is not None
        assert load_paired_items(path, pairs) is None
        replace_paired_items(path, pairs, expected)

    with h5py.File(target, "r") as file:
        assert set(file) == set(expected)
        for name, value in expected.items():
            np.testing.assert_array_equal(file[name][:], value)


def test_read_mode_rejects_entirely_missing_output_pair(tmp_path: Path) -> None:
    target = tmp_path / "stat.hdf5"
    with h5py.File(target, "w") as file:
        file.create_dataset("bias_atom_energy", data=np.zeros((2, 1)))
        file.create_dataset("std_atom_energy", data=np.ones((2, 1)))

    sampler = Mock()
    with open_stat_file(StatFileSpec(str(target), "read")) as path:
        assert path is not None
        with pytest.raises(FileNotFoundError, match="bias_atom_property"):
            compute_output_stats(
                sampler,
                ntypes=2,
                keys=["energy", "property"],
                stat_file_path=path,
            )

    sampler.assert_not_called()


def test_update_mode_preserves_absent_legacy_output_pair(tmp_path: Path) -> None:
    target = tmp_path / "stat.hdf5"
    with h5py.File(target, "w") as file:
        file.create_dataset("bias_atom_energy", data=np.zeros((2, 1)))
        file.create_dataset("std_atom_energy", data=np.ones((2, 1)))
    original = target.read_bytes()

    sampler = Mock()
    with open_stat_file(StatFileSpec(str(target))) as path:
        assert path is not None
        bias, std = compute_output_stats(
            sampler,
            ntypes=2,
            keys=["energy", "property"],
            stat_file_path=path,
        )

    sampler.assert_not_called()
    assert set(bias) == {"energy"}
    assert set(std) == {"energy"}
    assert target.read_bytes() == original


def test_update_mode_recomputes_partial_descriptor_group(tmp_path: Path) -> None:
    target = tmp_path / "stat.hdf5"
    with open_stat_file(StatFileSpec(str(target))) as path:
        assert path is not None
        get_model(energy_model_params()).compute_or_load_stat(energy_stat_sample, path)

    with h5py.File(target, "r+") as file:
        type_map_group = file["O H"]
        descriptor_group = next(
            item for item in type_map_group.values() if isinstance(item, h5py.Group)
        )
        del descriptor_group["r_0"]

    read_sampler = Mock()
    with open_stat_file(StatFileSpec(str(target), "read")) as path:
        assert path is not None
        with pytest.raises(FileNotFoundError, match="'r_0'"):
            get_model(energy_model_params()).compute_or_load_stat(read_sampler, path)
    read_sampler.assert_not_called()

    sampler = Mock(return_value=energy_stat_sample())
    with open_stat_file(StatFileSpec(str(target))) as path:
        assert path is not None
        get_model(energy_model_params()).compute_or_load_stat(sampler, path)

    sampler.assert_called_once_with()
    with h5py.File(target, "r") as file:
        type_map_group = file["O H"]
        descriptor_group = next(
            item for item in type_map_group.values() if isinstance(item, h5py.Group)
        )
        assert "r_0" in descriptor_group


def test_scoped_cache_matches_legacy_hdf5_layout(tmp_path: Path) -> None:
    legacy_file = tmp_path / "legacy.hdf5"
    scoped_file = tmp_path / "scoped.hdf5"
    with h5py.File(legacy_file, "w"):
        pass

    legacy_path = DPPath(str(legacy_file), "a")
    assert isinstance(legacy_path, DPH5Path)
    legacy_handle = legacy_path.root
    try:
        get_model(energy_model_params()).compute_or_load_stat(
            energy_stat_sample,
            legacy_path,
        )
    finally:
        legacy_handle.close()
        DPH5Path._load_h5py.cache_clear()
        DPH5Path._file_keys.cache_clear()
        DPH5Path._DPH5Path__file_new_keys.pop(legacy_handle, None)

    with open_stat_file(StatFileSpec(str(scoped_file))) as scoped_path:
        assert scoped_path is not None
        get_model(energy_model_params()).compute_or_load_stat(
            energy_stat_sample,
            scoped_path,
        )

    def read_layout(
        path: Path,
    ) -> tuple[set[str], dict[str, tuple[np.dtype, tuple[int, ...], np.ndarray]]]:
        groups: set[str] = set()
        datasets: dict[str, tuple[np.dtype, tuple[int, ...], np.ndarray]] = {}
        with h5py.File(path, "r") as file:

            def collect(name: str, item: h5py.Group | h5py.Dataset) -> None:
                if isinstance(item, h5py.Group):
                    groups.add(name)
                else:
                    datasets[name] = (item.dtype, item.shape, item[:])

            file.visititems(collect)
        return groups, datasets

    legacy_groups, legacy_datasets = read_layout(legacy_file)
    scoped_groups, scoped_datasets = read_layout(scoped_file)
    assert scoped_groups == legacy_groups
    assert scoped_datasets.keys() == legacy_datasets.keys()
    for name, (legacy_dtype, legacy_shape, legacy_value) in legacy_datasets.items():
        scoped_dtype, scoped_shape, scoped_value = scoped_datasets[name]
        assert scoped_dtype == legacy_dtype
        assert scoped_shape == legacy_shape
        np.testing.assert_array_equal(scoped_value, legacy_value)

    original = legacy_file.read_bytes()

    def unexpected_sample() -> list[dict[str, object]]:
        raise AssertionError("A complete legacy cache must not sample data.")

    for mode in ("update", "read"):
        with open_stat_file(StatFileSpec(str(legacy_file), mode)) as stat_path:
            assert stat_path is not None
            get_model(energy_model_params()).compute_or_load_stat(
                unexpected_sample,
                stat_path,
            )
        assert legacy_file.read_bytes() == original


def test_chief_failure_is_synchronized_before_reraising() -> None:
    synchronized: list[bool] = []

    def fail() -> None:
        raise ValueError("invalid statistics")

    def synchronize(failed: bool) -> bool:
        synchronized.append(failed)
        return failed

    with pytest.raises(ValueError, match="invalid statistics"):
        run_stat_on_chief(
            fail,
            is_chief=True,
            synchronize_failure=synchronize,
            operation="statistics initialization",
        )

    assert synchronized == [True]


def test_peer_raises_when_chief_reports_failure() -> None:
    action = Mock()

    with pytest.raises(RuntimeError, match="Rank 0 failed during statistics"):
        run_stat_on_chief(
            action,
            is_chief=False,
            synchronize_failure=lambda _: True,
            operation="statistics initialization",
        )

    action.assert_not_called()
