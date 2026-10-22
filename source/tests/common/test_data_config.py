# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared dataset configuration and sampling contracts."""

from pathlib import (
    Path,
)

import numpy as np
import pytest

from deepmd.utils import random as dp_random
from deepmd.utils.data_config import (
    resolve_data_groups,
)
from deepmd.utils.data_sampling import (
    group_probabilities,
    system_probabilities,
)
from deepmd.utils.data_system import (
    get_data,
)


def test_group_probabilities_do_not_equalize_unweighted_groups() -> None:
    np.testing.assert_allclose(group_probabilities([2, 8], [None, None]), [0.2, 0.8])
    np.testing.assert_allclose(group_probabilities([2, 8], [0.6, 0.4]), [0.6, 0.4])
    np.testing.assert_allclose(
        group_probabilities([2, 3, 9], [0.6, None, None]), [0.6, 0.1, 0.3]
    )
    np.testing.assert_allclose(group_probabilities([2, 8], [1.0, None]), [1.0, 0.0])


def test_group_and_system_probabilities_are_independent() -> None:
    outer = group_probabilities([10, 90], [0.6, 0.4])
    inner = system_probabilities([1, 2, 7], auto_prob="prob_sys_size;0:1:0.5;1:3:0.5")

    np.testing.assert_allclose(outer[0] * inner, [0.3, 0.6 / 9, 2.1 / 9])
    assert np.isclose((outer[0] * inner).sum(), 0.6)


@pytest.mark.parametrize("systems_as_list", [False, True])
def test_multi_lmdb_accepts_size_and_uniform_auto_prob(
    tmp_path: Path, systems_as_list: bool
) -> None:
    first = tmp_path / "first.lmdb"
    second = tmp_path / "nested" / "second.lmdb"
    first.mkdir()
    second.mkdir(parents=True)
    systems = [str(first), str(second)] if systems_as_list else str(tmp_path)

    groups = resolve_data_groups({"systems": systems})
    assert set(groups[0].systems) == {str(first), str(second)}
    assert groups[0].auto_prob is None
    explicit_default = resolve_data_groups(
        {"systems": systems, "auto_prob": "prob_sys_size"}
    )
    assert explicit_default[0].systems == groups[0].systems
    uniform = resolve_data_groups({"systems": systems, "auto_prob": "prob_uniform"})
    assert uniform[0].systems == groups[0].systems
    with pytest.raises(ValueError, match="auto_prob must be prob_sys_size"):
        resolve_data_groups({"systems": systems, "auto_prob": "prob_sys_size;0:1:1"})
    with pytest.raises(ValueError, match="sys_probs is not supported"):
        resolve_data_groups({"systems": systems, "sys_probs": [-1, -1]})


def test_empty_groups_cannot_receive_positive_probability() -> None:
    with pytest.raises(ValueError, match="must contain batches"):
        group_probabilities([0, 8], [0.5, 0.5])
    np.testing.assert_allclose(group_probabilities([0, 8], [None, None]), [0, 1])


def test_single_lmdb_groups_retain_internal_auto_prob() -> None:
    groups = resolve_data_groups(
        [
            {
                "systems": "first.lmdb",
                "auto_prob": "prob_sys_size;0:10:0.5;10:30:0.5",
                "prob": 0.6,
            },
            {"systems": "second.lmdb", "prob": 0.4},
        ]
    )

    assert groups[0].auto_prob == "prob_sys_size;0:10:0.5;10:30:0.5"
    assert groups[1].auto_prob is None
    assert [group.prob for group in groups] == [0.6, 0.4]


def test_numpy_groups_follow_the_training_seed() -> None:
    system = str(Path(__file__).resolve().parents[1] / "pt" / "property" / "double")
    config = [{"systems": system, "batch_size": size} for size in (1, 2)]
    sequences = []
    for _ in range(2):
        dp_random.seed(123)
        data = get_data(config, 0, None, None)
        sequences.append([data.get_batch()["coord"].shape[0] for _ in range(24)])
    assert sequences[0] == sequences[1]
    assert set(sequences[0]) == {1, 2}


def test_numpy_backend_parent_discovery_preserves_format_scope(tmp_path: Path) -> None:
    numpy = tmp_path / "numpy"
    numpy.mkdir()
    (numpy / "type.raw").touch()
    lmdb = tmp_path / "frames.lmdb"
    lmdb.mkdir()
    groups = resolve_data_groups({"systems": str(tmp_path)}, formats=("deepmd",))
    assert groups[0].systems == (str(numpy),)
