# SPDX-License-Identifier: LGPL-3.0-or-later
import pytest

from deepmd.dpmodel.utils.training_utils import (
    resolve_model_prob,
)


class UnsizedData:
    pass


class SizedData:
    def __len__(self) -> int:
        return 3


class SystemCountData:
    def get_nsystems(self) -> int:
        return 5

    def __len__(self) -> int:
        return 99


class BrokenLenData:
    def __len__(self) -> int:
        raise TypeError("broken length")


def test_resolve_model_prob_uses_get_nsystems() -> None:
    prob = resolve_model_prob(
        ["systems", "sized"],
        None,
        {
            "systems": SystemCountData(),
            "sized": SizedData(),
        },
        rank=1,
    )

    assert prob.tolist() == pytest.approx([5.0 / 8.0, 3.0 / 8.0])


def test_resolve_model_prob_falls_back_for_unsized_data_only() -> None:
    prob = resolve_model_prob(
        ["unsized", "sized"],
        None,
        {
            "unsized": UnsizedData(),
            "sized": SizedData(),
        },
        rank=1,
    )

    assert prob.tolist() == pytest.approx([0.25, 0.75])


def test_resolve_model_prob_propagates_broken_len() -> None:
    with pytest.raises(TypeError, match="broken length"):
        resolve_model_prob(
            ["broken"],
            None,
            {"broken": BrokenLenData()},
            rank=1,
        )
