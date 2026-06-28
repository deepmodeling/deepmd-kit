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


class BrokenLenData:
    def __len__(self) -> int:
        raise TypeError("broken length")


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
