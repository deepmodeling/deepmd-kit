# SPDX-License-Identifier: LGPL-3.0-or-later
"""Task A: composable pooling primitives (dpa_adapt).

parse_pooling + the strategy guard are pure-Python and run anywhere; the
byte-identical layout check needs torch (available without deepmd.lib).
"""

from __future__ import (
    annotations,
)

import numpy as np
import pytest

from dpa_adapt.finetuner import (
    POOLING_PRIMITIVES,
    DPAFineTuner,
    parse_pooling,
)


def test_parse_pooling_legacy_strings():
    # the 4 historical strings map to their exact primitive tuples
    assert parse_pooling("mean") == ("mean",)
    assert parse_pooling("sum") == ("sum",)
    assert parse_pooling("mean+std") == ("mean", "std")
    assert parse_pooling("mean+std+max+min") == ("mean", "std", "max", "min")


def test_parse_pooling_canonical_order_and_dedup():
    # output order follows POOLING_PRIMITIVES regardless of input order
    assert parse_pooling("std+mean") == ("mean", "std")
    assert parse_pooling("min+max+mean") == ("mean", "max", "min")
    assert parse_pooling(["std", "mean", "std"]) == ("mean", "std")  # dedup
    assert parse_pooling("mean+sum") == ("mean", "sum")  # legal, not redundant


def test_parse_pooling_list_input():
    assert parse_pooling(["mean", "std"]) == ("mean", "std")
    assert parse_pooling(("max", "mean")) == ("mean", "max")


def test_parse_pooling_unknown_token_raises():
    with pytest.raises(ValueError, match="unknown pooling primitive"):
        parse_pooling("mean+median")
    with pytest.raises(ValueError, match="unknown pooling primitive"):
        parse_pooling(["mean", "p95"])


def test_parse_pooling_empty_raises():
    with pytest.raises(ValueError, match="empty pooling spec"):
        parse_pooling("")
    with pytest.raises(ValueError, match="empty pooling spec"):
        parse_pooling([])


def test_pooling_strategy_guard():
    # composite/non-mean pooling with a training strategy raises, pointing to `intensive`
    for strat in ("frozen_head", "finetune", "mft"):
        with pytest.raises(ValueError, match="intensive"):
            DPAFineTuner(pretrained="x", strategy=strat, pooling="mean+std")
    # plain mean is a harmless default for training strategies (no raise)
    DPAFineTuner(pretrained="x", strategy="finetune", pooling="mean")
    # frozen_sklearn accepts any composite
    DPAFineTuner(pretrained="x", strategy="frozen_sklearn", pooling="mean+std+max+min")


def test_pool_descriptor_byte_identical_to_legacy():
    torch = pytest.importorskip("torch")
    torch.set_default_device("cpu")
    from dpa_adapt.finetuner import (
        _pool_descriptor,
    )

    rng = np.random.default_rng(0)
    descrpt = torch.tensor(rng.standard_normal((3, 5, 4)))

    def legacy(spec):
        if spec == "mean":
            return descrpt.mean(dim=1)
        if spec == "sum":
            return descrpt.sum(dim=1)
        if spec == "mean+std":
            m = descrpt.mean(dim=1)
            s = torch.nan_to_num(descrpt.std(dim=1), nan=0.0)
            return torch.cat([m, s], dim=-1)
        if spec == "mean+std+max+min":
            m = descrpt.mean(dim=1)
            s = torch.nan_to_num(descrpt.std(dim=1), nan=0.0)
            return torch.cat(
                [m, s, descrpt.max(dim=1).values, descrpt.min(dim=1).values], dim=-1
            )
        raise AssertionError(spec)

    for spec in ("mean", "sum", "mean+std", "mean+std+max+min"):
        new = _pool_descriptor(descrpt, parse_pooling(spec))
        assert torch.equal(new, legacy(spec)), spec


def test_pool_descriptor_is_nan_safe_for_masked_atoms():
    """A non-finite descriptor on a masked (virtual/padding) atom must not
    poison the pooled feature: ``descrpt * mask`` keeps ``0 * NaN == NaN``,
    which would corrupt the whole frame instead of just dropping that atom.
    """
    torch = pytest.importorskip("torch")
    torch.set_default_device("cpu")
    from dpa_adapt.finetuner import (
        _pool_descriptor,
    )

    # frame 0: atoms 0,1 kept ([1,2],[3,4]); atom 2 excluded and carries NaN.
    # frame 1: all 3 atoms kept, no NaN.
    descrpt = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [float("nan"), float("nan")]],
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ]
    )
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    for spec in ("mean", "sum", "mean+std", "mean+std+max+min"):
        out = _pool_descriptor(descrpt, parse_pooling(spec), mask=mask)
        assert torch.isfinite(out).all(), spec

    mean = _pool_descriptor(descrpt, parse_pooling("mean"), mask=mask)
    # frame 0 mean over the two kept atoms: ([1,2]+[3,4])/2 = [2,3]
    assert torch.allclose(mean[0], torch.tensor([2.0, 3.0]))
    # frame 1 unaffected: mean over all 3 kept atoms
    assert torch.allclose(mean[1], torch.tensor([2.0, 2.0]))

    summ = _pool_descriptor(descrpt, parse_pooling("sum"), mask=mask)
    assert torch.allclose(summ[0], torch.tensor([4.0, 6.0]))


def test_pooling_primitives_canonical_constant():
    # guards against accidental reordering that would shift feature columns
    assert POOLING_PRIMITIVES == ("mean", "sum", "std", "max", "min")
