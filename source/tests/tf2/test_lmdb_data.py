# SPDX-License-Identifier: LGPL-3.0-or-later
"""LMDB smoke coverage for the TF2 training entrypoint."""

import importlib
import os

import numpy as np
import pytest

if os.environ.get("DP_TEST_TF2_ONLY") != "1":
    pytest.skip("TF2 tests require DP_TEST_TF2_ONLY=1", allow_module_level=True)

from source.tests.common.lmdb_smoke import (
    check_legacy_lmdb_data,
    lmdb_training_config,
)


def test_lmdb_entrypoint_and_neighbor_statistics(tmp_path):
    module = importlib.import_module("deepmd.tf2.entrypoints.train")
    config = lmdb_training_config(tmp_path)
    updated, distance = module.update_sel(config)
    assert updated["model"]["descriptor"]["sel"] == [4]
    np.testing.assert_allclose(distance, 1.0)
    with module.get_data(
        config["training"]["training_data"], None, ["H"], None
    ) as data:
        check_legacy_lmdb_data(data)
        from deepmd.dpmodel.utils.batch import (
            normalize_batch,
        )

        batches = [normalize_batch(data.get_batch()) for _ in range(2)]
        assert {batch["box"] is None for batch in batches} == {False, True}
