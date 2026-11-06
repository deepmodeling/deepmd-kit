# SPDX-License-Identifier: LGPL-3.0-or-later
"""LMDB startup uses the actual TF entrypoint and neighbor-stat kernels."""

import importlib

import numpy as np

from ..common.lmdb_smoke import (
    check_legacy_lmdb_data,
    lmdb_training_config,
)


def test_lmdb_entrypoint_and_neighbor_statistics(tmp_path):
    module = importlib.import_module("deepmd.tf.entrypoints.train")
    config = lmdb_training_config(tmp_path)
    updated = module.update_sel(config)
    assert updated["model"]["descriptor"]["sel"] == [4]
    with module.get_data(config["training"]["training_data"], 2.0, ["H"], None) as data:
        check_legacy_lmdb_data(data)
        from deepmd.tf.utils.neighbor_stat import (
            NeighborStat,
        )

        distance, count = NeighborStat(1, 2.0, mixed_type=True).get_stat(data)
        np.testing.assert_allclose(distance, 1.0)
        np.testing.assert_array_equal(count, [1])
