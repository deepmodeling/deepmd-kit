# SPDX-License-Identifier: LGPL-3.0-or-later
"""LMDB startup uses the actual TF entrypoint and neighbor-stat kernels."""

import importlib

import numpy as np
import pytest

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

        from deepmd.tf.model.ener import (
            EnerModel,
        )

        # update_sel alone does not enter the mixed-type descriptor statistics
        # path that the default se_e2_a training model uses at startup.
        model_config = {**updated["model"]}
        model_config["fitting_net"] = {"type": "ener", **model_config["fitting_net"]}
        model = EnerModel(**model_config, data_stat_nbatch=1)
        with pytest.raises(ValueError, match="does not support mixed-type data"):
            model.data_stat(data)

        # LMDB's arbitrary frame composition requires an attention descriptor;
        # removing the argument-binding error alone would leave se_e2_a's
        # fixed-composition native operator receiving a mixed-type mesh.
        model_config["descriptor"] = {
            **model_config["descriptor"],
            "type": "se_atten",
            "sel": 4,
            "attn": 4,
            "attn_layer": 0,
        }
        attention_model = EnerModel(**model_config, data_stat_nbatch=1)
        attention_model.data_stat(data)
