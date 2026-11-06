# SPDX-License-Identifier: LGPL-3.0-or-later
"""LMDB startup and validation through the real JAX adapter surface."""

import importlib

import numpy as np

from ..common.lmdb_smoke import (
    check_legacy_lmdb_data,
    lmdb_training_config,
)


def test_lmdb_entrypoint_and_neighbor_statistics(tmp_path):
    module = importlib.import_module("deepmd.jax.entrypoints.train")
    config = lmdb_training_config(tmp_path)
    updated, distance = module.update_sel(config)
    assert updated["model"]["descriptor"]["sel"] == [4]
    np.testing.assert_allclose(distance, 1.0)
    with module.get_data(
        config["training"]["training_data"], None, ["H"], None
    ) as data:
        check_legacy_lmdb_data(data)
        from deepmd.jax.train.validation import (
            JAXFullValidator,
        )

        validator = JAXFullValidator.__new__(JAXFullValidator)
        validator.validation_data = data
        systems = list(validator._iter_validation_data_systems())
        assert {system.pbc for system in systems} == {False, True}
        assert sum(len(system.get_test()["coord"]) for system in systems) == 2
