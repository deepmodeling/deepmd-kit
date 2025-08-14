# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for TensorFlow training entrypoint."""

from pathlib import (
    Path,
)
from typing import (
    Any,
)

import yaml

from deepmd.tf.entrypoints.train import (  # type: ignore
    train,
)

from .common import (
    del_data,
    gen_data_type_specific,
    j_loader,
)


class TestYamlInput:
    """Ensure training entrypoint accepts YAML config."""

    def setup_method(self) -> None:
        gen_data_type_specific()
        config: dict[str, Any] = j_loader("water_se_atten.json")
        config["systems"] = ["system"]
        config["stop_batch"] = 1
        config["save_freq"] = 1
        yaml_file = Path("input.yaml")
        with open(yaml_file, "w") as fp:
            yaml.safe_dump(config, fp)
        self.yaml_file = yaml_file

    def teardown_method(self) -> None:
        del_data()
        for ff in [
            "out.json",
            "input.yaml",
            "lcurve.out",
            "model.ckpt.data-00000-of-00001",
            "model.ckpt.index",
            "model.ckpt.meta",
        ]:
            Path(ff).unlink(missing_ok=True)

    def test_yaml_input(self) -> None:
        train(
            INPUT=str(self.yaml_file),
            init_model=None,
            restart=None,
            output="out.json",
            init_frz_model=None,
            mpi_log="master",
            log_level=0,
            log_path=None,
            skip_neighbor_stat=True,
        )
        assert Path("out.json").exists()


__all__ = ["TestYamlInput"]
