# SPDX-License-Identifier: LGPL-3.0-or-later
"""DeePMD training entrypoint script.

Can handle local or distributed training.
"""

import json
import logging
import time
from typing import (
    Optional,
)

from deepmd.common import (
    j_loader,
)
from deepmd.jax.env import (
    jax,
    jax_export,
)
from deepmd.jax.train.trainer import (
    DPTrainer,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.data_system import (
    get_data,
)
from deepmd.utils.summary import SummaryPrinter as BaseSummaryPrinter

__all__ = ["train"]

log = logging.getLogger(__name__)


class SummaryPrinter(BaseSummaryPrinter):
    """Summary printer for JAX."""

    def is_built_with_cuda(self) -> bool:
        """Check if the backend is built with CUDA."""
        return jax_export.default_export_platform() == "cuda"

    def is_built_with_rocm(self) -> bool:
        """Check if the backend is built with ROCm."""
        return jax_export.default_export_platform() == "rocm"

    def get_compute_device(self) -> str:
        """Get Compute device."""
        return jax.default_backend()

    def get_ngpus(self) -> int:
        """Get the number of GPUs."""
        return jax.device_count()

    def get_backend_info(self) -> dict:
        """Get backend information."""
        return {
            "Backend": "JAX",
            "JAX ver": jax.__version__,
        }


def train(
    *,
    INPUT: str,
    init_model: Optional[str],
    restart: Optional[str],
    output: str,
    init_frz_model: str,
    mpi_log: str,
    log_level: int,
    log_path: Optional[str],
    skip_neighbor_stat: bool = False,
    finetune: Optional[str] = None,
    use_pretrain_script: bool = False,
    **kwargs,
) -> None:
    """Run DeePMD model training.

    Parameters
    ----------
    INPUT : str
        json/yaml control file
    init_model : Optional[str]
        path prefix of checkpoint files or None
    restart : Optional[str]
        path prefix of checkpoint files or None
    output : str
        path for dump file with arguments
    init_frz_model : str
        path to frozen model or None
    mpi_log : str
        mpi logging mode
    log_level : int
        logging level defined by int 0-3
    log_path : Optional[str]
        logging file path or None if logs are to be output only to stdout
    skip_neighbor_stat : bool, default=False
        skip checking neighbor statistics
    finetune : Optional[str]
        path to pretrained model or None
    use_pretrain_script : bool
        Whether to use model script in pretrained model when doing init-model or init-frz-model.
        Note that this option is true and unchangeable for fine-tuning.
    **kwargs
        additional arguments

    Raises
    ------
    RuntimeError
        if distributed training job name is wrong
    """
    # load json database
    jdata = j_loader(INPUT)

    origin_type_map = None

    jdata = update_deepmd_input(jdata, warning=True, dump="input_v2_compat.json")

    jdata = normalize(jdata)
    jdata = update_sel(jdata)

    with open(output, "w") as fp:
        json.dump(jdata, fp, indent=4)
    SummaryPrinter()()

    # make necessary checks
    assert "training" in jdata

    # init the model

    model = DPTrainer(jdata)
    rcut = model.model.get_rcut()
    type_map = model.model.get_type_map()
    if len(type_map) == 0:
        ipt_type_map = None
    else:
        ipt_type_map = type_map

    # init random seed of data systems
    seed = jdata["training"].get("seed", None)

    # init data
    train_data = get_data(jdata["training"]["training_data"], rcut, ipt_type_map, None)
    train_data.add_data_requirements(model.data_requirements)
    train_data.print_summary("training")
    if jdata["training"].get("validation_data", None) is not None:
        valid_data = get_data(
            jdata["training"]["validation_data"],
            rcut,
            train_data.type_map,
            None,
        )
        valid_data.add_data_requirements(model.data_requirements)
        valid_data.print_summary("validation")
    else:
        valid_data = None

    # get training info
    stop_batch = jdata["training"]["numb_steps"]
    origin_type_map = jdata["model"].get("origin_type_map", None)
    if (
        origin_type_map is not None and not origin_type_map
    ):  # get the type_map from data if not provided
        origin_type_map = get_data(
            jdata["training"]["training_data"], rcut, None, None
        ).get_type_map()

    # train the model with the provided systems in a cyclic way
    start_time = time.time()
    model.train(train_data, valid_data)
    end_time = time.time()
    log.info("finished training")
    log.info(f"wall time: {(end_time - start_time):.3f} s")


def update_sel(jdata):
    log.info(
        "Calculate neighbor statistics... (add --skip-neighbor-stat to skip this step)"
    )
    jdata_cpy = jdata.copy()
    type_map = jdata["model"].get("type_map")
    train_data = get_data(
        jdata["training"]["training_data"],
        0,  # not used
        type_map,
        None,  # not used
    )
    # TODO: OOM, need debug
    # jdata_cpy["model"], min_nbor_dist = BaseModel.update_sel(
    #     train_data, type_map, jdata["model"]
    # )
    return jdata_cpy
