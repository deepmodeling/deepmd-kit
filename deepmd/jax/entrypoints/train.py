# SPDX-License-Identifier: LGPL-3.0-or-later
"""DeePMD training entrypoint script.

Can handle local training.
"""

import json
import logging
import time
from typing import (
    Any,
)

from deepmd.common import (
    j_loader,
)
from deepmd.jax.env import (
    jax,
    jax_export,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.train.trainer import (
    DPTrainer,
)
from deepmd.jax.utils.update_sel import (
    use_jax_update_sel,
)
from deepmd.utils import random as dp_random
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

    def get_device_name(self) -> str:
        """Get the name of the device."""
        devices = jax.devices()
        if devices:
            return devices[0].device_kind
        else:
            return "Unknown"


def train(
    *,
    INPUT: str,
    init_model: str | None,
    restart: str | None,
    output: str,
    init_frz_model: str | None,
    mpi_log: str,
    log_level: int,
    log_path: str | None,
    skip_neighbor_stat: bool = False,
    finetune: str | None = None,
    use_pretrain_script: bool = False,
    **kwargs: Any,
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
    init_frz_model : str | None
        path to frozen model, or None if no frozen model is used
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
        if the training command fails.
    """
    # load json database
    jdata = j_loader(INPUT)

    if init_frz_model:
        raise NotImplementedError("JAX training does not support init_frz_model yet")
    if finetune:
        raise NotImplementedError("JAX training does not support finetune yet")
    if use_pretrain_script:
        raise NotImplementedError(
            "JAX training does not support use_pretrain_script yet"
        )

    jdata = update_deepmd_input(jdata, warning=True, dump="input_v2_compat.json")

    jdata = normalize(jdata)
    min_nbor_dist = None
    if not skip_neighbor_stat:
        jdata, min_nbor_dist = update_sel(jdata)

    with open(output, "w") as fp:
        json.dump(jdata, fp, indent=4)
    SummaryPrinter()()

    # make necessary checks
    assert "training" in jdata

    # init the model

    model = DPTrainer(
        jdata,
        init_model=init_model,
        restart=restart,
    )
    if min_nbor_dist is not None:
        model.model.min_nbor_dist = min_nbor_dist
    rcut = model.model.get_rcut()
    type_map = model.model.get_type_map()
    if len(type_map) == 0:
        ipt_type_map = None
    else:
        ipt_type_map = type_map

    # init random seed of data systems
    seed = jdata["training"].get("seed", None)
    if seed is not None:
        seed += jax.process_index()
        seed = seed % (2**32)
    dp_random.seed(seed)

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

    # train the model with the provided systems in a cyclic way
    start_time = time.time()
    model.train(train_data, valid_data)
    end_time = time.time()
    log.info("finished training")
    log.info(f"wall time: {(end_time - start_time):.3f} s")


def update_sel(jdata: dict) -> tuple[dict, float | None]:
    """Update descriptor selections from neighbor statistics when available."""
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
    with use_jax_update_sel():
        jdata_cpy["model"], min_nbor_dist = BaseModel.update_sel(
            train_data, type_map, jdata["model"]
        )
    return jdata_cpy, min_nbor_dist
