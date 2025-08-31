# SPDX-License-Identifier: LGPL-3.0-or-later
"""Evaluate descriptors using trained DeePMD model."""

import logging
import os
from pathlib import (
    Path,
)
from typing import (
    Any,
    Optional,
)

import numpy as np

from deepmd.common import (
    expand_sys_str,
)
from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.utils.data import (
    DeepmdData,
)

__all__ = ["eval_desc"]

log = logging.getLogger(__name__)


def eval_desc(
    *,
    model: str,
    system: str,
    datafile: str,
    output: str = "desc",
    head: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Evaluate descriptors for given systems.

    Parameters
    ----------
    model : str
        path where model is stored
    system : str
        system directory
    datafile : str
        the path to the list of systems to process
    output : str
        output directory for descriptor files
    head : Optional[str], optional
        (Supported backend: PyTorch) Task head if in multi-task mode.
    **kwargs
        additional arguments

    Notes
    -----
    Descriptors are saved as 3D numpy arrays with shape (nframes, natoms, ndesc)
    where each frame contains the descriptors for all atoms.

    Raises
    ------
    RuntimeError
        if no valid system was found
    """
    if datafile is not None:
        with open(datafile) as datalist:
            all_sys = datalist.read().splitlines()
    else:
        all_sys = expand_sys_str(system)

    if len(all_sys) == 0:
        raise RuntimeError("Did not find valid system")

    # init model
    dp = DeepEval(model, head=head)

    # create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cc, system_path in enumerate(all_sys):
        log.info("# -------output of dp eval_desc------- ")
        log.info(f"# processing system : {system_path}")

        # create data class
        tmap = dp.get_type_map()
        data = DeepmdData(
            system_path,
            set_prefix="set",
            shuffle_test=False,
            type_map=tmap,
            sort_atoms=False,
        )

        # get test data
        test_data = data.get_test()
        mixed_type = data.mixed_type
        natoms = len(test_data["type"][0])
        nframes = test_data["box"].shape[0]

        # prepare input data
        coord = test_data["coord"].reshape([nframes, -1])
        box = test_data["box"]
        if not data.pbc:
            box = None
        if mixed_type:
            atype = test_data["type"].reshape([nframes, -1])
        else:
            atype = test_data["type"][0]

        # handle optional parameters
        fparam = None
        if dp.get_dim_fparam() > 0:
            if "fparam" in test_data:
                fparam = test_data["fparam"]

        aparam = None
        if dp.get_dim_aparam() > 0:
            if "aparam" in test_data:
                aparam = test_data["aparam"]

        # evaluate descriptors
        log.info(f"# evaluating descriptors for {nframes} frames")
        descriptors = dp.eval_descriptor(
            coord,
            box,
            atype,
            fparam=fparam,
            aparam=aparam,
        )

        # descriptors are kept in 3D format (nframes, natoms, ndesc)

        # save descriptors
        system_name = os.path.basename(system_path.rstrip("/"))
        desc_file = output_dir / f"{system_name}.npy"
        np.save(desc_file, descriptors)

        log.info(f"# descriptors saved to {desc_file}")
        log.info(f"# descriptor shape: {descriptors.shape}")
        log.info("# ----------------------------------- ")

    log.info("# eval_desc completed successfully")
