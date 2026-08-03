# SPDX-License-Identifier: LGPL-3.0-or-later
"""Command-line entry point of ``dp test``.

System discovery and the run-level report live here; evaluating a system is
the business of :mod:`deepmd.infer.model_test`.
"""

import logging
from pathlib import (
    Path,
)
from typing import (
    Any,
)

from deepmd.common import (
    j_loader,
)
from deepmd.dpmodel.utils.lmdb_data import (
    LmdbTestData,
    LmdbTestDataNlocView,
    is_lmdb,
)
from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.infer.model_test import (
    build_tester,
)
from deepmd.utils import random as dp_random
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.data import (
    DeepmdData,
)
from deepmd.utils.data_system import (
    process_systems,
)
from deepmd.utils.weight_avg import (
    merge_weighted_errors,
)

__all__ = ["test"]

log = logging.getLogger(__name__)


def test(
    *,
    model: str,
    system: str | None,
    datafile: str | None,
    train_json: str | None = None,
    valid_json: str | None = None,
    numb_test: int,
    rand_seed: int | None,
    shuffle_test: bool,
    detail_file: str,
    atomic: bool,
    head: str | None = None,
    **kwargs: Any,
) -> None:
    """Test model predictions.

    Parameters
    ----------
    model : str
        path where model is stored
    system : str, optional
        system directory
    datafile : str, optional
        the path to the list of systems to test
    train_json : Optional[str]
        Path to the input.json file provided via ``--train-data``. Training systems will be used for testing.
    valid_json : Optional[str]
        Path to the input.json file provided via ``--valid-data``. Validation systems will be used for testing.
    numb_test : int
        number of tests to do. 0 means all data.
    rand_seed : Optional[int]
        seed for random generator
    shuffle_test : bool
        whether to shuffle tests
    detail_file : Optional[str]
        file where test details will be output
    atomic : bool
        whether per atom quantities should be computed
    head : Optional[str], optional
        (Supported backend: PyTorch) Task head to test if in multi-task mode.
    **kwargs
        additional arguments

    Raises
    ------
    RuntimeError
        if no valid system was found
    """
    if numb_test == 0:
        # only float has inf, but should work for min
        numb_test = float("inf")
    if train_json is not None:
        jdata = j_loader(train_json)
        jdata = update_deepmd_input(jdata)
        data_params = jdata.get("training", {}).get("training_data", {})
        systems = data_params.get("systems")
        if not systems:
            raise RuntimeError("No training data found in input json")
        root = Path(train_json).parent
        if isinstance(systems, str):
            systems = str((root / Path(systems)).resolve())
        else:
            systems = [str((root / Path(ss)).resolve()) for ss in systems]
        patterns = data_params.get("rglob_patterns", None)
        all_sys = process_systems(systems, patterns=patterns)
    elif valid_json is not None:
        jdata = j_loader(valid_json)
        jdata = update_deepmd_input(jdata)
        data_params = jdata.get("training", {}).get("validation_data", {})
        systems = data_params.get("systems")
        if not systems:
            raise RuntimeError("No validation data found in input json")
        root = Path(valid_json).parent
        if isinstance(systems, str):
            systems = str((root / Path(systems)).resolve())
        else:
            systems = [str((root / Path(ss)).resolve()) for ss in systems]
        patterns = data_params.get("rglob_patterns", None)
        all_sys = process_systems(systems, patterns=patterns)
    elif datafile is not None:
        with open(datafile) as datalist:
            all_sys = datalist.read().splitlines()
    elif system is not None:
        all_sys = process_systems(system)
    else:
        raise RuntimeError("No data source specified for testing")

    if len(all_sys) == 0:
        raise RuntimeError("Did not find valid system")
    err_coll = []

    # init random seed
    if rand_seed is not None:
        dp_random.seed(rand_seed % (2**32))

    # init model
    dp = DeepEval(model, head=head)
    tester = build_tester(dp, atomic=atomic)

    for cc, system in enumerate(all_sys):
        log.info("# ---------------output of dp test--------------- ")
        log.info(f"# testing system : {system}")

        # create data class
        tmap = dp.get_type_map()
        if is_lmdb(system):
            lmdb_data = LmdbTestData(
                system,
                type_map=tmap,
                shuffle_test=shuffle_test,
                max_frames=numb_test,
            )
            # For mixed-nloc LMDB, test each nloc group separately
            nloc_keys = sorted(lmdb_data.nloc_groups.keys())
            if len(nloc_keys) > 1:
                group_summary = {
                    k: len(v) for k, v in sorted(lmdb_data.nloc_groups.items())
                }
                log.info(
                    f"# mixed-nloc LMDB: testing {len(nloc_keys)} groups: "
                    f"{group_summary}"
                )
            data_items: list[tuple[Any, str]] = []
            for nloc_val in nloc_keys:
                label = f"{system} [nloc={nloc_val}]" if len(nloc_keys) > 1 else system
                # Create a thin wrapper that returns only this nloc group
                data_items.append((LmdbTestDataNlocView(lmdb_data, nloc_val), label))
        else:
            data = DeepmdData(
                system,
                set_prefix="set",
                shuffle_test=shuffle_test,
                type_map=tmap,
                sort_atoms=False,
            )
            data_items = [(data, system)]

        for data, sys_label in data_items:
            if sys_label != system:
                log.info(f"# testing sub-group : {sys_label}")
            # Only the very first tested group writes a fresh detail file; a
            # system split into sub-groups extends it like any later system.
            detail_group = len(err_coll)
            append_detail = bool(detail_group)

            err = tester.run(
                data,
                sys_label,
                numb_test,
                detail_file,
                append_detail=append_detail,
                detail_group=detail_group,
            )
            log.info("# ----------------------------------------------- ")
            err_coll.append(err)

    # For mixed-nloc LMDB, err_coll may have more entries than all_sys
    # (one per nloc group per system). Only warn if fewer.
    if len(err_coll) < len(all_sys):
        log.warning("Not all systems are tested! Check if the systems are valid")

    log.info("# ----------weighted average of errors----------- ")
    log.info(f"# number of systems : {len(all_sys)}")
    tester.log_errors(merge_weighted_errors(err_coll))
    log.info("# ----------------------------------------------- ")
