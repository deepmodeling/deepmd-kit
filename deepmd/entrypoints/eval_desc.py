# SPDX-License-Identifier: LGPL-3.0-or-later
"""Evaluate descriptors using trained DeePMD model."""

import logging
import os
from collections.abc import (
    Iterator,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.utils.lmdb_data import (
    LmdbTestData,
    LmdbTestDataNlocView,
    is_lmdb,
)
from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.utils.data import (
    DeepmdData,
)
from deepmd.utils.data_path import (
    process_systems,
)

__all__ = ["eval_desc"]

log = logging.getLogger(__name__)


def _system_base_name(system_path: str) -> str:
    """Return a filesystem-safe base name derived from a system path."""
    base = os.path.basename(system_path.rstrip("/\\"))
    if base in ("", ".", ".."):
        base = os.path.basename(os.path.abspath(system_path).rstrip("/\\"))
    return base or "system"


def _unique_output_name(
    system_path: str,
    used_names: set[str],
    *,
    nloc: int | None = None,
) -> str:
    """Return a collision-free output stem derived from a system path."""
    base = _system_base_name(system_path)
    if nloc is not None:
        base = f"{base}_nloc{nloc}"
    name = base
    idx = 1
    while name in used_names:
        name = f"{base}_{idx}"
        idx += 1
    used_names.add(name)
    return name


def _iter_system_data(
    system_path: str,
    type_map: list[str],
) -> Iterator[tuple[Any, str, int | None]]:
    """Yield ``(data, label, nloc)`` items for one DeePMD or LMDB system."""
    if is_lmdb(system_path):
        lmdb_data = LmdbTestData(
            system_path,
            type_map=type_map,
            shuffle_test=False,
        )
        nloc_keys = sorted(lmdb_data.nloc_groups.keys())
        if not nloc_keys:
            log.warning(f"# no frames found in LMDB system : {system_path}")
            return
        if len(nloc_keys) > 1:
            group_summary = {
                k: len(v) for k, v in sorted(lmdb_data.nloc_groups.items())
            }
            log.info(
                f"# mixed-nloc LMDB: evaluating {len(nloc_keys)} groups: "
                f"{group_summary}"
            )
        for nloc_val in nloc_keys:
            if len(nloc_keys) > 1:
                yield (
                    LmdbTestDataNlocView(lmdb_data, nloc_val),
                    f"{system_path} [nloc={nloc_val}]",
                    nloc_val,
                )
            else:
                yield LmdbTestDataNlocView(lmdb_data, nloc_val), system_path, None
        return

    data = DeepmdData(
        system_path,
        set_prefix="set",
        shuffle_test=False,
        type_map=type_map,
        sort_atoms=False,
    )
    yield data, system_path, None


def eval_desc(
    *,
    model: str,
    system: str | None,
    datafile: str | None,
    output: str = "desc",
    head: str | None = None,
    dtype: str = "native",
    **kwargs: Any,
) -> None:
    """Evaluate descriptors for given systems.

    Parameters
    ----------
    model : str
        path where model is stored
    system : str, optional
        system directory or LMDB dataset; systems are detected recursively
    datafile : str, optional
        the path to the list of systems to process
    output : str
        output directory for descriptor files
    head : Optional[str], optional
        (Supported backend: PyTorch) Task head if in multi-task mode.
    dtype : str
        Output dtype for descriptor arrays: ``"fp32"``, ``"fp64"``, or
        ``"native"``.
    **kwargs
        additional arguments

    Notes
    -----
    Descriptors are saved as 3D NumPy arrays with shape
    ``(nframes, natoms, ndesc)`` where each frame contains the descriptors for
    all atoms.

    LMDB datasets are read via :class:`~deepmd.dpmodel.utils.lmdb_data.LmdbTestData`
    without shuffling. A single-``nloc`` LMDB writes ``<basename>.npy`` in
    ascending LMDB storage order. A mixed-``nloc`` LMDB is partitioned by atom
    count in ascending ``nloc`` order, writing one ``<basename>_nloc<N>.npy``
    file per group with frames kept in ascending LMDB storage order. Colliding
    output names across discovered systems are de-duplicated with a numeric
    suffix.

    Raises
    ------
    RuntimeError
        if no valid system was found
    """
    if datafile is not None:
        with open(datafile) as datalist:
            all_sys = [line.strip() for line in datalist if line.strip()]
    elif system is not None:
        all_sys = process_systems(system)
    else:
        all_sys = []

    if len(all_sys) == 0:
        raise RuntimeError("Did not find valid system")

    # init model
    dp = DeepEval(model, head=head)

    # create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()

    for system_path in all_sys:
        log.info("# -------output of dp eval_desc------- ")
        log.info(f"# processing system : {system_path}")

        tmap = dp.get_type_map()
        for data, sys_label, nloc_val in _iter_system_data(system_path, tmap):
            if sys_label != system_path:
                log.info(f"# processing sub-group : {sys_label}")

            if dp.get_dim_fparam() > 0:
                data.add(
                    "fparam",
                    dp.get_dim_fparam(),
                    atomic=False,
                    must=False,
                    high_prec=False,
                )
            if dp.get_dim_aparam() > 0:
                data.add(
                    "aparam",
                    dp.get_dim_aparam(),
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            if dp.has_chg_spin_ebd():
                data.add(
                    "charge_spin",
                    2,
                    atomic=False,
                    must=False,
                    high_prec=False,
                )

            # get test data
            test_data = data.get_test()
            mixed_type = data.mixed_type
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
            fparam = (
                test_data["fparam"]
                if dp.get_dim_fparam() > 0 and test_data.get("find_fparam", 0.0) != 0.0
                else None
            )
            aparam = (
                test_data["aparam"]
                if dp.get_dim_aparam() > 0 and test_data.get("find_aparam", 0.0) != 0.0
                else None
            )
            eval_kwargs: dict[str, Any] = {}
            if dp.has_chg_spin_ebd() and test_data.get("find_charge_spin", 0.0) != 0.0:
                eval_kwargs["charge_spin"] = test_data["charge_spin"]

            # evaluate descriptors
            log.info(f"# evaluating descriptors for {nframes} frames")
            descriptors = dp.eval_descriptor(
                coord,
                box,
                atype,
                fparam=fparam,
                aparam=aparam,
                mixed_type=mixed_type,
                dtype=dtype,
                **eval_kwargs,
            )

            # descriptors are kept in 3D format (nframes, natoms, ndesc)

            # save descriptors
            system_name = _unique_output_name(system_path, used_names, nloc=nloc_val)
            desc_file = output_dir / f"{system_name}.npy"
            np.save(desc_file, descriptors)

            log.info(f"# descriptors saved to {desc_file}")
            log.info(f"# descriptor shape: {descriptors.shape}")
            log.info("# ----------------------------------- ")

    log.info("# eval_desc completed successfully")
