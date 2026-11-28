# SPDX-License-Identifier: LGPL-3.0-or-later
"""Evaluate model embeddings using a trained DeePMD-kit model."""

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

import h5py
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

__all__ = ["embedding"]

log = logging.getLogger(__name__)

# Byte shuffle plus gzip gives a strong compression ratio on floating-point
# embeddings without any optional HDF5 plugin: shuffle groups the equal-order
# bytes of neighboring values so the deflate stage finds longer runs.
_HDF5_DATASET_KWARGS = {
    "compression": "gzip",
    "compression_opts": 9,
    "shuffle": True,
}


def _system_base_name(system_path: str) -> str:
    """Return an HDF5-safe base name derived from a system path."""
    base = os.path.basename(system_path.rstrip("/\\"))
    if base in ("", ".", ".."):
        base = os.path.basename(os.path.abspath(system_path).rstrip("/\\"))
    return base or "system"


def _unique_group_name(
    system_path: str,
    used_names: set[str],
    *,
    nloc: int | None = None,
) -> str:
    """
    Return a collision-free HDF5 group name derived from a system path.

    Parameters
    ----------
    system_path : str
        The source system directory or LMDB dataset.
    used_names : set[str]
        Group names already assigned within the output file.
    nloc : int, optional
        Atom count suffix for a mixed-``nloc`` LMDB sub-group.

    Returns
    -------
    str
        A unique group name based on the system's base name.
    """
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


def embedding(
    *,
    model: str,
    system: str | None,
    datafile: str | None,
    output: str = "embedding.hdf5",
    head: str | None = None,
    dtype: str = "fp32",
    **kwargs: Any,
) -> None:
    """Evaluate embeddings for the given systems and store them in one HDF5 file.

    Three embeddings are produced per system in a single forward pass: the
    per-atom ``descriptor``, the per-atom ``atomic_feature`` (the activation
    after the last fitting hidden layer), and the per-structure
    ``structural_feature`` (the masked atom-sum of ``atomic_feature``).

    Parameters
    ----------
    model : str
        Path where the model is stored.
    system : str, optional
        System directory or LMDB dataset; systems are detected recursively.
    datafile : str, optional
        Path to a file listing system directories or LMDB datasets, one per line.
    output : str
        Output HDF5 file. Each system (or each atom-count group of a mixed-nloc
        LMDB dataset) becomes a group holding the three embedding datasets.
    head : str, optional
        (Supported backend: PyTorch, PyTorch Exportable) Task head if in
        multi-task mode.
    dtype : str
        Output dtype for embedding arrays: ``"fp32"``, ``"fp64"``, or
        ``"native"``.
    **kwargs
        Additional arguments.

    Notes
    -----
    Each system supplies the inputs the model conditions on, read as
    ``dp test`` reads them: the frame parameters (``fparam``), the atomic
    parameters (``aparam``), and the charge and spin multiplicity
    (``charge_spin``). An optional one that the system does not carry falls
    back to the default of the model.

    The output HDF5 file stores one group per system. The group name is the
    system directory's base name (de-duplicated on collision), and the source
    directory is recorded in the group's ``system`` attribute. LMDB datasets
    are read via :class:`~deepmd.dpmodel.utils.lmdb_data.LmdbTestData` without
    shuffling: a single-``nloc`` LMDB produces one group named after its base
    name with frames in ascending LMDB storage order, while a mixed-``nloc``
    LMDB produces one group per atom count in ascending ``nloc`` order named
    ``<basename>_nloc<N>`` (with ``system`` attribute ``<path> [nloc=<N>]``
    and frames in ascending LMDB storage order). Each group holds the datasets
    ``descriptor`` (nframes, natoms, dim_descriptor), ``atomic_feature``
    (nframes, natoms, dim_hidden), ``structural_feature``
    (nframes, dim_hidden), and ``atom_types`` (nframes, natoms), together with
    an ``nframes`` attribute; the frame axis follows the system's frame order.
    The model ``type_map`` is stored as a file-level attribute. The three
    embedding datasets are stored using the selected ``dtype``, and all
    datasets use gzip + shuffle compression.

    Raises
    ------
    RuntimeError
        If no valid system was found.
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

    dp = DeepEval(model, head=head)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5file:
        h5file.attrs["type_map"] = np.array(
            dp.get_type_map(), dtype=h5py.string_dtype()
        )
        used_names: set[str] = set()
        for system_path in all_sys:
            log.info("# -------output of embedding------- ")
            log.info(f"# processing system : {system_path}")

            tmap = dp.get_type_map()
            for data, sys_label, nloc_val in _iter_system_data(system_path, tmap):
                if sys_label != system_path:
                    log.info(f"# processing sub-group : {sys_label}")

                # The system supplies the frame and atomic parameters and the
                # charge/spin condition the model takes; an optional one it does
                # not carry falls back to the default of the model.
                if dp.get_dim_fparam() > 0:
                    data.add(
                        "fparam",
                        dp.get_dim_fparam(),
                        atomic=False,
                        must=not dp.has_default_fparam(),
                        high_prec=False,
                    )
                if dp.get_dim_aparam() > 0:
                    data.add(
                        "aparam",
                        dp.get_dim_aparam(),
                        atomic=True,
                        must=True,
                        high_prec=False,
                    )
                if dp.has_chg_spin_ebd():
                    data.add(
                        "charge_spin",
                        2,
                        atomic=False,
                        must=not dp.has_default_chg_spin(),
                        high_prec=False,
                    )

                test_data = data.get_test()
                mixed_type = data.mixed_type
                nframes = test_data["box"].shape[0]

                coord = test_data["coord"].reshape([nframes, -1])
                box = test_data["box"]
                if not data.pbc:
                    box = None
                if mixed_type:
                    atype = test_data["type"].reshape([nframes, -1])
                else:
                    atype = test_data["type"][0]

                fparam, aparam, charge_spin = (
                    test_data[key] if test_data.get(f"find_{key}", 0.0) != 0.0 else None
                    for key in ("fparam", "aparam", "charge_spin")
                )

                log.info(f"# evaluating embeddings for {nframes} frames")
                descriptor, atomic_feature, structural_feature = dp.eval_embedding(
                    coord,
                    box,
                    atype,
                    fparam=fparam,
                    aparam=aparam,
                    charge_spin=charge_spin,
                    mixed_type=mixed_type,
                    dtype=dtype,
                )

                group_name = _unique_group_name(system_path, used_names, nloc=nloc_val)
                group = h5file.create_group(group_name)
                group.create_dataset(
                    "descriptor", data=descriptor, **_HDF5_DATASET_KWARGS
                )
                group.create_dataset(
                    "atomic_feature", data=atomic_feature, **_HDF5_DATASET_KWARGS
                )
                group.create_dataset(
                    "structural_feature",
                    data=structural_feature,
                    **_HDF5_DATASET_KWARGS,
                )
                atom_types = np.asarray(atype, dtype=np.int32)
                if atom_types.ndim == 1:
                    atom_types = np.tile(atom_types, (nframes, 1))
                group.create_dataset(
                    "atom_types", data=atom_types, **_HDF5_DATASET_KWARGS
                )
                group.attrs["nframes"] = int(nframes)
                group.attrs["system"] = str(sys_label)

                log.info(
                    f"# stored group '{group_name}': "
                    f"descriptor {descriptor.shape}, "
                    f"atomic_feature {atomic_feature.shape}, "
                    f"structural_feature {structural_feature.shape}"
                )
                log.info("# ----------------------------------- ")

    log.info(f"# embeddings saved to {output_path}")
    log.info("# embedding completed successfully")
