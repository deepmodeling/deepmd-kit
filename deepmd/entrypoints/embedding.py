# SPDX-License-Identifier: LGPL-3.0-or-later
"""Evaluate model embeddings using a trained DeePMD-kit model."""

import logging
import os
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import h5py
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


def _unique_group_name(system_path: str, used_names: set[str]) -> str:
    """
    Return a collision-free HDF5 group name derived from a system path.

    Parameters
    ----------
    system_path : str
        The source system directory.
    used_names : set[str]
        Group names already assigned within the output file.

    Returns
    -------
    str
        A unique group name based on the system directory's base name.
    """
    base = os.path.basename(system_path.rstrip("/")) or "system"
    name = base
    idx = 1
    while name in used_names:
        name = f"{base}_{idx}"
        idx += 1
    used_names.add(name)
    return name


def embedding(
    *,
    model: str,
    system: str,
    datafile: str,
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
    system : str
        System directory; systems are detected recursively.
    datafile : str
        Path to a file listing system directories, one per line.
    output : str
        Output HDF5 file. Each system becomes a group holding the three
        embedding datasets.
    head : str, optional
        (Supported backend: PyTorch) Task head if in multi-task mode.
    dtype : str
        Output dtype for embedding arrays: ``"fp32"``, ``"fp64"``, or
        ``"native"``.
    **kwargs
        Additional arguments.

    Notes
    -----
    The output HDF5 file stores one group per system. The group name is the
    system directory's base name (de-duplicated on collision), and the source
    directory is recorded in the group's ``system`` attribute. Each group holds
    the datasets ``descriptor`` (nframes, natoms, dim_descriptor),
    ``atomic_feature`` (nframes, natoms, dim_hidden),
    ``structural_feature`` (nframes, dim_hidden), and ``atom_types``
    (nframes, natoms), together with an ``nframes`` attribute; the frame axis
    follows the system's frame order. The model ``type_map`` is stored as a
    file-level attribute. The three embedding datasets are stored using the
    selected ``dtype``, and all datasets use gzip + shuffle compression.

    Raises
    ------
    RuntimeError
        If no valid system was found.
    """
    if datafile is not None:
        with open(datafile) as datalist:
            all_sys = [line.strip() for line in datalist if line.strip()]
    else:
        all_sys = expand_sys_str(system)

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
            data = DeepmdData(
                system_path,
                set_prefix="set",
                shuffle_test=False,
                type_map=tmap,
                sort_atoms=False,
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

            fparam = None
            if dp.get_dim_fparam() > 0 and "fparam" in test_data:
                fparam = test_data["fparam"]
            aparam = None
            if dp.get_dim_aparam() > 0 and "aparam" in test_data:
                aparam = test_data["aparam"]

            log.info(f"# evaluating embeddings for {nframes} frames")
            descriptor, atomic_feature, structural_feature = dp.eval_embedding(
                coord,
                box,
                atype,
                fparam=fparam,
                aparam=aparam,
                mixed_type=mixed_type,
                dtype=dtype,
            )

            group_name = _unique_group_name(system_path, used_names)
            group = h5file.create_group(group_name)
            group.create_dataset("descriptor", data=descriptor, **_HDF5_DATASET_KWARGS)
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
            group.create_dataset("atom_types", data=atom_types, **_HDF5_DATASET_KWARGS)
            group.attrs["nframes"] = int(nframes)
            group.attrs["system"] = str(system_path)

            log.info(
                f"# stored group '{group_name}': "
                f"descriptor {descriptor.shape}, "
                f"atomic_feature {atomic_feature.shape}, "
                f"structural_feature {structural_feature.shape}"
            )
            log.info("# ----------------------------------- ")

    log.info(f"# embeddings saved to {output_path}")
    log.info("# embedding completed successfully")
