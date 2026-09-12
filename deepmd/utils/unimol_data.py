# SPDX-License-Identifier: LGPL-3.0-or-later
"""Convert Uni-Mol pretraining data into a deepmd LMDB dataset.

Uni-Mol ships its molecular pretraining set as a single LMDB file whose values
are pickled dicts, ``{"atoms": [element symbols], "coordinates": [n x 3 arrays],
"smi": str}``, with roughly ten RDKit conformers per molecule. deepmd reads a
different LMDB layout, so the data is converted once, offline, rather than
taught to the training loop.

One conformer becomes one frame, which is what makes the conversion streaming
and lets ordinary frame sampling stand in for Uni-Mol's per-epoch conformer
draw. Frames of the same molecule share a system id, so they can be kept
together if needed.

Upstream also appends a two-dimensional RDKit conformer to the pool while
loading. That belongs here rather than in the training loop, so the data path
never needs RDKit; pass ``add_2d_conformer=True`` to reproduce it.
"""

import argparse
import logging
import os
import pickle
import shutil
from collections.abc import (
    Iterator,
)
from typing import (
    Any,
)

import numpy as np

__all__ = ["convert_unimol_lmdb", "read_unimol_lmdb"]

log = logging.getLogger(__name__)


def _encode_array(arr: np.ndarray) -> dict[str, Any]:
    """Encode an array the way a deepmd LMDB frame stores it."""
    return {
        "nd": None,
        "type": str(arr.dtype),
        "kind": "",
        "shape": list(arr.shape),
        "data": arr.tobytes(),
    }


def read_unimol_lmdb(path: str) -> Iterator[dict[str, Any]]:
    """Stream molecules out of a Uni-Mol LMDB file.

    Parameters
    ----------
    path : str
        Path to the ``.lmdb`` file, which upstream writes as a single file
        rather than a directory.

    Yields
    ------
    dict
        ``atoms``, ``coordinates`` and ``smi`` as stored upstream.
    """
    import lmdb

    env = lmdb.open(
        path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False
    )
    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                # The upstream records are pickles written by the dataset
                # authors; only convert files you trust.
                yield pickle.loads(value)
    finally:
        env.close()


def _conformers(record: dict[str, Any], add_2d_conformer: bool) -> list[np.ndarray]:
    conformers = [np.asarray(c, dtype=np.float32) for c in record["coordinates"]]
    if add_2d_conformer:
        from rdkit import (
            Chem,
        )
        from rdkit.Chem import (
            AllChem,
        )

        mol = Chem.AddHs(Chem.MolFromSmiles(record["smi"]))
        AllChem.Compute2DCoords(mol)
        coords = mol.GetConformer().GetPositions().astype(np.float32)
        conformers.append(coords[: len(record["atoms"])])
    return conformers


def convert_unimol_lmdb(
    src: str,
    dst: str,
    type_map: list[str] | None = None,
    add_2d_conformer: bool = False,
    max_molecules: int | None = None,
    max_conformers: int | None = None,
    map_size: int = 1024**4,
) -> dict[str, int]:
    """Write a deepmd LMDB dataset from a Uni-Mol one.

    Parameters
    ----------
    src : str
        The Uni-Mol ``.lmdb`` file.
    dst : str
        Directory for the deepmd dataset, replaced if it exists.
    type_map : list[str], optional
        Element names for the output. Defaults to Uni-Mol's own 26 elements.
    add_2d_conformer : bool
        Append the RDKit two-dimensional conformer to each pool, as upstream
        does while loading. Needs RDKit.
    max_molecules : int, optional
        Stop after this many molecules, which is useful for a trial run.
    max_conformers : int, optional
        Keep at most this many conformers per molecule.
    map_size : int
        Maximum size of the output database.

    Returns
    -------
    dict
        Counts of molecules, frames and skipped records.
    """
    import lmdb
    import msgpack

    from deepmd.dpmodel.descriptor.unimol import (
        UNIMOL_ELEMENTS,
    )

    names = list(type_map) if type_map is not None else list(UNIMOL_ELEMENTS)
    index_of = {sym: i for i, sym in enumerate(names)}

    if os.path.exists(dst):
        shutil.rmtree(dst)
    env = lmdb.open(dst, map_size=map_size)
    fmt = "012d"
    frame_idx = 0
    molecules = 0
    skipped = 0
    frame_system_ids: list[int] = []
    frame_nlocs: list[int] = []

    try:
        with env.begin(write=True) as txn:
            for record in read_unimol_lmdb(src):
                if max_molecules is not None and molecules >= max_molecules:
                    break
                atoms = [str(a) for a in record["atoms"]]
                if len(atoms) < 2 or any(a not in index_of for a in atoms):
                    # The descriptor cannot tell a single real atom from
                    # padding, and an unmapped element would silently become
                    # [UNK]; skip both rather than write something misleading.
                    skipped += 1
                    continue
                atom_types = np.array([index_of[a] for a in atoms], dtype=np.int64)
                atom_numbs = [int((atom_types == i).sum()) for i in range(len(names))]
                pool = _conformers(record, add_2d_conformer)
                if max_conformers is not None:
                    pool = pool[:max_conformers]
                for coords in pool:
                    if coords.shape[0] != len(atoms):
                        skipped += 1
                        continue
                    frame = {
                        "atom_numbs": atom_numbs,
                        "atom_names": names,
                        "atom_types": _encode_array(atom_types),
                        "orig": _encode_array(np.zeros(3, dtype=np.float64)),
                        # A zero cell marks a molecule: the descriptor refuses
                        # periodic images anyway.
                        "cells": _encode_array(np.zeros((3, 3), dtype=np.float64)),
                        "coords": _encode_array(coords.astype(np.float64)),
                    }
                    txn.put(
                        format(frame_idx, fmt).encode(),
                        msgpack.packb(frame, use_bin_type=True),
                    )
                    frame_system_ids.append(molecules)
                    frame_nlocs.append(len(atoms))
                    frame_idx += 1
                molecules += 1
            metadata = {
                "nframes": frame_idx,
                "frame_idx_fmt": fmt,
                "type_map": names,
                "frame_system_ids": frame_system_ids,
                "frame_nlocs": frame_nlocs,
                "system_info": {"nframes": frame_idx},
            }
            txn.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
    finally:
        env.close()
    return {"molecules": molecules, "frames": frame_idx, "skipped": skipped}


def main(args: list[str] | None = None) -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="Convert a Uni-Mol pretraining LMDB into a deepmd LMDB dataset."
    )
    parser.add_argument("src", help="the Uni-Mol .lmdb file")
    parser.add_argument("dst", help="output directory for the deepmd dataset")
    parser.add_argument(
        "--add-2d-conformer",
        action="store_true",
        help="append the RDKit 2D conformer to every pool, as upstream does",
    )
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--max-conformers", type=int, default=None)
    parsed = parser.parse_args(args)
    counts = convert_unimol_lmdb(
        parsed.src,
        parsed.dst,
        add_2d_conformer=parsed.add_2d_conformer,
        max_molecules=parsed.max_molecules,
        max_conformers=parsed.max_conformers,
    )
    logging.basicConfig(level=logging.INFO)
    log.info(
        "converted %d molecules into %d frames (%d records skipped)",
        counts["molecules"],
        counts["frames"],
        counts["skipped"],
    )


if __name__ == "__main__":
    main()
