#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Benchmark dense and Cartesian-cell neighbor searches at rcut=6.

The benchmark isolates the search itself.  For periodic systems, ghost atoms are
constructed once with the common DeePMD routine before either search is timed.
This keeps the comparison focused on the O(N^2) dense distance matrix versus the
compact cell-list candidate construction.
"""

# ruff: noqa: T201 -- benchmark progress belongs on the command line.

from __future__ import (
    annotations,
)

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


RCUT = 6.0
DENSITY = 0.05
NSEL = 128
NONPERIODIC_SIZES = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384)
PERIODIC_SIZES = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("numpy", "torch", "jax", "tensorflow"), required=True
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument(
        "--scenario", choices=("nonperiodic", "periodic"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[4]
    )
    parser.add_argument(
        "--max-method-seconds",
        type=float,
        default=5.0,
        help="Stop increasing N for an algorithm after one median exceeds this time.",
    )
    return parser.parse_args()


def make_system(nloc: int, periodic: bool) -> tuple[np.ndarray, np.ndarray, int]:
    """Create one reproducible float32 frame at constant number density."""
    from deepmd.dpmodel.utils.nlist import (
        extend_coord_with_ghosts,
    )

    rng = np.random.default_rng(20260722 + nloc + (100000 if periodic else 0))
    side = (nloc / DENSITY) ** (1.0 / 3.0)
    coord = (rng.random((1, nloc, 3)) * side).astype(np.float32)
    atype = np.zeros((1, nloc), dtype=np.int64)
    if periodic:
        box = (np.eye(3, dtype=np.float32) * side)[None, :, :]
        extended_coord, extended_atype, _ = extend_coord_with_ghosts(
            coord, atype, box, RCUT
        )
    else:
        extended_coord = coord.reshape(1, -1)
        extended_atype = atype
    return extended_coord, extended_atype, extended_atype.shape[1]


def numpy_adapter(
    coord: np.ndarray, atype: np.ndarray, nloc: int
) -> tuple[
    Callable[[], Any],
    Callable[[], Any],
    Callable[[Any], None],
    Callable[[Any], np.ndarray],
]:
    from deepmd.dpmodel.utils.default_neighbor_list import (
        _build_neighbor_list_cell,
    )
    from deepmd.dpmodel.utils.nlist import (
        build_neighbor_list,
    )

    def dense() -> Any:
        return build_neighbor_list(
            coord, atype, nloc, RCUT, [NSEL], distinguish_types=False
        )

    def cell() -> Any:
        return _build_neighbor_list_cell(coord, atype, nloc, RCUT, NSEL)

    return dense, cell, lambda value: None, np.asarray


def torch_adapter(
    coord: np.ndarray, atype: np.ndarray, nloc: int, device_name: str
) -> tuple[
    Callable[[], Any],
    Callable[[], Any],
    Callable[[Any], None],
    Callable[[Any], np.ndarray],
]:
    import torch

    from deepmd.dpmodel.utils.default_neighbor_list import (
        _build_neighbor_list_cell,
    )
    from deepmd.dpmodel.utils.nlist import (
        build_neighbor_list,
    )

    device = torch.device(device_name)
    coord_t = torch.as_tensor(coord, device=device)
    atype_t = torch.as_tensor(atype, device=device)

    def dense() -> Any:
        with torch.no_grad():
            return build_neighbor_list(
                coord_t, atype_t, nloc, RCUT, [NSEL], distinguish_types=False
            )

    def cell() -> Any:
        with torch.no_grad():
            return _build_neighbor_list_cell(coord_t, atype_t, nloc, RCUT, NSEL)

    def synchronize(value: Any) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def to_numpy(value: Any) -> np.ndarray:
        return value.detach().cpu().numpy()

    return dense, cell, synchronize, to_numpy


def jax_adapter(
    coord: np.ndarray, atype: np.ndarray, nloc: int, device_kind: str
) -> tuple[
    Callable[[], Any],
    Callable[[], Any],
    Callable[[Any], None],
    Callable[[Any], np.ndarray],
]:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from deepmd.dpmodel.utils.default_neighbor_list import (
        _build_neighbor_list_cell,
    )
    from deepmd.dpmodel.utils.nlist import (
        build_neighbor_list,
    )

    devices = [device for device in jax.devices() if device.platform == device_kind]
    if not devices:
        raise RuntimeError(f"JAX {device_kind} device is unavailable")
    coord_j = jax.device_put(jnp.asarray(coord), devices[0])
    atype_j = jax.device_put(jnp.asarray(atype), devices[0])

    def dense() -> Any:
        return build_neighbor_list(
            coord_j, atype_j, nloc, RCUT, [NSEL], distinguish_types=False
        )

    def cell() -> Any:
        return _build_neighbor_list_cell(coord_j, atype_j, nloc, RCUT, NSEL)

    def synchronize(value: Any) -> None:
        value.block_until_ready()

    return dense, cell, synchronize, np.asarray


def tensorflow_adapter(
    coord: np.ndarray, atype: np.ndarray, nloc: int, device_name: str
) -> tuple[
    Callable[[], Any],
    Callable[[], Any],
    Callable[[Any], None],
    Callable[[Any], np.ndarray],
]:
    import tensorflow as tf

    from deepmd._vendors import ndtensorflow as xp
    from deepmd.dpmodel.utils.default_neighbor_list import (
        _build_neighbor_list_cell,
    )
    from deepmd.dpmodel.utils.nlist import (
        build_neighbor_list,
    )

    tf_device = "/GPU:0" if device_name == "gpu" else "/CPU:0"
    with tf.device(tf_device):
        coord_t = tf.convert_to_tensor(coord)
        atype_t = tf.convert_to_tensor(atype)

    @tf.function(autograph=False, reduce_retracing=True)
    def dense_graph(cc: Any, aa: Any) -> Any:
        return build_neighbor_list(
            xp.asarray(cc),
            xp.asarray(aa),
            nloc,
            RCUT,
            [NSEL],
            distinguish_types=False,
        ).unwrap()

    @tf.function(autograph=False, reduce_retracing=True)
    def cell_graph(cc: Any, aa: Any) -> Any:
        return _build_neighbor_list_cell(
            xp.asarray(cc), xp.asarray(aa), nloc, RCUT, NSEL
        ).unwrap()

    def dense() -> Any:
        with tf.device(tf_device):
            return dense_graph(coord_t, atype_t)

    def cell() -> Any:
        with tf.device(tf_device):
            return cell_graph(coord_t, atype_t)

    def synchronize(value: Any) -> None:
        value.numpy()

    def to_numpy(value: Any) -> np.ndarray:
        return value.numpy()

    return dense, cell, synchronize, to_numpy


def make_adapter(
    backend: str,
    device: str,
    coord: np.ndarray,
    atype: np.ndarray,
    nloc: int,
) -> tuple[
    Callable[[], Any],
    Callable[[], Any],
    Callable[[Any], None],
    Callable[[Any], np.ndarray],
]:
    if backend == "numpy":
        if device != "cpu":
            raise RuntimeError("NumPy has no GPU backend")
        return numpy_adapter(coord, atype, nloc)
    if backend == "torch":
        return torch_adapter(coord, atype, nloc, "cuda" if device == "gpu" else "cpu")
    if backend == "jax":
        return jax_adapter(coord, atype, nloc, device)
    if backend == "tensorflow":
        return tensorflow_adapter(coord, atype, nloc, device)
    raise AssertionError(backend)


def measure(
    fn: Callable[[], Any], synchronize: Callable[[Any], None]
) -> tuple[float, int, Any]:
    """Warm once, then return a median with an adaptive repetition count."""
    warm = fn()
    synchronize(warm)
    start = time.perf_counter()
    probe = fn()
    synchronize(probe)
    probe_seconds = time.perf_counter() - start
    repeats = max(1, min(15, int(0.5 / max(probe_seconds, 1.0e-6))))
    samples = [probe_seconds]
    last = probe
    for _ in range(repeats - 1):
        start = time.perf_counter()
        last = fn()
        synchronize(last)
        samples.append(time.perf_counter() - start)
    return float(np.median(samples) * 1000.0), repeats, last


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repo))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    periodic = args.scenario == "periodic"
    sizes = PERIODIC_SIZES if periodic else NONPERIODIC_SIZES
    rows: list[dict[str, Any]] = []
    active = {"dense": True, "cell": True}

    print(
        f"backend={args.backend} device={args.device} scenario={args.scenario} "
        f"rcut={RCUT} density={DENSITY} nsel={NSEL}",
        flush=True,
    )
    for nloc in sizes:
        coord, atype, nall = make_system(nloc, periodic)
        try:
            dense, cell, synchronize, to_numpy = make_adapter(
                args.backend, args.device, coord, atype, nloc
            )
        except Exception as exc:
            print(
                f"adapter failed at N={nloc}: {type(exc).__name__}: {exc}", flush=True
            )
            break

        outputs: dict[str, Any] = {}
        for algorithm, fn in (("dense", dense), ("cell", cell)):
            if not active[algorithm]:
                continue
            try:
                elapsed_ms, repeats, output = measure(fn, synchronize)
                outputs[algorithm] = output
                rows.append(
                    {
                        "backend": args.backend,
                        "device": args.device,
                        "scenario": args.scenario,
                        "nloc": nloc,
                        "nall": nall,
                        "algorithm": algorithm,
                        "time_ms": f"{elapsed_ms:.9g}",
                        "repeats": repeats,
                        "exact_match": "",
                        "same_neighbor_set": "",
                        "status": "ok",
                    }
                )
                print(
                    f"N={nloc:6d} nall={nall:7d} {algorithm:5s} "
                    f"{elapsed_ms:11.3f} ms ({repeats} reps)",
                    flush=True,
                )
                if elapsed_ms / 1000.0 > args.max_method_seconds:
                    active[algorithm] = False
            except Exception as exc:
                active[algorithm] = False
                rows.append(
                    {
                        "backend": args.backend,
                        "device": args.device,
                        "scenario": args.scenario,
                        "nloc": nloc,
                        "nall": nall,
                        "algorithm": algorithm,
                        "time_ms": "",
                        "repeats": 0,
                        "exact_match": "",
                        "same_neighbor_set": "",
                        "status": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
                    }
                )
                print(
                    f"N={nloc:6d} nall={nall:7d} {algorithm:5s} FAILED: "
                    f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
                    flush=True,
                )

        if "dense" in outputs and "cell" in outputs:
            dense_output = to_numpy(outputs["dense"])
            cell_output = to_numpy(outputs["cell"])
            exact = np.array_equal(dense_output, cell_output)
            same_neighbor_set = exact or np.array_equal(
                np.sort(dense_output, axis=-1), np.sort(cell_output, axis=-1)
            )
            for row in rows[-2:]:
                if row["nloc"] == nloc:
                    row["exact_match"] = str(exact).lower()
                    row["same_neighbor_set"] = str(same_neighbor_set).lower()
            if not exact:
                qualifier = "ordering only" if same_neighbor_set else "different set"
                print(
                    f"WARNING: exact output mismatch at N={nloc} ({qualifier})",
                    flush=True,
                )

        del coord, atype, outputs
        gc.collect()
        if not any(active.values()):
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "backend",
                "device",
                "scenario",
                "nloc",
                "nall",
                "algorithm",
                "time_ms",
                "repeats",
                "exact_match",
                "same_neighbor_set",
                "status",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
