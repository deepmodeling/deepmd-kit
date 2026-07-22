#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Create log-log runtime and speedup plots from benchmark CSV files."""

from __future__ import (
    annotations,
)

import argparse
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {
    "numpy": "#4C78A8",
    "torch": "#F58518",
    "jax": "#54A24B",
    "tensorflow": "#E45756",
}
LABELS = {
    "numpy": "NumPy",
    "torch": "PyTorch",
    "jax": "JAX eager",
    "tensorflow": "TensorFlow tf.function",
}
ALGORITHM_STYLE = {"dense": ("-", "o"), "cell": ("--", "^")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args()


def load_data(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("result_*.csv"))
    consolidated = directory / "default_neighbor_list_benchmark_rcut6.csv"
    if files:
        data = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    elif consolidated.exists():
        data = pd.read_csv(consolidated)
    else:
        raise FileNotFoundError(f"no benchmark CSV files in {directory}")
    data["time_ms"] = pd.to_numeric(data["time_ms"], errors="coerce")
    data = data[(data["status"] == "ok") & data["time_ms"].notna()].copy()
    return data


def setup_axes() -> tuple[plt.Figure, dict[tuple[str, str], plt.Axes]]:
    fig, grid = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)
    axes = {
        ("cpu", "nonperiodic"): grid[0, 0],
        ("cpu", "periodic"): grid[0, 1],
        ("gpu", "nonperiodic"): grid[1, 0],
        ("gpu", "periodic"): grid[1, 1],
    }
    return fig, axes


def collect_legend(
    axes: dict[tuple[str, str], plt.Axes],
) -> tuple[list[Any], list[str]]:
    """Collect each plotted series once across all benchmark panels."""
    handles_by_label: dict[str, Any] = {}
    for axis in axes.values():
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            handles_by_label.setdefault(label, handle)
    return list(handles_by_label.values()), list(handles_by_label)


def plot_runtime(data: pd.DataFrame, directory: Path) -> None:
    fig, axes = setup_axes()
    for (device, scenario), axis in axes.items():
        subset = data[(data.device == device) & (data.scenario == scenario)]
        for backend in COLORS:
            for algorithm in ("dense", "cell"):
                series = subset[
                    (subset.backend == backend) & (subset.algorithm == algorithm)
                ].sort_values("nloc")
                if series.empty:
                    continue
                linestyle, marker = ALGORITHM_STYLE[algorithm]
                axis.plot(
                    series.nloc,
                    series.time_ms,
                    color=COLORS[backend],
                    linestyle=linestyle,
                    marker=marker,
                    markersize=4.5,
                    linewidth=1.8,
                    label=f"{LABELS[backend]} {algorithm}",
                )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.set_title(f"{device.upper()} — {scenario}")
        axis.set_xlabel("Local atoms N")
        axis.set_ylabel("Neighbor-search time (ms)")
    handles, labels = collect_legend(axes)
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle(
        "DeePMD dense vs Cartesian cell-list neighbor search\n"
        "rcut=6, density=0.05, nsel=128, float32, one frame",
        fontsize=14,
    )
    for suffix in ("png",):
        fig.savefig(directory / f"neighbor_list_rcut6_runtime_loglog.{suffix}", dpi=180)
    plt.close(fig)


def plot_speedup(data: pd.DataFrame, directory: Path) -> None:
    pivot = data.pivot_table(
        index=["backend", "device", "scenario", "nloc"],
        columns="algorithm",
        values="time_ms",
        aggfunc="first",
    ).dropna(subset=["dense", "cell"])
    pivot["speedup"] = pivot["dense"] / pivot["cell"]
    pivot = pivot.reset_index()
    fig, axes = setup_axes()
    for (device, scenario), axis in axes.items():
        subset = pivot[(pivot.device == device) & (pivot.scenario == scenario)]
        for backend in COLORS:
            series = subset[subset.backend == backend].sort_values("nloc")
            if series.empty:
                continue
            axis.plot(
                series.nloc,
                series.speedup,
                color=COLORS[backend],
                marker="o",
                markersize=4.5,
                linewidth=1.8,
                label=LABELS[backend],
            )
        axis.axhline(1.0, color="black", linestyle=":", linewidth=1.2)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.set_title(f"{device.upper()} — {scenario}")
        axis.set_xlabel("Local atoms N")
        axis.set_ylabel("Speedup: dense time / cell time")
    handles, labels = collect_legend(axes)
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle(
        "Cartesian cell-list speedup over dense neighbor search\n"
        "Values above 1 favor the cell-list algorithm",
        fontsize=14,
    )
    for suffix in ("png",):
        fig.savefig(directory / f"neighbor_list_rcut6_speedup_loglog.{suffix}", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = load_data(args.directory)
    output_directory = args.output_directory or args.directory
    output_directory.mkdir(parents=True, exist_ok=True)
    plot_runtime(data, output_directory)
    plot_speedup(data, output_directory)


if __name__ == "__main__":
    main()
