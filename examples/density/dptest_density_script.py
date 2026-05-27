#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Evaluate a charge-density model on validation/test datasets.

Usage
-----
python test_density_new.py model.pt /path/to/data --ratio 0.1 --output result.txt
"""

import argparse
import glob
import math
import os
import random
import sys

import dpdata
import numpy as np
from dpdata.data_type import (
    Axis,
    DataType,
)
from tqdm import (
    tqdm,
)

from deepmd.infer import (
    DeepPot,
)

# Register custom dpdata types for grid density
_GRID_DATA_TYPE = DataType(
    "grid",
    np.ndarray,
    shape=(Axis.NFRAMES, 125, 3),
    required=False,
)
_DENSITY_DATA_TYPE = DataType(
    "density",
    np.ndarray,
    shape=(Axis.NFRAMES, 125, 1),
    required=False,
)
dpdata.System.register_data_type(_GRID_DATA_TYPE)
dpdata.System.register_data_type(_DENSITY_DATA_TYPE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate charge-density DeepPot model."
    )
    parser.add_argument("model", type=str, help="Path to the model file (.pt or .pth).")
    parser.add_argument(
        "data_dir", type=str, help="Root directory of deepmd/npy datasets."
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.1,
        help="Fraction of frames to randomly sample from each system (default: 0.1).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="If provided, write screen output to this file as well.",
    )
    parser.add_argument(
        "--pred-file",
        type=str,
        default="result.d.out",
        help="File to save paired [prediction, label] array (default: result.d.out).",
    )
    return parser.parse_args()


class TeeLogger:
    """Redirect stdout to both the terminal and a file."""

    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def evaluate(
    model_path: str, data_dir: str, ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return (predictions, labels)."""
    dm = DeepPot(model_path)
    type_map = dm.get_type_map()

    pred_list, label_list = [], []
    pattern = os.path.join(data_dir, "**/type.raw")
    systems = glob.glob(pattern, recursive=True)

    if not systems:
        raise RuntimeError(f"No deepmd/npy systems found under '{data_dir}'")

    for f in tqdm(systems, desc="Systems"):
        sys_name = os.path.dirname(f)
        s = dpdata.System(sys_name, fmt="deepmd/npy", type_map=type_map)

        n_frames = len(s)
        n_sample = max(1, math.floor(n_frames * ratio))
        indices = random.sample(range(n_frames), n_sample)
        s = s.sub_system(indices)

        coord = s.data["coords"].reshape(len(s), -1)
        atype = list(s.data["atom_types"])
        cell = s.data["cells"].reshape(len(s), -1)
        grid = s.data["grid"].reshape(len(s), -1)

        density_pred = dm.eval(coord, cell, atype, grid=grid)
        density_label = s.data["density"].reshape(len(s), -1)

        pred_list.append(density_pred)
        label_list.append(density_label)
        print(f"  {sys_name:60s}  frames={n_sample}/{n_frames}")

    predictions = np.concatenate(pred_list)
    labels = np.concatenate(label_list)
    return predictions, labels


def print_summary(pred: np.ndarray, label: np.ndarray) -> None:
    diff = pred - label
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    label_mean_abs = np.mean(np.abs(label))
    label_std = np.std(label)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Number of grid points : {label.size}")
    print(f"  Label std             : {label_std:.6e}")
    print(f"  RMSE                  : {rmse:.6e}")
    print(f"  MAE                   : {mae:.6e}")
    print(f"  Mean |label|          : {label_mean_abs:.6e}")
    print(f"  epsilon_MAE (MAE/Mean|label|) : {mae / label_mean_abs:.6e}")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    if args.output:
        tee = TeeLogger(args.output)
        sys.stdout = tee
        print(f"[INFO] Screen output will also be saved to: {args.output}\n")

    print(f"[INFO] Model : {args.model}")
    print(f"[INFO] Data  : {args.data_dir}")
    print(f"[INFO] Ratio : {args.ratio}\n")

    pred, label = evaluate(args.model, args.data_dir, args.ratio)

    # Save paired predictions & labels
    out_array = np.stack([pred.reshape(-1), label.reshape(-1)], axis=1)
    np.savetxt(args.pred_file, out_array)
    print(f"\n[INFO] Paired [pred, label] saved to: {args.pred_file}")

    print_summary(pred, label)

    if args.output:
        tee.close()
        sys.stdout = tee.terminal


if __name__ == "__main__":
    main()
