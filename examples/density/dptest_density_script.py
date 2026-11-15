#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Evaluate a charge-density model on validation/test datasets.

Usage
-----
python dptest_density_script.py model.pt /path/to/data --ratio 0.1 --output result.txt
"""

import argparse
import glob
import math
import os
import random
import sys

import numpy as np

from deepmd.infer import (
    DeepEval,
)
from deepmd.utils.data import (
    DeepmdData,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a charge-density model on deepmd/npy datasets."
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

    def __init__(self, filepath: str) -> None:
        self.terminal = sys.stdout
        self.log = open(filepath, "w")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()


def evaluate(
    model_path: str, data_dir: str, ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return (predictions, labels)."""
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    # DeepEval dispatches density models to DeepDensity automatically.
    dm = DeepEval(model_path)
    type_map = dm.get_type_map()

    pred_list, label_list = [], []
    pattern = os.path.join(data_dir, "**/type.raw")
    systems = glob.glob(pattern, recursive=True)

    if not systems:
        raise RuntimeError(f"No deepmd/npy systems found under '{data_dir}'")

    for f in systems:
        sys_name = os.path.dirname(f)
        data = DeepmdData(sys_name, type_map=type_map)
        data.add(
            "grid",
            3,
            atomic=False,
            must=True,
            high_prec=True,
            special_shape="frame_major",
        )
        data.add(
            "density",
            1,
            atomic=False,
            must=True,
            high_prec=True,
            special_shape="frame_major",
        )
        test_data = data.get_test()

        n_frames = test_data["coord"].shape[0]
        n_sample = max(1, math.floor(n_frames * ratio))
        indices = random.sample(range(n_frames), n_sample)

        coord = test_data["coord"][indices].reshape(n_sample, -1)
        if data.mixed_type:
            atype = test_data["type"][indices].reshape(n_sample, -1)
        else:
            atype = test_data["type"][0]
        cell = test_data["box"][indices] if data.pbc else None
        grid = test_data["grid"][indices].reshape(n_sample, -1, 3)

        density_pred = dm.eval(
            coord,
            cell,
            atype,
            grid=grid,
            mixed_type=data.mixed_type,
        )
        density_label = test_data["density"][indices].reshape(n_sample, -1)

        # Flatten each system before concatenating so systems may carry
        # different grid extents.
        pred_list.append(density_pred.reshape(-1))
        label_list.append(density_label.reshape(-1))
        sys.stdout.write(f"  {sys_name:60s}  frames={n_sample}/{n_frames}" + "\n")

    predictions = np.concatenate(pred_list)
    labels = np.concatenate(label_list)
    return predictions, labels


def print_summary(pred: np.ndarray, label: np.ndarray) -> None:
    diff = pred - label
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    label_mean_abs = np.mean(np.abs(label))
    label_std = np.std(label)

    sys.stdout.write("\n" + "=" * 60 + "\n")
    sys.stdout.write("Summary" + "\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write(f"  Number of grid points : {label.size}" + "\n")
    sys.stdout.write(f"  Label std             : {label_std:.6e}" + "\n")
    sys.stdout.write(f"  RMSE                  : {rmse:.6e}" + "\n")
    sys.stdout.write(f"  MAE                   : {mae:.6e}" + "\n")
    sys.stdout.write(f"  Mean |label|          : {label_mean_abs:.6e}" + "\n")
    sys.stdout.write(
        f"  epsilon_MAE (MAE/Mean|label|) : {mae / label_mean_abs:.6e}" + "\n"
    )
    sys.stdout.write("=" * 60 + "\n")


def main() -> None:
    args = parse_args()

    if args.output:
        tee = TeeLogger(args.output)
        sys.stdout = tee
        sys.stdout.write(f"[INFO] Screen output will also be saved to: {args.output}\n")

    sys.stdout.write(f"[INFO] Model : {args.model}" + "\n")
    sys.stdout.write(f"[INFO] Data  : {args.data_dir}" + "\n")
    sys.stdout.write(f"[INFO] Ratio : {args.ratio}\n")

    pred, label = evaluate(args.model, args.data_dir, args.ratio)

    # Save paired predictions & labels
    out_array = np.stack([pred.reshape(-1), label.reshape(-1)], axis=1)
    np.savetxt(args.pred_file, out_array)
    sys.stdout.write(f"\n[INFO] Paired [pred, label] saved to: {args.pred_file}" + "\n")

    print_summary(pred, label)

    if args.output:
        tee.close()
        sys.stdout = tee.terminal


if __name__ == "__main__":
    main()
