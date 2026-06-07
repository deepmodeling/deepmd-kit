#!/usr/bin/env python3
"""Fit a frozen DPA descriptor + Ridge regressor on the quickstart demo data.

Requires the DPA-3.1-3M pretrained checkpoint.  Provide it via ``--model`` or
set the ``DPA_MODEL_PATH`` environment variable.

Usage (from the demo directory)::

    python fit_evaluate.py --model /path/to/DPA-3.1-3M.pt

(or set the ``DPA_MODEL_PATH`` environment variable instead of ``--model``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels.npy"
TEST_LABELS_PATH = DATA_DIR / "test_labels.npy"
FROZEN_MODEL_PATH = HERE / "frozen_model.pth"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dp dpa fit",
        description="Quickstart: fit frozen DPA descriptor + Ridge on QM9 HOMO-LUMO gap.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to DPA-3.1-3M.pt checkpoint.  Falls back to $DPA_MODEL_PATH.",
    )
    args = parser.parse_args()

    # --- resolve model path ---
    model_path = args.model or os.environ.get("DPA_MODEL_PATH")
    if not model_path:
        print(
            "error: DPA-3.1-3M checkpoint not specified.\n"
            "  Provide it via --model or set the DPA_MODEL_PATH environment variable.\n"
            "  Example: python fit_evaluate.py --model /path/to/DPA-3.1-3M.pt",
            file=sys.stderr,
        )
        sys.exit(1)

    if not Path(model_path).is_file():
        print(f"error: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Model checkpoint: {model_path}")

    # --- verify data ---
    if not TRAIN_DIR.is_dir():
        print(
            f"error: training data not found at {TRAIN_DIR}\n"
            "  Run scripts/prepare_data.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not TEST_DIR.is_dir():
        print(
            f"error: test data not found at {TEST_DIR}\n"
            "  Run scripts/prepare_data.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- load labels ---
    train_labels = np.load(str(TRAIN_LABELS_PATH)).astype(np.float32)
    test_labels = np.load(str(TEST_LABELS_PATH)).astype(np.float32)

    # --- build model ---
    from deepmd.dpa_tools import DPAFineTuner

    model = DPAFineTuner(
        pretrained=model_path,
        model_branch="Domains_Drug",
        pooling="mean",
        predictor="linear",
        seed=42,
    )

    # --- fit ---
    print("Fitting …")
    model.fit(
        train_data=str(TRAIN_DIR),
        labels=train_labels,
        target_key="gap",
    )

    # --- evaluate ---
    print("Evaluating …")
    metrics = model.evaluate(data=str(TEST_DIR))

    print()
    print("=" * 50)
    print(f"MAE  : {metrics.mae:.4f} eV")
    print(f"R²   : {metrics.r2:.4f}")
    print(f"RMSE : {metrics.rmse:.4f} eV")
    print(f"N    : {metrics.predictions.shape[0]}")
    print("=" * 50)

    # --- freeze ---
    out = model.freeze(str(FROZEN_MODEL_PATH))
    print(f"Frozen model → {out}")


if __name__ == "__main__":
    main()
