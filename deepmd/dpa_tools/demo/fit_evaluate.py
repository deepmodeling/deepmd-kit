#!/usr/bin/env python3
"""Fit a pretrained DPA descriptor on the quickstart QM9 demo data.

All four fine-tuning strategies are demonstrated.  The default
(``frozen_sklearn``) runs on CPU in under 5 minutes and requires no GPU.
``linear_probe``, ``finetune``, and ``mft`` use ``dp --pt train`` under the
hood and need a GPU to finish in reasonable time.

Usage (from the demo directory)::

    python fit_evaluate.py --model /path/to/DPA-3.1-3M.pt
    python fit_evaluate.py --model /path/to/DPA-3.1-3M.pt --strategy finetune

Set ``DPA_MODEL_PATH`` instead of ``--model`` to avoid typing it every time.
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_model(args: argparse.Namespace) -> str:
    path = args.model or os.environ.get("DPA_MODEL_PATH")
    if not path:
        print(
            "error: DPA checkpoint not specified.\n"
            "  Provide it via --model or set $DPA_MODEL_PATH.\n"
            "  Example: python fit_evaluate.py --model /path/to/DPA-3.1-3M.pt",
            file=sys.stderr,
        )
        sys.exit(1)
    if not Path(path).is_file():
        print(f"error: model file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return path


def _verify_data() -> None:
    for name, d in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
        if not d.is_dir():
            print(
                f"error: {name} data not found at {d}\n"
                "  Run scripts/prepare_data.py first.",
                file=sys.stderr,
            )
            sys.exit(1)


def _load_labels() -> tuple[np.ndarray, np.ndarray]:
    train = np.load(str(TRAIN_LABELS_PATH)).astype(np.float32)
    test = np.load(str(TEST_LABELS_PATH)).astype(np.float32)
    return train, test


def _print_metrics(metrics, label: str = "") -> None:
    tag = f" [{label}]" if label else ""
    print()
    print("=" * 50)
    print(f"MAE{tag}  : {metrics.mae:.4f} eV")
    print(f"R²{tag}   : {metrics.r2:.4f}")
    print(f"RMSE{tag} : {metrics.rmse:.4f} eV")
    print(f"N{tag}    : {metrics.predictions.shape[0]}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Strategy 1 — frozen_sklearn (default, CPU)
# ---------------------------------------------------------------------------


def demo_frozen_sklearn(model_path: str, train_labels: np.ndarray) -> None:
    """Freeze the DPA backbone, extract descriptors once, fit a sklearn Ridge.

    Fastest iteration.  No GPU, no ``dp train`` subprocess.  The frozen
    bundle (``.pth``) is portable and can be loaded with ``DPAPredictor``.
    """
    from deepmd.dpa_tools import DPAFineTuner

    model = DPAFineTuner(
        pretrained=model_path,
        model_branch="Domains_Drug",
        strategy="frozen_sklearn",
        predictor="linear",   # "linear" (Ridge) | "rf" | "mlp"
        pooling="mean",        # "mean" | "sum" | "mean+std" | "mean+std+max+min"
        seed=42,
    )

    print("frozen_sklearn — fitting …")
    model.fit(
        train_data=str(TRAIN_DIR),
        labels=train_labels,
        target_key="gap",
    )

    print("frozen_sklearn — evaluating …")
    metrics = model.evaluate(data=str(TEST_DIR))
    _print_metrics(metrics, "frozen_sklearn")

    out = model.freeze(str(FROZEN_MODEL_PATH))
    print(f"Frozen model → {out}")


# ---------------------------------------------------------------------------
# Strategy 2 — linear_probe (GPU recommended)
# ---------------------------------------------------------------------------


def demo_linear_probe(model_path: str) -> None:
    """Freeze the DPA backbone, train only a neural property fitting net.

    Uses ``dp --pt train --finetune`` under the hood.  A GPU is recommended.
    """
    from deepmd.dpa_tools import DPAFineTuner

    model = DPAFineTuner(
        pretrained=model_path,
        strategy="linear_probe",
        property_name="gap",
        task_dim=1,
        intensive=True,
        output_dir=str(HERE / "output_lp"),
    )

    print("linear_probe — fitting (dp --pt train) …")
    model.fit(
        train_data=str(TRAIN_DIR),
        valid_data=str(TEST_DIR),
        target_key="gap",
    )

    print("linear_probe — evaluating …")
    metrics = model.evaluate(data=str(TEST_DIR))
    _print_metrics(metrics, "linear_probe")


# ---------------------------------------------------------------------------
# Strategy 3 — finetune (GPU recommended)
# ---------------------------------------------------------------------------


def demo_finetune(model_path: str) -> None:
    """Load the pretrained backbone and fine-tune the full network.

    Uses ``dp --pt train --finetune`` under the hood.  A GPU is strongly
    recommended — this trains all parameters.
    """
    from deepmd.dpa_tools import DPAFineTuner

    model = DPAFineTuner(
        pretrained=model_path,
        strategy="finetune",
        property_name="gap",
        task_dim=1,
        intensive=True,
        output_dir=str(HERE / "output_ft"),
    )

    print("finetune — fitting (dp --pt train) …")
    model.fit(
        train_data=str(TRAIN_DIR),
        valid_data=str(TEST_DIR),
        target_key="gap",
    )

    print("finetune — evaluating …")
    metrics = model.evaluate(data=str(TEST_DIR))
    _print_metrics(metrics, "finetune")


# ---------------------------------------------------------------------------
# Strategy 4 — mft (multi-task fine-tuning, GPU + aux data required)
# ---------------------------------------------------------------------------


def demo_mft(model_path: str) -> None:
    """Multi-task fine-tuning: property head + auxiliary force-field head.

    Requires auxiliary training data (e.g. SPICE2) via ``--aux-data``.
    Prevents representation collapse on small property datasets.
    """
    from deepmd.dpa_tools import DPAFineTuner

    model = DPAFineTuner(
        pretrained=model_path,
        strategy="mft",
        property_name="gap",
        aux_branch="MP_traj_v024_alldata_mixu",
    )

    print("mft — fitting …")
    # NOTE: you must supply real aux_data; this is a placeholder.
    model.fit(
        train_data=str(TRAIN_DIR),
        aux_data=str(HERE / "data" / "aux"),   # ← replace with your aux data path
    )

    print("mft — evaluating …")
    metrics = model.evaluate(data=str(TEST_DIR))
    _print_metrics(metrics, "mft")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dp dpa fit",
        description="Quickstart: fit a DPA model on QM9 HOMO-LUMO gap.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to DPA checkpoint (.pt).  Falls back to $DPA_MODEL_PATH.",
    )
    parser.add_argument(
        "--strategy",
        default="frozen_sklearn",
        choices=["frozen_sklearn", "linear_probe", "finetune", "mft"],
        help="Fine-tuning strategy (default: %(default)s).",
    )
    args = parser.parse_args()

    model_path = _resolve_model(args)
    _verify_data()
    train_labels, _test_labels = _load_labels()

    print(f"Model   : {model_path}")
    print(f"Strategy: {args.strategy}")

    if args.strategy == "frozen_sklearn":
        demo_frozen_sklearn(model_path, train_labels)
    elif args.strategy == "linear_probe":
        demo_linear_probe(model_path)
    elif args.strategy == "finetune":
        demo_finetune(model_path)
    elif args.strategy == "mft":
        demo_mft(model_path)


if __name__ == "__main__":
    main()
