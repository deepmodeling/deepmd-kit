#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
)

from deepmd_property_tools import (
    PropertyPredict,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "DATA"
MODEL_PATH = ROOT / "exp_property_20" / "model.ckpt-10.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Train first; checkpoint not found: {MODEL_PATH}")

predictor = PropertyPredict(load_model=MODEL_PATH)

y_pred = predictor.predict(
    {
        "dataset": DATA_DIR / "dataset_demo.csv",
    },
    save_path=ROOT / "pred_property_20",
)

print(y_pred)
