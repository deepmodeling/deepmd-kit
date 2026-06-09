#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Minimal demo: frozen_sklearn + Ridge on QM9 HOMO–LUMO gap."""

import sys
from pathlib import (
    Path,
)

# Ensure repo root is on sys.path so `dpa_adapt` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np

from dpa_adapt import (
    DPAFineTuner,
)

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"

model = DPAFineTuner(
    pretrained="DPA-3.1-3M",
    model_branch="Domains_Drug",
    strategy="frozen_sklearn",
    predictor="linear",
    seed=42,
)
model.fit(train_data=str(DATA / "train" / "*"), target_key="gap")

m = model.evaluate(data=str(DATA / "test" / "*"))
true = np.load(DATA / "test_labels.npy")
print(f"MAE  = {m.mae:.4f} eV")
print(f"RMSE = {m.rmse:.4f} eV")
print(f"R²   = {m.r2:.4f}")
