#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Minimal demo: frozen_sklearn + Ridge on QM9 HOMO-LUMO gap."""

from pathlib import (
    Path,
)

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
print(f"MAE  = {m.mae:.4f} eV")
print(f"RMSE = {m.rmse:.4f} eV")
print(f"R2   = {m.r2:.4f}")
