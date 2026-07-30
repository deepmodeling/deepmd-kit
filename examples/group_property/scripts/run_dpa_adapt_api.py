#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Train a lightweight grouped embedding regressor through the dpa-adapt API."""

from __future__ import (
    annotations,
)

import logging
from pathlib import (
    Path,
)

import prepare_data

from dpa_adapt import (
    DPAFineTuner,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEMO_DIR = Path(__file__).resolve().parent.parent
DATA = DEMO_DIR / "data"
TARGET = "target_property"

if not (DATA / "train").is_dir() or not (DATA / "valid").is_dir():
    prepare_data.main()

model = DPAFineTuner(
    pretrained="DPA-3.1-3M",
    model_branch="Domains_Drug",
    strategy="frozen_sklearn",
    predictor="linear",
    seed=42,
)
model.fit(train_data=str(DATA / "train" / "systems" / "*"), target_key=TARGET)

metrics = model.evaluate(data=str(DATA / "valid" / "systems" / "*"))
logger.info("MAE  = %.4f", metrics.mae)
logger.info("RMSE = %.4f", metrics.rmse)
logger.info("R2   = %.4f", metrics.r2)
