#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Minimal demo: frozen_sklearn + Ridge on QM9 HOMO-LUMO gap."""

import logging
from pathlib import (
    Path,
)

from dpa_adapt import (
    DPAFineTuner,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

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
logger.info("MAE  = %.4f eV", m.mae)
logger.info("RMSE = %.4f eV", m.rmse)
logger.info("R2   = %.4f", m.r2)
