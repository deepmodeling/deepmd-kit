#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Minimal demo: frozen_head fine-tuning on QM9 HOMO-LUMO gap."""

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
    strategy="frozen_head",
    property_name="gap",
    learning_rate=1e-3,
    batch_size=128,
    max_steps=5,
)
model.fit(train_data=str(DATA / "train" / "*"), valid_data=str(DATA / "test" / "*"))

pred = model.predict(data=str(DATA / "test" / "*"))
metrics = model.evaluate(data=str(DATA / "test" / "*"))

logger.info(pred.predictions)
logger.info("%s %s %s", metrics.mae, metrics.rmse, metrics.r2)
