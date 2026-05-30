#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
)

from deepmd_property_tools import (
    PropertyPredict,
    PropertyTrain,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "DATA"
EXP_DIR = ROOT / "exp_property_20"
PRED_DIR = ROOT / "pred_property_20"
PRETRAINED_MODEL = "DPA-3.2-5M"
TRAIN_DATA = {
    "dataset": DATA_DIR / "dataset_demo.csv",
}
PREDICT_DATA = {
    "dataset": DATA_DIR / "dataset_demo.csv",
}

trainer = PropertyTrain(
    task="regression",
    data_type="molecule",
    property_name="Property",
    property_col="Property",
    save_path=EXP_DIR,
    epochs=1,
    numb_steps=10,
    batch_size=1,
    model_name="dpa3",
    model_size="5m",
    freeze=False,
    finetune=PRETRAINED_MODEL,
    use_pretrain_script=False,
    input_updates={
        "learning_rate": {
            "type": "exp",
            "decay_steps": 1000,
            "start_lr": 1e-4,
            "stop_lr": 1e-6,
            "warmup_steps": 0,
        }
    },
)

trainer.fit(TRAIN_DATA)

checkpoints = sorted(
    EXP_DIR.glob("model.ckpt-*.pt"), key=lambda path: path.stat().st_mtime
)
if not checkpoints:
    raise FileNotFoundError(f"No checkpoint found in {EXP_DIR}")
model_path = checkpoints[-1]
print(f"Using trained model for prediction: {model_path}")

predictor = PropertyPredict(load_model=model_path)
y_pred = predictor.predict(PREDICT_DATA, save_path=PRED_DIR)
print(y_pred)
