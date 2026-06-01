# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA tools — fine-tuning, descriptor extraction, cross-validation, and data
utilities for DPA-3 pretrained models.
"""

__version__ = "0.1.0"

from .conditions import ConditionManager, DPAConditionError
from .cv import cross_validate, train_test_split
from .data import (
    SmilesDataResult,
    attach_labels,
    auto_convert,
    batch_convert,
    check_data,
    convert,
    load_dataset,
    smiles_to_npy,
)
from .finetuner import DPAFineTuner, extract_descriptors
from .mft import MFTFineTuner
from .predictor import DPAPredictor
from .trainer import DPATrainer

__all__ = [
    "ConditionManager",
    "DPAConditionError",
    "DPAFineTuner",
    "DPAPredictor",
    "DPATrainer",
    "MFTFineTuner",
    "SmilesDataResult",
    "attach_labels",
    "auto_convert",
    "batch_convert",
    "check_data",
    "convert",
    "cross_validate",
    "extract_descriptors",
    "load_dataset",
    "smiles_to_npy",
    "train_test_split",
]
