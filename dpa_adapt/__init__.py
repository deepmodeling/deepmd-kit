# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA tools — fine-tuning, descriptor extraction, cross-validation, and data
utilities for DPA-3 pretrained models.

All public names are lazily imported: ``import dpa_adapt`` does not load
torch, dpdata, or any other heavy dependency until you actually access
a specific class or function.
"""

__version__ = "0.1.0"

_LAZY = {
    "Assembly": (".grouped", "Assembly"),
    "ComponentSpec": (".grouped", "ComponentSpec"),
    "GroupSpec": (".grouped", "GroupSpec"),
    "PoolMask": (".grouped", "PoolMask"),
    "SiteSelector": (".grouped", "SiteSelector"),
    "SubstitutionSpec": (".grouped", "SubstitutionSpec"),
    "ConditionManager": (".conditions", "ConditionManager"),
    "DPAConditionError": (".conditions", "DPAConditionError"),
    "cross_validate": (".cv", "cross_validate"),
    "train_test_split": (".cv", "train_test_split"),
    "SmilesDataResult": (".data", "SmilesDataResult"),
    "attach_labels": (".data", "attach_labels"),
    "check_data": (".data", "check_data"),
    "convert": (".data", "convert"),
    "formula_to_npy": (".data", "formula_to_npy"),
    "load_dataset": (".data", "load_dataset"),
    "smiles_to_npy": (".data", "smiles_to_npy"),
    "DPAFineTuner": (".finetuner", "DPAFineTuner"),
    "extract_descriptors": (".finetuner", "extract_descriptors"),
    "MFTFineTuner": (".mft", "MFTFineTuner"),
    "DPAPredictor": (".predictor", "DPAPredictor"),
    "DPATrainer": (".trainer", "DPATrainer"),
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> object:
    if name in _LAZY:
        import importlib

        mod_name, attr_name = _LAZY[name]
        mod = importlib.import_module(mod_name, __package__)
        attr = getattr(mod, attr_name)
        # Cache in the module namespace so __getattr__ is only called once
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
