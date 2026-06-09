# SPDX-License-Identifier: LGPL-3.0-or-later
"""Data loading, conversion, validation, and SMILES/type-map utilities.

All public names are lazily imported so that ``import dpa_adapt.data``
(and therefore ``dpa --help``) does not pull in dpdata, torch, or rdkit.
"""

__all__ = [
    "DPADataError",
    "Issue",
    "SmilesDataResult",
    "attach_labels",
    "auto_convert",
    "batch_convert",
    "check_data",
    "convert",
    "formula_to_npy",
    "load_data",
    "load_dataset",
    "read_checkpoint_type_map",
    "read_data_type_map_union",
    "read_mol_coords",
    "smiles_to_3d_coords",
    "smiles_to_npy",
    "validate_type_map_subset",
]

_LAZY = {
    "load_data": (".loader", "load_data"),
    "load_dataset": (".dataset", "load_dataset"),
    "read_checkpoint_type_map": (".type_map", "read_checkpoint_type_map"),
    "read_data_type_map_union": (".type_map", "read_data_type_map_union"),
    "validate_type_map_subset": (".type_map", "validate_type_map_subset"),
    "auto_convert": (".convert", "auto_convert"),
    "convert": (".convert", "convert"),
    "attach_labels": (".convert", "attach_labels"),
    "batch_convert": (".convert", "batch_convert"),
    "formula_to_npy": (".formula", "formula_to_npy"),
    "check_data": (".validate", "check_data"),
    "Issue": (".validate", "Issue"),
    "DPADataError": (".errors", "DPADataError"),
    "SmilesDataResult": (".smiles", "SmilesDataResult"),
    "read_mol_coords": (".smiles", "read_mol_coords"),
    "smiles_to_3d_coords": (".smiles", "smiles_to_3d_coords"),
    "smiles_to_npy": (".smiles", "smiles_to_npy"),
    "predict_records_from_data": (".smiles", "predict_records_from_data"),
    "records_from_direct_data": (".smiles", "records_from_direct_data"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        mod_name, attr_name = _LAZY[name]
        mod = importlib.import_module(mod_name, __package__)
        attr = getattr(mod, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
