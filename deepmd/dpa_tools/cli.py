# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dispatch for ``dp dpa`` subcommands.

This module is imported lazily by ``deepmd.entrypoints.main`` only when
``dp dpa ...`` is invoked — never at ``dp`` startup, so ``torch`` and the
rest of the DPA stack are not loaded until needed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Sequence

import numpy as np

_LOG = logging.getLogger("dpa_tools")


def _maybe_split_list(val: str | None) -> list[str] | None:
    """``"a,b,c"`` → ``["a","b","c"]``; ``None`` → ``None``."""
    if val is None:
        return None
    return [x.strip() for x in val.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Subcommand handlers — each lazy-imports its dependencies
# ---------------------------------------------------------------------------


def _cmd_fit(args: argparse.Namespace) -> int:
    from deepmd.dpa_tools import DPAFineTuner

    train = _maybe_split_list(args.train_data) or [args.train_data]
    valid = _maybe_split_list(args.valid_data) if args.valid_data else None
    type_map = _maybe_split_list(args.type_map)

    # Parse target_key: comma-separated → list[str] (multi-property),
    # single value → str (single-property, backward compat).
    target_keys = _maybe_split_list(args.target_key)
    if target_keys is None:
        target_key = "property"
        prop_name = "property"
    elif len(target_keys) == 1:
        target_key = target_keys[0]
        prop_name = target_key
    else:
        target_key = target_keys
        prop_name = target_keys[0]

    model = DPAFineTuner(
        pretrained=args.pretrained,
        model_branch=args.model_branch,
        predictor=args.predictor,
        pooling=args.pooling,
        seed=args.seed,
        strategy=args.strategy,
        property_name=prop_name,
        task_dim=args.task_dim,
        intensive=args.intensive,
        learning_rate=args.learning_rate,
        stop_lr=args.stop_lr,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        save_freq=args.save_freq,
        disp_freq=args.disp_freq,
        # MFT
        aux_branch=args.aux_branch,
        aux_prob=args.aux_prob,
        aux_type_map=_maybe_split_list(args.aux_type_map),
        downstream_type_map=_maybe_split_list(args.downstream_type_map),
        downstream_task_type=args.downstream_task_type,
        aux_batch_size=args.aux_batch_size,
        downstream_batch_size=args.downstream_batch_size,
    )
    aux_data = (_maybe_split_list(args.aux_data) or [args.aux_data]
                if args.aux_data else None)
    model.fit(train_data=train, valid_data=valid, type_map=type_map,
              target_key=target_key, aux_data=aux_data)
    if args.strategy == "frozen_sklearn":
        out = model.freeze(args.output)
        _LOG.info("Frozen model → %s", out)
    else:
        _LOG.info("Checkpoint → %s", args.output_dir)
    return 0


def _cmd_cv(args: argparse.Namespace) -> int:
    from deepmd.dpa_tools import DPAFineTuner, cross_validate, load_dataset

    systems = load_dataset(args.data, label_key=args.label_key)
    print(f"{len(systems)} systems")

    model = DPAFineTuner(
        pretrained=args.pretrained,
        model_branch=args.model_branch,
        predictor=args.predictor,
        pooling=args.pooling,
        seed=args.seed,
    )
    result = cross_validate(
        model, systems,
        label_key=args.label_key,
        cv=args.cv if args.cv == "holdout" else int(args.cv),
        group_by=args.group_by or "formula",
        granularity=args.granularity,
        seed=args.seed,
    )
    a = result["aggregate"]
    print(f"R²  = {a.get('r2_mean', float('nan')):.4f} ± {a.get('r2_std', float('nan')):.4f}")
    print(f"MAE = {a.get('mae_mean', float('nan')):.4f} ± {a.get('mae_std', float('nan')):.4f}")
    print(f"RMSE= {a.get('rmse_mean', float('nan')):.4f} ± {a.get('rmse_std', float('nan')):.4f}")
    print(f"n   = {result['n_independent']} independent groups")
    for w in result.get("warnings", []):
        print(f"[!] {w}")
    return 0


def _cmd_extract_descriptors(args: argparse.Namespace) -> int:
    from deepmd.dpa_tools.finetuner import extract_descriptors

    X = extract_descriptors(
        args.data,
        pretrained=args.pretrained,
        model_branch=args.model_branch,
        pooling=args.pooling,
        cache=not args.no_cache,
    )
    np.save(args.output, X)
    print(f"Descriptors shape={X.shape} → {args.output}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from deepmd.dpa_tools import DPAPredictor

    predictor = DPAPredictor(args.model)
    result = predictor.predict(args.data)
    np.save(args.output, result.predictions)
    _LOG.info("Predictions shape=%s → %s", result.predictions.shape, args.output)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from deepmd.dpa_tools import DPAPredictor

    predictor = DPAPredictor(args.model)
    metrics = predictor.evaluate(args.data)
    print(f"MAE  : {metrics.mae:.6f}")
    print(f"RMSE : {metrics.rmse:.6f}")
    print(f"R²   : {metrics.r2:.6f}")
    print(f"N    : {metrics.predictions.shape[0]}")
    return 0


def _cmd_data_convert(args: argparse.Namespace) -> int:
    import glob as _glob

    type_map = _maybe_split_list(args.type_map)
    input_val = args.input

    # Detect glob patterns — batch mode.
    if any(ch in input_val for ch in "*?["):
        from deepmd.dpa_tools import batch_convert

        outputs = batch_convert(
            glob_pattern=input_val, output_dir=args.output, fmt=args.fmt or "auto",
            type_map=type_map, validate=args.validate, strict=args.strict,
        )
        _LOG.info("Wrote %d deepmd/npy dirs under %s", len(outputs), args.output)
        return 0

    # Single-file mode.
    from deepmd.dpa_tools.data.convert import auto_convert

    result = auto_convert(
        input_path=input_val,
        output_dir=args.output,
        fmt=args.fmt,
        type_map=type_map,
        property_name=args.property_name,
        property_col=args.property_col,
        train_ratio=args.train_ratio,
        smiles_col=args.smiles_col,
        mol_dir=args.mol_dir,
        seed=args.seed,
        poscar=args.poscar,
        formula_col=args.formula_col,
        base_element=args.base_element,
        sets=args.sets,
        overwrite=args.overwrite,
        validate=args.validate,
        strict=args.strict,
        verbose=False,
    )
    if result["method"] == "smiles":
        print(f"Train systems: {len(result['train_systems'])}")
        print(f"Valid systems: {len(result['valid_systems'])}")
        print(f"Type map     : {result['type_map']}")
        print(f"Samples used : {result['samples_used']}")
        print(f"Failed rows  : {len(result['failed_rows'])}")
        print(f"Skipped zero : {result['skipped_zero']}")
        print(f"Skipped overlap: {result['skipped_overlap']}")
    else:
        _LOG.info("Wrote deepmd/npy → %s", result["output_dir"])
    return 0


def _cmd_data_validate(args: argparse.Namespace) -> int:
    from deepmd.dpa_tools import check_data
    from deepmd.dpa_tools.data.loader import load_data

    systems = load_data(args.data)
    issues = check_data(systems, strict=False)
    if not issues:
        print(f"OK: {len(systems)} system(s) clean.")
        return 0
    n_err = sum(1 for i in issues if i.severity == "error")
    for i in issues:
        tag = "ERROR" if i.severity == "error" else "warn"
        print(f"[{tag}] {i.system}/{i.set_dir} :: {i.description}")
    print(f"\n{len(issues)} issue(s): {n_err} error, {len(issues) - n_err} warning")
    return 1 if (n_err > 0 or (args.strict and issues)) else 0


def _cmd_data_attach_labels(args: argparse.Namespace) -> int:
    from deepmd.dpa_tools import attach_labels
    from deepmd.dpa_tools.data.loader import load_data

    values = np.load(args.values)
    if args.head_json:
        head = json.loads(args.head)
    else:
        head = args.head
    systems = load_data(args.data)
    if len(systems) != 1:
        _LOG.warning(
            "attach-labels: expected 1 system from %r, got %d; "
            "attaching to first.",
            args.data, len(systems),
        )
    attach_labels(systems[0], head=head, values=values)
    _LOG.info("Labels attached to %s", args.data)
    return 0


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_DISPATCH = {
    "extract-descriptors": _cmd_extract_descriptors,
    "fit": _cmd_fit,
    "cv": _cmd_cv,
    "predict": _cmd_predict,
    "evaluate": _cmd_evaluate,
}

_DATA_DISPATCH = {
    "convert": _cmd_data_convert,
    "validate": _cmd_data_validate,
    "attach-labels": _cmd_data_attach_labels,
}


# ---------------------------------------------------------------------------
# Entry point (called from deepmd.entrypoints.main)
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    """Dispatch a ``dp dpa`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from the ``dp`` CLI.  Must carry ``dpa_command``
        and, for data subcommands, ``dpa_data_command``.

    Raises
    ------
    SystemExit
        Propagated from subcommand handlers on failure.
    """
    from deepmd.dpa_tools.data.errors import DPADataError

    try:
        if args.dpa_command == "data":
            handler = _DATA_DISPATCH.get(args.dpa_data_command)
            if handler is None:
                print(f"Unknown data command: {args.dpa_data_command}", file=sys.stderr)
                sys.exit(1)
            sys.exit(handler(args))
        else:
            handler = _DISPATCH.get(args.dpa_command)
            if handler is None:
                print(f"Unknown dpa command: {args.dpa_command}", file=sys.stderr)
                sys.exit(1)
            sys.exit(handler(args))
    except DPADataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
