# SPDX-License-Identifier: LGPL-3.0-or-later
"""CLI entry point for the ``dpa-adapt`` and ``dpaad`` commands.

Unlike the deepmd-kit ``dp`` command, ``dpa-adapt`` is a standalone CLI that
focuses solely on DPA model fine-tuning, descriptor extraction,
cross-validation, prediction, evaluation, and data preparation.

``dpa-adapt --help`` and ``dpaad --help`` do not load torch — the parser is
pure argparse and the handlers (and the DPA stack) are imported lazily only
when a subcommand actually runs.
"""

from __future__ import (
    annotations,
)

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence

import numpy as np

_LOG = logging.getLogger("dpa_adapt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_ll(log_level: str) -> int:
    """Convert string to python logging level.

    Parameters
    ----------
    log_level : str
        allowed input values are: DEBUG, INFO, WARNING, ERROR, 3, 2, 1, 0

    Returns
    -------
    int
        one of python logging module log levels - 10, 20, 30 or 40
    """
    if log_level.isdigit():
        int_level = (4 - int(log_level)) * 10
    else:
        int_level = getattr(logging, log_level)
    return int_level


def _set_log_handles(level: int, log_path: str | None = None) -> None:
    """Set up logging to console and optionally a file."""
    logger = logging.getLogger("dpa_adapt")
    logger.setLevel(level)
    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def _maybe_split_list(val: str | None) -> list[str] | None:
    """``"a,b,c"`` → ``["a","b","c"]``; ``None`` → ``None``."""
    if val is None:
        return None
    return [x.strip() for x in val.split(",") if x.strip()]


class _RawTextArgDefaultsHelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """Formatter for multi-line help with default values."""


# ---------------------------------------------------------------------------
# Subcommand handlers — each lazy-imports its dependencies
# ---------------------------------------------------------------------------


def _cmd_fit(args: argparse.Namespace) -> int:
    from dpa_adapt import (
        DPAFineTuner,
    )

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
        fparam_dim=args.fparam_dim,
    )
    aux_data = (
        _maybe_split_list(args.aux_data) or [args.aux_data] if args.aux_data else None
    )
    model.fit(
        train_data=train,
        valid_data=valid,
        type_map=type_map,
        target_key=target_key,
        aux_data=aux_data,
    )
    if args.strategy == "frozen_sklearn":
        out = model.freeze(args.output)
        _LOG.info("Frozen model → %s", out)
    else:
        _LOG.info("Checkpoint → %s", args.output_dir)
    return 0


def _cmd_cv(args: argparse.Namespace) -> int:
    from dpa_adapt import (
        DPAFineTuner,
        cross_validate,
        load_dataset,
    )

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
        model,
        systems,
        label_key=args.label_key,
        cv=args.cv if args.cv == "holdout" else int(args.cv),
        group_by=args.group_by or "formula",
        granularity=args.granularity,
        seed=args.seed,
    )
    a = result["aggregate"]
    print(
        f"R²  = {a.get('r2_mean', float('nan')):.4f} ± {a.get('r2_std', float('nan')):.4f}"
    )
    print(
        f"MAE = {a.get('mae_mean', float('nan')):.4f} ± {a.get('mae_std', float('nan')):.4f}"
    )
    print(
        f"RMSE= {a.get('rmse_mean', float('nan')):.4f} ± {a.get('rmse_std', float('nan')):.4f}"
    )
    print(f"n   = {result['n_independent']} independent groups")
    for w in result.get("warnings", []):
        print(f"[!] {w}")
    return 0


def _cmd_extract_descriptors(args: argparse.Namespace) -> int:
    from dpa_adapt.finetuner import (
        extract_descriptors,
    )

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
    from dpa_adapt import (
        DPAPredictor,
    )

    predictor = DPAPredictor(args.model)
    result = predictor.predict(args.data)
    np.save(args.output, result.predictions)
    _LOG.info("Predictions shape=%s → %s", result.predictions.shape, args.output)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from dpa_adapt import (
        DPAPredictor,
    )

    predictor = DPAPredictor(args.model)
    metrics = predictor.evaluate(args.data)
    print(f"MAE  : {metrics.mae:.6f}")
    print(f"RMSE : {metrics.rmse:.6f}")
    print(f"R²   : {metrics.r2:.6f}")
    print(f"N    : {metrics.predictions.shape[0]}")
    return 0


def _cmd_data_convert(args: argparse.Namespace) -> int:

    type_map = _maybe_split_list(args.type_map)

    from dpa_adapt import (
        convert,
    )

    result = convert(
        input_path=args.input,
        output_dir=args.output,
        fmt=args.fmt,
        type_map=type_map,
        property_name=args.property_name or args.property_col,
        property_col=args.property_col,
        train_ratio=args.train_ratio,
        smiles_col=args.smiles_col,
        mol_dir=args.mol_dir,
        mol_template=args.mol_template,
        split_seed=args.split_seed,
        conformer_seed=args.conformer_seed,
        poscar=args.poscar,
        formula_col=args.formula_col,
        base_element=args.base_element,
        sets=args.sets,
        seed=args.seed,
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
    elif result["method"] == "batch_dpdata":
        print(f"Output dirs  : {len(result['output_dirs'])}")
        print(f"Manifest     : {result['manifest']}")
    else:
        _LOG.info("Wrote deepmd/npy → %s", result["output_dir"])
    return 0


def _cmd_data_validate(args: argparse.Namespace) -> int:
    from dpa_adapt import (
        check_data,
    )
    from dpa_adapt.data.loader import (
        load_data,
    )

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
    from dpa_adapt import (
        attach_labels,
    )
    from dpa_adapt.data.loader import (
        load_data,
    )

    values = np.load(args.values)
    if args.head_json:
        head = json.loads(args.head)
    else:
        head = args.head
    systems = load_data(args.data)
    if len(systems) != 1:
        _LOG.warning(
            "attach-labels: expected 1 system from %r, got %d; attaching to first.",
            args.data,
            len(systems),
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
# Argument parser
# ---------------------------------------------------------------------------


def get_parser() -> argparse.ArgumentParser:
    """Build the standalone ``dpa-adapt`` / ``dpaad`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The fully configured parser for the ``dpa-adapt`` / ``dpaad`` CLI.
    """
    try:
        from dpa_adapt import (
            __version__,
        )
    except ImportError:
        __version__ = "unknown"

    parser = argparse.ArgumentParser(
        description="DPA tools — fine-tune pre-trained DPA models, extract descriptors, "
        "cross-validate, predict, evaluate, and prepare data.",
        formatter_class=_RawTextArgDefaultsHelpFormatter,
    )

    # Logging options (shared across all subcommands)
    parser_log = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_log.add_argument(
        "-v",
        "--log-level",
        choices=["DEBUG", "3", "INFO", "2", "WARNING", "1", "ERROR", "0"],
        default="INFO",
        help="set verbosity level by string or number, 0=ERROR, 1=WARNING, "
        "2=INFO and 3=DEBUG",
    )
    parser_log.add_argument(
        "-l",
        "--log-path",
        type=str,
        default=None,
        help="set log file to log messages to disk, if not specified, "
        "the logs will only be output to console",
    )

    parser.add_argument(
        "--version", action="version", version=f"dpa-adapt v{__version__}"
    )

    subparsers = parser.add_subparsers(title="subcommands", dest="command")

    # -- extract-descriptors -------------------------------------------------
    parser_extract = subparsers.add_parser(
        "extract-descriptors",
        help="Extract pooled DPA descriptors to .npy",
        parents=[parser_log],
    )
    parser_extract.add_argument(
        "--data", required=True, nargs="+", help="System directories."
    )
    parser_extract.add_argument(
        "--pretrained", required=True, help="Path to DPA checkpoint (.pt)."
    )
    parser_extract.add_argument("--model-branch", default=None)
    parser_extract.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "sum", "mean+std", "mean+std+max+min"],
    )
    parser_extract.add_argument("--output", required=True, help="Output .npy path.")
    parser_extract.add_argument(
        "--no-cache", action="store_true", help="Bypass descriptor cache."
    )

    # -- fit -----------------------------------------------------------------
    parser_fit = subparsers.add_parser(
        "fit",
        help="Train a model (any strategy)",
        parents=[parser_log],
    )
    parser_fit.add_argument(
        "--train-data", required=True, nargs="+", help="Training system directories."
    )
    parser_fit.add_argument(
        "--valid-data", default=None, nargs="+", help="Validation system directories."
    )
    parser_fit.add_argument(
        "--pretrained", default="DPA-3.1-3M", help="Path to DPA checkpoint (.pt)."
    )
    parser_fit.add_argument("--model-branch", default=None)
    parser_fit.add_argument(
        "--strategy",
        default="frozen_sklearn",
        choices=["frozen_sklearn", "frozen_head", "finetune", "mft"],
    )
    parser_fit.add_argument(
        "--predictor", default="rf", choices=["rf", "linear", "ridge", "mlp"]
    )
    parser_fit.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "sum", "mean+std", "mean+std+max+min"],
    )
    parser_fit.add_argument(
        "--target-key",
        default=None,
        help="Label key under set.*/ (e.g. energy, homo, bandgap).",
    )
    parser_fit.add_argument("--output", default="frozen_model.pth")
    parser_fit.add_argument("--type-map", default=None)
    parser_fit.add_argument("--task-dim", type=int, default=1)
    parser_fit.add_argument(
        "--intensive", action=argparse.BooleanOptionalAction, default=True
    )
    parser_fit.add_argument("--max-steps", type=int, default=100_000)
    parser_fit.add_argument("--learning-rate", type=float, default=1e-3)
    parser_fit.add_argument("--stop-lr", type=float, default=1e-5)
    parser_fit.add_argument("--batch-size", default="auto:512")
    parser_fit.add_argument("--seed", type=int, default=42)
    parser_fit.add_argument("--output-dir", default="./dpa_output")
    parser_fit.add_argument("--save-freq", type=int, default=10_000)
    parser_fit.add_argument("--disp-freq", type=int, default=1_000)
    # MFT-only flags
    parser_fit.add_argument(
        "--aux-data",
        default=None,
        nargs="+",
        help="(mft) Auxiliary system directories.",
    )
    parser_fit.add_argument(
        "--aux-branch",
        default="MP_traj_v024_alldata_mixu",
        help="(mft) Aux branch name in checkpoint.",
    )
    parser_fit.add_argument(
        "--aux-prob",
        type=float,
        default=0.5,
        help="(mft) Sampling weight for aux branch.",
    )
    parser_fit.add_argument(
        "--aux-type-map",
        default=None,
        help="(mft) Comma-separated aux element symbols.",
    )
    parser_fit.add_argument(
        "--downstream-type-map",
        default=None,
        help="(mft) Comma-separated downstream element symbols.",
    )
    parser_fit.add_argument(
        "--downstream-task-type",
        default="property",
        choices=["ener", "property"],
        help="(mft) Downstream head type.",
    )
    parser_fit.add_argument(
        "--aux-batch-size", default=None, help="(mft) Batch size for aux branch."
    )
    parser_fit.add_argument(
        "--downstream-batch-size",
        type=int,
        default=None,
        help="(mft) Batch size for downstream.",
    )
    parser_fit.add_argument(
        "--fparam-dim",
        type=int,
        default=0,
        help="(frozen_head/finetune/mft) Dimensionality of per-frame condition "
        "inputs (fparam). Requires set.*/fparam.npy in training data. Default: 0.",
    )

    # -- cv ------------------------------------------------------------------
    parser_cv = subparsers.add_parser(
        "cv",
        help="Cross-validate frozen_sklearn baseline",
        parents=[parser_log],
    )
    parser_cv.add_argument(
        "--data", required=True, nargs="+", help="System directories."
    )
    parser_cv.add_argument("--label-key", default="energy")
    parser_cv.add_argument(
        "--pretrained", default="DPA-3.1-3M", help="Path to DPA checkpoint (.pt)."
    )
    parser_cv.add_argument("--model-branch", default=None)
    parser_cv.add_argument(
        "--predictor", default="rf", choices=["rf", "linear", "ridge", "mlp"]
    )
    parser_cv.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "sum", "mean+std", "mean+std+max+min"],
    )
    parser_cv.add_argument("--cv", default="5")
    parser_cv.add_argument("--group-by", default="formula")
    parser_cv.add_argument(
        "--granularity", default="composition", choices=["frame", "composition"]
    )
    parser_cv.add_argument("--seed", type=int, default=42)

    # -- predict -------------------------------------------------------------
    parser_predict = subparsers.add_parser(
        "predict",
        help="Predict with a frozen .pth bundle",
        parents=[parser_log],
    )
    parser_predict.add_argument("--model", required=True, help="Path to frozen .pth.")
    parser_predict.add_argument(
        "--data", required=True, nargs="+", help="System directories."
    )
    parser_predict.add_argument("--output", required=True, help="Output .npy path.")

    # -- evaluate ------------------------------------------------------------
    parser_evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate a frozen .pth against stored labels",
        parents=[parser_log],
    )
    parser_evaluate.add_argument("--model", required=True, help="Path to frozen .pth.")
    parser_evaluate.add_argument(
        "--data", required=True, nargs="+", help="System directories."
    )

    # -- data (nested group) -------------------------------------------------
    parser_data = subparsers.add_parser(
        "data",
        help="Data conversion and validation tools",
        parents=[parser_log],
    )
    data_subparsers = parser_data.add_subparsers(
        dest="data_command",
        required=True,
    )

    # data convert
    parser_data_convert = data_subparsers.add_parser(
        "convert",
        help="Convert structure/CSV file → deepmd/npy (format auto-detected)",
        parents=[parser_log],
    )
    parser_data_convert.add_argument("--input", required=True)
    parser_data_convert.add_argument("--output", required=True)
    parser_data_convert.add_argument(
        "--fmt",
        default=None,
        help="Format hint (auto-detected if omitted). "
        "Use 'smiles' for CSV+SMILES, 'formula' for "
        "CSV+POSCAR composition formulas, otherwise "
        "dpdata format string (extxyz, vasp/poscar, …).",
    )
    parser_data_convert.add_argument("--type-map", default=None)
    parser_data_convert.add_argument(
        "--no-validate", dest="validate", action="store_false"
    )
    parser_data_convert.add_argument("--strict", action="store_true")
    parser_data_convert.add_argument("--property-name", default=None)
    parser_data_convert.add_argument("--property-col", default="Property")
    parser_data_convert.add_argument("--smiles-col", default="SMILES")
    parser_data_convert.add_argument("--mol-dir", default=None)
    parser_data_convert.add_argument("--mol-template", default="id{row}.mol",
                                     help="Filename template under --mol-dir; use {row} for the CSV row index.")
    parser_data_convert.add_argument("--train-ratio", type=float, default=0.9)
    parser_data_convert.add_argument("--split-seed", type=int, default=None,
                                     help="Random seed for train/valid split (SMILES input).")
    parser_data_convert.add_argument("--conformer-seed", type=int, default=None,
                                     help="Random seed for RDKit conformer generation (SMILES input).")
    parser_data_convert.add_argument("--poscar", default=None,
                                     help="Template POSCAR for fmt=formula.")
    parser_data_convert.add_argument("--base-element", default=None,
                                     help="Sublattice element to substitute "
                                          "(fmt=formula). Auto-inferred if omitted.")
    parser_data_convert.add_argument("--formula-col", default="formula",
                                     help="Column index or name for the formula "
                                          "(fmt=formula, default: formula).")
    parser_data_convert.add_argument("--sets", type=int, default=1,
                                     help="Random structures per formula "
                                          "(fmt=formula, default: 1).")
    parser_data_convert.add_argument("--seed", type=int, default=42,
                                     help="Random seed for selecting substituted host-atom sites "
                                          "(fmt=formula, default: 42).")
    parser_data_convert.add_argument("--overwrite", action="store_true")

    # data validate
    parser_data_validate = data_subparsers.add_parser(
        "validate",
        help="Sanity-check deepmd/npy directories",
        parents=[parser_log],
    )
    parser_data_validate.add_argument("--data", required=True, nargs="+")
    parser_data_validate.add_argument("--strict", action="store_true")

    # data attach-labels
    parser_data_attach = data_subparsers.add_parser(
        "attach-labels",
        help="Attach .npy labels to deepmd/npy directory",
        parents=[parser_log],
    )
    parser_data_attach.add_argument("--data", required=True)
    parser_data_attach.add_argument("--head", required=True)
    parser_data_attach.add_argument("--head-json", action="store_true")
    parser_data_attach.add_argument("--values", required=True)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args: Sequence[str] | None = None) -> None:
    """Entry point for the ``dpa-adapt`` / ``dpaad`` CLI.

    Parameters
    ----------
    args : list[str], optional
        Command-line arguments. If ``None``, ``sys.argv[1:]`` is used.
    """
    parser = get_parser()
    parsed_args = parser.parse_args(args)

    # Set up logging
    log_level = _get_ll(parsed_args.log_level)
    _set_log_handles(log_level, parsed_args.log_path)

    if parsed_args.command is None:
        parser.print_help()
        return

    try:
        if parsed_args.command == "data":
            handler = _DATA_DISPATCH.get(parsed_args.data_command)
            if handler is None:
                print(
                    f"Unknown data command: {parsed_args.data_command}", file=sys.stderr
                )
                sys.exit(1)
            sys.exit(handler(parsed_args))
        else:
            handler = _DISPATCH.get(parsed_args.command)
            if handler is None:
                print(f"Unknown dpa-adapt command: {parsed_args.command}", file=sys.stderr)
                sys.exit(1)
            sys.exit(handler(parsed_args))
    except Exception as exc:
        # Lazy-import DPADataError so that --help doesn't trigger heavy imports.
        from dpa_adapt.data.errors import (
            DPADataError,
        )

        if isinstance(exc, DPADataError):
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        raise
