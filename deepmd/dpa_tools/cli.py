# dpa_tools/cli.py
#
# Command-line interface.  Mirrors the Python API — every subcommand maps
# directly to a public function or method.

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Sequence

import numpy as np

from deepmd.dpa_tools import (
    DPAFineTuner,
    DPAPredictor,
    attach_labels,
    batch_convert,
    check_data,
    convert,
    cross_validate,
    load_dataset,
    train_test_split,
)
from deepmd.dpa_tools.data.errors import DPADataError
from deepmd.dpa_tools.data.loader import load_data
from deepmd.dpa_tools.finetuner import extract_descriptors
from deepmd.dpa_tools.mft import MFTFineTuner

_LOG = logging.getLogger("dpa_tools")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Shared argument helpers — keep subcommand flags consistent
# ---------------------------------------------------------------------------

def _add_data_args(parser, valid: bool = False):
    parser.add_argument("--train-data", required=True,
                        help="Path(s) to deepmd/npy system directories (space-separated).")
    if valid:
        parser.add_argument("--valid-data", default=None,
                            help="Validation system directories.")


def _add_type_map_arg(parser):
    parser.add_argument("--type-map", default=None,
                        help="Comma-separated element symbols.  Auto-inferred from "
                             "checkpoint + data type_map.raw when omitted.")


def _add_property_args(parser):
    parser.add_argument("--property-name", default="property",
                        help="Label key under set.*/ (default: property).")
    parser.add_argument("--task-dim", type=int, default=1,
                        help="Output dim of property head (default: 1).")
    parser.add_argument("--intensive", action=argparse.BooleanOptionalAction, default=True,
                        help="Intensive (mean-pool) vs extensive (sum). Default: intensive.")


def _add_training_args(parser, default_steps: int = 100_000):
    parser.add_argument("--max-steps", type=int, default=default_steps)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--stop-lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", default="auto:512")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./dpa_output")
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--disp-freq", type=int, default=1_000)


def _maybe_split_list(val: str | None) -> list[str] | None:
    """'a,b,c' → ['a','b','c']; None → None."""
    if val is None:
        return None
    return [x.strip() for x in val.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Subcommand: fit (all strategies)
# ---------------------------------------------------------------------------

def _cmd_fit(args: argparse.Namespace) -> int:
    train = _maybe_split_list(args.train_data) or [args.train_data]
    valid = _maybe_split_list(args.valid_data) if args.valid_data else None
    type_map = _maybe_split_list(args.type_map)

    model = DPAFineTuner(
        pretrained=args.pretrained,
        model_branch=args.model_branch,
        predictor=args.predictor,
        pooling=args.pooling,
        seed=args.seed,
        strategy=args.strategy,
        property_name=args.property_name,
        task_dim=args.task_dim,
        intensive=args.intensive,
        learning_rate=args.learning_rate,
        stop_lr=args.stop_lr,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        save_freq=args.save_freq,
        disp_freq=args.disp_freq,
    )

    model.fit(train_data=train, valid_data=valid, type_map=type_map,
              target_key=args.target_key)

    if args.strategy == "frozen_sklearn":
        out = model.freeze(args.output)
        _LOG.info("Frozen model → %s", out)
    else:
        _LOG.info("Checkpoint → %s", args.output_dir)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: cv (cross_validate)
# ---------------------------------------------------------------------------

def _cmd_cv(args: argparse.Namespace) -> int:
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


# ---------------------------------------------------------------------------
# Subcommand: mft
# ---------------------------------------------------------------------------

def _cmd_mft(args: argparse.Namespace) -> int:
    systems = load_dataset(args.data, label_key=args.label_key)
    train, valid, test = train_test_split(
        systems,
        group_by=args.group_by or "formula",
        manifest=args.manifest,
        test_size=args.test_size,
        valid_size=args.valid_size,
        seed=args.seed,
    )
    print(f"train={len(train)} valid={len(valid)} test={len(test)}")

    aux = _maybe_split_list(args.aux_data) or [args.aux_data]

    mft = MFTFineTuner(
        pretrained=args.pretrained,
        aux_branch=args.aux_branch,
        aux_prob=args.aux_prob,
        aux_type_map=_maybe_split_list(args.aux_type_map),
        downstream_type_map=_maybe_split_list(args.downstream_type_map),
        downstream_task_type=args.downstream_task_type,
        property_name=args.property_name,
        task_dim=args.task_dim,
        intensive=args.intensive,
        learning_rate=args.learning_rate,
        stop_lr=args.stop_lr,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        aux_batch_size=args.aux_batch_size,
        downstream_batch_size=args.downstream_batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
        save_freq=args.save_freq,
        disp_freq=args.disp_freq,
    )
    mft.fit(train_data=train, aux_data=aux, valid_data=valid)

    if test:
        res = mft.evaluate(test)
        print(f"test MAE = {float(res['mae']):.4f}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: extract-descriptors
# ---------------------------------------------------------------------------

def _cmd_extract_descriptors(args: argparse.Namespace) -> int:
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


# ---------------------------------------------------------------------------
# Subcommand: predict (frozen .pth)
# ---------------------------------------------------------------------------

def _cmd_predict(args: argparse.Namespace) -> int:
    predictor = DPAPredictor(args.model)
    result = predictor.predict(args.data)
    np.save(args.output, result.predictions)
    _LOG.info("Predictions shape=%s → %s", result.predictions.shape, args.output)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: evaluate (frozen .pth)
# ---------------------------------------------------------------------------

def _cmd_evaluate(args: argparse.Namespace) -> int:
    predictor = DPAPredictor(args.model)
    metrics = predictor.evaluate(args.data)
    print(f"MAE  : {metrics.mae:.6f}")
    print(f"RMSE : {metrics.rmse:.6f}")
    print(f"R²   : {metrics.r2:.6f}")
    print(f"N    : {metrics.predictions.shape[0]}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: convert / batch-convert / check-data / attach-labels
# (unchanged logic, preserved from original)
# ---------------------------------------------------------------------------

def _cmd_convert(args: argparse.Namespace) -> int:
    type_map = _maybe_split_list(args.type_map)
    _LOG.info("Converting %s (fmt=%s) → %s", args.input, args.fmt, args.output)
    output = convert(
        input_path=args.input, output_dir=args.output, fmt=args.fmt,
        type_map=type_map, validate=args.validate, strict=args.strict,
    )
    _LOG.info("Wrote deepmd/npy → %s", output)
    return 0


def _cmd_batch_convert(args: argparse.Namespace) -> int:
    type_map = _maybe_split_list(args.type_map)
    outputs = batch_convert(
        glob_pattern=args.glob, output_dir=args.output, fmt=args.fmt,
        type_map=type_map, validate=args.validate, strict=args.strict,
    )
    _LOG.info("Wrote %d deepmd/npy dirs under %s", len(outputs), args.output)
    return 0


def _cmd_check_data(args: argparse.Namespace) -> int:
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


def _cmd_attach_labels(args: argparse.Namespace) -> int:
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
# Parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dpa-tools",
        description="Fine-tuning helpers for DPA-3.1 pretrained descriptors.",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Debug-level logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- fit ---------------------------------------------------------------
    fit_p = sub.add_parser("fit", help="Train a model (any strategy).")
    _add_data_args(fit_p, valid=True)
    fit_p.add_argument("--pretrained", default="DPA-3.1-3M",
                       help="Path to DPA checkpoint (.pt).")
    fit_p.add_argument("--model-branch", default=None,
                       help="Branch for multi-task ckpts (frozen_sklearn).")
    fit_p.add_argument("--strategy", default="frozen_sklearn",
                       choices=["frozen_sklearn", "linear_probe", "finetune", "scratch"])
    fit_p.add_argument("--predictor", default="rf",
                       choices=["rf", "linear", "ridge", "mlp"],
                       help="sklearn head type (frozen_sklearn only).")
    fit_p.add_argument("--pooling", default="mean",
                       choices=["mean", "sum", "mean+std", "mean+std+max+min"])
    fit_p.add_argument("--target-key", default=None,
                       help="Label key (frozen_sklearn only).")
    fit_p.add_argument("--output", default="frozen_model.pth",
                       help="Output .pth path (frozen_sklearn only).")
    _add_type_map_arg(fit_p)
    _add_property_args(fit_p)
    _add_training_args(fit_p)
    fit_p.set_defaults(func=_cmd_fit)

    # ---- cv ----------------------------------------------------------------
    cv_p = sub.add_parser("cv", help="Cross-validate frozen_sklearn baseline.")
    cv_p.add_argument("--data", required=True,
                      help="dpdata root or system directory list.")
    cv_p.add_argument("--label-key", default="energy",
                      help="Label filename under set.*/ (default: energy).")
    cv_p.add_argument("--pretrained", default="DPA-3.1-3M",
                      help="Path to DPA checkpoint (.pt).")
    cv_p.add_argument("--model-branch", default=None,
                      help="Branch for multi-task ckpts.")
    cv_p.add_argument("--predictor", default="rf",
                      choices=["rf", "linear", "ridge", "mlp"])
    cv_p.add_argument("--pooling", default="mean",
                      choices=["mean", "sum", "mean+std", "mean+std+max+min"])
    cv_p.add_argument("--cv", default="5", help="'holdout' or int >= 2.")
    cv_p.add_argument("--group-by", default="formula",
                      help="Grouping: 'formula' or comma-separated list.")
    cv_p.add_argument("--granularity", default="composition",
                      choices=["frame", "composition"])
    cv_p.add_argument("--seed", type=int, default=42)
    cv_p.set_defaults(func=_cmd_cv)

    # ---- mft ---------------------------------------------------------------
    mft_p = sub.add_parser("mft", help="Multi-task fine-tuning.")
    mft_p.add_argument("--data", required=True,
                       help="dpdata root or system directory list (downstream).")
    mft_p.add_argument("--aux-data", required=True,
                       help="Aux data system directory.")
    mft_p.add_argument("--label-key", default="energy",
                       help="Label key (default: energy).")
    mft_p.add_argument("--pretrained", required=True,
                       help="Path to DPA checkpoint (.pt).")
    mft_p.add_argument("--aux-branch", default="MP_traj_v024_alldata_mixu",
                       help="Aux branch name in checkpoint.")
    mft_p.add_argument("--aux-prob", type=float, default=0.5,
                       help="Sampling weight for aux branch.")
    mft_p.add_argument("--aux-type-map", default=None,
                       help="Comma-separated aux element symbols (auto if omitted).")
    mft_p.add_argument("--downstream-type-map", default=None,
                       help="Comma-separated downstream element symbols (auto if omitted).")
    mft_p.add_argument("--downstream-task-type", default="property",
                       choices=["ener", "property"])
    mft_p.add_argument("--group-by", default="formula")
    mft_p.add_argument("--manifest", default=None,
                       help="Path to split_manifest.json for fixed splits.")
    mft_p.add_argument("--test-size", type=float, default=0.1)
    mft_p.add_argument("--valid-size", type=float, default=0.1)
    mft_p.add_argument("--aux-batch-size", default=None,
                       help="Batch size for aux branch (e.g. auto:128).")
    mft_p.add_argument("--downstream-batch-size", type=int, default=None,
                       help="Batch size for downstream (e.g. 3).")
    _add_property_args(mft_p)
    _add_training_args(mft_p)
    mft_p.set_defaults(func=_cmd_mft)

    # ---- extract-descriptors -----------------------------------------------
    ext_p = sub.add_parser("extract-descriptors",
                           help="Extract pooled DPA descriptors to .npy.")
    ext_p.add_argument("--data", required=True,
                       help="System directory or dpdata root.")
    ext_p.add_argument("--pretrained", required=True,
                       help="Path to DPA checkpoint (.pt).")
    ext_p.add_argument("--model-branch", default=None)
    ext_p.add_argument("--pooling", default="mean",
                       choices=["mean", "sum", "mean+std", "mean+std+max+min"])
    ext_p.add_argument("--output", required=True,
                       help="Output .npy path.")
    ext_p.add_argument("--no-cache", action="store_true",
                       help="Bypass descriptor cache.")
    ext_p.set_defaults(func=_cmd_extract_descriptors)

    # ---- predict (frozen .pth) ---------------------------------------------
    pred_p = sub.add_parser("predict",
                            help="Predict with a frozen .pth bundle.")
    pred_p.add_argument("--model", required=True,
                        help="Path to frozen .pth.")
    pred_p.add_argument("--data", required=True,
                        help="System directory or dpdata root.")
    pred_p.add_argument("--output", required=True,
                        help="Output .npy path.")
    pred_p.set_defaults(func=_cmd_predict)

    # ---- evaluate (frozen .pth) --------------------------------------------
    eval_p = sub.add_parser("evaluate",
                            help="Evaluate a frozen .pth against stored labels.")
    eval_p.add_argument("--model", required=True,
                        help="Path to frozen .pth.")
    eval_p.add_argument("--data", required=True,
                        help="System directory or dpdata root.")
    eval_p.set_defaults(func=_cmd_evaluate)

    # ---- convert -----------------------------------------------------------
    conv_p = sub.add_parser("convert",
                            help="Convert structure file → deepmd/npy.")
    conv_p.add_argument("--input", required=True)
    conv_p.add_argument("--output", required=True)
    conv_p.add_argument("--fmt", required=True)
    conv_p.add_argument("--type-map", default=None,
                        help="Comma-separated element symbols.")
    conv_p.add_argument("--no-validate", dest="validate", action="store_false")
    conv_p.add_argument("--strict", action="store_true")
    conv_p.set_defaults(func=_cmd_convert)

    # ---- batch-convert -----------------------------------------------------
    bat_p = sub.add_parser("batch-convert",
                            help="Batch-convert glob → deepmd/npy.")
    bat_p.add_argument("--glob", required=True)
    bat_p.add_argument("--output", required=True)
    bat_p.add_argument("--fmt", required=True)
    bat_p.add_argument("--type-map", default=None)
    bat_p.add_argument("--no-validate", dest="validate", action="store_false")
    bat_p.add_argument("--strict", action="store_true")
    bat_p.set_defaults(func=_cmd_batch_convert)

    # ---- check-data --------------------------------------------------------
    chk_p = sub.add_parser("check-data",
                            help="Sanity-check deepmd/npy directories.")
    chk_p.add_argument("--data", required=True, nargs="+")
    chk_p.add_argument("--strict", action="store_true")
    chk_p.set_defaults(func=_cmd_check_data)

    # ---- attach-labels -----------------------------------------------------
    att_p = sub.add_parser("attach-labels",
                            help="Attach .npy labels to deepmd/npy directory.")
    att_p.add_argument("--data", required=True)
    att_p.add_argument("--head", required=True)
    att_p.add_argument("--head-json", action="store_true")
    att_p.add_argument("--values", required=True)
    att_p.set_defaults(func=_cmd_attach_labels)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    try:
        return args.func(args)
    except DPADataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, TypeError) as exc:
        allowed = {"attach-labels", "convert", "batch-convert", "fit", "cv", "mft"}
        if args.command in allowed:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
