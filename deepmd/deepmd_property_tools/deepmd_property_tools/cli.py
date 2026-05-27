# SPDX-License-Identifier: LGPL-3.0-or-later
"""Command line interface for DeePMD property tools."""

from __future__ import (
    annotations,
)

import argparse
from pathlib import (
    Path,
)
from collections.abc import Sequence

from deepmd_property_tools import (
    PropertyPredict,
    PropertyTrain,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser containing training and prediction subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="deepmd-property-tools",
        description="DeePMD molecular property training and prediction helpers.",
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a property model")
    train_parser.add_argument(
        "--dataset", required=True, type=Path, help="CSV dataset path"
    )
    train_parser.add_argument(
        "--mol-dir", required=True, type=Path, help="MOL directory path"
    )
    train_parser.add_argument(
        "--save-path", required=True, type=Path, help="Experiment output directory"
    )
    train_parser.add_argument(
        "--property-col", default="Property", help="CSV property column"
    )
    train_parser.add_argument(
        "--property-name", default="Property", help="DeePMD property name"
    )
    train_parser.add_argument(
        "--finetune", default=None, help="Pretrained model name or path"
    )
    train_parser.add_argument(
        "--numb-steps", type=int, default=None, help="Number of training steps"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=None, help="Training batch size"
    )
    train_parser.set_defaults(func=_run_train)

    predict_parser = subparsers.add_parser("predict", help="Predict properties")
    predict_parser.add_argument(
        "--model", required=True, type=Path, help="Model file or experiment directory"
    )
    predict_parser.add_argument(
        "--dataset", required=True, type=Path, help="CSV dataset path"
    )
    predict_parser.add_argument(
        "--mol-dir", required=True, type=Path, help="MOL directory path"
    )
    predict_parser.add_argument(
        "--save-path", default=None, type=Path, help="Prediction output directory"
    )
    predict_parser.set_defaults(func=_run_predict)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface.

    Parameters
    ----------
    argv
        Optional argument list. When omitted, arguments are read from the command
        line.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    args.func(args)
    return 0


def _run_train(args: argparse.Namespace) -> None:
    trainer = PropertyTrain(
        property_name=args.property_name,
        property_col=args.property_col,
        save_path=args.save_path,
        numb_steps=args.numb_steps,
        batch_size=args.batch_size,
        finetune=args.finetune,
    )
    trainer.fit({"dataset": args.dataset, "mol_dir": args.mol_dir})


def _run_predict(args: argparse.Namespace) -> None:
    predictor = PropertyPredict(load_model=args.model)
    y_pred = predictor.predict(
        {"dataset": args.dataset, "mol_dir": args.mol_dir},
        save_path=args.save_path,
    )
    print(y_pred)


if __name__ == "__main__":
    raise SystemExit(main())
