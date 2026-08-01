# SPDX-License-Identifier: LGPL-3.0-or-later
"""DeePMD-kit entry point for the TensorFlow 2 backend."""

import argparse
from pathlib import (
    Path,
)

from deepmd.loggers.loggers import (
    set_log_handles,
)
from deepmd.main import (
    parse_args,
)
from deepmd.tf2.entrypoints.compress import (
    enable_compression,
)
from deepmd.tf2.entrypoints.freeze import (
    freeze,
)
from deepmd.tf2.entrypoints.train import (
    train,
)

__all__ = ["main"]


def main(args: list[str] | argparse.Namespace | None = None) -> None:
    """TensorFlow 2 backend command dispatcher."""
    if not isinstance(args, argparse.Namespace):
        args = parse_args(args=args)

    set_log_handles(
        args.log_level,
        Path(args.log_path) if args.log_path else None,
        mpi_log=None,
    )

    if args.command == "train":
        train(**vars(args))
    elif args.command == "freeze":
        freeze(**vars(args))
    elif args.command == "compress":
        enable_compression(
            input_file=args.input,
            output=args.output,
            stride=args.step,
            extrapolate=args.extrapolate,
            check_frequency=args.frequency,
            training_script=args.training_script,
            head=args.head,
        )
    elif args.command is None:
        pass
    else:
        raise RuntimeError(
            f"Unsupported command '{args.command}' for the TensorFlow 2 backend."
        )
