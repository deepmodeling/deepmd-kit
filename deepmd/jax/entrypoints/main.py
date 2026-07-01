# SPDX-License-Identifier: LGPL-3.0-or-later
"""DeePMD-Kit entry point module."""

import argparse
from pathlib import (
    Path,
)

from deepmd.backend.suffix import (
    format_model_suffix,
)
from deepmd.jax.entrypoints.compress import (
    enable_compression,
)
from deepmd.jax.entrypoints.freeze import (
    freeze,
)
from deepmd.jax.entrypoints.train import (
    train,
)
from deepmd.loggers.loggers import (
    set_log_handles,
)
from deepmd.main import (
    parse_args,
)

__all__ = ["main"]


def main(args: list[str] | argparse.Namespace | None = None) -> None:
    """DeePMD-Kit entry point.

    Parameters
    ----------
    args : list[str] or argparse.Namespace, optional
        list of command line arguments, used to avoid calling from the subprocess,
        as it is quite slow to import tensorflow; if Namespace is given, it will
        be used directly

    Raises
    ------
    RuntimeError
        if no command was input
    """
    if not isinstance(args, argparse.Namespace):
        args = parse_args(args=args)

    dict_args = vars(args)
    set_log_handles(
        args.log_level,
        Path(args.log_path) if args.log_path else None,
        mpi_log=None,
    )

    if args.command == "train":
        train(**dict_args)
    elif args.command == "freeze":
        freeze(**dict_args)
    elif args.command == "compress":
        enable_compression(
            input_file=format_model_suffix(
                args.input,
                preferred_backend="jax",
                strict_prefer=True,
            ),
            output=format_model_suffix(
                args.output,
                preferred_backend="jax",
                strict_prefer=True,
            ),
            stride=args.step,
            extrapolate=args.extrapolate,
            check_frequency=args.frequency,
            training_script=args.training_script,
        )
    elif args.command is None:
        pass
    else:
        raise RuntimeError(f"unknown command {args.command}")
