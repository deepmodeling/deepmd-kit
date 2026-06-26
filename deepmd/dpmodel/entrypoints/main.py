# SPDX-License-Identifier: LGPL-3.0-or-later
"""DeePMD-kit entry point for the DPModel backend."""

import argparse
from pathlib import (
    Path,
)

from deepmd.backend.suffix import (
    format_model_suffix,
)
from deepmd.dpmodel.entrypoints.compress import (
    enable_compression,
)
from deepmd.loggers.loggers import (
    set_log_handles,
)
from deepmd.main import (
    parse_args,
)

__all__ = ["main"]


def main(args: list[str] | argparse.Namespace | None = None) -> None:
    """DPModel backend command dispatcher."""
    if not isinstance(args, argparse.Namespace):
        args = parse_args(args=args)

    set_log_handles(
        args.log_level,
        Path(args.log_path) if args.log_path else None,
        mpi_log=None,
    )

    if args.command == "compress":
        enable_compression(
            input_file=format_model_suffix(
                args.input,
                preferred_backend="dp",
                strict_prefer=True,
            ),
            output=format_model_suffix(
                args.output,
                preferred_backend="dp",
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
        raise RuntimeError(
            f"Unsupported command '{args.command}' for the DPModel backend."
        )
