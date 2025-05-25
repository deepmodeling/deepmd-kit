# SPDX-License-Identifier: LGPL-3.0-or-later
"""DeePMD-Kit entry point module."""

import argparse
from typing import (
    Optional,
    Union,
)

from deepmd.backend.suffix import (
    format_model_suffix,
)
from deepmd.jax.entrypoints.train import (
    train,
)
from deepmd.main import (
    parse_args,
)

__all__ = ["main"]


def main(args: Optional[Union[list[str], argparse.Namespace]] = None) -> None:
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

    if args.command == "train":
        train(**dict_args)
    elif args.command == "freeze":
        raise
        dict_args["output"] = format_model_suffix(
            dict_args["output"], preferred_backend=args.backend, strict_prefer=True
        )
        # freeze(**dict_args)
    elif args.command is None:
        pass
    else:
        raise RuntimeError(f"unknown command {args.command}")
