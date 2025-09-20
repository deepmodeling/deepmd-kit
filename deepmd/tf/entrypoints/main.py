# SPDX-License-Identifier: LGPL-3.0-or-later
"""DeePMD-Kit entry point module."""

import argparse
from pathlib import (
    Path,
)
from typing import (
    Optional,
    Union,
)

from deepmd.backend.suffix import (
    format_model_suffix,
)
from deepmd.main import (
    get_ll,
    main_parser,
    parse_args,
)
from deepmd.tf.common import (
    clear_session,
)
from deepmd.tf.entrypoints import (
    compress,
    convert,
    freeze,
    train_dp,
    transfer,
)
from deepmd.tf.loggers import (
    set_log_handles,
)
from deepmd.tf.nvnmd.entrypoints.train import (
    train_nvnmd,
)

__all__ = ["get_ll", "main", "main_parser", "parse_args"]


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
    if args is not None:
        clear_session()

    if not isinstance(args, argparse.Namespace):
        args = parse_args(args=args)

    # do not set log handles for None, it is useless
    # log handles for train will be set separately
    # when the use of MPI will be determined in `RunOptions`
    if args.command not in (None, "train"):
        set_log_handles(args.log_level, Path(args.log_path) if args.log_path else None)

    dict_args = vars(args)

    if args.command == "train":
        train_dp(**dict_args)
    elif args.command == "freeze":
        dict_args["output"] = format_model_suffix(
            dict_args["output"], preferred_backend=args.backend, strict_prefer=True
        )
        freeze(**dict_args)
    elif args.command == "transfer":
        transfer(**dict_args)
    elif args.command == "compress":
        dict_args["input"] = format_model_suffix(
            dict_args["input"], preferred_backend=args.backend, strict_prefer=True
        )
        dict_args["output"] = format_model_suffix(
            dict_args["output"], preferred_backend=args.backend, strict_prefer=True
        )
        compress(**dict_args)
    elif args.command == "convert-from":
        convert(**dict_args)
    elif args.command == "train-nvnmd":  # nvnmd
        train_nvnmd(**dict_args)
    elif args.command is None:
        pass
    else:
        raise RuntimeError(f"unknown command {args.command}")

    if args is not None:
        clear_session()
