# SPDX-License-Identifier: LGPL-3.0-or-later
"""DeePMD-Kit entry point module."""

import argparse
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Union,
)

from deepmd.common import (
    clear_session,
)
from deepmd.entrypoints import (
    compress,
    convert,
    doc_train_input,
    freeze,
    make_model_devi,
    neighbor_stat,
    start_dpgui,
    test,
    train_dp,
    transfer,
)
from deepmd.loggers import (
    set_log_handles,
)
from deepmd.nvnmd.entrypoints.train import (
    train_nvnmd,
)
from deepmd_utils.main import (
    get_ll,
    main_parser,
    parse_args,
)

__all__ = ["main", "parse_args", "get_ll", "main_parser"]


def main(args: Optional[Union[List[str], argparse.Namespace]] = None):
    """DeePMD-Kit entry point.

    Parameters
    ----------
    args : List[str] or argparse.Namespace, optional
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
    # log handles for train will be set separatelly
    # when the use of MPI will be determined in `RunOptions`
    if args.command not in (None, "train"):
        set_log_handles(args.log_level, Path(args.log_path) if args.log_path else None)

    dict_args = vars(args)

    if args.command == "train":
        train_dp(**dict_args)
    elif args.command == "freeze":
        freeze(**dict_args)
    elif args.command == "test":
        test(**dict_args)
    elif args.command == "transfer":
        transfer(**dict_args)
    elif args.command == "compress":
        compress(**dict_args)
    elif args.command == "doc-train-input":
        doc_train_input(**dict_args)
    elif args.command == "model-devi":
        make_model_devi(**dict_args)
    elif args.command == "convert-from":
        convert(**dict_args)
    elif args.command == "neighbor-stat":
        neighbor_stat(**dict_args)
    elif args.command == "train-nvnmd":  # nvnmd
        train_nvnmd(**dict_args)
    elif args.command == "gui":
        start_dpgui(**dict_args)
    elif args.command is None:
        pass
    else:
        raise RuntimeError(f"unknown command {args.command}")

    if args is not None:
        clear_session()
