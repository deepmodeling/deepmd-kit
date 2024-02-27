# SPDX-License-Identifier: LGPL-3.0-or-later
"""Common entrypoints."""

import argparse
from pathlib import (
    Path,
)

from deepmd.backend.backend import (
    Backend,
)
from deepmd.backend.suffix import (
    format_model_suffix,
)
from deepmd.entrypoints.convert_backend import (
    convert_backend,
)
from deepmd.entrypoints.doc import (
    doc_train_input,
)
from deepmd.entrypoints.gui import (
    start_dpgui,
)
from deepmd.entrypoints.neighbor_stat import (
    neighbor_stat,
)
from deepmd.entrypoints.test import (
    test,
)
from deepmd.infer.model_devi import (
    make_model_devi,
)
from deepmd.loggers.loggers import (
    set_log_handles,
)


def main(args: argparse.Namespace):
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
    set_log_handles(args.log_level, Path(args.log_path) if args.log_path else None)

    dict_args = vars(args)

    if args.command == "test":
        dict_args["model"] = format_model_suffix(
            dict_args["model"],
            feature=Backend.Feature.DEEP_EVAL,
            preferred_backend=args.backend,
            strict_prefer=False,
        )
        test(**dict_args)
    elif args.command == "doc-train-input":
        doc_train_input(**dict_args)
    elif args.command == "model-devi":
        dict_args["models"] = [
            format_model_suffix(
                mm,
                feature=Backend.Feature.DEEP_EVAL,
                preferred_backend=args.backend,
                strict_prefer=False,
            )
            for mm in dict_args["models"]
        ]
        make_model_devi(**dict_args)
    elif args.command == "neighbor-stat":
        neighbor_stat(**dict_args)
    elif args.command == "gui":
        start_dpgui(**dict_args)
    elif args.command == "convert-backend":
        convert_backend(**dict_args)
    else:
        raise ValueError(f"Unknown command: {args.command}")
