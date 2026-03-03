# SPDX-License-Identifier: LGPL-3.0-or-later
"""CLI entrypoint for pretrained model operations."""

from __future__ import (
    annotations,
)

import logging
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
)

from deepmd.pretrained.download import (
    download_model,
)

if TYPE_CHECKING:
    import argparse


def pretrained_entrypoint(args: argparse.Namespace) -> None:
    """Handle `dp pretrained ...` subcommands."""
    if args.pretrained_command == "download":
        cache_dir = Path(args.cache_dir) if args.cache_dir else None
        path = download_model(args.MODEL, cache_dir=cache_dir)
        logging.getLogger(__name__).info("Pretrained model path: %s", path)
        print(path)  # noqa: T201
        return

    raise ValueError(f"Unknown pretrained subcommand: {args.pretrained_command}")
