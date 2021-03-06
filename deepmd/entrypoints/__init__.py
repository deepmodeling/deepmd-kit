"""Submodule that contains all the DeePMD-Kit entry point scripts."""

from .compress import compress
from .config import config
from .doc import doc_train_input
from .freeze import freeze
from .test import test
from .train import train
from .transfer import transfer

__all__ = [
    "config",
    "doc_train_input",
    "freeze",
    "test",
    "train",
    "transfer",
    "compress",
    "doc_train_input",
]
