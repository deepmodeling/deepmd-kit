"""Submodule that contains all the DeePMD-Kit entry point scripts."""

from .compress import compress
from .config import config
from .doc import doc_train_input
from .freeze import freeze
from .test import test
# import `train` as `train_dp` to avoid the conflict of the
# module name `train` and the function name `train`
from .train import train as train_dp
from .transfer import transfer
from ..infer.model_devi import make_model_devi
from .convert import convert

__all__ = [
    "config",
    "doc_train_input",
    "freeze",
    "test",
    "train",
    "transfer",
    "compress",
    "doc_train_input",
    "make_model_devi",
    "convert",
]
