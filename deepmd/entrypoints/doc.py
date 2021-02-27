"""Module that prints train input arguments docstrings."""

from deepmd.utils.argcheck import gen_doc

__all__ = ["doc_train_input"]


def doc_train_input():
    """Print out trining input arguments to console."""
    doc_str = gen_doc(make_anchor=True)
    print(doc_str)
