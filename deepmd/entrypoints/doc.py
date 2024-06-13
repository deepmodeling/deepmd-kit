# SPDX-License-Identifier: LGPL-3.0-or-later
"""Module that prints train input arguments docstrings."""

from deepmd.utils.argcheck import (
    gen_doc,
    gen_json,
    gen_json_schema,
)

__all__ = ["doc_train_input"]


def doc_train_input(*, out_type: str = "rst", **kwargs):
    """Print out trining input arguments to console."""
    if out_type == "rst":
        doc_str = gen_doc(make_anchor=True)
    elif out_type == "json":
        doc_str = gen_json()
    elif out_type == "json_schema":
        doc_str = gen_json_schema()
    else:
        raise RuntimeError(f"Unsupported out type {out_type}")
    print(doc_str)  # noqa: T201
