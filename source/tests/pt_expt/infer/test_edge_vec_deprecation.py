# SPDX-License-Identifier: LGPL-3.0-or-later
"""The edge_vec .pt2 schema (pt-backend SeZM freeze) is a deprecated
legacy format; loading one must warn, loading anything else must not.
"""

import warnings

import pytest


def test_edge_vec_metadata_warns() -> None:
    from deepmd.pt_expt.infer.deep_eval import (
        _warn_legacy_edge_vec,
    )

    with pytest.warns(DeprecationWarning, match="edge_vec"):
        _warn_legacy_edge_vec({"lower_input_kind": "edge_vec"})


@pytest.mark.parametrize(
    "kind",
    [
        "nlist",  # pt_expt dense freeze
        "graph",  # pt_expt graph freeze
        "dpa1_canonical",  # dpa1 compact-canonical freeze
        None,  # pre-metadata archives
    ],
)
def test_other_kinds_do_not_warn(kind) -> None:
    from deepmd.pt_expt.infer.deep_eval import (
        _warn_legacy_edge_vec,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_legacy_edge_vec({"lower_input_kind": kind} if kind else {})
