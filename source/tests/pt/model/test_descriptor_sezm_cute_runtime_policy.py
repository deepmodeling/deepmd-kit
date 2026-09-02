# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Behavioral tests for CuTe inference feature policy."""

from __future__ import (
    annotations,
)

import importlib.util
import os
import sys
from pathlib import (
    Path,
)
from unittest import (
    mock,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
POLICY_PATH = REPO_ROOT / "deepmd/pt_expt/kernels/cute/sezm/runtime_policy.py"


def _load_policy():
    assert POLICY_PATH.is_file(), f"CuTe runtime policy is missing: {POLICY_PATH}"
    name = "sezm_cute_runtime_policy_test"
    spec = importlib.util.spec_from_file_location(name, POLICY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_cute_master_gate_controls_sezm_path() -> None:
    policy = _load_policy()
    cases = (
        ({}, False),
        ({"DP_CUTE_INFER": "0"}, False),
        ({"DP_CUTE_INFER": "1"}, True),
        ({"DP_CUTE_INFER": "true"}, True),
        ({"DP_CUTE_INFER": "unsupported"}, False),
        ({"DP_CUTE_INFER": "1", "DP_TRITON_INFER": "2"}, True),
    )
    for environment, expected in cases:
        with mock.patch.dict(os.environ, environment, clear=True):
            assert policy.is_cute_infer_enabled() is expected


def test_master_switch_controls_every_sm80_subfeature() -> None:
    policy = _load_policy()
    with mock.patch.dict(
        os.environ,
        {
            "DP_CUTE_GIE": "1",
            "DP_CUTE_K1_PACKED_WIGNER": "1",
            "DP_CUTE_K1_THIN_WRAPPER": "1",
            "DP_CUTE_OUTPUT_GRID_BWD_SM80_C96_N48_PANEL": "1",
            "DP_CUTE_OUTPUT_GRID_FWD_SM80_C96_N48": "1",
            "DP_CUTE_READOUT_INPUT_FOLD_SM80": "1",
        },
        clear=True,
    ):
        assert not policy.is_cute_infer_enabled()
        for capability in policy.SM80_PROFILE_CAPABILITIES:
            assert not policy.is_sm80_profile_enabled(capability)
            assert not policy.is_gie_enabled(capability)
            assert not policy.is_packed_wigner_enabled(capability)
            assert not policy.is_k1_thin_wrapper_enabled(capability)
            assert not policy.is_output_grid_bwd_sm80_c96_n48_panel_enabled(capability)
            assert not policy.is_output_grid_fwd_sm80_c96_n48_enabled(capability)
            assert not policy.is_readout_input_fold_sm80_enabled(capability)


def test_sm80_family_profile_defaults_from_master_only() -> None:
    policy = _load_policy()
    with mock.patch.dict(os.environ, {"DP_CUTE_INFER": "1"}, clear=True):
        assert policy.SM80_PROFILE_CAPABILITIES == frozenset({(8, 0), (8, 6)})
        for capability in policy.SM80_PROFILE_CAPABILITIES:
            assert policy.is_sm80_profile_enabled(capability)
            assert policy.is_gie_enabled(capability)
            assert policy.is_packed_wigner_enabled(capability)
            assert policy.is_k1_thin_wrapper_enabled(capability)
            assert policy.is_output_grid_bwd_sm80_c96_n48_panel_enabled(capability)
            assert policy.is_output_grid_fwd_sm80_c96_n48_enabled(capability)
            assert policy.is_readout_input_fold_sm80_enabled(capability)


def test_every_sm80_profile_feature_can_be_disabled() -> None:
    policy = _load_policy()
    checks = {
        "DP_CUTE_GIE": lambda: policy.is_gie_enabled((8, 0)),
        "DP_CUTE_K1_PACKED_WIGNER": lambda: policy.is_packed_wigner_enabled((8, 0)),
        "DP_CUTE_K1_THIN_WRAPPER": lambda: policy.is_k1_thin_wrapper_enabled((8, 0)),
        "DP_CUTE_OUTPUT_GRID_BWD_SM80_C96_N48_PANEL": lambda: (
            policy.is_output_grid_bwd_sm80_c96_n48_panel_enabled((8, 0))
        ),
        "DP_CUTE_OUTPUT_GRID_FWD_SM80_C96_N48": lambda: (
            policy.is_output_grid_fwd_sm80_c96_n48_enabled((8, 0))
        ),
        "DP_CUTE_READOUT_INPUT_FOLD_SM80": lambda: (
            policy.is_readout_input_fold_sm80_enabled((8, 0))
        ),
    }
    for name, checker in checks.items():
        with mock.patch.dict(
            os.environ,
            {"DP_CUTE_INFER": "1", name: "0"},
            clear=True,
        ):
            assert not checker()


def test_non_sm80_family_uses_architecture_defaults_without_sm80_features() -> None:
    policy = _load_policy()
    with mock.patch.dict(os.environ, {"DP_CUTE_INFER": "1"}, clear=True):
        capability = (8, 9)
        assert not policy.is_sm80_profile_enabled(capability)
        assert not policy.is_gie_enabled(capability)
        assert policy.is_packed_wigner_enabled(capability)
        assert not policy.is_k1_thin_wrapper_enabled(capability)
        assert not policy.is_output_grid_bwd_sm80_c96_n48_panel_enabled(capability)
        assert not policy.is_output_grid_fwd_sm80_c96_n48_enabled(capability)
        assert not policy.is_readout_input_fold_sm80_enabled(capability)


def test_sm90_c96_asymmetric_panels_default_is_exact_arch_disable_only() -> None:
    policy = _load_policy()
    switch = policy.OUTPUT_GRID_SM90_C96_ASYMMETRIC_PANELS_ENV

    with mock.patch.dict(os.environ, {"DP_CUTE_INFER": "1"}, clear=True):
        assert policy.is_output_grid_sm90_c96_asymmetric_panels_enabled((9, 0))
        assert not policy.is_output_grid_sm90_c96_asymmetric_panels_enabled((8, 9))
        assert not policy.is_output_grid_sm90_c96_asymmetric_panels_enabled((9, 1))

    with mock.patch.dict(
        os.environ,
        {"DP_CUTE_INFER": "1", switch: "0"},
        clear=True,
    ):
        assert not policy.is_output_grid_sm90_c96_asymmetric_panels_enabled((9, 0))

    with mock.patch.dict(os.environ, {switch: "1"}, clear=True):
        assert not policy.is_output_grid_sm90_c96_asymmetric_panels_enabled((9, 0))

    with mock.patch.dict(
        os.environ,
        {"DP_CUTE_INFER": "1", switch: "1"},
        clear=True,
    ):
        assert policy.is_output_grid_sm90_c96_asymmetric_panels_enabled((9, 0))
        assert not policy.is_output_grid_sm90_c96_asymmetric_panels_enabled((8, 0))


def test_overrides_remain_architecture_safe() -> None:
    policy = _load_policy()
    with mock.patch.dict(
        os.environ,
        {
            "DP_CUTE_INFER": "1",
            "DP_CUTE_K1_THIN_WRAPPER": "1",
            "DP_CUTE_OUTPUT_GRID_BWD_SM80_C96_N48_PANEL": "1",
            "DP_CUTE_OUTPUT_GRID_FWD_SM80_C96_N48": "1",
            "DP_CUTE_READOUT_INPUT_FOLD_SM80": "1",
        },
        clear=True,
    ):
        assert policy.is_k1_thin_wrapper_enabled((9, 0))
        assert not policy.is_output_grid_bwd_sm80_c96_n48_panel_enabled((9, 0))
        assert not policy.is_output_grid_fwd_sm80_c96_n48_enabled((9, 0))
        assert not policy.is_readout_input_fold_sm80_enabled((9, 0))

    with mock.patch.dict(
        os.environ,
        {
            "DP_CUTE_INFER": "1",
            "DP_CUTE_GIE": "1",
            "DP_CUTE_K1_PACKED_WIGNER": "1",
        },
        clear=True,
    ):
        assert not policy.is_gie_enabled((9, 0))
        assert not policy.is_packed_wigner_enabled((10, 1))


def test_int32_k1_capacity_checks_every_flattened_axis() -> None:
    policy = _load_policy()
    max_edges = policy.INT32_MAX // policy.K1_VALUES_PER_EDGE
    max_nodes = policy.INT32_MAX // policy.K1_VALUES_PER_NODE

    assert policy.k1_int32_indexing_is_safe(
        edge_count=max_edges,
        node_count=max_nodes,
    )
    assert not policy.k1_int32_indexing_is_safe(
        edge_count=max_edges + 1,
        node_count=1,
    )
    assert not policy.k1_int32_indexing_is_safe(
        edge_count=1,
        node_count=max_nodes + 1,
    )
    assert not policy.k1_int32_indexing_is_safe(
        edge_count=-1,
        node_count=1,
    )
    assert not policy.k1_int32_indexing_is_safe(
        edge_count=1,
        node_count=-1,
    )


def test_strict_mode_is_explicitly_opt_in() -> None:
    policy = _load_policy()
    with mock.patch.dict(os.environ, {}, clear=True):
        assert not policy.is_cute_strict_enabled()
    with mock.patch.dict(os.environ, {"DP_CUTE_STRICT": "1"}, clear=True):
        assert policy.is_cute_strict_enabled()


def test_neighbor_list_eager_island_policy_is_owned_by_neo_runtime() -> None:
    policy = _load_policy()
    cases = (
        ({}, (8, 0), False),
        ({"DP_CUTE_INFER": "1"}, (8, 0), True),
        ({"DP_CUTE_INFER": "1"}, (8, 6), False),
        ({"DP_CUTE_INFER": "1"}, (9, 0), False),
        (
            {
                "DP_CUTE_INFER": "1",
                "DP_CUTE_K1_EAGER_ISLANDS": "0",
            },
            (8, 0),
            False,
        ),
        (
            {
                "DP_CUTE_INFER": "1",
                "DP_CUTE_K1_EAGER_ISLANDS": "1",
            },
            (9, 0),
            True,
        ),
    )
    for environment, capability, expected in cases:
        with mock.patch.dict(os.environ, environment, clear=True):
            assert policy.is_k1_eager_island_enabled(capability) is expected
