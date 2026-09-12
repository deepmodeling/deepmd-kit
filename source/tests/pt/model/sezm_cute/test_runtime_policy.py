# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Behavioral tests for CuTe inference feature policy."""

from __future__ import (
    annotations,
)

import os
from unittest import (
    mock,
)

from deepmd.pt_expt.kernels.cute.sezm import runtime_policy as policy


def test_cute_master_gate_controls_sezm_path() -> None:
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
    with mock.patch.dict(
        os.environ,
        {
            "DP_CUTE_GIE": "1",
            "DP_CUTE_SO2_PACKED_WIGNER": "1",
            "DP_CUTE_SO2_THIN_WRAPPER": "1",
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
            assert not policy.is_so2_thin_wrapper_enabled(capability)
            assert not policy.is_output_grid_bwd_sm80_c96_n48_panel_enabled(capability)
            assert not policy.is_output_grid_fwd_sm80_c96_n48_enabled(capability)
            assert not policy.is_readout_input_fold_sm80_enabled(capability)


def test_sm80_family_profile_defaults_from_master_only() -> None:
    with mock.patch.dict(os.environ, {"DP_CUTE_INFER": "1"}, clear=True):
        assert policy.SM80_PROFILE_CAPABILITIES == frozenset({(8, 0), (8, 6)})
        for capability in policy.SM80_PROFILE_CAPABILITIES:
            assert policy.is_sm80_profile_enabled(capability)
            assert policy.is_gie_enabled(capability)
            assert policy.is_packed_wigner_enabled(capability)
            assert policy.is_so2_thin_wrapper_enabled(capability)
            assert policy.is_output_grid_bwd_sm80_c96_n48_panel_enabled(capability)
            assert policy.is_output_grid_fwd_sm80_c96_n48_enabled(capability)
            assert policy.is_readout_input_fold_sm80_enabled(capability)


def test_every_sm80_profile_feature_can_be_disabled() -> None:
    checks = {
        "DP_CUTE_GIE": lambda: policy.is_gie_enabled((8, 0)),
        "DP_CUTE_SO2_PACKED_WIGNER": lambda: policy.is_packed_wigner_enabled((8, 0)),
        "DP_CUTE_SO2_THIN_WRAPPER": lambda: policy.is_so2_thin_wrapper_enabled((8, 0)),
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
    with mock.patch.dict(os.environ, {"DP_CUTE_INFER": "1"}, clear=True):
        capability = (8, 9)
        assert not policy.is_sm80_profile_enabled(capability)
        assert not policy.is_gie_enabled(capability)
        assert policy.is_packed_wigner_enabled(capability)
        assert not policy.is_so2_thin_wrapper_enabled(capability)
        assert not policy.is_output_grid_bwd_sm80_c96_n48_panel_enabled(capability)
        assert not policy.is_output_grid_fwd_sm80_c96_n48_enabled(capability)
        assert not policy.is_readout_input_fold_sm80_enabled(capability)


def test_sm90_c96_asymmetric_panels_default_is_exact_arch_disable_only() -> None:
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
    with mock.patch.dict(
        os.environ,
        {
            "DP_CUTE_INFER": "1",
            "DP_CUTE_SO2_THIN_WRAPPER": "1",
            "DP_CUTE_OUTPUT_GRID_BWD_SM80_C96_N48_PANEL": "1",
            "DP_CUTE_OUTPUT_GRID_FWD_SM80_C96_N48": "1",
            "DP_CUTE_READOUT_INPUT_FOLD_SM80": "1",
        },
        clear=True,
    ):
        assert policy.is_so2_thin_wrapper_enabled((9, 0))
        assert not policy.is_output_grid_bwd_sm80_c96_n48_panel_enabled((9, 0))
        assert not policy.is_output_grid_fwd_sm80_c96_n48_enabled((9, 0))
        assert not policy.is_readout_input_fold_sm80_enabled((9, 0))

    with mock.patch.dict(
        os.environ,
        {
            "DP_CUTE_INFER": "1",
            "DP_CUTE_GIE": "1",
            "DP_CUTE_SO2_PACKED_WIGNER": "1",
        },
        clear=True,
    ):
        assert not policy.is_gie_enabled((9, 0))
        assert not policy.is_packed_wigner_enabled((10, 1))


def test_int32_so2_capacity_checks_every_flattened_axis() -> None:
    max_edges = policy.INT32_MAX // policy.SO2_VALUES_PER_EDGE
    max_nodes = policy.INT32_MAX // policy.SO2_VALUES_PER_NODE

    assert policy.so2_int32_indexing_is_safe(
        edge_count=max_edges,
        node_count=max_nodes,
    )
    assert not policy.so2_int32_indexing_is_safe(
        edge_count=max_edges + 1,
        node_count=1,
    )
    assert not policy.so2_int32_indexing_is_safe(
        edge_count=1,
        node_count=max_nodes + 1,
    )
    assert not policy.so2_int32_indexing_is_safe(
        edge_count=-1,
        node_count=1,
    )
    assert not policy.so2_int32_indexing_is_safe(
        edge_count=1,
        node_count=-1,
    )


def test_strict_mode_is_explicitly_opt_in() -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
        assert not policy.is_cute_strict_enabled()
    with mock.patch.dict(os.environ, {"DP_CUTE_STRICT": "1"}, clear=True):
        assert policy.is_cute_strict_enabled()


def test_neighbor_list_eager_island_policy_is_owned_by_neo_runtime() -> None:
    cases = (
        ({}, (8, 0), False),
        ({"DP_CUTE_INFER": "1"}, (8, 0), True),
        ({"DP_CUTE_INFER": "1"}, (8, 6), False),
        ({"DP_CUTE_INFER": "1"}, (9, 0), False),
        (
            {
                "DP_CUTE_INFER": "1",
                "DP_CUTE_SO2_EAGER_ISLANDS": "0",
            },
            (8, 0),
            False,
        ),
        (
            {
                "DP_CUTE_INFER": "1",
                "DP_CUTE_SO2_EAGER_ISLANDS": "1",
            },
            (9, 0),
            True,
        ),
    )
    for environment, capability, expected in cases:
        with mock.patch.dict(os.environ, environment, clear=True):
            assert policy.is_so2_eager_island_enabled(capability) is expected
