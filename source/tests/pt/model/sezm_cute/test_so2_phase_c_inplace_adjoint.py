# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Storage-contract tests for the exact in-place Phase-C adjoint."""

from __future__ import (
    annotations,
)

import itertools
from dataclasses import (
    fields,
    replace,
)

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in {(8, 0), (9, 0)},
    reason="in-place Phase-C tests require sm_80 or sm_90",
)

OUTPUT_FIELDS = (
    "grad_stack",
    "grad_wigner_dt",
    "grad_logits",
    "grad_edge",
    "grad_z_partial",
    "grad_z",
    "grad_focus_src",
)
NONALIASED_OUTPUT_FIELDS = OUTPUT_FIELDS[1:]
INPUT_FIELDS = (
    "grad_out",
    "stack",
    "wigner_dt",
    "alpha",
    "focus_alpha",
    "dst_ptr",
    "rotate_inv_rescale",
    "edge_gate",
    "z_bias_raw",
    "group_max",
    "denom",
    "focus_src",
    "focus_weight",
    "focus_scale",
)
REDZONE_ELEMENTS = 4
SENTINEL = 937.25


def _phase_c_api():
    pytest.importorskip("cutlass")
    from deepmd.pt_expt.kernels.cute.sezm.so2.phase_c import (
        CuteNeoPhaseCBackwardLayout,
        NeoPhaseCBackwardLayoutOutputs,
    )

    return CuteNeoPhaseCBackwardLayout, NeoPhaseCBackwardLayoutOutputs


def _runner():
    Runner, _ = _phase_c_api()
    return Runner(
        focus_eps=1.0e-8,
        focus_tau=1.0,
        focus_label_smoothing=0.0,
    )


def _inputs(dst_ptr_values: list[int]):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(
        20260722 + dst_ptr_values[-1]
    )
    node_count = len(dst_ptr_values) - 1
    edge_count = dst_ptr_values[-1]
    rand = lambda *shape: torch.randn(  # noqa: E731
        *shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    return {
        "grad_out": rand(node_count, 16, 64),
        "stack": rand(edge_count, 2, 10, 32),
        "wigner_dt": rand(edge_count, 46),
        "alpha": torch.softmax(rand(edge_count, 2), dim=0),
        "focus_alpha": torch.softmax(rand(edge_count, 2), dim=1),
        "dst_ptr": torch.tensor(dst_ptr_values, device=device, dtype=torch.int32),
        "rotate_inv_rescale": rand(16),
        "edge_gate": rand(edge_count).abs(),
        "z_bias_raw": rand(2),
        "group_max": rand(node_count, 2),
        "denom": rand(node_count, 2).abs().add_(0.5),
        "focus_src": rand(edge_count, 2, 32),
        "focus_weight": rand(32, 2),
        "focus_scale": rand(2, 32),
    }


def _clone_inputs(inputs):
    return {name: value.clone() for name, value in inputs.items()}


def _allocate_outputs(inputs):
    _, Outputs = _phase_c_api()
    edge_count = inputs["stack"].shape[0]
    node_count = inputs["grad_out"].shape[0]
    options = {"device": inputs["stack"].device, "dtype": torch.float32}
    return Outputs(
        grad_stack=inputs["stack"],
        grad_wigner_dt=torch.empty(edge_count, 46, **options),
        grad_logits=torch.empty(edge_count, 2, **options),
        grad_edge=torch.empty(edge_count, **options),
        grad_z_partial=torch.empty(node_count, 2, **options),
        grad_z=torch.empty(2, **options),
        grad_focus_src=torch.empty(2, edge_count, 32, **options),
    )


def _call(runner, inputs, outputs):
    return runner(
        inputs["grad_out"],
        inputs["stack"],
        inputs["wigner_dt"],
        inputs["alpha"],
        inputs["focus_alpha"],
        inputs["dst_ptr"],
        inputs["rotate_inv_rescale"],
        inputs["edge_gate"],
        inputs["z_bias_raw"],
        inputs["group_max"],
        inputs["denom"],
        inputs["focus_src"],
        inputs["focus_weight"],
        inputs["focus_scale"],
        outputs,
    )


def _redzoned_like(tensor: torch.Tensor):
    flat = torch.full(
        (tensor.numel() + 2 * REDZONE_ELEMENTS,),
        SENTINEL,
        device=tensor.device,
        dtype=tensor.dtype,
    )
    value = flat[REDZONE_ELEMENTS : REDZONE_ELEMENTS + tensor.numel()].view_as(tensor)
    return value, (flat[:REDZONE_ELEMENTS], flat[-REDZONE_ELEMENTS:])


def _misaligned_like(tensor: torch.Tensor):
    flat = torch.empty(tensor.numel() + 1, device=tensor.device, dtype=tensor.dtype)
    value = flat[1:].view_as(tensor)
    value.copy_(tensor)
    assert value.data_ptr() % 16 != 0
    return value


def _assert_outputs_close(actual, expected):
    for field in fields(type(expected)):
        torch.testing.assert_close(
            getattr(actual, field.name),
            getattr(expected, field.name),
            atol=1.0e-6,
            rtol=1.0e-6,
        )


@pytest.mark.parametrize(
    "dst_ptr_values",
    (
        [0, 0, 2, 2, 7],
        [0, 1, 1, 4, 12, 12],
        [0, 0, 37, 37],
    ),
)
def test_exact_inplace_matches_independent_run_and_preserves_redzones(
    dst_ptr_values,
):
    runner = _runner()
    reference_inputs = _inputs(dst_ptr_values)
    reference_outputs = _allocate_outputs(reference_inputs)
    _call(runner, reference_inputs, reference_outputs)

    inputs = _inputs(dst_ptr_values)
    stack, stack_redzones = _redzoned_like(inputs["stack"])
    stack.copy_(inputs["stack"])
    inputs["stack"] = stack
    outputs = _allocate_outputs(inputs)
    redzones = {"grad_stack": stack_redzones}
    for field_name in NONALIASED_OUTPUT_FIELDS:
        value, field_redzones = _redzoned_like(getattr(outputs, field_name))
        outputs = replace(outputs, **{field_name: value})
        redzones[field_name] = field_redzones

    _call(runner, inputs, outputs)
    torch.cuda.synchronize()

    assert outputs.grad_stack.data_ptr() == inputs["stack"].data_ptr()
    _assert_outputs_close(outputs, reference_outputs)
    for field_name, (prefix, suffix) in redzones.items():
        assert torch.equal(prefix, torch.full_like(prefix, SENTINEL)), field_name
        assert torch.equal(suffix, torch.full_like(suffix, SENTINEL)), field_name


def test_grad_stack_must_be_the_exact_input_view():
    inputs = _inputs([0, 3, 7])
    outputs = replace(_allocate_outputs(inputs), grad_stack=inputs["stack"].clone())

    with pytest.raises(ValueError, match="exact in-place stack view"):
        _call(_runner(), inputs, outputs)


def test_partial_or_shifted_stack_alias_is_rejected():
    inputs = _inputs([0, 3, 7])
    edge_count = inputs["stack"].shape[0]
    base = torch.empty(
        edge_count + 1,
        2,
        10,
        32,
        device="cuda",
        dtype=torch.float32,
    )
    base[1:].copy_(inputs["stack"])
    inputs["stack"] = base[1:]
    outputs = replace(_allocate_outputs(inputs), grad_stack=base[:-1])

    with pytest.raises(ValueError, match="exact in-place stack view"):
        _call(_runner(), inputs, outputs)


@pytest.mark.parametrize(
    "tensor_name",
    INPUT_FIELDS + tuple(f"outputs.{name}" for name in NONALIASED_OUTPUT_FIELDS),
)
def test_compiled_tensors_require_16_byte_alignment(tensor_name):
    inputs = _inputs([0, 3, 7])
    outputs = _allocate_outputs(inputs)

    if tensor_name.startswith("outputs."):
        field_name = tensor_name.removeprefix("outputs.")
        outputs = replace(
            outputs,
            **{field_name: _misaligned_like(getattr(outputs, field_name))},
        )
    else:
        inputs[tensor_name] = _misaligned_like(inputs[tensor_name])
        if tensor_name == "stack":
            outputs = replace(outputs, grad_stack=inputs["stack"])

    with pytest.raises(ValueError, match="must be 16-byte aligned"):
        _call(_runner(), inputs, outputs)


@pytest.mark.parametrize("field_name", NONALIASED_OUTPUT_FIELDS)
def test_nonaliased_outputs_reject_input_overlap(field_name):
    inputs = _inputs([0, 3, 7])
    outputs = _allocate_outputs(inputs)
    target = getattr(outputs, field_name)
    overlapping = inputs["stack"].view(-1)[: target.numel()].view_as(target)
    outputs = replace(outputs, **{field_name: overlapping})

    with pytest.raises(ValueError, match="must not overlap"):
        _call(_runner(), inputs, outputs)


@pytest.mark.parametrize(
    ("first_name", "second_name"),
    tuple(itertools.combinations(NONALIASED_OUTPUT_FIELDS, 2)),
)
def test_nonaliased_output_pairs_must_not_overlap(first_name, second_name):
    inputs = _inputs([0, 3, 7])
    outputs = _allocate_outputs(inputs)
    first = getattr(outputs, first_name)
    second = getattr(outputs, second_name)
    slab = torch.empty(
        max(first.numel(), second.numel()),
        device="cuda",
        dtype=torch.float32,
    )
    outputs = replace(
        outputs,
        **{
            first_name: slab[: first.numel()].view_as(first),
            second_name: slab[: second.numel()].view_as(second),
        },
    )

    with pytest.raises(ValueError, match="must not overlap output"):
        _call(_runner(), inputs, outputs)


def test_retained_runner_supports_dynamic_edge_counts():
    runner = _runner()
    for dst_ptr_values in ([0, 2, 7], [0, 0, 1, 9, 9]):
        inputs = _inputs(dst_ptr_values)
        original = _clone_inputs(inputs)
        outputs = _allocate_outputs(inputs)
        _call(runner, inputs, outputs)
        first = {
            field.name: getattr(outputs, field.name).clone()
            for field in fields(outputs)
        }

        inputs = original
        outputs = _allocate_outputs(inputs)
        _call(runner, inputs, outputs)
        torch.cuda.synchronize()
        for field_name, expected in first.items():
            torch.testing.assert_close(
                getattr(outputs, field_name),
                expected,
                atol=1.0e-6,
                rtol=1.0e-6,
            )
