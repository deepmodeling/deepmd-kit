# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Exact normalized-Gaunt message-grid product for the Neo SM90 path."""

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
)

import cutlass
import cutlass.cute as cute
import cutlass.utils as cute_utils
import torch
from cuda.bindings.driver import (
    CUstream,
)
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
)

from ...compile_cache import (
    device_aware_lru_cache,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


# CuTe JIT functions use DSL-inferred argument and return types.
# ruff: noqa: ANN202, ANN204, TC002

COEFF_DIM = 48
CHANNELS = 64
THREADS = 256
GROUPS = 4
THREADS_PER_GROUP = CHANNELS
VALUES_PER_NODE = COEFF_DIM * CHANNELS
EXPECTED_COMPACT_PATHS = 1968
EXPECTED_ORDERED_PATHS = 3833
SUPPORT_GAP_THRESHOLD = 1.0e-4
SM90_CAPABILITY = (9, 0)
FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}

# (left row, right row, C[p,i,j], d[i]G[p,i,j], d[j]G[p,i,j]).
# The last value is zero on a diagonal path.
GauntTerm = tuple[int, int, float, float, float]


@dataclass(frozen=True, eq=False)
class Sm90GauntSchedule:
    """Normalized-Gaunt rows captured as CuTe compile-time constants."""

    rows: tuple[tuple[GauntTerm, ...], ...]
    output_groups: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if len(self.rows) != COEFF_DIM:
            raise ValueError("Gaunt schedule must contain 48 rows")
        if len(self.output_groups) != GROUPS:
            raise ValueError("Gaunt schedule must contain four groups")
        assigned = tuple(row for group in self.output_groups for row in group)
        if tuple(sorted(assigned)) != tuple(range(COEFF_DIM)):
            raise ValueError("each coefficient row must occur in exactly one group")
        if sum(len(row) for row in self.rows) != EXPECTED_COMPACT_PATHS:
            raise ValueError("Gaunt schedule must contain 1,968 paths")
        for output_row, row in enumerate(self.rows):
            for left_row, right_row, forward, adj_left, adj_right in row:
                if not (0 <= left_row <= right_row < COEFF_DIM):
                    raise ValueError(
                        f"invalid compact path in row {output_row}: "
                        f"({left_row}, {right_row})"
                    )
                if forward == 0.0 or adj_left == 0.0:
                    raise ValueError("Gaunt paths must not contain zero weights")
                if (left_row == right_row) != (adj_right == 0.0):
                    raise ValueError(
                        "only diagonal Gaunt paths may omit the mirrored adjoint"
                    )


def build_sm90_gaunt_schedule(
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
) -> Sm90GauntSchedule:
    """Recover and certify the fixed Neo Gaunt support from its projectors."""
    if (
        tuple(to_grid.shape) != (152, COEFF_DIM)
        or tuple(from_grid.shape) != (COEFF_DIM, 152)
        or to_grid.dtype != torch.float32
        or from_grid.dtype != torch.float32
    ):
        raise ValueError("Neo Gaunt requires FP32 (152,48)/(48,152) projectors")

    to_cpu = to_grid.detach().to(device="cpu", dtype=torch.float64)
    from_cpu = from_grid.detach().to(device="cpu", dtype=torch.float64)
    tensor = torch.einsum("pg,gi,gj->pij", from_cpu, to_cpu, to_cpu)
    support = tensor.abs() > SUPPORT_GAP_THRESHOLD
    ordered_paths = int(support.sum())
    if ordered_paths != EXPECTED_ORDERED_PATHS:
        raise ValueError(
            "Neo projector support changed: expected "
            f"{EXPECTED_ORDERED_PATHS} paths, got {ordered_paths}"
        )
    minimum_signal = float(tensor.abs()[support].min())
    maximum_residual = float(tensor.abs()[~support].max())
    if minimum_signal <= 10.0 * SUPPORT_GAP_THRESHOLD:
        raise ValueError("Neo Gaunt support no longer has a certified magnitude gap")
    if maximum_residual >= SUPPORT_GAP_THRESHOLD:
        raise ValueError("Neo Gaunt structural-zero residual exceeds its certificate")

    degree_weight = tuple(
        2 * degree + 1
        for degree in range(4)
        for _order in range(-degree, degree + 1)
        for _frame in range(3)
    )
    weights = torch.tensor(degree_weight, device="cpu", dtype=torch.float64)
    normalized = tensor / weights[:, None, None]
    permutation_error = max(
        float((normalized - normalized.permute(permutation)).abs().max())
        for permutation in (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        )
    )
    if permutation_error > 2.0e-6:
        raise ValueError(
            f"Neo normalized-Gaunt symmetry changed: max error {permutation_error:.3e}"
        )

    rows: list[tuple[GauntTerm, ...]] = []
    for output_row in range(COEFF_DIM):
        terms: list[GauntTerm] = []
        for left_row in range(COEFF_DIM):
            for right_row in range(left_row, COEFF_DIM):
                if not bool(support[output_row, left_row, right_row]):
                    continue
                forward = float(tensor[output_row, left_row, right_row].float())
                symmetric = normalized[output_row, left_row, right_row]
                adj_left = float((symmetric * degree_weight[left_row]).float())
                adj_right = (
                    0.0
                    if left_row == right_row
                    else float((symmetric * degree_weight[right_row]).float())
                )
                terms.append((left_row, right_row, forward, adj_left, adj_right))
        rows.append(tuple(terms))

    group_rows: list[list[int]] = [[] for _ in range(GROUPS)]
    group_loads = [0] * GROUPS
    for output_row in sorted(
        range(COEFF_DIM),
        key=lambda row: (-len(rows[row]), row),
    ):
        group = min(range(GROUPS), key=lambda item: (group_loads[item], item))
        group_rows[group].append(output_row)
        group_loads[group] += len(rows[output_row])
    if max(group_loads) - min(group_loads) > 32:
        raise ValueError(f"Neo Gaunt static groups are imbalanced: {group_loads}")
    return Sm90GauntSchedule(
        tuple(rows),
        tuple(tuple(sorted(group)) for group in group_rows),
    )


class _Sm90GauntForward:
    def __init__(self, schedule: Sm90GauntSchedule) -> None:
        self.schedule = schedule

    @cute.jit
    def __call__(
        self,
        left: cute.Tensor,
        right: cute.Tensor,
        output: cute.Tensor,
        stream: CUstream,
    ):
        operand_layout = cute.make_layout(
            (2, COEFF_DIM, CHANNELS),
            stride=(VALUES_PER_NODE, CHANNELS, 1),
        )
        self.kernel(left, right, output, operand_layout).launch(
            grid=(left.shape[0], 1, 1),
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        left: cute.Tensor,
        right: cute.Tensor,
        output: cute.Tensor,
        operand_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        node, _, _ = cute.arch.block_idx()
        group = tidx >> 6
        channel = tidx & (CHANNELS - 1)

        smem = cute_utils.SmemAllocator()
        operands = smem.allocate_tensor(cutlass.Float32, operand_layout, 16)
        for side in cutlass.range_constexpr(2):
            for slot in cutlass.range_constexpr(VALUES_PER_NODE // THREADS):
                linear = tidx + slot * THREADS
                row = linear >> 6
                scalar_channel = linear & (CHANNELS - 1)
                if cutlass.const_expr(side == 1):
                    value = right[node, row, scalar_channel].to(cutlass.Float32)
                else:
                    value = left[node, row, scalar_channel].to(cutlass.Float32)
                operands[side, row, scalar_channel] = value
        cute.arch.sync_threads()

        if group == 0:
            self._accumulate_group(operands, output, node, channel, 0)
        elif group == 1:
            self._accumulate_group(operands, output, node, channel, 1)
        elif group == 2:
            self._accumulate_group(operands, output, node, channel, 2)
        else:
            self._accumulate_group(operands, output, node, channel, 3)

    @cute.jit
    def _accumulate_group(
        self,
        operands: cute.Tensor,
        output: cute.Tensor,
        node: cutlass.Int32,
        channel: cutlass.Int32,
        group: cutlass.Constexpr,
    ):
        rows = self.schedule.output_groups[group]
        for row_slot in cutlass.range_constexpr(len(rows)):
            output_row = rows[row_slot]
            accumulator = cutlass.Float32(0.0)
            terms = self.schedule.rows[output_row]
            for path in cutlass.range_constexpr(len(terms)):
                left_row, right_row, coefficient_value, _, _ = terms[path]
                product = (
                    operands[0, left_row, channel] * operands[1, right_row, channel]
                )
                if cutlass.const_expr(left_row != right_row):
                    product = product + (
                        operands[0, right_row, channel] * operands[1, left_row, channel]
                    )
                accumulator = accumulator + cutlass.Float32(coefficient_value) * product
            output[node, output_row, channel] = accumulator


def _fake_dense() -> cute.Tensor:
    return make_fake_compact_tensor(
        cutlass.Float32,
        (cute.sym_int64(), COEFF_DIM, CHANNELS),
        stride_order=(2, 1, 0),
        **FAKE_TENSOR_KW,
    )


@device_aware_lru_cache(maxsize=8)
def _compiled_sm90_gaunt_forward(schedule: Sm90GauntSchedule) -> Callable:
    dense = _fake_dense()
    return cute.compile(
        _Sm90GauntForward(schedule),
        dense,
        dense,
        dense,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _validate_dense(*tensors: torch.Tensor) -> None:
    reference = tensors[0]
    if (
        tuple(reference.shape[1:]) != (COEFF_DIM, CHANNELS)
        or reference.shape[0] <= 0
        or not reference.is_cuda
        or reference.dtype != torch.float32
        or tuple(torch.cuda.get_device_capability(reference.device)) != SM90_CAPABILITY
    ):
        raise ValueError("Gaunt operands must be SM90 FP32 (N,48,64)")
    if any(
        tensor.shape != reference.shape
        or tensor.device != reference.device
        or tensor.dtype != torch.float32
        or not tensor.is_contiguous()
        or tensor.data_ptr() % 16 != 0
        for tensor in tensors
    ):
        raise ValueError("Gaunt operands must match and be 16-byte aligned")


def run_sm90_gaunt_forward(
    left: torch.Tensor,
    right: torch.Tensor,
    schedule: Sm90GauntSchedule,
) -> torch.Tensor:
    """Evaluate the exact normalized-Gaunt product on SM90."""
    output = torch.empty_like(left)
    _validate_dense(left, right, output)
    with torch.cuda.device(left.device):
        _compiled_sm90_gaunt_forward(schedule)(left, right, output)
    return output


__all__ = [
    "Sm90GauntSchedule",
    "build_sm90_gaunt_schedule",
    "run_sm90_gaunt_forward",
]
