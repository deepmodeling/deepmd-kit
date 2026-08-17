# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Safety coverage for the fixed pitch-68 radial backward path."""

from __future__ import (
    annotations,
)

import ast
import unittest
from pathlib import (
    Path,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
CUTE_PATH = REPO_ROOT / "deepmd/kernels/cute/neo"
WRAPPER_PATH = CUTE_PATH / "k1_radial_phase_a_node.py"
KERNEL_PATH = CUTE_PATH / "k1_kernels/cute_neo_radial_phase_a_backward_node.py"

WARP_HELPERS = {"_warp_owned_grad_compact", "_warp_owned_grad_d"}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_constexpr_if(node: ast.If) -> bool:
    test = node.test
    return isinstance(test, ast.Call) and _call_name(test) == "const_expr"


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _dynamic_if_ancestors(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    boundary: ast.FunctionDef,
) -> list[ast.If]:
    result: list[ast.If] = []
    current = parents.get(node)
    while current is not None and current is not boundary:
        if isinstance(current, ast.If) and not _is_constexpr_if(current):
            result.append(current)
        current = parents.get(current)
    return result


class TestRadialPitch68SafeStatic(unittest.TestCase):
    def test_pitch68_specialization_contract(self) -> None:
        wrapper_source = WRAPPER_PATH.read_text(encoding="utf-8")
        kernel_source = KERNEL_PATH.read_text(encoding="utf-8")

        self.assertIn("SHARED_ROW_PITCH = HIDDEN + 4", kernel_source)
        self.assertNotIn("pitch68_safe", wrapper_source)
        self.assertNotIn("pitch68_safe", kernel_source)

    def test_warp_collectives_are_not_runtime_predicated(self) -> None:
        source = KERNEL_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        kernel = _function(tree, "neo_radial_phase_a_backward_node_kernel")

        helper_calls = [
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.Call) and _call_name(node) in WARP_HELPERS
        ]
        self.assertEqual(
            sorted(_call_name(node) for node in helper_calls),
            sorted(WARP_HELPERS),
        )
        for call in helper_calls:
            self.assertEqual(
                _dynamic_if_ancestors(call, parents, kernel),
                [],
                f"{_call_name(call)} must be reached by every warp lane",
            )

        compact_call = next(
            node
            for node in helper_calls
            if _call_name(node) == "_warp_owned_grad_compact"
        )
        panel_call = next(
            node for node in helper_calls if _call_name(node) == "_warp_owned_grad_d"
        )
        self.assertEqual(ast.unparse(compact_call.args[3]), "safe_compact_idx")
        self.assertEqual(ast.unparse(panel_call.args[2]), "safe_panel_idx")

        for helper_name in WARP_HELPERS:
            helper = _function(tree, helper_name)
            reductions = [
                node
                for node in ast.walk(helper)
                if isinstance(node, ast.Call)
                and _call_name(node) == "warp_reduction_sum"
            ]
            self.assertEqual(len(reductions), 1)
            self.assertEqual(
                _dynamic_if_ancestors(reductions[0], parents, helper),
                [],
                f"warp collective in {helper_name} must be unconditional",
            )

        assignments = [
            node for node in ast.walk(kernel) if isinstance(node, ast.Assign)
        ]
        compact_write = next(
            node
            for node in assignments
            if ast.unparse(node.targets[0]) == "grad_compact[compact_idx]"
        )
        panel_write = next(
            node
            for node in assignments
            if ast.unparse(node.targets[0]) == "params.grad_d_full[edge, panel_idx]"
        )
        self.assertEqual(
            {
                ast.unparse(guard.test)
                for guard in _dynamic_if_ancestors(compact_write, parents, kernel)
            },
            {"compact_idx < COMPACT_WIDTH", "subgroup_lane == 0"},
        )
        self.assertEqual(
            {
                ast.unparse(guard.test)
                for guard in _dynamic_if_ancestors(panel_write, parents, kernel)
            },
            {"panel_idx < PACKED_WIGNER_VALUES", "subgroup_lane == 0"},
        )


if __name__ == "__main__":
    unittest.main()
