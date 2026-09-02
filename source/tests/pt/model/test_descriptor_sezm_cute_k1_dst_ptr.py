# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Contracts for asynchronous K1 destination-pointer construction."""

from __future__ import (
    annotations,
)

import ast
import importlib
import unittest
from pathlib import (
    Path,
)
from types import (
    SimpleNamespace,
)
from typing import (
    Any,
)
from unittest import (
    mock,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - lightweight source-test host
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[4]
K1_PATH = REPO_ROOT / "deepmd/pt_expt/kernels/cute/sezm/k1.py"
EDGE_CACHE_PATH = REPO_ROOT / "deepmd/pt/model/descriptor/sezm_nn/edge_cache.py"


class _Tensor:
    pass


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} is missing")


def _load_extracted_dst_ptr_function(*, strict: bool = False):
    source = K1_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = _function(tree, "_dst_ptr_from_sorted")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "Any": Any,
        "Tensor": _Tensor,
        "runtime_policy": SimpleNamespace(
            is_cute_strict_enabled=lambda: strict,
        ),
    }
    exec(compile(module, str(K1_PATH), "exec"), namespace)
    return namespace["_dst_ptr_from_sorted"]


def _load_extracted_sorted_metadata_function():
    source = EDGE_CACHE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = _function(tree, "build_sorted_edge_index_metadata")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(EDGE_CACHE_PATH), "exec"), namespace)
    return namespace["build_sorted_edge_index_metadata"]


class _FakeDst:
    device = "cuda:0"
    dtype = "int64"

    def __init__(self):
        self.contiguous_calls = 0

    def contiguous(self):
        self.contiguous_calls += 1
        return self


class _RecordingTorch:
    int64 = "int64"

    def __init__(self):
        self.calls: list[tuple[Any, ...]] = []
        self.boundaries = object()
        self.result = object()

    def arange(self, stop, *, device, dtype):
        self.calls.append(("arange", stop, device, dtype))
        return self.boundaries

    def searchsorted(self, sorted_sequence, values):
        self.calls.append(("searchsorted", sorted_sequence, values))
        return self.result

    def bincount(self, *args, **kwargs):
        raise AssertionError("CUDA bincount must not construct sorted dst_ptr")


class TestK1DstPtrExtracted(unittest.TestCase):
    def test_sorted_path_operator_and_output_contract(self):
        helper = _load_extracted_dst_ptr_function()
        torch_module = _RecordingTorch()
        dst = _FakeDst()

        result = helper(
            torch_module,
            dst,
            8,
            destinations_sorted=True,
        )

        self.assertIs(result, torch_module.result)
        self.assertEqual(dst.contiguous_calls, 1)
        self.assertEqual(
            torch_module.calls,
            [
                ("arange", 9, dst.device, torch_module.int64),
                ("searchsorted", dst, torch_module.boundaries),
            ],
        )

    def test_unsorted_path_returns_before_tensor_work(self):
        helper = _load_extracted_dst_ptr_function()
        torch_module = _RecordingTorch()

        self.assertIsNone(
            helper(
                torch_module,
                _FakeDst(),
                8,
                destinations_sorted=False,
            )
        )
        self.assertEqual(torch_module.calls, [])


@unittest.skipIf(torch is None, "destination-pointer differential requires PyTorch")
class TestK1DstPtrTorch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = staticmethod(_load_extracted_dst_ptr_function())

    def test_matches_bincount_reference_and_preserves_layout(self):
        assert torch is not None
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        cases = (
            (1, []),
            (1, [0, 0, 0]),
            (8, [0, 0, 2, 5, 5, 5, 7]),
            (9, [1, 1, 1, 4, 8]),
            (17, [0, 3, 3, 7, 7, 7, 15]),
        )
        for device in devices:
            for n_node, values in cases:
                with self.subTest(device=device, n_node=n_node, values=values):
                    dst = torch.tensor(values, device=device, dtype=torch.int64)
                    counts = torch.bincount(dst, minlength=n_node)
                    expected = torch.empty(
                        n_node + 1,
                        device=device,
                        dtype=torch.int64,
                    )
                    expected[0] = 0
                    expected[1:] = counts.cumsum(0)

                    actual = self.helper(
                        torch,
                        dst,
                        n_node,
                        destinations_sorted=True,
                    )

                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(actual.dtype, torch.int64)
                    self.assertEqual(actual.device, dst.device)
                    self.assertTrue(actual.is_contiguous())
                    self.assertEqual(actual.stride(), (1,))

    def test_strict_mode_rejects_false_sorted_provenance(self):
        assert torch is not None
        helper = _load_extracted_dst_ptr_function(strict=True)
        dst = torch.tensor([0, 2, 1, 3], dtype=torch.int64, device="cpu")

        with self.assertRaisesRegex(RuntimeError, "monotonically nondecreasing"):
            helper(
                torch,
                dst,
                4,
                destinations_sorted=True,
            )

    def test_strict_mode_accepts_duplicate_sorted_destinations(self):
        assert torch is not None
        helper = _load_extracted_dst_ptr_function(strict=True)
        dst = torch.tensor([0, 0, 2, 3], dtype=torch.int64, device="cpu")

        actual = helper(
            torch,
            dst,
            4,
            destinations_sorted=True,
        )

        self.assertTrue(
            torch.equal(
                actual,
                torch.tensor([0, 2, 2, 3, 4], device="cpu"),
            )
        )

    def test_fullgraph_dynamic_compile_preserves_pointer_contract(self):
        assert torch is not None
        helper = self.helper
        if torch.cuda.is_available():
            device = torch.device("cuda")
            backend = "inductor"
        else:
            device = torch.device("cpu")
            backend = "aot_eager"

        def build_ptr(dst):
            return helper(
                torch,
                dst,
                8,
                destinations_sorted=True,
            )

        compiled = torch.compile(
            build_ptr,
            backend=backend,
            dynamic=True,
            fullgraph=True,
        )
        dst = torch.tensor(
            [0, 0, 2, 5, 5, 5, 7],
            device=device,
            dtype=torch.int64,
        )
        actual = compiled(dst)
        expected = torch.tensor(
            [0, 2, 2, 3, 3, 3, 6, 6, 7],
            device=device,
            dtype=torch.int64,
        )

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.stride(), (1,))

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "K1 custom-op registry lifetime regression requires CUDA",
    )
    def test_compile_cold_registration_survives_into_packed_custom_op_runtime(self):
        assert torch is not None
        k1 = importlib.import_module("deepmd.pt_expt.kernels.cute.sezm.k1")
        prior_registry = dict(k1._REGISTRY)
        prior_next_handle = k1._NEXT_HANDLE

        def cleanup():
            torch._dynamo.reset()
            k1._PACKED_RUNNER_CACHE.clear()
            k1._REGISTRY.clear()
            k1._REGISTRY.update(prior_registry)
            k1._NEXT_HANDLE = prior_next_handle

        self.addCleanup(cleanup)
        k1._REGISTRY.clear()
        k1._PACKED_RUNNER_CACHE.clear()
        torch._dynamo.reset()
        block = torch.nn.Module()
        config = k1.NeoK1RuntimeConfig()
        runtime_handles = []

        class FakeRunner:
            pass

        def fake_build(handle, x_arg, *args):
            del args
            entry = k1._REGISTRY[int(handle)]
            self.assertIs(entry.block, block)
            runtime_handles.append(int(handle))
            runner = FakeRunner()
            runner.final = x_arg.detach().clone()
            return runner

        def invoke(
            x_arg,
            d_arg,
            dt_arg,
            radial_arg,
            edge_arg,
            src_arg,
            dst_arg,
        ):
            state = getattr(block, "_deepmd_cute_k1_state", None)
            if state is None:
                state = k1._register_cute_k1_state(
                    block,
                    x_arg.device.index,
                    config,
                )
            dst_ptr = k1._dst_ptr_from_sorted(
                torch,
                dst_arg,
                x_arg.shape[0],
                destinations_sorted=True,
            )
            edge_src_gate = edge_arg.new_empty((0,))
            return k1.cute_k1(
                state.handle,
                x_arg,
                d_arg,
                dt_arg,
                radial_arg,
                edge_arg,
                src_arg,
                dst_arg,
                dst_ptr,
                edge_src_gate,
            )

        disabled_invoke = torch.compiler.disable(invoke)

        def compiled_entry(*args):
            return disabled_invoke(*args)

        device = torch.device("cuda:0")
        x = torch.randn(2, 16, 1, 32, device=device)
        d_full = torch.randn(3, 46, device=device)
        dt_full = torch.randn_like(d_full)
        radial = torch.randn(3, 4, 32, device=device)
        edge_env = torch.ones(3, 1, device=device)
        src = torch.tensor((0, 1, 0), dtype=torch.int64, device=device)
        dst = torch.tensor((0, 0, 1), dtype=torch.int64, device=device)

        with mock.patch.object(k1, "_build_runner", new=fake_build):
            compiled = torch.compile(compiled_entry, backend="eager", dynamic=True)
            first = compiled(x, d_full, dt_full, radial, edge_env, src, dst)
            second = compiled(x, d_full, dt_full, radial, edge_env, src, dst)

        state = block._deepmd_cute_k1_state
        self.assertEqual(runtime_handles, [state.handle, state.handle])
        self.assertEqual(list(k1._REGISTRY), [state.handle])
        self.assertIs(k1._REGISTRY[state.handle].block, block)
        torch.testing.assert_close(first, x, rtol=0.0, atol=0.0)
        torch.testing.assert_close(second, x, rtol=0.0, atol=0.0)


@unittest.skipIf(torch is None, "sorted edge metadata requires PyTorch")
class TestSortedEdgeIndexMetadata(unittest.TestCase):
    @staticmethod
    def _cache(src, dst, node_count):
        assert torch is not None
        edge_cache = importlib.import_module(
            "deepmd.pt.model.descriptor.sezm_nn.edge_cache"
        )
        device = torch.device("cpu")
        src = src.to(device=device)
        dst = dst.to(device=device)
        edge_count = src.numel()
        return edge_cache.EdgeFeatureCache(
            src=src,
            dst=dst,
            edge_type_feat=torch.empty(edge_count, 1, device=device),
            edge_vec=torch.empty(edge_count, 3, device=device),
            edge_rbf=torch.empty(edge_count, 1, device=device),
            edge_env=torch.ones(edge_count, 1, device=device),
            deg=torch.zeros(node_count, device=device),
            inv_sqrt_deg=torch.ones(node_count, 1, 1, device=device),
            destinations_sorted=True,
        )

    def test_builds_destination_and_indirect_source_csr(self):
        assert torch is not None
        edge_cache = importlib.import_module(
            "deepmd.pt.model.descriptor.sezm_nn.edge_cache"
        )
        src = torch.tensor(
            [2, 0, 3, 0, 1, 2],
            dtype=torch.int64,
            device="cpu",
        )
        dst = torch.tensor(
            [0, 0, 1, 2, 2, 3],
            dtype=torch.int64,
            device="cpu",
        )
        dst_ptr, source_order, source_ptr = edge_cache.build_sorted_edge_index_metadata(
            src, dst, 4
        )

        self.assertEqual(dst_ptr.dtype, torch.int32)
        self.assertEqual(source_order.dtype, torch.int32)
        self.assertEqual(source_ptr.dtype, torch.int32)
        torch.testing.assert_close(
            dst_ptr,
            torch.tensor(
                [0, 2, 3, 5, 6],
                dtype=torch.int32,
                device="cpu",
            ),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            src.index_select(0, source_order.to(torch.int64)),
            torch.tensor(
                [0, 0, 1, 2, 2, 3],
                dtype=torch.int64,
                device="cpu",
            ),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            source_ptr,
            torch.tensor(
                [0, 2, 3, 5, 6],
                dtype=torch.int32,
                device="cpu",
            ),
            rtol=0.0,
            atol=0.0,
        )

    def test_empty_edges_build_valid_zero_csr(self):
        assert torch is not None
        edge_cache = importlib.import_module(
            "deepmd.pt.model.descriptor.sezm_nn.edge_cache"
        )
        dst_ptr, source_order, source_ptr = edge_cache.build_sorted_edge_index_metadata(
            torch.empty(0, dtype=torch.int64, device="cpu"),
            torch.empty(0, dtype=torch.int64, device="cpu"),
            4,
        )

        torch.testing.assert_close(
            dst_ptr,
            torch.zeros(5, dtype=torch.int32, device="cpu"),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(source_order.numel(), 0)
        torch.testing.assert_close(
            source_ptr,
            torch.zeros(5, dtype=torch.int32, device="cpu"),
            rtol=0.0,
            atol=0.0,
        )

    def test_strict_metadata_rejects_unsorted_destinations(self):
        assert torch is not None
        builder = _load_extracted_sorted_metadata_function()
        src = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cpu")
        dst = torch.tensor([0, 2, 1, 3], dtype=torch.int64, device="cpu")

        with self.assertRaisesRegex(RuntimeError, "monotonically nondecreasing"):
            builder(
                src,
                dst,
                4,
                validate_sorted=True,
            )

    def test_strict_metadata_builder_is_fullgraph_compilable(self):
        assert torch is not None
        builder = _load_extracted_sorted_metadata_function()
        dynamo = getattr(torch, "_dynamo", None)
        if dynamo is None:
            self.skipTest("torch._dynamo is unavailable")

        def build_ptr(src, dst):
            dst_ptr, _, _ = builder(
                src,
                dst,
                4,
                validate_sorted=True,
            )
            return dst_ptr

        compiled = torch.compile(
            build_ptr,
            backend="eager",
            dynamic=True,
            fullgraph=True,
        )
        src = torch.tensor([2, 0, 1], dtype=torch.int64, device="cpu")
        dst = torch.tensor([0, 1, 2], dtype=torch.int64, device="cpu")
        actual = compiled(src, dst)

        torch.testing.assert_close(
            actual,
            torch.tensor([0, 1, 2, 3, 3], dtype=torch.int32, device="cpu"),
            rtol=0.0,
            atol=0.0,
        )

    def test_dynamic_node_and_edge_counts_build_independent_metadata(self):
        assert torch is not None
        edge_cache = importlib.import_module(
            "deepmd.pt.model.descriptor.sezm_nn.edge_cache"
        )
        first = edge_cache.build_sorted_edge_index_metadata(
            torch.tensor([0, 1, 2], dtype=torch.int64, device="cpu"),
            torch.tensor([0, 1, 2], dtype=torch.int64, device="cpu"),
            4,
        )
        second = edge_cache.build_sorted_edge_index_metadata(
            torch.tensor(
                [0, 2, 4, 1, 3],
                dtype=torch.int64,
                device="cpu",
            ),
            torch.tensor(
                [0, 0, 2, 4, 5],
                dtype=torch.int64,
                device="cpu",
            ),
            6,
        )

        torch.testing.assert_close(
            first[0],
            torch.tensor([0, 1, 2, 3, 3], dtype=torch.int32, device="cpu"),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            second[0],
            torch.tensor(
                [0, 2, 2, 3, 3, 4, 5],
                dtype=torch.int32,
                device="cpu",
            ),
            rtol=0.0,
            atol=0.0,
        )
        self.assertNotEqual(first[0].data_ptr(), second[0].data_ptr())
        self.assertNotEqual(first[1].data_ptr(), second[1].data_ptr())
        self.assertNotEqual(first[2].data_ptr(), second[2].data_ptr())

    def test_explicit_tensor_metadata_survives_disabled_k1_boundary(self):
        assert torch is not None
        edge_cache = importlib.import_module(
            "deepmd.pt.model.descriptor.sezm_nn.edge_cache"
        )
        dynamo = getattr(torch, "_dynamo", None)
        if dynamo is None:
            self.skipTest("torch._dynamo is unavailable")

        def k1_boundary(
            x,
            cache,
            radial,
            dst_ptr,
            source_order,
            source_ptr,
        ):
            del radial
            assert dst_ptr is not None
            assert source_order is not None
            assert source_ptr is not None
            return (
                x
                + cache.edge_env.sum()
                + dst_ptr.sum()
                + source_order.sum()
                + source_ptr.sum()
            )

        disabled_k1_boundary = torch.compiler.disable(k1_boundary)

        def consume(cache, x, radial, dst_ptr, source_order, source_ptr):
            before_break = x + cache.edge_env.sum()
            return disabled_k1_boundary(
                before_break,
                cache,
                radial,
                dst_ptr,
                source_order,
                source_ptr,
            )

        compiled = torch.compile(consume, backend="eager", dynamic=True)
        cases = (
            (
                torch.tensor([0, 1, 2], dtype=torch.int64, device="cpu"),
                torch.tensor([0, 1, 2], dtype=torch.int64, device="cpu"),
                4,
            ),
            (
                torch.tensor(
                    [0, 2, 4, 1, 3],
                    dtype=torch.int64,
                    device="cpu",
                ),
                torch.tensor(
                    [0, 0, 2, 4, 5],
                    dtype=torch.int64,
                    device="cpu",
                ),
                6,
            ),
        )
        for src, dst, node_count in cases:
            cache = self._cache(src, dst, node_count)
            dst_ptr, source_order, source_ptr = (
                edge_cache.build_sorted_edge_index_metadata(
                    src,
                    dst,
                    node_count,
                )
            )
            radial = torch.ones(src.numel(), 4, 1, device="cpu")
            args = (
                cache,
                torch.ones(1, device="cpu"),
                radial,
                dst_ptr,
                source_order,
                source_ptr,
            )
            eager = consume(*args)
            actual = compiled(*args)
            torch.testing.assert_close(
                actual,
                eager,
                rtol=0.0,
                atol=0.0,
            )


if __name__ == "__main__":
    unittest.main()
