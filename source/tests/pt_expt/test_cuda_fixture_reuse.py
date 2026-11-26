# SPDX-License-Identifier: LGPL-3.0-or-later
"""CPU checks of CUDA test setup reuse and per-case isolation."""

import copy
import dataclasses
from types import (
    SimpleNamespace,
)

import pytest
import torch

from .descriptor import test_dpa4c_cuda as dpa4c


@pytest.fixture
def graph_cache():
    dpa4c._graph_template.cache_clear()
    yield dpa4c._graph_template
    dpa4c._graph_template.cache_clear()


def _assert_same_graph(left, right):
    for field in dataclasses.fields(left[0]):
        actual, expected = getattr(left[0], field.name), getattr(right[0], field.name)
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        else:
            assert actual == expected
    torch.testing.assert_close(left[1], right[1], atol=0, rtol=0)


@pytest.mark.parametrize(
    "index,replacement",
    [
        (0, 2.5),
        (1, False),
        (2, 25),
        (3, True),
        (4, 6.0),
        (5, 2),
        (6, (0, 2, 3)),
        (7, torch.device("cpu:0")),
    ],
    ids=[
        "cutoff",
        "canonical",
        "nodes",
        "one-type",
        "side",
        "frames",
        "types",
        "device",
    ],
)
def test_graph_cache_keys_every_geometry_input(graph_cache, index, replacement):
    """Every geometry input misses the cache and agrees with a fresh build."""
    arguments = [3.0, True, 24, False, 5.0, 1, (0, 1), torch.device("cpu")]
    before = torch.random.get_rng_state().clone()
    first = graph_cache(*arguments)
    assert graph_cache(*arguments) is first
    assert graph_cache.cache_info().hits == 1
    arguments[index] = replacement
    changed = graph_cache(*arguments)
    assert graph_cache.cache_info().misses == 2
    assert graph_cache.cache_info().currsize == 1
    _assert_same_graph(changed, graph_cache.__wrapped__(*arguments))
    torch.testing.assert_close(torch.random.get_rng_state(), before, atol=0, rtol=0)


def test_graph_cache_returns_independent_tensors(graph_cache, monkeypatch):
    """Mutating one caller's graph cannot change the template or next caller."""
    devices = []

    def on_cpu(*arguments):
        devices.append(arguments[-1])
        return graph_cache(*arguments[:-1], torch.device("cpu"))

    monkeypatch.setattr(dpa4c, "_graph_template", on_cpu)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    descriptor = SimpleNamespace(rcut=3.0)
    first = dpa4c._build_graph(descriptor, canonical=True)
    expected = copy.deepcopy(first)
    for field in dataclasses.fields(first[0]):
        tensor = getattr(first[0], field.name)
        if isinstance(tensor, torch.Tensor):
            tensor.zero_()
    first[1].fill_(7)
    second = dpa4c._build_graph(descriptor, canonical=True)
    _assert_same_graph(second, expected)
    assert devices == [torch.device("cuda:0"), torch.device("cuda:0")]
    assert graph_cache.cache_info().misses == graph_cache.cache_info().hits == 1
    second[0].edge_vec.requires_grad_(True)
    second[0].edge_vec.square().sum().backward()
    third = dpa4c._build_graph(descriptor, canonical=True)
    assert not third[0].edge_vec.requires_grad
    assert third[0].edge_vec.grad is None
    _assert_same_graph(third, expected)
