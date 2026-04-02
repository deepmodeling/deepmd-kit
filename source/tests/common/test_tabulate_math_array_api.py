# SPDX-License-Identifier: LGPL-3.0-or-later
import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np


def _load_tabulate_math_module():
    root = Path(__file__).resolve().parents[3]
    module_path = root / "deepmd" / "utils" / "tabulate_math.py"

    deepmd_pkg = types.ModuleType("deepmd")
    deepmd_pkg.__path__ = []
    sys.modules.setdefault("deepmd", deepmd_pkg)

    dpmodel_pkg = types.ModuleType("deepmd.dpmodel")
    dpmodel_pkg.__path__ = []
    sys.modules["deepmd.dpmodel"] = dpmodel_pkg

    dpmodel_common = types.ModuleType("deepmd.dpmodel.common")
    dpmodel_common.to_numpy_array = np.asarray
    sys.modules["deepmd.dpmodel.common"] = dpmodel_common

    dpmodel_utils_pkg = types.ModuleType("deepmd.dpmodel.utils")
    dpmodel_utils_pkg.__path__ = []
    sys.modules["deepmd.dpmodel.utils"] = dpmodel_utils_pkg

    dpmodel_network = types.ModuleType("deepmd.dpmodel.utils.network")

    def get_activation_fn(name: str):
        name = name.lower()
        if name == "tanh":
            return np.tanh
        if name in ("none", "linear"):
            return lambda x: x
        raise NotImplementedError(name)

    dpmodel_network.get_activation_fn = get_activation_fn
    sys.modules["deepmd.dpmodel.utils.network"] = dpmodel_network

    utils_pkg = types.ModuleType("deepmd.utils")
    utils_pkg.__path__ = []
    sys.modules["deepmd.utils"] = utils_pkg

    utils_tabulate = types.ModuleType("deepmd.utils.tabulate")

    class BaseTabulate:
        def __init__(self, descrpt, neuron, type_one_side, exclude_types):
            self.descrpt = descrpt
            self.neuron = neuron
            self.type_one_side = type_one_side
            self.exclude_types = exclude_types

    utils_tabulate.BaseTabulate = BaseTabulate
    sys.modules["deepmd.utils.tabulate"] = utils_tabulate

    spec = importlib.util.spec_from_file_location(
        "deepmd.utils.tabulate_math_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tm = _load_tabulate_math_module()


class FakeArray:
    def __init__(self, data, namespace):
        self._data = np.asarray(data)
        self._namespace = namespace

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def astype(self, dtype):
        return FakeArray(self._data.astype(dtype), self._namespace)

    def __array__(self, dtype=None):
        if dtype is None:
            return self._data
        return self._data.astype(dtype)

    def __getitem__(self, item):
        return FakeArray(self._data[item], self._namespace)

    def __add__(self, other):
        return FakeArray(self._data + _to_np(other), self._namespace)

    def __radd__(self, other):
        return FakeArray(_to_np(other) + self._data, self._namespace)

    def __sub__(self, other):
        return FakeArray(self._data - _to_np(other), self._namespace)

    def __rsub__(self, other):
        return FakeArray(_to_np(other) - self._data, self._namespace)

    def __mul__(self, other):
        return FakeArray(self._data * _to_np(other), self._namespace)

    def __rmul__(self, other):
        return FakeArray(_to_np(other) * self._data, self._namespace)

    def __truediv__(self, other):
        return FakeArray(self._data / _to_np(other), self._namespace)

    def __rtruediv__(self, other):
        return FakeArray(_to_np(other) / self._data, self._namespace)

    def __pow__(self, other):
        return FakeArray(self._data ** _to_np(other), self._namespace)

    def __neg__(self):
        return FakeArray(-self._data, self._namespace)

    def __gt__(self, other):
        return FakeArray(self._data > _to_np(other), self._namespace)

    def __ge__(self, other):
        return FakeArray(self._data >= _to_np(other), self._namespace)

    def __lt__(self, other):
        return FakeArray(self._data < _to_np(other), self._namespace)

    def __and__(self, other):
        return FakeArray(self._data & _to_np(other), self._namespace)


class FakeXP:
    def __init__(self):
        self.calls = []
        self.bool = np.bool_
        self.float64 = np.float64
        self.float32 = np.float32
        self.int64 = np.int64
        self.pi = math.pi

    def _wrap(self, data):
        return FakeArray(data, self)

    def asarray(self, value, device=None, copy=None, dtype=None):
        self.calls.append("asarray")
        arr = np.asarray(value, dtype=dtype)
        if copy:
            arr = arr.copy()
        return self._wrap(arr)

    def exp(self, x):
        self.calls.append("exp")
        return self._wrap(np.exp(_to_np(x)))

    def where(self, cond, x, y):
        self.calls.append("where")
        return self._wrap(np.where(_to_np(cond), _to_np(x), _to_np(y)))

    def tanh(self, x):
        self.calls.append("tanh")
        return self._wrap(np.tanh(_to_np(x)))

    def ones_like(self, x):
        self.calls.append("ones_like")
        return self._wrap(np.ones_like(_to_np(x)))

    def zeros_like(self, x):
        self.calls.append("zeros_like")
        return self._wrap(np.zeros_like(_to_np(x)))

    def astype(self, x, dtype):
        self.calls.append("astype")
        return self._wrap(_to_np(x).astype(dtype))

    def reshape(self, x, shape):
        self.calls.append("reshape")
        return self._wrap(np.reshape(_to_np(x), shape))

    def broadcast_to(self, x, shape):
        self.calls.append("broadcast_to")
        return self._wrap(np.broadcast_to(_to_np(x), shape))

    def matmul(self, x, y):
        self.calls.append("matmul")
        return self._wrap(np.matmul(_to_np(x), _to_np(y)))

    def concat(self, xs, axis=0):
        self.calls.append("concat")
        return self._wrap(np.concatenate([_to_np(v) for v in xs], axis=axis))

    def ones(self, shape, dtype=None, device=None):
        self.calls.append("ones")
        return self._wrap(np.ones(shape, dtype=dtype))

    def empty(self, shape, dtype=None, device=None):
        self.calls.append("empty")
        return self._wrap(np.empty(shape, dtype=dtype))


def _to_np(value):
    if isinstance(value, FakeArray):
        return value._data
    return np.asarray(value)


def test_grad_and_chain_rule_helpers_use_array_api(monkeypatch):
    fake_xp = FakeXP()

    monkeypatch.setattr(tm.array_api_compat, "array_namespace", lambda *args: fake_xp)
    monkeypatch.setattr(tm.array_api_compat, "device", lambda arr: "fake-device")

    xbar = fake_xp.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=np.float64)
    y = fake_xp.asarray(np.tanh(_to_np(xbar)), dtype=np.float64)
    w = fake_xp.asarray([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64)

    dy_s = tm.unaggregated_dy_dx_s(y, w, xbar, 1)
    dy2_s = tm.unaggregated_dy2_dx_s(y, dy_s, w, xbar, 1)
    dy = tm.unaggregated_dy_dx(y, w, dy_s, xbar, 1)
    dy2 = tm.unaggregated_dy2_dx(y, w, dy_s, dy2_s, xbar, 1)

    np.testing.assert_allclose(
        np.asarray(dy_s),
        (1 - np.asarray(y) ** 2) * np.broadcast_to(np.asarray(w).reshape(-1)[:2], (2, 2)),
    )
    np.testing.assert_equal(np.asarray(dy2_s).shape, (2, 2))
    np.testing.assert_equal(np.asarray(dy).shape, (2, 2))
    np.testing.assert_equal(np.asarray(dy2).shape, (2, 2))

    assert "reshape" in fake_xp.calls
    assert "broadcast_to" in fake_xp.calls
    assert "matmul" in fake_xp.calls


def test_stable_sigmoid_and_silu_match_numpy(monkeypatch):
    fake_xp = FakeXP()

    monkeypatch.setattr(tm.array_api_compat, "array_namespace", lambda *args: fake_xp)

    xbar = fake_xp.asarray([[-1000.0, -1.0, 0.0, 1.0, 1000.0]], dtype=np.float64)
    stable = tm._stable_sigmoid(xbar)
    silu_grad = tm.grad(xbar, stable, 7)

    x_np = np.asarray(xbar)
    stable_np = np.empty_like(x_np)
    positive = x_np >= 0
    stable_np[positive] = 1.0 / (1.0 + np.exp(-x_np[positive]))
    exp_x = np.exp(x_np[~positive])
    stable_np[~positive] = exp_x / (1.0 + exp_x)
    silu_grad_np = stable_np + x_np * stable_np * (1 - stable_np)

    np.testing.assert_allclose(np.asarray(stable), stable_np)
    np.testing.assert_allclose(np.asarray(silu_grad), silu_grad_np)
    assert fake_xp.calls.count("exp") >= 2
    assert fake_xp.calls.count("where") >= 2
