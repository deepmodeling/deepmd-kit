# SPDX-License-Identifier: LGPL-3.0-or-later
"""HybridMuon trajectories against the production PyTorch implementation."""

import copy
import unittest
from importlib.util import (
    find_spec,
)

import numpy as np
import optax
import pytest

from deepmd.jax.env import (
    jax,
    jnp,
    nnx,
)
from deepmd.jax.optimizer import (
    build_hybrid_muon,
    export_hybrid_muon_state,
    import_hybrid_muon_state,
)
from deepmd.jax.optimizer.hybrid_muon import (
    MuonState,
    _gram_newton_schulz_orth,
    _newton_schulz_orth,
    _parameter_name,
    _route_parameter,
)
from deepmd.jax.utils.network import (
    NativeLayer,
)

TEST_PT = find_spec("torch") is not None

if TEST_PT:
    import torch

    from deepmd.pt.optimizer.hybrid_muon import (
        HybridMuonOptimizer,
        _batched_newton_schulz_orth,
        _GramNewtonSchulzOrthogonalizer,
    )

    PT_DEVICE = torch.device("cuda" if jax.default_backend() == "gpu" else "cpu")


@pytest.mark.skipif(not TEST_PT, reason="PyTorch is required")
@pytest.mark.parametrize("mode", ["2d", "flat", "slice"])
@pytest.mark.parametrize("gram", [False, True])
@pytest.mark.parametrize("magma", [False, True])
def test_torch_trajectory(mode: str, gram: bool, magma: bool) -> None:
    """Mixed routes follow the same eight-step, changing-rate trajectory."""
    rng = np.random.default_rng(183)
    shapes = {
        "square.weight": (4, 4),
        "wide.weight": (4, 8),
        "tall.weight": (8, 4),
        "tensor.weight": (2, 3, 4),
        "singleton.weight": (1, 3, 1, 4),
        "scalar": (),
        "vector": (5,),
        "ADAM_scale.0": (3, 4),
        "adamw_gate": (3,),
        "attention_bias": (3, 4),
        "layer.b": (4,),
        "radial_embedding.net.0.w": (4, 6),
    }
    initial = {
        name: rng.normal(size=shape).astype(np.float32)
        for name, shape in shapes.items()
    }
    params = {name: jnp.asarray(value) for name, value in initial.items()}
    torch_params = {
        name: torch.nn.Parameter(torch.tensor(value, device=PT_DEVICE))
        for name, value in initial.items()
    }
    options = {
        "momentum": 0.91,
        "weight_decay": 0.08,
        "adam_betas": (0.85, 0.97),
        "lr_adjust": 2.0 if magma else 0.0,
        "lr_adjust_coeff": 0.18,
        "muon_mode": mode,
        "enable_gram": gram,
        "magma_muon": magma,
        "adam_patterns": ("radial_embedding.net.0.",),
    }
    torch_opt = HybridMuonOptimizer(
        torch_params.values(),
        named_parameters=torch_params.items(),
        flash_muon=False,
        **options,
    )
    transform = build_hybrid_muon(**options)
    state = transform.init(params)
    update = jax.jit(transform.update)
    accumulated_rate = 0.0
    for step in range(8):
        rate = 0.003 * 0.8**step
        accumulated_rate += rate
        gradients = {
            name: rng.normal(size=shape).astype(np.float32)
            for name, shape in shapes.items()
        }
        for name, value in torch_params.items():
            value.grad = torch.tensor(gradients[name], device=PT_DEVICE)
        torch_opt.param_groups[0]["lr"] = rate
        torch_opt.step()
        changes, state = update(gradients, state, params, learning_rate=rate)
        params = optax.apply_updates(params, changes)
        for name, value in params.items():
            # NS uses bfloat16 internally; the two compilers may round a
            # recurrence at adjacent bfloat16 values. Adam stays in float32.
            pt_state = torch_opt.state[torch_params[name]]
            muon_route = "momentum_buffer" in pt_state
            absolute_tolerance = 1e-8
            if muon_route:
                route = _route_parameter(
                    name, shapes[name], mode, options["adam_patterns"]
                )
                _, rows, cols = route.matrix_shape
                scale = (
                    0.18 * np.sqrt(max(rows, cols))
                    if options["lr_adjust"] <= 0
                    else np.sqrt(max(1.0, rows / cols))
                )
                # The direct orthogonalization test bounds the bfloat16
                # recurrence error by 0.02. Integration accumulates that
                # bound with the scheduled rates and matrix scale.
                absolute_tolerance = accumulated_rate * scale * 0.02
            np.testing.assert_allclose(
                value,
                torch_params[name].detach().cpu().numpy(),
                rtol=0.0 if muon_route else 8 * np.finfo(np.float32).eps,
                atol=absolute_tolerance,
                err_msg=f"{name}, step={step}, mode={mode}, gram={gram}, magma={magma}",
            )
        flat, _ = jax.tree_util.tree_flatten_with_path(params)
        for (path, _), local_state in zip(flat, state.parameters, strict=True):
            reference = torch_opt.state[torch_params[_parameter_name(path)]]
            if isinstance(local_state, MuonState):
                np.testing.assert_allclose(
                    local_state.momentum,
                    reference["momentum_buffer"].cpu().numpy(),
                    atol=2e-7,
                    rtol=1e-6,
                )
                if magma:
                    np.testing.assert_allclose(
                        local_state.magma_score,
                        reference["magma_score"].cpu().numpy(),
                        atol=2e-7,
                        rtol=1e-6,
                    )
            else:
                np.testing.assert_allclose(
                    local_state.exp_avg,
                    reference["exp_avg"].cpu().numpy(),
                    atol=2e-7,
                    rtol=1e-6,
                )
                np.testing.assert_allclose(
                    local_state.exp_avg_sq,
                    reference["exp_avg_sq"].cpu().numpy(),
                    atol=2e-7,
                    rtol=1e-6,
                )


class TestHybridMuonState(unittest.TestCase):
    def test_routes(self) -> None:
        cases = (
            ("x.b", (4, 4), "adam", None),
            ("x.bias_matrix", (4, 4), "adam", None),
            ("x.adam_scale.0", (4, 4), "adam", None),
            ("x.ADAMW_scale.1", (4,), "adamw", None),
            ("x.weight", (1, 4, 1), "adam", None),
            ("x.weight", (1, 2, 4, 1, 3), "muon", (2, 4, 3)),
            ("x.radial.w", (4, 4), "adamw", None),
        )
        for name, shape, kind, matrix_shape in cases:
            self.assertEqual(
                _route_parameter(name, shape, "slice", ("radial.",)),
                (kind, matrix_shape),
            )
        self.assertEqual(
            _route_parameter("w", (2, 3, 4), "flat", ()), ("muon", (1, 6, 4))
        )
        self.assertEqual(_route_parameter("w", (2, 3, 4), "2d", ()), ("adamw", None))

    def test_nnx_update(self) -> None:
        model = NativeLayer(3, 4)
        transform = build_hybrid_muon()
        optimizer = nnx.Optimizer(model, transform, wrt=nnx.Param)

        @nnx.jit
        def step(model: NativeLayer, optimizer: nnx.Optimizer, rate: jax.Array) -> None:
            gradients = nnx.grad(lambda m: jnp.sum(m(jnp.ones((2, 3))) ** 2))(model)
            optimizer.update(model, gradients, learning_rate=rate)

        before = np.asarray(model.w[...])
        step(model, optimizer, jnp.asarray(0.01))
        self.assertGreater(np.max(np.abs(np.asarray(model.w[...]) - before)), 0.0)
        state = nnx.state(optimizer)
        self.assertTrue(
            all(np.all(np.isfinite(v)) for v in jax.tree_util.tree_leaves(state))
        )

    def test_resume_and_inactive_state(self) -> None:
        params = {
            "w": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 12,
            "bias": jnp.ones(4),
        }
        gradients = jax.tree_util.tree_map(jnp.ones_like, params)
        transform = build_hybrid_muon()
        state = transform.init(params)
        update = jax.jit(transform.update)
        for _ in range(3):
            changes, state = update(gradients, state, params, learning_rate=0.002)
            params = optax.apply_updates(params, changes)
        saved = import_hybrid_muon_state(
            copy.deepcopy(export_hybrid_muon_state(state, params)),
            params,
            transform.init(params),
        )
        frozen, inactive = update(
            gradients,
            state,
            params,
            learning_rate=0.01,
            active={"w": False, "bias": False},
        )
        for value in jax.tree_util.tree_leaves(frozen):
            np.testing.assert_array_equal(value, 0.0)
        for left, right in zip(
            jax.tree_util.tree_leaves(inactive),
            jax.tree_util.tree_leaves(state),
            strict=True,
        ):
            np.testing.assert_array_equal(left, right)
        expected, expected_state = update(gradients, state, params, learning_rate=0.001)
        resumed, resumed_state = update(gradients, saved, params, learning_rate=0.001)
        for left, right in zip(
            jax.tree_util.tree_leaves((expected, expected_state)),
            jax.tree_util.tree_leaves((resumed, resumed_state)),
            strict=True,
        ):
            np.testing.assert_array_equal(left, right)

    def test_zero_gradients_are_finite(self) -> None:
        params = {
            "square": jnp.zeros((4, 4)),
            "wide": jnp.zeros((4, 8)),
            "bias": jnp.zeros(4),
        }
        transform = build_hybrid_muon()
        update, state = jax.jit(transform.update)(
            params, transform.init(params), params, learning_rate=0.01
        )
        for value in jax.tree_util.tree_leaves((update, state)):
            self.assertTrue(np.all(np.isfinite(value)))
        for value in jax.tree_util.tree_leaves(update):
            np.testing.assert_array_equal(value, 0.0)

    def test_restore_rejects_incompatible_state(self) -> None:
        params = {"weight": jnp.zeros((3, 5)), "bias": jnp.zeros(3)}
        transform = build_hybrid_muon()
        template = transform.init(params)
        exported = export_hybrid_muon_state(template, params)
        with self.assertRaisesRegex(ValueError, "parameter mismatch"):
            import_hybrid_muon_state({}, params, template)
        malformed = copy.deepcopy(exported)
        malformed["weight"]["momentum_buffer"] = np.zeros((5, 3))
        with self.assertRaisesRegex(ValueError, "expects shape"):
            import_hybrid_muon_state(malformed, params, template)
        malformed = copy.deepcopy(exported)
        del malformed["bias"]["beta1_pow"]
        with self.assertRaisesRegex(ValueError, "missing its beta1 clock"):
            import_hybrid_muon_state(malformed, params, template)

    def test_independent_adam_clocks(self) -> None:
        params = {"left_bias": jnp.ones(3), "right_bias": jnp.ones(4)}
        transform = build_hybrid_muon()
        state = transform.init(params)
        update = jax.jit(transform.update)
        for index in range(4):
            _, state = update(
                params,
                state,
                params,
                learning_rate=0.001,
                active={"left_bias": True, "right_bias": index % 2 == 0},
            )
        np.testing.assert_allclose(state.parameters[0].beta1_power, 0.9**4, rtol=1e-6)
        np.testing.assert_allclose(state.parameters[1].beta1_power, 0.9**2, rtol=1e-6)


@pytest.mark.skipif(not TEST_PT, reason="PyTorch is required")
@pytest.mark.parametrize("shape", [(2, 4, 4), (2, 4, 8), (2, 8, 4)])
def test_orthogonalization(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(17)
    values = rng.normal(size=shape).astype(np.float32)
    for jax_func, torch_func, tolerance in (
        (_newton_schulz_orth, _batched_newton_schulz_orth, 0.02),
        (_gram_newton_schulz_orth, _GramNewtonSchulzOrthogonalizer(), 0.004),
    ):
        expected = (
            torch_func(torch.tensor(values, device=PT_DEVICE)).float().cpu().numpy()
        )
        result = jax.jit(jax_func)(jnp.asarray(values))
        np.testing.assert_allclose(
            np.asarray(result, dtype=np.float32), expected, rtol=0, atol=tolerance
        )


@pytest.mark.skipif(not TEST_PT, reason="PyTorch is required")
def test_cross_backend_optimizer_resume() -> None:
    """Optimizer state transfer preserves momentum, Magma, and Adam clocks."""
    rng = np.random.default_rng(79)
    initial = {
        "weight": rng.normal(size=(3, 5)).astype(np.float32),
        "bias": rng.normal(size=(3,)).astype(np.float32),
    }
    pt_params = {
        name: torch.nn.Parameter(torch.tensor(value, device=PT_DEVICE))
        for name, value in initial.items()
    }
    reference = HybridMuonOptimizer(
        pt_params.values(), named_parameters=pt_params.items(), flash_muon=False
    )
    gradients = [
        {
            name: rng.normal(size=value.shape).astype(np.float32)
            for name, value in initial.items()
        }
        for _ in range(9)
    ]
    rates = [0.002 * 0.8**step for step in range(9)]

    def torch_step(optimizer: HybridMuonOptimizer, params: dict, step: int) -> None:
        optimizer.param_groups[0]["lr"] = rates[step]
        for name, value in params.items():
            value.grad = torch.tensor(gradients[step][name], device=PT_DEVICE)
        optimizer.step()

    for step in range(3):
        torch_step(reference, pt_params, step)
    params = {
        name: jnp.asarray(value.detach().cpu().numpy())
        for name, value in pt_params.items()
    }
    named = {}
    group = reference.param_groups[0]
    for name, value in pt_params.items():
        named[name] = {
            key: array.cpu().numpy() for key, array in reference.state[value].items()
        }
        if "exp_avg" in named[name]:
            for key in ("beta1_pow_device", "beta2_pow_device"):
                named[name][key] = group[key].cpu().numpy()
    transform = build_hybrid_muon()
    state = import_hybrid_muon_state(named, params, transform.init(params))
    update = jax.jit(transform.update)
    for step in range(3, 6):
        torch_step(reference, pt_params, step)
        changes, state = update(
            gradients[step], state, params, learning_rate=rates[step]
        )
        params = optax.apply_updates(params, changes)
        for name, value in params.items():
            np.testing.assert_allclose(
                value, pt_params[name].detach().cpu().numpy(), rtol=1e-6, atol=3e-6
            )

    transferred = {
        name: torch.nn.Parameter(torch.tensor(np.asarray(value), device=PT_DEVICE))
        for name, value in params.items()
    }
    target = HybridMuonOptimizer(
        transferred.values(), named_parameters=transferred.items(), flash_muon=False
    )
    exported = export_hybrid_muon_state(state, params)
    state_dict = target.state_dict()
    for parameter_id, name in zip(
        state_dict["param_groups"][0]["params"], transferred, strict=True
    ):
        state_dict["state"][parameter_id] = {
            key: torch.tensor(value, device=PT_DEVICE)
            for key, value in exported[name].items()
        }
    target.load_state_dict(state_dict)
    for step in range(6, 9):
        torch_step(target, transferred, step)
        changes, state = update(
            gradients[step], state, params, learning_rate=rates[step]
        )
        params = optax.apply_updates(params, changes)
        for name, value in params.items():
            np.testing.assert_allclose(
                value, transferred[name].detach().cpu().numpy(), rtol=1e-6, atol=3e-6
            )


@pytest.mark.skipif(not TEST_PT, reason="PyTorch is required")
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "bfloat16"])
def test_momentum_dtype(dtype: str) -> None:
    """Low-precision momentum rounds the interpolated value, not its coefficient."""
    torch_dtype = getattr(torch, dtype)
    jax_dtype = getattr(jnp, dtype)
    rng = np.random.default_rng(18)
    value = torch.nn.Parameter(torch.zeros((3, 5), dtype=torch_dtype, device=PT_DEVICE))
    optimizer = HybridMuonOptimizer([value], flash_muon=False)
    params = {"w": jnp.zeros((3, 5), dtype=jax_dtype)}
    transform = build_hybrid_muon()
    state = transform.init(params)
    for _ in range(3):
        value.grad = torch.tensor(
            rng.normal(size=(3, 5)), dtype=torch_dtype, device=PT_DEVICE
        )
        gradients = {
            "w": jnp.asarray(value.grad.double().cpu().numpy(), dtype=jax_dtype)
        }
        optimizer.step()
        _, state = transform.update(gradients, state, params, learning_rate=0.0005)
        self_momentum = np.asarray(state.parameters[0].momentum, dtype=np.float64)
        torch_momentum = (
            optimizer.state[value]["momentum_buffer"].double().cpu().numpy()
        )
        np.testing.assert_allclose(self_momentum, torch_momentum, atol=1e-8, rtol=1e-7)
