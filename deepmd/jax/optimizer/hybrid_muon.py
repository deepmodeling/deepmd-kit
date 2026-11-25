# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pure JAX HybridMuon with the PyTorch backend's routing and update rules."""

import math
from collections.abc import (
    Mapping,
    Sequence,
)
from typing import (
    Any,
    NamedTuple,
)

import numpy as np
import optax
from jax.sharding import (
    NamedSharding,
)

from deepmd.jax.env import (
    jax,
    jnp,
)

NS_EPS = 1e-7
ADAM_EPS = 1e-20
NS_STEPS_FAST = 8
NS_STEPS_POLISH = 2
NS_COEFF_FAST = (3.4445, -4.7750, 2.0315)
NS_COEFF_POLISH = (2.0, -1.5, 0.5)
POLAR_EXPRESS_COEFFICIENTS = tuple(
    (a / 1.05, b / 1.05**3, c / 1.05**5)
    for a, b, c in (
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    )
)


class MuonState(NamedTuple):
    """Per-parameter momentum and per-matrix Magma alignment averages."""

    momentum: jax.Array
    magma_score: jax.Array


class AdamState(NamedTuple):
    """Float32 Adam moments and bias-correction powers."""

    exp_avg: jax.Array
    exp_avg_sq: jax.Array
    beta1_power: jax.Array
    beta2_power: jax.Array


class HybridMuonState(NamedTuple):
    """Optimizer state in the deterministic flattened parameter order."""

    parameters: tuple[MuonState | AdamState, ...]


class _Route(NamedTuple):
    kind: str
    matrix_shape: tuple[int, int, int] | None


def export_hybrid_muon_state(
    state: HybridMuonState, params: Any
) -> dict[str, dict[str, np.ndarray]]:
    """
    Export named optimizer arrays in the PyTorch HybridMuon state vocabulary.

    Parameters
    ----------
    state : HybridMuonState
        Optimizer state associated with ``params``.
    params : Any
        NNX parameter state or a pytree of arrays. Its deterministic tree paths
        supply the names; NNX variable wrappers are omitted.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Per-parameter state on the host. The caller maps model-specific parameter
        names before assigning states to another backend's parameter IDs. Adam
        powers use the legacy per-parameter representation accepted by the
        PyTorch optimizer's checkpoint migration.
    """
    items, _ = jax.tree_util.tree_flatten_with_path(params)
    exported = {}
    for (path, _), current in zip(items, state.parameters, strict=True):
        if isinstance(current, MuonState):
            arrays = {
                "momentum_buffer": current.momentum,
                "magma_score": current.magma_score,
            }
        else:
            arrays = {
                "exp_avg": current.exp_avg,
                "exp_avg_sq": current.exp_avg_sq,
                "beta1_pow": current.beta1_power,
                "beta2_pow": current.beta2_power,
            }
        exported[_parameter_name(path)] = {
            key: np.asarray(value).copy() for key, value in arrays.items()
        }
    return exported


def import_hybrid_muon_state(
    named_state: Mapping[str, Mapping[str, Any]],
    params: Any,
    template: HybridMuonState,
) -> HybridMuonState:
    """
    Restore a named checkpoint into a JAX HybridMuon state.

    Parameters
    ----------
    named_state : Mapping[str, Mapping[str, Any]]
        States keyed by JAX parameter names, after any model-specific mapping.
        Values are NumPy arrays or scalars. PyTorch's group powers must be copied
        into the corresponding Adam states under ``beta1_pow_device`` and
        ``beta2_pow_device``. Per-parameter powers and bias corrections are also
        accepted. An empty state denotes a parameter that has never been active.
    params : Any
        Target parameter tree.
    template : HybridMuonState
        State from the target transformation's ``init(params)``. Routing and
        array shapes must agree with the checkpoint.

    Returns
    -------
    HybridMuonState
        Restored arrays with the target state dtypes and fixed pytree structure.

    Raises
    ------
    ValueError
        If parameter names, routes, shapes, or Adam clocks are incompatible.
    """
    items, _ = jax.tree_util.tree_flatten_with_path(params)
    names = [_parameter_name(path) for path, _ in items]
    if set(names) != set(named_state):
        missing = sorted(set(names) - set(named_state))
        extra = sorted(set(named_state) - set(names))
        raise ValueError(
            f"HybridMuon checkpoint parameter mismatch: missing={missing}, extra={extra}."
        )

    def restore_array(value: Any, target: jax.Array, name: str, key: str) -> jax.Array:
        # Host checkpoint arrays may alias an active optimizer's mutable
        # buffers. The restored functional state owns an independent copy.
        array = jnp.array(value, dtype=target.dtype, copy=True)
        if array.shape != target.shape:
            raise ValueError(
                f"HybridMuon state {name}.{key} expects shape {target.shape}, got {array.shape}."
            )
        return array

    restored = []
    for name, target in zip(names, template.parameters, strict=True):
        source = named_state[name]
        if not source:
            restored.append(target)
            continue
        if isinstance(target, MuonState):
            if "momentum_buffer" not in source or "exp_avg" in source:
                raise ValueError(
                    f"HybridMuon checkpoint route for {name} must be Muon."
                )
            restored.append(
                MuonState(
                    restore_array(
                        source["momentum_buffer"],
                        target.momentum,
                        name,
                        "momentum_buffer",
                    ),
                    restore_array(
                        source.get("magma_score", target.magma_score),
                        target.magma_score,
                        name,
                        "magma_score",
                    ),
                )
            )
            continue
        if (
            "exp_avg" not in source
            or "exp_avg_sq" not in source
            or "momentum_buffer" in source
        ):
            raise ValueError(f"HybridMuon checkpoint route for {name} must be Adam.")
        powers = []
        for index, target_power in enumerate(
            (target.beta1_power, target.beta2_power), 1
        ):
            correction = f"bias_correction{index}"
            power = f"beta{index}_pow"
            group_power = f"beta{index}_pow_device"
            if correction in source:
                value = 1.0 - np.asarray(source[correction], dtype=np.float32)
            elif power in source:
                value = source[power]
            elif group_power in source:
                value = source[group_power]
            else:
                raise ValueError(
                    f"HybridMuon Adam state for {name} is missing its beta{index} clock."
                )
            powers.append(restore_array(value, target_power, name, power))
        restored.append(
            AdamState(
                restore_array(source["exp_avg"], target.exp_avg, name, "exp_avg"),
                restore_array(
                    source["exp_avg_sq"], target.exp_avg_sq, name, "exp_avg_sq"
                ),
                *powers,
            )
        )
    return HybridMuonState(tuple(restored))


def _parameter_name(path: tuple[Any, ...]) -> str:
    """Remove NNX variable wrappers from a JAX tree path."""
    segments = []
    for entry in path:
        if isinstance(entry, jax.tree_util.DictKey):
            segments.append(str(entry.key))
        elif isinstance(entry, jax.tree_util.SequenceKey):
            segments.append(str(entry.idx))
        elif isinstance(entry, jax.tree_util.GetAttrKey):
            if entry.name != "value":
                segments.append(entry.name)
        else:
            raise TypeError(f"Unsupported optimizer parameter path entry: {entry!r}")
    return ".".join(segments)


def _route_parameter(
    name: str,
    shape: tuple[int, ...],
    muon_mode: str,
    adam_patterns: tuple[str, ...],
) -> _Route:
    segments = name.lower().split(".")
    while len(segments) > 1 and segments[-1].isdigit():
        segments.pop()
    leaf = segments[-1]
    if leaf == "b" or "bias" in leaf or leaf.startswith("adam_"):
        return _Route("adam", None)
    if leaf.startswith("adamw_"):
        return _Route("adamw", None)
    effective = tuple(dim for dim in shape if dim != 1)
    if len(effective) < 2:
        return _Route("adam", None)
    if (muon_mode == "2d" and len(effective) > 2) or any(
        pattern in name.lower() for pattern in adam_patterns
    ):
        return _Route("adamw", None)
    if muon_mode == "flat":
        return _Route("muon", (1, math.prod(effective[:-1]), effective[-1]))
    return _Route("muon", (math.prod(effective[:-2]), effective[-2], effective[-1]))


def _matrix_product(left: jax.Array, right: jax.Array) -> jax.Array:
    """Accumulate the BF16 recurrence in FP32 on the target hardware."""
    # An FP32 accumulator does not change the input type passed to cuBLAS.
    # Pre-Ampere CUDA devices have no BF16 matrix multiply. Widening the
    # already rounded operands is exact; the caller retains each BF16
    # iteration boundary.
    if left.dtype == jnp.bfloat16 and any(
        int(device.compute_capability.split(".")[0]) < 8
        for device in jax.local_devices()
        if "cuda" in device.client.platform_version.lower()
    ):
        left = left.astype(jnp.float32)
        right = right.astype(jnp.float32)
    return jnp.matmul(
        left,
        right,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


def _matmul(left: jax.Array, right: jax.Array) -> jax.Array:
    product = _matrix_product(left, right)
    return jax.lax.optimization_barrier(product.astype(left.dtype))


def _addmm(
    base: jax.Array,
    left: jax.Array,
    right: jax.Array,
    beta: float,
    alpha: float = 1.0,
) -> jax.Array:
    """Accumulate matrix products and scalar coefficients before rounding."""
    product = _matrix_product(left, right)
    result = (beta * base.astype(jnp.float32) + alpha * product).astype(base.dtype)
    # Each recurrence consumes the rounded low-precision matrix. XLA may
    # otherwise retain excess precision across fused GEMM epilogues, changing
    # the Gram recurrence after its cancellation-sensitive polynomial step.
    return jax.lax.optimization_barrier(result)


def _scale(value: jax.Array, scalar: float | jax.Array) -> jax.Array:
    """Round each low-precision update stage before the next scalar operation."""
    low_precision = value.dtype in (jnp.bfloat16, jnp.float16)
    dtype = jnp.float32 if low_precision else value.dtype
    scaled = (value.astype(dtype) * jnp.asarray(scalar, dtype=dtype)).astype(
        value.dtype
    )
    return jax.lax.optimization_barrier(scaled) if low_precision else scaled


def _lerp(start: jax.Array, end: jax.Array, weight: float) -> jax.Array:
    """Interpolate in the accumulator dtype, retaining low-precision momentum."""
    dtype = start.dtype
    if dtype in (jnp.bfloat16, jnp.float16):
        start = start.astype(jnp.float32)
        end = end.astype(jnp.float32)
    value = (
        start + weight * (end - start)
        if weight < 0.5
        else end + (1.0 - weight) * (start - end)
    )
    return value.astype(dtype)


def _newton_schulz_orth(value: jax.Array) -> jax.Array:
    """Apply eight fast and two polish iterations to a batch of matrices."""
    value = jax.lax.optimization_barrier(value.astype(jnp.bfloat16))
    transpose = value.shape[-2] > value.shape[-1]
    if transpose:
        value = jnp.swapaxes(value, -2, -1)
    norm = jnp.linalg.norm(value.astype(jnp.float32), axis=(-2, -1), keepdims=True)
    norm = jax.lax.optimization_barrier(
        jnp.maximum(norm.astype(value.dtype), jnp.asarray(NS_EPS, value.dtype))
    )
    value = jax.lax.optimization_barrier(
        (value.astype(jnp.float32) / norm.astype(jnp.float32)).astype(value.dtype)
    )
    # The loop carry is a bfloat16 matrix at every iteration boundary. An
    # unrolled graph permits sharding-dependent fusion across iterations,
    # which changes the recurrence for nearly rank-deficient gradients.
    for steps, (a, b, c) in (
        (NS_STEPS_FAST, NS_COEFF_FAST),
        (NS_STEPS_POLISH, NS_COEFF_POLISH),
    ):

        def iteration(
            index: int,
            current: jax.Array,
            coefficients: tuple[float, float, float] = (a, b, c),
        ) -> jax.Array:
            del index
            coeff_a, coeff_b, coeff_c = coefficients
            gram = _matmul(current, jnp.swapaxes(current, -2, -1))
            polynomial = _addmm(gram, gram, gram, coeff_b, coeff_c)
            return _addmm(current, polynomial, current, coeff_a)

        value = jax.lax.fori_loop(0, steps, iteration, value)
    return jnp.swapaxes(value, -2, -1) if transpose else value


def _gram_newton_schulz_orth(value: jax.Array) -> jax.Array:
    """Apply the five-step Polar Express Gram recurrence, restarting at step 2."""
    dtype = value.dtype
    value = value.astype(jnp.float32)
    transpose = value.shape[-2] > value.shape[-1]
    if transpose:
        value = jnp.swapaxes(value, -2, -1)
    norm = jnp.linalg.norm(value, axis=(-2, -1), keepdims=True)
    value = jax.lax.optimization_barrier((value / (norm + NS_EPS)).astype(jnp.float16))
    gram = _matmul(value, jnp.swapaxes(value, -2, -1))
    identity = jnp.eye(value.shape[-2], dtype=value.dtype)
    transform = jnp.broadcast_to(identity, gram.shape)
    for index, (a, b, c) in enumerate(POLAR_EXPRESS_COEFFICIENTS):
        if index == 2:
            value = _matmul(transform, value)
            gram = _matmul(value, jnp.swapaxes(value, -2, -1))
        polynomial = _addmm(gram, gram, gram, b, c)
        if index in (0, 2):
            transform = polynomial + _scale(identity, a)
        else:
            transform = _addmm(transform, transform, polynomial, a)
        if index < len(POLAR_EXPRESS_COEFFICIENTS) - 1 and index != 1:
            gram_poly = _addmm(gram, gram, polynomial, a)
            gram = _addmm(gram_poly, polynomial, gram_poly, a)
    value = _matmul(transform, value)
    if transpose:
        value = jnp.swapaxes(value, -2, -1)
    return value.astype(dtype)


def _magma_score(
    gradient: jax.Array,
    momentum: jax.Array,
    score: jax.Array,
    matrix_shape: tuple[int, int, int],
) -> jax.Array:
    """Advance the per-slice EMA of range-stretched alignment scores."""
    gradient = gradient.reshape(matrix_shape).astype(jnp.float32)
    momentum = momentum.reshape(matrix_shape).astype(jnp.float32)
    dot = jnp.sum(momentum * gradient, axis=(-2, -1))
    norm = jnp.linalg.norm(momentum, axis=(-2, -1)) * jnp.linalg.norm(
        gradient, axis=(-2, -1)
    )
    cosine = jnp.clip(dot / jnp.maximum(norm, 1e-12), -1.0, 1.0)
    lower = 1.0 / (1.0 + math.exp(0.5))
    upper = 1.0 / (1.0 + math.exp(-0.5))
    raw = jnp.clip((jax.nn.sigmoid(cosine / 2.0) - lower) / (upper - lower), 0.0, 1.0)
    return 0.9 * score + 0.1 * raw


def _orthogonalize_bucket(
    entries: list[tuple[int, jax.Array]],
    use_gram: bool,
    matrix_sharding: NamedSharding | None = None,
) -> list[tuple[int, jax.Array]]:
    """Batch compatible matrices and restore their individual orientations.

    Gram buckets share the smaller matrix dimension. Zero-padding the columns
    of a wide matrix leaves both its Frobenius norm and ``X @ X.T`` unchanged,
    so rectangular matrices with different widths share one Gram recurrence.
    Standard buckets have identical matrix dimensions and require no padding.
    """
    matrices = [matrix for _, matrix in entries]
    if use_gram:
        width = max(max(matrix.shape[-2:]) for matrix in matrices)
        matrices = [
            jnp.swapaxes(matrix, -2, -1)
            if matrix.shape[-2] > matrix.shape[-1]
            else matrix
            for matrix in matrices
        ]
        matrices = [
            jnp.pad(matrix, ((0, 0), (0, 0), (0, width - matrix.shape[-1])))
            for matrix in matrices
        ]
    stacked = jnp.concatenate(matrices, axis=0)
    if matrix_sharding is not None:
        device_count = matrix_sharding.num_devices
        padding = (-stacked.shape[0]) % device_count
        stacked = jnp.pad(stacked, ((0, padding), (0, 0), (0, 0)))
        stacked = jax.lax.with_sharding_constraint(stacked, matrix_sharding)
    orthogonal = (
        _gram_newton_schulz_orth(stacked) if use_gram else _newton_schulz_orth(stacked)
    )
    if matrix_sharding is not None:
        orthogonal = jax.lax.with_sharding_constraint(orthogonal, matrix_sharding)
    result = []
    offset = 0
    for index, matrix in entries:
        batch, rows, cols = matrix.shape
        value = orthogonal[offset : offset + batch]
        offset += batch
        if use_gram:
            value = value[..., : max(rows, cols)]
            if rows > cols:
                value = jnp.swapaxes(value, -2, -1)
        result.append((index, value))
    return result


def build_hybrid_muon(
    *,
    momentum: float = 0.95,
    weight_decay: float = 0.001,
    adam_betas: tuple[float, float] = (0.9, 0.95),
    lr_adjust: float = 0.0,
    lr_adjust_coeff: float = 0.18,
    muon_mode: str = "slice",
    enable_gram: bool = True,
    magma_muon: bool = True,
    adam_patterns: Sequence[str] = (),
    matrix_sharding: NamedSharding | None = None,
) -> optax.GradientTransformationExtraArgs:
    """
    Construct the HybridMuon transformation for an NNX model or an array tree.

    Parameter names and matrix shapes are taken from the parameter tree.

    Parameters
    ----------
    momentum : float
        Muon EMA coefficient used in the Nesterov update.
    weight_decay : float
        Decoupled decay for Muon and AdamW routes.
    adam_betas : tuple[float, float]
        Adam first- and second-moment EMA coefficients.
    lr_adjust : float
        Non-positive values select match-RMS scaling; positive values select
        rectangular scaling and divide the Adam learning rate by this value.
    lr_adjust_coeff : float
        Multiplier of ``sqrt(max(rows, cols))`` in match-RMS mode.
    muon_mode : {"2d", "flat", "slice"}
        Matrix layout after removing singleton dimensions from parameters.
    enable_gram : bool
        Use the float16 Gram recurrence for rectangular matrices. Other matrices
        use the bfloat16 two-stage Newton-Schulz recurrence.
    magma_muon : bool
        Apply per-matrix momentum-gradient alignment damping on the Muon route.
    adam_patterns : Sequence[str]
        Additional case-insensitive substrings selecting AdamW for matrix weights.
    matrix_sharding : NamedSharding, optional
        Batch-axis placement for independent orthogonalization matrices. Matrix
        rows and columns remain local, preserving the low-precision recurrence
        when persistent momentum arrays are partitioned by ZeRO stage one.

    Returns
    -------
    optax.GradientTransformationExtraArgs
        Transformation accepting a scalar ``learning_rate`` extra argument on
        every update. The learning rate may be a JAX tracer. Optional ``active``
        supplies a boolean array tree with one scalar per parameter; inactive
        parameters retain both their values and their complete optimizer state.

    Notes
    -----
    Momentum follows parameter dtype; Adam moments, clocks, and Magma averages
    use float32. Matrix routes are batched by shape during tracing, so the
    compiled update contains no parameter-value-dependent Python control flow.
    """
    muon_mode = muon_mode.lower()
    if muon_mode not in ("2d", "flat", "slice"):
        raise ValueError(
            f"Invalid muon_mode {muon_mode!r}; expected '2d', 'flat', or 'slice'."
        )
    patterns = tuple(sorted({pattern.lower() for pattern in adam_patterns}))

    def routes_for(params: Any) -> tuple[list[jax.Array], Any, list[_Route]]:
        items, tree = jax.tree_util.tree_flatten_with_path(params)
        routes = [
            _route_parameter(_parameter_name(path), value.shape, muon_mode, patterns)
            for path, value in items
        ]
        return [value for _, value in items], tree, routes

    def init_fn(params: Any) -> HybridMuonState:
        values, _, routes = routes_for(params)
        states = []
        for value, route in zip(values, routes, strict=True):
            if route.kind == "muon":
                states.append(
                    MuonState(
                        jnp.zeros_like(value),
                        jnp.full((route.matrix_shape[0],), 0.5, dtype=jnp.float32),
                    )
                )
            else:
                states.append(
                    AdamState(
                        jnp.zeros_like(value, dtype=jnp.float32),
                        jnp.zeros_like(value, dtype=jnp.float32),
                        jnp.ones((), dtype=jnp.float32),
                        jnp.ones((), dtype=jnp.float32),
                    )
                )
        return HybridMuonState(tuple(states))

    def update_fn(
        updates: Any,
        state: HybridMuonState,
        params: Any = None,
        *,
        learning_rate: float | jax.Array,
        active: Any = None,
        **extra_args: Any,
    ) -> tuple[Any, HybridMuonState]:
        del extra_args
        if params is None:
            raise ValueError(
                "HybridMuon requires current parameters for decoupled weight decay."
            )
        values, tree, routes = routes_for(params)
        gradients = tree.flatten_up_to(updates)
        active_leaves = (
            [True] * len(values) if active is None else tree.flatten_up_to(active)
        )
        if any(getattr(enabled, "shape", ()) != () for enabled in active_leaves):
            raise ValueError(
                "HybridMuon active mask must contain one scalar per parameter."
            )
        lr = jnp.asarray(learning_rate, dtype=jnp.float32)
        deltas: list[jax.Array | None] = [None] * len(values)
        next_states = list(state.parameters)
        buckets: dict[
            tuple[tuple[int, ...], Any, bool], list[tuple[int, jax.Array]]
        ] = {}
        for index, (value, grad, previous, route) in enumerate(
            zip(values, gradients, state.parameters, routes, strict=True)
        ):
            if route.kind != "muon":
                beta1, beta2 = adam_betas
                grad = grad.astype(jnp.float32)
                first = previous.exp_avg + (1.0 - beta1) * (grad - previous.exp_avg)
                second = previous.exp_avg_sq + (1.0 - beta2) * (
                    grad * grad - previous.exp_avg_sq
                )
                power1 = previous.beta1_power * beta1
                power2 = previous.beta2_power * beta2
                rate = lr if lr_adjust <= 0 else lr / lr_adjust
                denominator = jnp.sqrt(second / (1.0 - power2)) + ADAM_EPS
                delta = (first / denominator) * (rate / (1.0 - power1))
                decayed = (
                    _scale(value, 1.0 - rate * weight_decay)
                    if route.kind == "adamw" and weight_decay > 0
                    else value
                )
                deltas[index] = decayed - delta.astype(value.dtype) - value
                next_states[index] = AdamState(first, second, power1, power2)
                continue
            grad = grad.astype(previous.momentum.dtype)
            buffer = _lerp(previous.momentum, grad, 1.0 - momentum)
            nesterov = _lerp(grad, buffer, momentum)
            score = previous.magma_score
            if magma_muon:
                score = _magma_score(grad, buffer, score, route.matrix_shape)
            next_states[index] = MuonState(buffer, score)
            batch, rows, cols = route.matrix_shape
            use_gram = enable_gram and rows != cols
            orthogonal_shape = (min(rows, cols),) if use_gram else (rows, cols)
            buckets.setdefault((orthogonal_shape, value.dtype, use_gram), []).append(
                (index, nesterov.reshape(batch, rows, cols))
            )

        for (_, _, use_gram), entries in buckets.items():
            for index, delta in _orthogonalize_bucket(
                entries, use_gram, matrix_sharding
            ):
                _, rows, cols = routes[index].matrix_shape
                scale = (
                    lr_adjust_coeff * math.sqrt(max(rows, cols))
                    if lr_adjust <= 0
                    else math.sqrt(max(1.0, rows / cols))
                )
                delta = _scale(delta, scale)
                if magma_muon:
                    # Damping is cast and multiplied in the orthogonalized
                    # matrix dtype before the learning-rate multiplication.
                    damping = jax.lax.optimization_barrier(
                        (0.1 + 0.9 * next_states[index].magma_score).astype(delta.dtype)
                    )
                    delta = jax.lax.optimization_barrier(delta * damping[:, None, None])
                value = values[index]
                delta = _scale(delta, lr).reshape(value.shape).astype(value.dtype)
                decayed = (
                    _scale(value, 1.0 - lr * weight_decay)
                    if weight_decay > 0
                    else value
                )
                deltas[index] = decayed - delta - value

        for index, enabled in enumerate(active_leaves):
            deltas[index] = jnp.where(
                enabled, deltas[index], jnp.zeros_like(values[index])
            )
            next_states[index] = jax.tree_util.tree_map(
                lambda new, old, mask=enabled: jnp.where(mask, new, old),
                next_states[index],
                state.parameters[index],
            )
        return tree.unflatten(deltas), HybridMuonState(tuple(next_states))

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
