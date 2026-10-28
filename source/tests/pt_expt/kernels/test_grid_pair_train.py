# SPDX-License-Identifier: LGPL-3.0-or-later
"""Correctness of the fused coefficient-grid pair operator used in training.

The operator evaluates ``from_grid(to_grid(left) * to_grid(right))`` without
materializing the grid field, which is 39 times larger than its coefficient
operand at the production SO(3) shape. It is a fixed multilinear composition,
so the eager autograd of the same expression is exact and arbitrates the
operator's forward, backward and second order.

The comparison follows the conditioning argument of :mod:`.conditioning`: both
sides run in the working precision and are judged against the float64
evaluation of the reference, with the bound expressed as a multiple of the
eager reference's own distance from that truth.
"""

from __future__ import (
    annotations,
)

from contextlib import (
    nullcontext,
)
from types import (
    SimpleNamespace,
)
from typing import (
    TYPE_CHECKING,
)

import pytest
import torch
from torch._dynamo.testing import (
    CompileCounterWithBackend,
)
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.pt_expt.kernels.triton.sezm import grid_pair as grid_pair_module
from deepmd.pt_expt.kernels.triton.sezm.grid_pair import (
    GRID_PAIR_TRITON_AVAILABLE,
    _built_in_launch_config,
    grid_pair_train,
)

from .conditioning import (
    assert_conditioned,
    deviations,
    grad_chain,
    median_deviations,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterator,
    )

    from triton.compiler.compiler import (
        CompiledKernel,
    )

pytestmark = [
    pytest.mark.skipif(not GRID_PAIR_TRITON_AVAILABLE, reason="Triton is unavailable"),
]
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

# ``(lmax, n_frames, n_focus, channels, n_grid)`` spanning the deployed grid
# shapes. The slot count ``(lmax + 1)^2 * n_frames`` drives the operator's
# two-stage tiling of the contraction axis, so the set covers a power-of-two
# slot count, counts that force the split, and the single-frame degree-five
# layout whose four-slot remainder must retain Triton's tensor-core floor.
GRID_SHAPES = [
    (3, 3, 2, 32, 152),
    (5, 1, 2, 64, 344),
    (5, 3, 2, 64, 344),
    (5, 3, 1, 64, 344),
    (6, 3, 2, 96, 460),
]

# Independent operand draws the verdict is taken over; see
# :func:`.conditioning.median_deviations`.
DRAW_SEEDS = (11, 2027, 40529)


@pytest.fixture
def isolated_launch_selection(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep selector tests independent of CUDA hardware and prior cache entries."""
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "test device")
    grid_pair_module._select_launch_config.cache_clear()
    yield
    grid_pair_module._select_launch_config.cache_clear()


@pytest.mark.usefixtures("isolated_launch_selection")
@pytest.mark.parametrize("failure_stage", ["warmup", "handles"])
def test_launch_selection_retries_and_caches(
    failure_stage: str,
) -> None:
    """Resource rejection tries the next candidate and caches its admission."""
    attempts = []

    class Kernel:
        def warmup(self, *args: object, **kwargs: object) -> SimpleNamespace:
            attempts.append((kwargs["C_BLK"], kwargs["BLOCK_G"], kwargs["num_stages"]))

            def init_handles() -> None:
                if len(attempts) <= 2:
                    raise grid_pair_module.triton.runtime.errors.OutOfResources(
                        2, 1, "test"
                    )

            if failure_stage == "warmup":
                init_handles()
                return SimpleNamespace(_init_handles=lambda: None)
            return SimpleNamespace(_init_handles=init_handles)

    kernel = Kernel()
    shape = ("test_kernel", 147, 64, 2, 3, False, 587, torch.float32)
    select = grid_pair_module._select_launch_config
    result = select(kernel, torch.device("cuda:0"), shape, (torch.float32, 2), 1)
    assert result == (32, 32, 1)
    assert attempts == [(32, 64, 2), (32, 32, 2), (32, 32, 1)]
    assert (
        select(kernel, torch.device("cuda:0"), shape, (torch.float32, 2), 1) == result
    )
    assert len(attempts) == 3


@pytest.mark.usefixtures("isolated_launch_selection")
def test_admitted_launch_preserves_runtime_arguments_and_reuses_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An admitted tile executes with each input's actual layout and pair count."""
    probes = []
    loaded = []
    launches = []

    class Kernel:
        fn = SimpleNamespace(__name__="test_kernel")

        def warmup(self, *args: object, **kwargs: object) -> SimpleNamespace:
            probes.append((args, kwargs))
            return SimpleNamespace(_init_handles=lambda: loaded.append(True))

    class Launcher:
        def __getitem__(self, grid: tuple[int, int]) -> Callable[..., None]:
            def execute(*args: object, **kwargs: object) -> None:
                launches.append((grid, args, kwargs))

            return execute

    kernel = Kernel()
    monkeypatch.setattr(grid_pair_module, "wrap_triton", lambda candidate: Launcher())
    for n_node in (2, 3):
        value = torch.empty(49, n_node, 2, 192, device="cpu").permute(1, 0, 2, 3)
        strides = value.stride()
        grid_pair_module._launch(kernel, value, 587, 3, False, (value, *strides), 1)
        grid, args, kwargs = launches[-1]
        assert grid == (2 * n_node, 2)
        assert args[0] is value
        assert args[1:-1] == strides
        assert args[-1] == 2 * n_node
        assert kwargs == {
            key: val for key, val in probes[0][1].items() if key != "grid"
        }
    assert len(probes) == len(loaded) == 1
    assert len(launches) == 2
    assert grid_pair_module._select_launch_config.cache_info().hits == 1


@pytest.mark.usefixtures("isolated_launch_selection")
def test_launch_rejection_is_cached_and_reported() -> None:
    """An exhausted selector prevents execution and reuses its cached miss."""
    attempts = []

    class Kernel:
        fn = SimpleNamespace(__name__="test_kernel")

        def warmup(self, *args: object, **kwargs: object) -> None:
            attempts.append((kwargs["C_BLK"], kwargs["BLOCK_G"], kwargs["num_stages"]))
            raise grid_pair_module.triton.runtime.errors.OutOfResources(2, 1, "test")

    kernel = Kernel()
    select = grid_pair_module._select_launch_config
    value = torch.empty(2, 49, 2, 192, device="cpu")
    for _ in range(2):
        with pytest.raises(grid_pair_module._NoViableConfig, match="P=147, C=64"):
            grid_pair_module._launch(kernel, value, 587, 3, False, (value,), 1)
    assert len(attempts) == len(set(attempts)) == 8
    assert select.cache_info().hits == 1


@pytest.mark.usefixtures("isolated_launch_selection")
def test_launch_selection_preserves_compiler_errors() -> None:
    """A compiler defect must neither select a fallback nor enter the cache."""
    attempts = []

    class Kernel:
        def warmup(self, *args: object, **kwargs: object) -> None:
            attempts.append(kwargs)
            raise RuntimeError("compiler defect")

    kernel = Kernel()
    select = grid_pair_module._select_launch_config
    shape = ("test_kernel", 147, 64, 2, 3, False, 587, torch.float32)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="compiler defect"):
            select(kernel, torch.device("cuda:0"), shape, (torch.float32, 2), 1)
    assert len(attempts) == 2
    assert select.cache_info().currsize == 0


def test_blackwell_launch_table_uses_exact_grid_shape() -> None:
    """Keep production launch pins scoped to the swept device and grid."""
    shape_key = (
        "_grid_pair_bwd2_kernel",
        147,
        96,
        2,
        3,
        False,
        584,
        torch.bfloat16,
    )
    assert _built_in_launch_config(
        "NVIDIA RTX PRO 6000 Blackwell Server Edition", shape_key
    ) == (16, 32, 2)
    assert _built_in_launch_config("NVIDIA H20", shape_key) is None
    alternate_grid = (*shape_key[:-2], 460, shape_key[-1])
    assert (
        _built_in_launch_config(
            "NVIDIA RTX PRO 6000 Blackwell Server Edition", alternate_grid
        )
        is None
    )


def _eager_pair(
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
    n_frames: int,
) -> torch.Tensor:
    """Reference composition on the frame-packed ``(N, D, F, K * C)`` layout."""
    n_batch, coeff_dim, n_focus, packed = left.shape
    n_grid = to_grid.shape[0]
    to_slots = to_grid.reshape(n_grid, coeff_dim, n_frames)
    from_slots = from_grid.reshape(n_grid, coeff_dim, n_frames)
    left_view = left.reshape(n_batch, coeff_dim, n_focus, n_frames, -1)
    right_view = right.reshape(n_batch, coeff_dim, n_focus, n_frames, -1)
    left_grid = torch.einsum("gdk,ndfkc->ngfc", to_slots, left_view)
    right_grid = torch.einsum("gdk,ndfkc->ngfc", to_slots, right_view)
    out = torch.einsum("gdk,ngfc->ndfkc", from_slots, left_grid * right_grid)
    return out.reshape(n_batch, coeff_dim, n_focus, packed)


class _GridPairCase:
    """One grid shape with operands shared by every evaluation of it.

    ``seed`` selects the draw, so a comparison can be repeated over
    independent operands.
    """

    def __init__(
        self,
        lmax: int,
        n_frames: int,
        n_focus: int,
        channels: int,
        n_grid: int,
        *,
        seed: int,
        n_node: int = 300,
    ) -> None:
        device = torch.device("cuda")
        torch.manual_seed(seed)
        self.n_frames = n_frames
        coeff_dim = (lmax + 1) ** 2
        double = {"device": device, "dtype": torch.float64}

        self.left = torch.randn(
            n_node, coeff_dim, n_focus, n_frames * channels, **double
        )
        self.right = torch.randn_like(self.left)
        # The projectors are scaled by the grid count so the round trip keeps
        # the operands' magnitude and the comparison is not dominated by one
        # side's dynamic range.
        self.to_grid = torch.randn(n_grid, coeff_dim * n_frames, **double) / (
            n_grid**0.5
        )
        self.from_grid = torch.randn_like(self.to_grid) / n_grid**0.5
        self.cotangent = torch.randn_like(self.left)
        self.second_cotangents = (
            (0, torch.randn_like(self.left)),
            (1, torch.randn_like(self.right)),
        )

    @staticmethod
    def quantity_names() -> list[str]:
        """Labels of every quantity the evaluation reports."""
        return ["fwd", "d/d left", "d/d right", "d2/d left", "d2/d right"]

    def evaluate(
        self,
        *,
        fused: bool,
        dtype: torch.dtype,
        amp: bool,
        strided: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        Run one evaluation of the pair product and its differentiated forms.

        Parameters
        ----------
        fused : bool
            Whether to call the fused operator or the eager composition.
        dtype : torch.dtype
            Working precision of the leaves.
        amp : bool
            Whether to run inside bfloat16 autocast. Both sides lower to the
            same reduced-precision regime there, so the comparison stays
            inside one ambient mode.
        strided : bool, default=False
            Whether coefficient operands use a non-contiguous trailing stride,
            as the channel slices entering the production grid nets do.

        Returns
        -------
        tuple of torch.Tensor
            The output and its first and second order gradients.
        """

        def make_leaf(value: torch.Tensor) -> torch.Tensor:
            value = value.to(dtype)
            if not strided:
                return value.clone().requires_grad_(True)
            storage = torch.empty(
                (*value.shape[:-1], value.shape[-1] * 2),
                device=value.device,
                dtype=value.dtype,
            )
            view = storage[..., ::2]
            view.copy_(value)
            return view.requires_grad_(True)

        left = make_leaf(self.left)
        right = make_leaf(self.right)
        to_grid, from_grid = self.to_grid.to(dtype), self.from_grid.to(dtype)
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if amp
            else torch.autocast("cuda", enabled=False)
        )
        with context:
            evaluate = grid_pair_train if fused else _eager_pair
            out = evaluate(left, right, to_grid, from_grid, self.n_frames)
        return grad_chain(out, [left, right], self.cotangent, self.second_cotangents)


def _compare(
    shape: tuple[int, int, int, int, int],
    *,
    amp: bool,
    strided: bool = False,
) -> None:
    """Arbitrate the fused pair product against the eager composition."""
    working = torch.bfloat16 if amp else torch.float32
    runs = []
    for seed in DRAW_SEEDS:
        case = _GridPairCase(*shape, seed=seed)
        runs.append(
            deviations(
                case.quantity_names(),
                case.evaluate(
                    fused=False, dtype=torch.float64, amp=False, strided=strided
                ),
                case.evaluate(
                    fused=False, dtype=torch.float32, amp=amp, strided=strided
                ),
                case.evaluate(
                    fused=True, dtype=torch.float32, amp=amp, strided=strided
                ),
                # The operator walks the grid axis in its natural order while
                # the eager chain reduces through cuBLAS, so the two agree to
                # the conditioning of the same contraction. Under bfloat16 the
                # fused walk keeps float32 partials where the eager chain
                # rounds the grid field itself, so it normally sits closer to
                # the truth than the reference does.
                factor=5.0,
                working_dtype=working,
            )
        )
    assert_conditioned(median_deviations(runs))


@pytest.mark.parametrize(
    ("lmax", "n_frames", "n_focus", "channels", "n_grid"), GRID_SHAPES
)
@requires_cuda
def test_float32_matches_eager_conditioning(
    lmax: int, n_frames: int, n_focus: int, channels: int, n_grid: int
) -> None:
    """Hold the fused pair product to the eager composition's float32 error."""
    _compare((lmax, n_frames, n_focus, channels, n_grid), amp=False)


@pytest.mark.parametrize(
    ("lmax", "n_frames", "n_focus", "channels", "n_grid"), GRID_SHAPES
)
@requires_cuda
def test_autocast_bfloat16_matches_eager_conditioning(
    lmax: int, n_frames: int, n_focus: int, channels: int, n_grid: int
) -> None:
    """Hold the same bound under the bfloat16 autocast of production training."""
    _compare((lmax, n_frames, n_focus, channels, n_grid), amp=True)


@requires_cuda
def test_noncontiguous_operands_match_eager_conditioning() -> None:
    """Cover the channel-slice strides supplied by production grid nets."""
    _compare(GRID_SHAPES[1], amp=True, strided=True)


@pytest.mark.parametrize(
    ("n_focus", "degree_major"),
    [(1, False), (2, True)],
)
@requires_cuda
def test_symbolic_graph_reuses_layout_across_batch_shapes(
    n_focus: int, degree_major: bool
) -> None:
    """Keep batch-dependent sizes and strides symbolic through every order."""
    n_frames, coeff_dim, channels, n_grid = 3, 4, 16, 32
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(1729)
    projector_shape = (n_grid, coeff_dim * n_frames)
    to_grid = torch.randn(projector_shape, device=device, generator=generator)
    from_grid = torch.randn(projector_shape, device=device, generator=generator)

    def make_inputs(n_node: int) -> tuple[torch.Tensor, ...]:
        shape = (
            (coeff_dim, n_node, n_focus, n_frames * channels)
            if degree_major
            else (n_node, coeff_dim, n_focus, n_frames * channels)
        )

        def leaf() -> torch.Tensor:
            return torch.randn(
                shape, device=device, generator=generator, requires_grad=True
            )

        return leaf(), leaf(), to_grid, from_grid, leaf(), leaf(), leaf()

    def logical_layout(value: torch.Tensor) -> torch.Tensor:
        if degree_major:
            return value.permute(1, 0, 2, 3)
        return value

    def evaluate(
        left: torch.Tensor,
        right: torch.Tensor,
        to_grid: torch.Tensor,
        from_grid: torch.Tensor,
        cotangent: torch.Tensor,
        h_left: torch.Tensor,
        h_right: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        left = logical_layout(left)
        right = logical_layout(right)
        cotangent = logical_layout(cotangent)
        h_left = logical_layout(h_left)
        h_right = logical_layout(h_right)
        out = grid_pair_train(left, right, to_grid, from_grid, n_frames)
        grad_left, grad_right = torch.autograd.grad(
            (out * cotangent).sum(), (left, right), create_graph=True
        )
        grad_cotangent, grad2_left, grad2_right = torch.autograd.grad(
            (grad_left * h_left).sum() + (grad_right * h_right).sum(),
            (cotangent, left, right),
        )
        return (
            out,
            grad_left,
            grad_right,
            grad_cotangent,
            grad2_left,
            grad2_right,
        )

    traced_inputs = make_inputs(7)
    graph = make_fx(evaluate, tracing_mode="symbolic")(*traced_inputs)
    compile_counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(
        graph, backend=compile_counter, dynamic=True, fullgraph=True
    )
    with torch.no_grad():
        compiled(*traced_inputs)
        runtime_inputs = make_inputs(11)
        actual = compiled(*runtime_inputs)
    expected = evaluate(*runtime_inputs)

    assert compile_counter.frame_count == 1
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want)


@requires_cuda
def test_aotautograd_reuses_multifocus_stride_across_batch_shapes() -> None:
    """Keep a producer's batch-dependent coefficient stride out of guards."""
    n_frames, coeff_dim, n_focus, channels, n_grid = 3, 4, 2, 16, 32
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(40529)
    projector_shape = (n_grid, coeff_dim * n_frames)
    to_grid = torch.randn(projector_shape, device=device, generator=generator)
    from_grid = torch.randn(projector_shape, device=device, generator=generator)

    def evaluate(
        left_storage: torch.Tensor, right_storage: torch.Tensor
    ) -> torch.Tensor:
        left = left_storage.permute(1, 0, 2, 3)
        right = right_storage.permute(1, 0, 2, 3)
        return grid_pair_train(left, right, to_grid, from_grid, n_frames).square().sum()

    compile_counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(
        evaluate, backend=compile_counter, dynamic=True, fullgraph=True
    )
    for n_node in (7, 11, 13):
        shape = (coeff_dim, n_node, n_focus, n_frames * channels)
        left = torch.randn(
            shape, device=device, generator=generator, requires_grad=True
        )
        right = torch.randn(
            shape, device=device, generator=generator, requires_grad=True
        )
        expected_loss = evaluate(left, right)
        expected_grad = torch.autograd.grad(expected_loss, (left, right))
        actual_loss = compiled(left, right)
        actual_grad = torch.autograd.grad(actual_loss, (left, right))
        torch.testing.assert_close(actual_loss, expected_loss)
        for got, want in zip(actual_grad, expected_grad, strict=True):
            torch.testing.assert_close(got, want)

    assert compile_counter.frame_count == 1


@pytest.mark.parametrize(
    ("n_focus", "channels", "amp", "force_reference"),
    [
        (2, 96, False, False),
        (1, 256, False, False),
        (2, 96, True, False),
        (1, 256, True, False),
        (2, 16, False, True),
    ],
    ids=[
        "multifocus-fp32",
        "singlefocus-fp32",
        "multifocus-amp",
        "singlefocus-amp",
        "reference",
    ],
)
@requires_cuda
def test_cold_compiled_force_backward(
    n_focus: int,
    channels: int,
    amp: bool,
    force_reference: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve device resources before tracing a production grid's second order.

    The first execution is compiled, so an eager pass cannot populate the
    launch cache first. The two production layouts cover multi-focus messages
    and the single-focus feed-forward grid. A device with no viable fused
    configuration must retain the reference composition through compilation.
    """
    # Inductor requires native BF16 support, not autocast's software emulation.
    if amp and not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("Compiled BF16 requires native device support")
    torch._dynamo.reset()
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    fallback_ops = set()
    probed_ops = set()
    admitted_ops = set()
    probe_inputs = {}
    reference = grid_pair_module._eager_packed

    def record_fallback(op: str, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        fallback_ops.add(op)
        return reference(op, *args)

    monkeypatch.setattr(grid_pair_module, "_eager_packed", record_fallback)
    for op, kernel in (
        ("fwd", grid_pair_module._grid_pair_fwd_kernel),
        ("bwd", grid_pair_module._grid_pair_bwd_kernel),
        ("bwd2", grid_pair_module._grid_pair_bwd2_kernel),
    ):

        def validate_candidate(
            *args: object,
            _op: str = op,
            _warmup: Callable[..., CompiledKernel] = kernel.warmup,
            **kwargs: object,
        ) -> CompiledKernel:
            probed_ops.add(_op)
            probe_inputs[_op] = (_warmup, args, kwargs)
            if force_reference:
                raise grid_pair_module.triton.runtime.errors.OutOfResources(
                    2, 1, "test"
                )
            candidate = _warmup(*args, **kwargs)
            candidate._init_handles()
            admitted_ops.add(_op)
            return candidate

        monkeypatch.setattr(kernel, "warmup", validate_candidate)

    # Degree six reproduces the P=147 resource pressure. The distinct grid
    # count prevents eager production-shape tests from warming this launch key.
    n_frames, coeff_dim, n_grid = 3, 49, 587
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(40529)
    projectors = tuple(
        torch.randn(
            n_grid,
            coeff_dim * n_frames,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        / n_grid**0.5
        for _ in range(2)
    )

    def make_inputs(n_node: int) -> tuple[torch.Tensor, ...]:
        # Degree-major storage gives the coefficient axis a batch-dependent
        # stride; the logical entry remains (N, D, F, K * C).
        shape = (coeff_dim, n_node, n_focus, n_frames * channels)
        return tuple(
            torch.randn(
                shape,
                device=device,
                dtype=torch.float32,
                generator=generator,
                requires_grad=True,
            )
            for _ in range(2)
        )

    def force_forward(
        left: torch.Tensor,
        right: torch.Tensor,
        to_grid: torch.Tensor,
        from_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = grid_pair_train(
                left.permute(1, 0, 2, 3),
                right.permute(1, 0, 2, 3),
                to_grid,
                from_grid,
                n_frames,
            )
        gradients = torch.autograd.grad(out.sum(), (left, right), create_graph=True)
        return out, *gradients

    def force_loss_gradient(
        outputs: tuple[torch.Tensor, ...],
        inputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        loss = sum(value.float().square().mean() for value in outputs)
        return torch.autograd.grad(loss, inputs)

    traced_inputs = make_inputs(7)
    graph = make_fx(force_forward, tracing_mode="symbolic")(*traced_inputs, *projectors)
    counter = CompileCounterWithBackend("inductor")
    with torch._inductor.config.patch(fx_graph_cache=False):
        compiled = torch.compile(graph, backend=counter, dynamic=True, fullgraph=True)
        for n_node in (7, 11):
            inputs = traced_inputs if n_node == 7 else make_inputs(n_node)
            actual = compiled(*inputs, *projectors)
            actual_gradients = force_loss_gradient(actual, inputs)
            # Observe compiled selection before eager evaluation can add events.
            # A cached/unconditional rejection without validation cannot pass.
            assert probed_ops == {"fwd", "bwd", "bwd2"}
            assert fallback_ops == probed_ops - admitted_ops
            if force_reference:
                assert not admitted_ops
            else:
                for op in probed_ops - admitted_ops:
                    warmup, args, kwargs = probe_inputs[op]
                    # Probe only after the cold compiled execution. This uses
                    # the original kernel, independently of the selector's
                    # admission decision, without warming the traced path.
                    try:
                        minimum = warmup(
                            *args,
                            **(
                                kwargs
                                | {
                                    "C_BLK": 16,
                                    "BLOCK_G": 16,
                                    "num_stages": 1,
                                    "num_warps": 4,
                                }
                            ),
                        )
                        minimum._init_handles()
                    except grid_pair_module.triton.runtime.errors.OutOfResources:
                        continue
                    pytest.fail(
                        f"{op} rejected every tile although its minimum tile loads"
                    )
            expected = force_forward(*inputs, *projectors)
            expected_gradients = force_loss_gradient(expected, inputs)
            for got, want in zip(
                (*actual, *actual_gradients),
                (*expected, *expected_gradients),
                strict=True,
            ):
                torch.testing.assert_close(got, want)
    assert counter.frame_count == 1
