# SPDX-License-Identifier: LGPL-3.0-or-later
"""PyTorch ``torch.compile`` support for the deepmd backend.

The deepmd PyTorch backend traces selected compute functions with ``make_fx``
and lowers them through Inductor while preserving the second-order autograd
graph that force training requires. This module gathers the helpers that
support that pipeline together with the PyTorch defect workarounds it needs, so
the model code stays free of compiler plumbing.

The contents fall into two kinds:

* helpers and workarounds common to the supported releases -- trace-shape and
  trace-input preparation, per-task buffer promotion, FX graph repair, the
  Inductor option lockdown, and the process-global configuration; and
* a workaround specific to PyTorch 2.12, which must not be applied on 2.11.

Only PyTorch 2.11.x and 2.12.x are permitted for compilation (see
:func:`check_compile_torch_version`).
"""

from __future__ import (
    annotations,
)

import os
from typing import (
    Any,
)

import torch
from packaging.version import (
    Version,
)

__all__ = [
    "AM_PREFIX",
    "FIT_PREFIX",
    "apply_global_compile_patches",
    "build_inductor_compile_options",
    "check_compile_torch_version",
    "get_task_buffer_names",
    "get_task_buffer_values",
    "is_prime",
    "next_safe_prime",
    "patch_inductor_force_int64_indexing",
    "patch_inductor_symbolic_divisibility",
    "rebuild_graph_module",
    "relax_views_to_reshapes",
    "strip_saved_tensor_detach",
    "trace_pad_dim",
]


# =============================================================================
# Common workarounds (PyTorch 2.11 and 2.12)
# =============================================================================
def apply_global_compile_patches() -> None:
    """Apply every process-global PyTorch adjustment the compile path needs.

    The adjustments are mutually independent and individually idempotent. The
    function is intended to run exactly once, when the model module is
    imported, so that the global state is established before the first
    compilation. The symbolic-divisibility repair is applied only on PyTorch
    2.12, where the regression exists.
    """
    # Silence Inductor / Triton autotune console dumps.  ``torch.compile``
    # reads these environment variables once, when its backend is first
    # initialised, so they must be set before the first compilation; setting
    # them afterwards has no effect in the current run.  ``setdefault``
    # preserves any explicit user-level override.
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_REPORT_CHOICES_STATS", "0")
    os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")

    # Disable DDPOptimizer graph splitting globally.  The inner
    # ``torch.compile`` calls sit *inside* a DDP-wrapped model; DDPOptimizer
    # assumes it sees the *whole* model and splits the FX graph at DDP bucket
    # boundaries.  For an inner submodule that heuristic produces subgraphs
    # whose outputs include symbolic integers, which then crash aot_autograd
    # with ``'int' object has no attribute 'meta'``.
    # See https://github.com/pytorch/pytorch/issues/134182.  Turning the
    # optimizer off globally is safe because the compile region always owns its
    # own boundary and the surrounding DDP wrapper operates on the full model
    # call.
    import torch._dynamo.config as dynamo_config

    dynamo_config.optimize_ddp = False

    # Force int64 tensor indexing in every compiled kernel.  Applies on all
    # supported PyTorch versions and is independent of runtime shapes.
    patch_inductor_force_int64_indexing()

    # The symbolic-divisibility regression exists only on PyTorch 2.12; the
    # 2.11 backend evaluates the same predicate correctly and must not be
    # patched.
    if Version(torch.__version__).release[:2] == (2, 12):
        patch_inductor_symbolic_divisibility()


def patch_inductor_force_int64_indexing() -> None:
    """Force Inductor to emit int64 tensor indexing in every compiled kernel.

    Inductor selects the index dtype from static size hints. The compiled
    ``core_compute`` graph is traced with the small placeholder shapes returned
    by :func:`next_safe_prime`, from which Inductor infers that the
    data-dependent edge and node axes fit in int32. At runtime those axes grow
    large enough that the flattened index of a tensor such as ``(E, D, D, C)``
    exceeds ``2**31`` and wraps to an out-of-range address, which surfaces
    asynchronously as a CUDA illegal memory access. Forcing int64 indexing
    removes this dependence on the trace-time size hints at the cost of a small
    amount of additional address arithmetic. The patch is idempotent and
    complements ``triton.max_tiles=1`` in :func:`build_inductor_compile_options`.
    """
    try:
        from torch._inductor.codegen.simd import (
            SIMDScheduling,
        )
    except Exception:
        return

    if getattr(SIMDScheduling, "_dp_force_int64_patched", False):
        return

    # ``can_use_32bit_indexing`` gates int32 selection; returning ``False``
    # forces int64 indexing in every generated kernel.
    SIMDScheduling.can_use_32bit_indexing = staticmethod(lambda numel, buffers: False)
    SIMDScheduling._dp_force_int64_patched = True


def check_compile_torch_version() -> None:
    """Fail fast when ``torch.compile`` is requested on an unsupported PyTorch."""
    version = Version(torch.__version__).release
    if len(version) < 2 or (version[:2] != (2, 11) and version[:2] != (2, 12)):
        raise RuntimeError(
            "deepmd `torch.compile` support requires PyTorch 2.11.x or 2.12.x; "
            f"found torch {torch.__version__}."
        )


def is_prime(n: int) -> bool:
    """Return True when ``n`` is a prime integer (``n >= 2``)."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    k = 3
    while k * k <= n:
        if n % k == 0:
            return False
        k += 2
    return True


def forbidden_dims_from_model(
    model: torch.nn.Module,
    task_buf_vals: tuple[torch.Tensor, ...] = (),
) -> set[int]:
    """Prime-collision set for trace-dim selection.

    Collects every ``> 1`` dim of the model's parameters/buffers (so
    :func:`next_safe_prime` never aliases an internal dim like ``g2_dim`` /
    ``axis_neuron`` / ``attn_head`` without a hardcoded list), plus
    ``dim_fparam``/``dim_aparam`` and the task-buffer dims.  Shared by the
    compiled-training traces (``_trace_and_compile`` /
    ``_trace_and_compile_graph``) and the graph ``.pt2`` export trace
    (``_trace_and_export``); each caller adds its path-specific dims
    (nall/nloc/nsel for dense, charge_spin for both) on top of this base set.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose parameter/buffer/conditioning dims to collect.
    task_buf_vals : tuple of torch.Tensor
        Per-task buffers promoted to FX placeholders (multi-task compiled
        training); their dims join the forbidden set.

    Returns
    -------
    set of int
        Every ``> 1`` dimension a trace-time size must not collide with.
    """
    forbidden: set[int] = {
        int(_d)
        for _src in (model.parameters(), model.buffers())
        for _p in _src
        for _d in _p.shape
        if _d > 1
    }
    for _getter_name in ("get_dim_fparam", "get_dim_aparam"):
        try:
            # resolve inside the try: a model without the accessor must fall
            # through the best-effort path, not raise during tuple building
            _dim = getattr(model, _getter_name)()
            if _dim > 1:
                forbidden.add(int(_dim))
        except Exception:
            pass  # best-effort: dim unavailable -> nothing to forbid
    for _tbv in task_buf_vals:
        for _d in _tbv.shape:
            if _d > 1:
                forbidden.add(int(_d))
    return forbidden


def next_safe_prime(start: int, forbidden: set[int]) -> int:
    """Return the smallest prime ``>= max(start, 5)`` not in ``forbidden``.

    Used by the ``make_fx`` symbolic-tracing path to choose collision-free
    trace-time sizes for ``nf``, ``nall`` and ``nloc``.  Primes ``>= 5``
    avoid every dim PyTorch specializes on (``1`` → broadcasting,
    ``2``/``3``/``9`` → Cartesian / virial / charge_spin literals baked
    into model code) and guarantee distinct values, which suppresses
    make_fx's duck-shape unification without needing the
    ``ShapeEnv(duck_shape=False)`` patch.
    """
    n = max(start, 5)
    while not is_prime(n) or n in forbidden:
        n += 1
    return n


def trace_pad_dim(t: torch.Tensor, dim: int, target: int) -> torch.Tensor:
    """Pad or trim ``t`` along ``dim`` so ``t.shape[dim] == target``.

    Padding duplicates the last slice along ``dim``; trimming drops
    trailing slices.  Used to coerce real-data trace inputs into the
    prime-numbered shapes chosen by :func:`next_safe_prime`.

    Duplicating the last slice preserves valid index values inside
    index-bearing tensors (``nlist`` neighbor indices, ``mapping``
    extended-to-local indices) because the duplicated row reuses the
    previously-valid row's values.  Trimming likewise never invalidates
    indices.

    The result is always contiguous, which matters as much as its shape.
    Trimming a non-leading dimension by slicing returns a view whose stride
    still encodes the *pre-trim* length; ``make_fx`` symbolic tracing records
    that stale stride as a free symbol, and duck-shaping then unifies it with
    any size symbol that happens to share the same trace-time value -- e.g. the
    trimmed ``atype`` stride (= the frame's ``nloc``) colliding with the edge
    count when both equal a ``next_safe_prime`` value. The compiled graph would
    then guard unrelated axes against one another and fail ``assert_size_stride``
    at runtime. Materializing a contiguous copy keeps the trace inputs' memory
    layout identical to the contiguous runtime inputs, so strides never carry a
    stale length into the symbol pool.
    """
    cur = int(t.shape[dim])
    if cur == target:
        return t
    if cur > target:
        sl: list[slice] = [slice(None)] * t.ndim
        sl[dim] = slice(None, target)
        return t[tuple(sl)].contiguous()
    sl = [slice(None)] * t.ndim
    sl[dim] = slice(-1, None)
    last = t[tuple(sl)]
    repeats = target - cur
    return torch.cat([t, *([last] * repeats)], dim=dim)


def strip_saved_tensor_detach(
    gm: torch.fx.GraphModule, *, remove_all: bool = False
) -> None:
    """Strip ``aten.detach`` nodes that ``make_fx`` inserts for saved tensors.

    When ``make_fx`` decomposes ``autograd.grad(..., create_graph=True)``,
    the autograd engine wraps every saved forward activation in a double-detach
    chain (e.g. ``tanh -> detach_A -> detach_B -> tanh_backward``).  These
    detach nodes block the second-order gradient path from the loss back to
    model parameters, causing incorrect parameter updates during force-loss
    training.

    With ``remove_all=False`` (default), user-explicit ``.detach()`` calls are
    preserved.  The make_fx-inserted and user-explicit detaches are
    distinguished by graph topology alone — no hard-coded op names — using
    three rules:

    * *Chain inner*: input is another detach node.
    * *Dead node*: no downstream users.
    * *Chain head*: *all* users are detach nodes.

    Any detach that does **not** match these rules is treated as user-explicit
    and left untouched.  This is the right behaviour for the SeZM model
    inference compile path, which contains legitimate user ``.detach()`` calls.

    With ``remove_all=True``, *every* detach node is removed unconditionally.
    The pt_expt training trace is invoked with already-detached, grad-enabled
    inputs and opens with ``coord.detach().requires_grad_(True)``; that
    boundary detach must also go or the force-loss gradient path is severed, so
    the training path passes ``remove_all=True``.
    """
    _DETACH = torch.ops.aten.detach.default

    def _is_detach(n: torch.fx.Node) -> bool:
        return n.op == "call_function" and n.target == _DETACH

    # Pass 1 classifies every detach against the original graph.  Erasing
    # nodes eagerly would let later classifications inspect a mutated
    # neighbourhood and misjudge the chain-interior / chain-head / dead
    # boundaries; the double-detach pattern in particular changes category
    # within a single erase.  Classifying first and mutating second keeps the
    # topology rules well defined.
    to_remove: list[torch.fx.Node] = []
    for node in gm.graph.nodes:
        if not _is_detach(node):
            continue
        if remove_all:
            to_remove.append(node)
            continue
        input_node = node.args[0]
        users = list(node.users.keys())
        is_chain_inner = _is_detach(input_node)
        is_dead = len(users) == 0
        is_chain_head = len(users) > 0 and all(_is_detach(u) for u in users)
        if is_chain_inner or is_dead or is_chain_head:
            to_remove.append(node)

    # Pass 2 rewires and erases after classification is complete.
    # ``replace_all_uses_with`` forwards every consumer to the detach's input
    # and ``erase_node`` removes the now-dead detach, so the graph never holds
    # a partially redirected state.
    for node in to_remove:
        node.replace_all_uses_with(node.args[0])
        gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()


def rebuild_graph_module(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Return a fresh ``GraphModule`` whose node linked-list is newly allocated.

    After ``strip_saved_tensor_detach`` erases nodes via
    ``Graph.erase_node()``, the internal doubly-linked list may retain
    stale pointers to erased nodes.  When ``torch.compile`` later
    triggers dynamo re-tracing and iterates ``graph.nodes`` to read
    ``nd.meta`` (``output_graph.py:_create_proxy``), accessing these
    stale entries causes a segfault.

    Copying every node into a brand-new ``Graph`` builds a clean linked
    list from scratch, side-stepping the corruption entirely.
    """
    old_graph = gm.graph
    new_graph = torch.fx.Graph()
    # node_copy needs a mapper from old nodes to their copies in new_graph.
    val_map: dict[torch.fx.Node, torch.fx.Node] = {}
    for node in old_graph.nodes:
        val_map[node] = new_graph.node_copy(node, lambda n: val_map[n])
    new_graph.lint()
    new_gm = torch.fx.GraphModule(gm, new_graph)
    return new_gm


def relax_views_to_reshapes(gm: torch.fx.GraphModule) -> None:
    """Rewrite every ``aten.view`` in a ``make_fx`` graph to ``aten.reshape``.

    ``make_fx`` lowers ``Tensor.reshape`` to ``aten.view`` whenever the traced
    ``FakeTensor`` is view-compatible. The lowering is unsound when the fake
    stride differs from the eager stride -- a permuted tensor that ``FakeTensor``
    keeps strided while eager materializes contiguous -- since the baked
    ``aten.view`` is accepted during tracing yet rejected at runtime for
    incompatible size and stride. ``aten.reshape`` coincides with ``aten.view``
    on view-compatible strides (and is elided by Inductor in that case) and
    copies only when a view is impossible; the rewrite is therefore
    semantics-preserving and free on the fast path.

    Parameters
    ----------
    gm : torch.fx.GraphModule
        The ``make_fx`` graph to rewrite in place.
    """
    view = torch.ops.aten.view.default
    reshape = torch.ops.aten.reshape.default
    relaxed = False
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is view:
            node.target = reshape
            relaxed = True
    if relaxed:
        gm.graph.lint()
        gm.recompile()


def build_inductor_compile_options(*, inference: bool = False) -> dict[str, Any]:
    """Return the conservative Inductor options used to lower the dynamic graph.

    The option set disables every Inductor and Triton feature that has
    misbehaved on the combination of data-dependent edge counts and a
    second-order autograd graph -- most visibly the oversized fused Triton
    reduction kernels that fail ``make_ttgir`` (``PassManager::run failed``) on
    some GPU/Triton combinations. Options absent from the running PyTorch's
    configuration registry are dropped so the returned dictionary stays valid
    across releases.

    Parameters
    ----------
    inference : bool
        Whether the options lower an inference graph (the ``make_fx`` +
        ``aot_module_simplified`` path and the AOTInductor freeze) rather
        than the ``torch.compile`` training graph.  Inference graphs enter
        Inductor with hint-less data-dependent symbols, which breaks the
        peak-memory reordering pass (see below); training graphs carry real
        size hints from the first traced call and benefit from the pass.

    Returns
    -------
    dict[str, Any]
        Keyword options accepted by ``torch.compile(options=...)`` and by
        ``torch._inductor.config.patch``.
    """
    compile_options: dict[str, Any] = {
        "max_autotune": False,
        "shape_padding": True,
        "epilogue_fusion": False,
        "triton.cudagraphs": False,
        "max_fusion_size": 8,
        "triton.persistent_reductions": False,
        # ``mix_order_reduction`` is defective under data-dependent symbolic
        # shapes on PyTorch 2.11 and earlier (pytorch/pytorch#174379, #178080,
        # #179494); the edge count is exactly that kind of shape.
        "triton.mix_order_reduction": False,
        # Constrain every generated kernel to a 1D launch grid. The default
        # 2D/3D tiling can place the data-dependent edge or node axis on the y
        # or z launch dimension, whose limit is 65535; a larger axis then
        # launches an out-of-range grid that surfaces as a CUDA illegal memory
        # access. A 1D grid keeps that axis on the x dimension (limit 2**31-1).
        # The option is shared by the training and evaluation graphs.
        "triton.max_tiles": 1,
    }
    if inference:
        # The peak-memory reordering pass sizes buffers through
        # ``sizevars.size_hint(numel, fallback=0)``.  The inference graph is
        # lowered from ``make_fx`` fake placeholders whose edge-count symbols
        # carry no hint, so every dynamically shaped buffer is costed as zero
        # bytes, the candidate orders become indistinguishable to the cost
        # model, and the pass rewrites the schedule into an order that hoists
        # the dynamic allocations to the head of the generated ``call()`` --
        # all forward/backward intermediates then coexist, more than doubling
        # peak memory on the SeZM inference graph.  Training compiles through
        # Dynamo with real hints from the first call and measurably benefits
        # from the pass, so it keeps the upstream default.
        compile_options["reorder_for_peak_memory"] = False
    try:
        from torch._inductor import config as inductor_config

        valid_options = inductor_config.get_config_copy()
        compile_options = {
            key: value
            for key, value in compile_options.items()
            if key.replace("-", "_") in valid_options
        }
    except Exception:
        # Older/future PyTorch builds may not expose the config registry.
        # In that case keep the curated option set and let torch.compile
        # surface any real backend error.
        pass
    return compile_options


# Prefix namespace for promoted per-task buffer names.
AM_PREFIX = "am/"
FIT_PREFIX = "fit/"


def get_task_buffer_names(model: Any) -> tuple[str, ...]:
    """Return the ordered names of per-task buffers to promote as FX placeholders.

    ``model`` is any deepmd model exposing ``atomic_model`` and
    ``atomic_model.fitting_net``.  Promoting these buffers as explicit graph
    inputs lets one compiled graph stay correct across tasks that differ only
    in their values.  Always promotes:

    * ``out_bias``, ``out_std`` on ``atomic_model`` -- may be replaced
      out-of-place by ``model_change_out_bias``, so the compiled graph must
      never bake them as constants.
    * ``bias_atom_e`` on the fitting net -- task-specific per-type bias that
      differs across tasks after ``share_params``.
    * ``case_embd`` on the fitting net -- task-identity vector used for
      multi-task case conditioning.
    """
    names: list[str] = []
    atomic_model = model.atomic_model
    fitting = atomic_model.fitting_net
    for bname in ("out_bias", "out_std"):
        if atomic_model._buffers.get(bname) is not None:
            names.append(AM_PREFIX + bname)
    for bname in ("bias_atom_e", "case_embd"):
        if fitting._buffers.get(bname) is not None:
            names.append(FIT_PREFIX + bname)
    return tuple(names)


def get_task_buffer_values(
    model: Any,
    names: tuple[str, ...],
) -> tuple[torch.Tensor, ...]:
    """Return the current tensor values for the given promoted-buffer names."""
    if not names:
        return ()
    atomic_model = model.atomic_model
    fitting = atomic_model.fitting_net
    vals: list[torch.Tensor] = []
    for name in names:
        if name.startswith(AM_PREFIX):
            vals.append(atomic_model._buffers[name[len(AM_PREFIX) :]])
        elif name.startswith(FIT_PREFIX):
            vals.append(fitting._buffers[name[len(FIT_PREFIX) :]])
        else:
            raise ValueError(f"Unknown task-buffer name: {name}")
    return tuple(vals)


# =============================================================================
# PyTorch 2.12-specific workarounds
# =============================================================================
def patch_inductor_symbolic_divisibility() -> None:
    """Repair the PyTorch 2.12 Inductor symbolic-divisibility regression.

    ``SizeVarAllocator.statically_known_multiple_of`` determines whether one
    symbolic size is an exact multiple of another. ``SIMDKernel`` consults it
    while splitting a fused iteration space into kernel groups and raises
    ``CantSplit`` whenever the test reports a non-multiple.

    PyTorch 2.11 evaluated the test with sympy's native modulo operator, which
    factors polynomials, so an expression such as ``(32*s + 64) % (s + 2)``
    reduces to ``0`` and the split proceeds. PyTorch 2.12 rewrote the helper
    and, for symbolic denominators, routes the test through Inductor's own
    ``Mod`` implementation, which does not factor. ``Mod(32*s + 64, s + 2)``
    therefore stays unevaluated, the test returns ``False``, and lowering
    aborts with::

        CantSplit: 32*s38 + 64 not divisible by s38 + 2

    Such ``c * (s + k)`` over ``(s + k)`` patterns arise whenever a padded axis
    is multiplied by a constant channel count, which is common in the compiled
    descriptor graph.

    The wrapper re-tests a symbolic denominator that the original rejects, this
    time with sympy's simplifying modulo. It reports a multiple only when sympy
    proves the remainder is identically zero, so it never asserts an unsound
    divisibility, and it leaves the 2.11 behaviour unchanged because the
    original test already succeeds there. A sentinel attribute on the class
    ensures the patch is installed at most once.
    """
    try:
        import sympy
        from torch._inductor.sizevars import (
            SizeVarAllocator,
        )
    except Exception:
        return

    if getattr(SizeVarAllocator, "_dp_divisibility_patched", False):
        return

    original_known_multiple_of = SizeVarAllocator.statically_known_multiple_of

    def statically_known_multiple_of(
        self: Any, numerator: Any, denominator: Any
    ) -> bool:
        if original_known_multiple_of(self, numerator, denominator):
            return True
        # Integer denominators use the structural divisibility path introduced
        # in 2.12, which is unaffected by the regression and needs no retry.
        if isinstance(denominator, (int, sympy.Integer)):
            return False
        try:
            num = sympy.sympify(numerator)
            den = sympy.sympify(denominator)
            # The bound mirrors Inductor's own guard against the cost of
            # symbolic evaluation on expressions with many free symbols.
            if len(num.free_symbols) > 20:
                return False
            # sympy's modulo factors the numerator, so (32*s + 64) % (s + 2)
            # reduces to 0 and the divisibility is proven.
            return bool(self.statically_known_true(sympy.Eq(num % den, 0)))
        except Exception:
            return False

    statically_known_multiple_of.__doc__ = original_known_multiple_of.__doc__
    SizeVarAllocator.statically_known_multiple_of = statically_known_multiple_of
    SizeVarAllocator._dp_divisibility_patched = True
