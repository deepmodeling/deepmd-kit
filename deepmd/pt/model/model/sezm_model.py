# SPDX-License-Identifier: LGPL-3.0-or-later
"""SeZM: Smooth equivariant Zone-bridging Model.

This module hosts the ``make_fx`` + Inductor pipeline that runs the SeZM
energy (``ener``) path on the GPU.  To the authors' knowledge this is the
first public implementation of a compiled, dynamically shaped
machine-learning potential whose *second-order* derivatives -- required by
force-loss training -- travel end-to-end through Inductor without any
eager fallback.

After ``make_fx`` captures the graph the two modes diverge at the backend.
**Training** lowers it with ``torch.compile``: the Dynamo frontend builds
the optimizer's second backward through the already-materialised first
derivative.  **Inference** lowers the same graph with
``aot_module_simplified`` -- AOTAutograd's forward-only path -- which skips
the Dynamo frontend entirely.  Dynamo's shape-guard production aborts on
the forward-only graph (``sources must not be empty for symbol s...``),
and its activation handling costs ~3x the peak memory; the AOTAutograd
inference path avoids both.  The ``dens`` path below uses a plain
``torch.compile`` wrapper and is not covered by the rest of this docstring.

Why force-loss training is hard to compile
==========================================

An ML potential models atomic energy ``E(x, theta)`` from coordinates
``x`` and parameters ``theta``.  Force-loss training minimizes

::

    L = alpha * ||E_pred - E_label||^2 + beta * ||f_pred - f_label||^2

with ``f_pred = -dE/dx``.  The parameter update needs ``dL/dtheta``,
which contains ``d(f_pred)/dtheta = -d^2 E / (dx dtheta)`` -- a full
second-order derivative of the network with respect to one input and
one parameter axis.

The standard ``torch.compile`` stack (AOT Autograd) captures forward and
first backward; it does *not* natively handle an
``autograd.grad(..., create_graph=True)`` call nested *inside* the
compiled region.  So we compose two lower-level tools:

1. ``make_fx`` traces the compute function *after* the inner
   ``autograd.grad`` has been materialised, producing an FX graph whose
   forward already contains the first-derivative graph as ordinary ops.
2. ``torch.compile(..., dynamic=True)`` lowers that traced FX graph to
   Inductor.  Because the graph no longer hides an autograd call,
   Inductor's normal backward pipeline can differentiate the whole
   thing a second time for the optimizer step.

Everything else in this file exists to make that composition correct
under dynamic shapes, FSDP/DDP, and the list of PyTorch bugs that
surface along the way.  Every non-obvious choice is pinned to a source
comment tagged ``NOTE:``; the numbered catalogue at the bottom of this
docstring explains each tag in depth.

The PyTorch ``torch.compile`` workarounds themselves -- the version
guard, the process-global patches, the post-``make_fx`` FX graph
repair, the trace-shape selection, and the Inductor option lockdown --
live in :mod:`deepmd.pt.utils.compile_compat` so they can be reused
independently of this model.

Pipeline for one training batch
===============================

::

    forward(coord, atype, ...)
        |-- input dtype cast
        |-- neighbor list built in the extended region
        '-- forward_common -- ener branch
              |-- extended_coord.detach().requires_grad_(True)       (NOTE 9)
              |-- should_use_compile()?  yes ->
              |     |-- trace_and_compile() on cache miss
              |     |     |-- make_fx(compute_fn,
              |     |     |           tracing_mode="symbolic",
              |     |     |           _allow_non_fake_inputs=True,
              |     |     |           decomposition_table=<silu>)    (NOTE 0)
              |     |     |      * trace inputs use safe prime dims   (NOTE 1)
              |     |     |      * silu_backward is decomposed        (NOTE 2)
              |     |     |      * traced graph already contains the
              |     |     |        first autograd.grad over coords
              |     |     |-- strip_saved_tensor_detach (train only) (NOTE 3)
              |     |     |-- rebuild_graph_module      (train only) (NOTE 4)
              |     |     |-- train: torch.compile(backend="inductor",
              |     |     |              dynamic=True, options=<locked>)  (NOTE 6)
              |     |     '-- eval:  aot_module_simplified (forward-only) (NOTE 13)
              |     |           stored in compiled_core_compute_cache[key]    (NOTE 8)
              |     '-- compiled_core_compute_cache[key](...)  under no_grad in eval
              '-- communicate_extended_output

Subsequent batches look up the cached callable at the same
``(training, do_atomic_virial, has_coord_corr)`` slot of
``compiled_core_compute_cache``.  Each slot is retained independently, so
train <-> eval toggles around every ``disp_freq`` / full-validation checkpoint
reuse the other slot's compile product instead of evicting it (NOTE 7).

Body of the traced compute
==========================

``compute_fn`` (defined inside ``trace_and_compile``) wraps
``core_compute`` so that make_fx sees a pure tensor-in / tensor-out
function:

* ``core_compute`` rebuilds a compact, GPU-friendly edge list from the
  padded DeePMD neighbor list (``build_edge_list_from_nlist``), with
  two masked dummy edges appended so the edge tensor has a non-singular
  symbolic lower bound (NOTE 10).  Edge vectors come from
  ``index_select`` on the extended
  coordinate tensor, which keeps the gradient path back to coordinates
  explicit and safe under symbolic shapes (NOTE 11).
* The SeZM descriptor consumes the edge list and produces per-atom
  features.
* The fitting network predicts per-atom energy; ``apply_out_stat`` adds
  the per-type statistics and the atom mask zeroes out padding atoms.
* ``fit_output_to_model_output(..., create_graph=self.training)`` calls
  ``autograd.grad`` internally to compute ``force = -dE/dx``.
  ``create_graph`` is the single toggle that activates the
  second-derivative branch for training and omits it at inference
  (NOTE 12).

Because ``make_fx`` traces *after* that inner ``autograd.grad`` has
executed, the resulting FX graph encodes both the forward and the first
derivative as ordinary ops.  Any further ``.backward()`` on the compiled
output therefore just walks an FX-level backward that Inductor is
perfectly capable of lowering.

The ``NOTE:`` catalogue
=======================

NOTE 0 -- ``make_fx(tracing_mode="symbolic", _allow_non_fake_inputs=True)``
--------------------------------------------------------------------------

``tracing_mode="symbolic"`` tells the proxy tensor that shapes are
sympy-backed symbols; it is what makes ``dynamic=True`` compile work
later.  ``_allow_non_fake_inputs=True`` lets us feed *real* tensors
(not FakeTensors) to the trace.  We need real data because the edge
compactor contains data-dependent operations (``torch.nonzero``,
``index_select``) that cannot be executed on FakeTensors; the shapes
become symbolic immediately after the first op, so only the control
flow is decided by concrete values.

NOTE 1 -- Trace inputs with safe prime dimensions
-------------------------------------------------

``make_fx(tracing_mode="symbolic")`` replaces tensor shapes with sympy
symbols at trace time.  If two input dimensions happen to share the same
concrete value, PyTorch may assign them one duck-shaped symbol and bake a
false equality such as ``nloc == nall`` into the graph.  Dimensions that
match internal literals are also fragile: ``1`` triggers 0/1
specialization, while ``2`` / ``3`` / ``9`` commonly appear as
charge-spin, Cartesian-coordinate, and virial widths.

Before tracing, SeZM pads or trims real batch tensors so ``nf``, ``nloc``
and ``nall`` become pairwise-distinct primes >= 5 that do not collide
with fixed model dimensions.  ``nlist`` and ``mapping`` are clamped after
the shape coercion so their index values stay valid for the trace-only
sample.

NOTE 2 -- Decomposing ``silu_backward``
---------------------------------------

PyTorch ships forward and first-order backward for SiLU but *no*
symbolic higher-order derivative.  make_fx therefore emits
``aten.silu_backward.default`` opaquely inside the first-derivative
graph.  When Inductor later has to differentiate that op again for the
optimizer step, it refuses because silu_backward is not differentiable
in its registered form.  We pass an explicit decomposition
``silu_backward -> sigmoid + pointwise mul`` to ``make_fx``; every
pointwise piece then has a well-defined higher derivative of its own.

NOTE 3 -- Stripping autograd-inserted detach chains
---------------------------------------------------

When ``autograd.grad(create_graph=True)`` runs under make_fx, the
autograd engine wraps every saved forward activation in a double-detach
chain, e.g.::

    tanh  ->  detach_A  ->  detach_B  ->  tanh_backward

In eager autograd those detaches are informational -- they mark saved
tensors as belonging to a different graph.  After tracing, however,
they become ordinary ops inside the FX graph and sever the gradient
path from the force loss back to ``theta``; training then silently
produces zero parameter updates for the second-derivative term.

``strip_saved_tensor_detach`` removes them by pure graph topology --
no op-name matching -- so that user-explicit ``.detach()`` calls
(e.g. cached SO2 weights, activation lookup matrices) survive:

* *Chain inner*: input is another detach.
* *Dead node*:   no downstream users.
* *Chain head*:  every user is a detach.

Any detach that matches none of the three is treated as user intent and
is kept verbatim.  Stripping is guarded by ``self.training`` because
eval mode does not set ``create_graph=True``; the chain is never
inserted and removing it would be incorrect.

NOTE 4 -- Rebuilding the FX graph from scratch
----------------------------------------------

``Graph.erase_node`` inside ``strip_saved_tensor_detach`` unlinks nodes
from the doubly linked list that represents the graph.  On several
PyTorch builds (observed on 2.11+cu130) it leaves the C-level
``prev/next`` pointers of *neighbouring* Node objects stale.  Dynamo,
when it later re-traces the ``GraphModule`` and walks ``graph.nodes``
inside ``output_graph.py:_create_proxy`` to read ``nd.meta``,
dereferences one of those stale pointers and segfaults.

``rebuild_graph_module`` does a single ``node_copy`` pass into a
freshly allocated ``torch.fx.Graph``.  The result is an equivalent graph
whose linked list contains no erased entries, so dynamo can iterate it
safely.  We always rebuild -- including in eval -- because a fresh
graph is cheap while a segfault is fatal.

NOTE 5 -- Disabling ``DDPOptimizer``
------------------------------------

``torch._dynamo.config.optimize_ddp = False`` is set unconditionally at
import time.  DDPOptimizer is designed to split a DDP-wrapped model's
graph at bucket boundaries so that gradients can overlap with
all-reduce.  But here the compile region is *inside* the DDP-wrapped
model -- it wraps only ``core_compute``.  DDPOptimizer assumes it owns
the whole model, splits our inner graph at its internal bucket
heuristic, and the split produces subgraphs whose outputs include
symbolic integers.  AOT Autograd then crashes with
``'int' object has no attribute 'meta'`` (pytorch/pytorch#134182).
Disabling the optimizer globally is safe because SeZM always owns its
own compile boundary and the surrounding DDP wrapper operates on the
full model call.

NOTE 6 -- Inductor / Triton option lockdown
-------------------------------------------

One Inductor option set governs both backends: ``torch.compile`` takes it
as ``options=`` for training, and the eval path (NOTE 13) applies it via
``torch._inductor.config.patch`` around ``compile_fx_inner``, adding
``triton.max_tiles=1``.  The options are:

* ``max_autotune=False``
      Autotune regresses on dynamic shapes because each recompile rolls
      the search; deterministic kernels compiled once are consistently
      faster on our edge-level reductions.
* ``shape_padding=True``
      Pads tensors to SIMD-friendly sizes when symbolic shapes
      fluctuate batch-to-batch, eliminating tail-kernel generation cost.
* ``epilogue_fusion=False``
      Two independent reasons to keep it off.  (a) Inductor only
      enables epilogue fusion when ``max_autotune`` is on, and we
      deliberately disable autotune above; leaving the flag on would
      pay the scheduling cost without ever activating the fusion.
      (b) Fused epilogues occasionally reorder saved tensors in ways
      the second backward cannot recover; disabling the fusion keeps
      the backward graph shape-stable under make_fx.
* ``triton.cudagraphs=False``
      cudagraphs capture autograd metadata only once.  Higher-order
      gradients need fresh metadata per call, so cudagraphs would feed
      stale autograd state into the second backward.
* ``max_fusion_size=8``
      Caps kernel fusion complexity so Inductor's scheduler does not
      time out on the large edge-level reductions inside the
      descriptor when nsel is big.  The tighter value keeps both
      training and inference fusions small enough for Triton IR
      generation on GPU backends that are sensitive to large dynamic
      edge graphs.
* ``triton.persistent_reductions=False``
      Inductor's persistent-reduction scheduler fuses a ``sum`` with
      *all* neighbouring pointwise ops (``tanh_backward``, ``pow``,
      ``exp``, ``mul``, ``select``, ``slice``, ``view`` ...) into one
      ``triton_per_fused_...`` kernel.  On SeZM's dynamic edge graph
      this can hit Triton bug ``PassManager::run failed`` inside
      ``make_ttgir``.  Disabling it forces the reduction into its own
      kernel before either training or inference can form the
      pathological fused IR.
* ``triton.mix_order_reduction=False``
      Workaround for PyTorch <=2.11 bugs pytorch/pytorch#174379,
      #178080, #179494.  All three manifest only under data-dependent
      symbolic shapes -- exactly our edge count.

NOTE 7 -- Multi-slot compile cache key
--------------------------------------

The key is ``(training, do_atomic_virial, has_coord_corr)`` because all three
fields alter the traced graph topology:

* ``self.training`` switches ``create_graph`` in
  ``fit_output_to_model_output`` -- it toggles the entire
  second-derivative branch on or off.
* ``do_atomic_virial`` adds or removes an extra per-atom virial tensor
  in the compute output.
* ``has_coord_corr`` selects the spin-virial correction branch, changing the
  compiled callable arity from six tensor inputs to seven.

No single compiled graph can serve both variants, so the cache is a
``dict[tuple[bool, bool, bool], Callable]`` named
``compiled_core_compute_cache``.  A single-slot
cache would have to evict on every flip, which turns the normal
training-loop pattern -- ``train -> eval at every disp_freq -> train``
and an occasional full validation on top of that -- into
two-recompile-per-disp_freq thrashing (each recompile costs tens of
seconds to minutes on SeZM).  With multi-slot caching the first
encounter of each mode pays the compile cost once, and every later
toggle is a dict lookup.

Multi-task runs add one module-level sharing layer on top of this
per-instance cache.  Tasks whose descriptor and fitting parameters are
the same Python objects after ``share_params(level=0)`` reuse a single
compiled callable.  Per-task tensors that must remain distinct
(``out_bias``, ``out_std``, ``bias_atom_e`` and ``case_embd``) are
promoted to explicit FX placeholders, so the shared graph reads their
current values at each call instead of baking the first task's tensors as
constants.

Enabling compile for eval is an opt-in via ``DP_COMPILE_INFER=1``
(``should_use_compile`` returns ``_env_use_compile_infer`` when
``self.training`` is ``False``).  Once enabled, regular validation,
full validation and EMA full validation all reuse the eval slot.
Eval TF32 is separately controlled by ``DP_TF32_INFER``:
``0 -> highest``, ``1 -> high``, ``2 -> medium``.

NOTE 8 -- Storing the compile cache outside the ``nn.Module`` tree
------------------------------------------------------------------

The cache dict is installed via ``object.__setattr__(self, ...)`` at
__init__ time rather than plain ``self.compiled_core_compute_cache = {}``, and
every later mutation writes into that same dict in place.
``nn.Module.__setattr__`` would register any module-looking value as a
submodule; the compiled wrappers held as *values* of this dict carry
duplicated flat views of the trainable parameters, and FSDP2 / DDP
would then shard or synchronise those duplicates and silently corrupt
training.  A plain ``dict`` container escapes parameter discovery
entirely because ``nn.Module.__setattr__`` only recognises
``nn.Parameter`` / ``nn.Module`` values, and ``named_parameters`` /
``named_modules`` walk ``self._parameters`` / ``self._modules``, never
arbitrary attributes; ``object.__setattr__`` merely belt-and-braces
this invariant for readers of the constructor.

NOTE 9 -- Graph restart via ``detach().requires_grad_(True)``
-------------------------------------------------------------

Before calling into the traced graph we rebind the extended coordinates
to a fresh leaf tensor: ``detach()`` breaks any upstream autograd graph
carried over from the data pipeline, and ``requires_grad_(True)``
reinstates a grad-endpoint owned by this forward.  The subsequent
``autograd.grad`` in ``fit_output_to_model_output`` therefore computes
``dE/dx`` against a graph of known shape and ownership -- the essential
precondition for make_fx symbolic tracing.

In eval the rebound coordinate still requires grad for the eager
(non-compiled) path's ``autograd.grad``, but the compiled callable runs
under ``torch.no_grad`` so its AOTAutograd inference lowering builds no
outer backward (NOTE 13).

NOTE 10 -- Tail dummy edges
---------------------------

``build_edge_list_from_nlist`` appends two masked edges at the end of
every batch.  Real edge compaction happens via
``torch.nonzero(valid_mask)``, whose output length is data-dependent
and can be zero in sparse or single-atom systems (e.g. isolated-atom
reference frames in training data).  make_fx cannot trace an
"if n_edges == 0: skip" branch symbolically; without the dummies it
would fall back to concrete shape specialization and break
``dynamic=True``.  A pair of dummy slots also gives Inductor's batched
matmul lowering a static ``E >= 2`` edge-axis bound, avoiding
data-dependent layout guards on ``E == 1`` that would otherwise cause
an extra recompile when the first batch contains no real edges.  Each
dummy's ``edge_mask`` is ``False`` so it contributes exactly zero to
every downstream sum or gather.

NOTE 11 -- ``index_select`` for coordinate gradients
----------------------------------------------------

Edge geometry is built with ``coord_flat.index_select(0, src)`` instead
of advanced indexing ``coord_flat[src]``.  ``index_select`` registers
an explicit backward that routes gradient cleanly back to the original
extended coordinate tensor.  Advanced indexing combined with make_fx
symbolic shapes has previously produced silent gradient truncation in
this project -- the second-derivative gradient over coordinates was
effectively zero, with no error raised.

NOTE 12 -- ``create_graph=self.training``
-----------------------------------------

The single toggle that turns force-loss training on.  When ``True``,
``autograd.grad`` keeps the graph over the first derivative alive so
the outer optimizer's ``.backward()`` can continue walking it into the
parameters.  When ``False`` the double-backward graph is never built,
saving memory during inference.

NOTE 13 -- Inference lowering through ``aot_module_simplified``
---------------------------------------------------------------

Training lowers the traced graph with ``torch.compile`` so the Dynamo
frontend can build the optimizer's second backward.  Inference needs no
such backward, and routing it through Dynamo is actively harmful:

* Dynamo re-traces the already-symbolic ``make_fx`` graph and re-runs
  shape-guard production.  On the forward-only graph one intermediate
  view's extended-atom (``nall``) axis becomes a backed symbol with no
  input source, and ``produce_guards`` aborts with ``sources must not be
  empty for symbol s...``.
* Even when it compiles, the grad-bearing input makes AOTAutograd treat
  the call as forward+backward and keep the whole forward activation set
  alive -- ~3x the eager peak memory, OOM-ing on large inference sweeps.

So eval lowers the graph with ``aot_module_simplified`` -- AOTAutograd's
inference path -- with no Dynamo frontend.  It still functionalizes the
graph (in-place ops become out-of-place, so Inductor reuses buffers and
peak memory matches eager), and compiling under ``torch.no_grad`` selects
the forward-only partition.  The make_fx placeholders' fake values carry
the symbolic sizes, so the single lowered artifact stays dynamic across
``nframes`` / ``nall`` / edge count without recompiles.

Four details make the hand-wired AOTAutograd + Inductor path work:

* **Flat output.**  AOTAutograd rejects the dict ``core_compute``
  returns, so the graph output is rewritten to a tuple of the dict
  values and the dict is re-packed around the compiled callable.
* **Decomposition table.**  ``select_decomp_table()`` is passed so the
  decomposition set matches Inductor's fallback set; the default
  core-aten table clashes (``both a fallback and a decomp for same op:
  aten._to_copy.default``).
* **Compile-time default device.**  ``aot_module_simplified`` enters a
  ``PhiloxStateTracker`` that allocates an RNG-state tensor without an
  explicit device, so the compile runs under ``torch.device(model
  device)`` to keep it off any stray ambient default.
* **1D pointwise grids.**  ``triton.max_tiles=1`` keeps the
  data-dependent edge axis on Triton's ``x`` grid (limit ``2**31``); the
  default tiling places it on the ``y``/``z`` grid (limit ``65535``),
  which overflows past ~2e4 atoms with ``CUDA error: invalid argument``.

The random local-Z roll (``random_gamma``) is gated to training in the
descriptor, so the inference graph holds no ``aten.rand`` at all: the
model is roll-equivariant, which makes the roll a pure training
augmentation and inference deterministic.
"""

from __future__ import (
    annotations,
)

import logging
import os
import time
from contextlib import (
    contextmanager,
    nullcontext,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
from einops import (
    rearrange,
)
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from jaxtyping import Float, Int
    from torch import Tensor

from deepmd.pt.model.atomic_model.sezm_atomic_model import (
    SeZMAtomicModel,
)
from deepmd.pt.model.descriptor.sezm_nn import (
    nvtx_range,
)
from deepmd.pt.model.model.dp_model import (
    DPModelCommon,
)
from deepmd.pt.model.model.make_model import (
    make_model,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.model.transform_output import (
    communicate_extended_output,
    fit_output_to_model_output,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.compile_compat import (
    AM_PREFIX,
    FIT_PREFIX,
    apply_global_compile_patches,
    build_inductor_compile_options,
    check_compile_torch_version,
    get_task_buffer_names,
    get_task_buffer_values,
    next_safe_prime,
    rebuild_graph_module,
    strip_saved_tensor_detach,
    trace_pad_dim,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt.utils.nv_nlist import (
    NvNeighborList,
    is_nv_available,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

log = logging.getLogger(__name__)

SeZMModel_ = make_model(SeZMAtomicModel)

# Local-atom counts above which Toolkit-Ops replaces the dense all-pairs builder.
# Non-periodic systems switch when dense all-pairs transients become memory-heavy,
# even though the dense path remains slightly faster at medium sizes.
SEZM_NV_NLIST_THRESHOLD = 1024
SEZM_NV_NONPERIODIC_NLIST_THRESHOLD = 2048

# Apply the process-global PyTorch workarounds the compile pipeline relies on
# (autotune log suppression, DDP optimiser, and the 2.12 divisibility repair)
# once, before the first compilation in this run.
apply_global_compile_patches()

# ---------------------------------------------------------------------------
# Multi-task compile sharing
# ---------------------------------------------------------------------------
# Maps (structure_key..., training, do_atomic_virial, has_coord_corr) to the
# compiled callable.  Tasks whose descriptor and fitting parameters share the
# same Python-object identity after ``share_params(level=0)`` reuse one compiled
# graph, avoiding N x compile-cache growth and duplicated DDP graph boundaries.
_SEZM_COMPILE_CACHE: dict[tuple[Any, ...], Any] = {}

# Maps structure_key -> task_buf_order so every instance in the same group
# knows which buffers were promoted and in what order.
_SEZM_TASK_BUF_ORDER: dict[tuple[Any, ...], tuple[str, ...]] = {}

_ENV_BOOL_CHOICES = {
    "1": True,
    "true": True,
    "yes": True,
    "on": True,
    "0": False,
    "false": False,
    "no": False,
    "off": False,
}
_TF32_INFER_PRECISION_CHOICES = {
    "0": "highest",
    "1": "high",
    "2": "medium",
}


def _module_shared_key(module: torch.nn.Module) -> tuple[int, ...]:
    """Return the direct identities that define a shared compiled structure."""
    child_ids = tuple(id(child) for child in module.children())
    param_ids = tuple(id(param) for param in module.parameters(recurse=False))
    if child_ids or param_ids:
        return child_ids + param_ids
    return (id(module),)


def _int_tuple(values: Any) -> tuple[int, ...]:
    """Return a stable integer tuple for graph-state keys."""
    return tuple(int(value) for value in values)


def _int_pair_tuple(values: Any) -> tuple[tuple[int, int], ...]:
    """Return a stable integer-pair tuple for graph-state keys."""
    return tuple(sorted(tuple(int(type_id) for type_id in pair) for pair in values))


def _sezm_structure_key(model: SeZMModel) -> tuple[Any, ...]:
    """Return a key that is equal iff two SeZMModel instances can share a compiled graph.

    After ``share_params``, the descriptor and fitting-net module objects
    themselves remain *different* Python objects per task; only their
    submodules / parameters are replaced with shared references.  Using
    ``id(descriptor)`` or ``id(fitting_net)`` would therefore always differ
    between tasks and defeat the cache.

    The key uses direct child-module identities plus direct parameter
    identities.  This matches SeZM's ``share_params(level=0)`` implementation
    and avoids false sharing when only part of the descriptor is linked.
    Non-module state that changes ``core_compute`` branches or masks is
    included explicitly.
    """
    atomic_model = model.atomic_model
    descriptor = atomic_model.descriptor
    fitting = atomic_model.fitting_net
    descriptor_key = _module_shared_key(descriptor)
    fitting_key = _module_shared_key(fitting)
    descriptor_state = (
        _int_pair_tuple(descriptor.exclude_types),
        bool(descriptor.use_env_seed),
        bool(descriptor.use_gie),
        bool(descriptor.random_gamma),
        descriptor.charge_spin_embedding is not None,
        descriptor.inner_clamp is not None,
        descriptor.bridging_switch is not None,
        descriptor.inner_clamp_r_inner,
        descriptor.inner_clamp_r_outer,
        int(descriptor.get_dim_chg_spin()),
    )
    fitting_state = (
        _int_tuple(fitting.exclude_types),
        bool(fitting.eval_return_middle_output),
    )
    atomic_state = (
        _int_tuple(atomic_model.atom_exclude_types),
        bool(atomic_model.enable_eval_descriptor_hook),
        bool(atomic_model.enable_eval_fitting_last_layer_hook),
    )
    model_state = (
        str(model.bridging_method),
        model.inter_potential is not None,
        float(model.bridging_r_inner),
        float(model.bridging_r_outer),
        tuple(model.get_type_map()),
    )
    return (
        descriptor_key
        + fitting_key
        + (
            descriptor_state,
            fitting_state,
            atomic_state,
            model_state,
        )
    )


@BaseModel.register("SeZM")
@BaseModel.register("sezm")
@BaseModel.register("DPA4")
@BaseModel.register("dpa4")
class SeZMModel(DPModelCommon, SeZMModel_):
    """
    SeZM energy model with an optional compiled sparse-edge path.

    By default it uses the traditional DeePMD neighbor list path with ghost atoms
    and padded neighbor matrix, compatible with LAMMPS and other MD engines.
    When `use_compile=True`, it builds a compact sparse edge list from the
    standard neighbor list and traces the local graph with ``make_fx`` for
    higher-order force training. Evaluation/inference compile usage is
    controlled by the `DP_COMPILE_INFER` environment variable read at model
    initialization time. This path is experimental, requires ``torch==2.11``,
    may still expose PyTorch compiler bugs, and can improve training speed by
    roughly 2-3x on supported workloads.
    """

    model_type = "SeZM"

    def __init__(
        self,
        *args: Any,
        use_compile: bool = False,
        enable_tf32: bool = True,
        bridging_method: str = "none",
        bridging_r_inner: float = 0.5,
        bridging_r_outer: float = 0.8,
        lora: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        SeZMModel_.__init__(self, *args, **kwargs)
        self.redu_prec = env.GLOBAL_PT_ENER_FLOAT_PRECISION
        self.use_compile = bool(use_compile)
        self.enable_tf32 = bool(enable_tf32)
        # LoRA injection happens in Trainer.__init__ after pre-trained state is loaded.
        self.lora_config: dict[str, Any] | None = None if lora is None else dict(lora)
        self._dens_compiled = False
        self._core_compute_pending_compile_t0: float | None = None
        self._core_compute_pending_compile_key: tuple[bool, bool, bool] | None = None
        self._dens_pending_compile_t0: float | None = None
        # Store compiled callables outside the nn.Module tree so that
        # FSDP2 / DDP do not shard or sync its duplicated parameters.
        # ``compiled_core_compute_cache`` is keyed on
        # ``(training, do_atomic_virial, has_coord_corr)`` so every graph
        # topology has its own slot; flipping between train and eval for
        # validation -- regular, full, or EMA full -- therefore reuses cached
        # compile products instead of evicting the other mode.
        object.__setattr__(self, "compiled_core_compute_cache", {})
        object.__setattr__(self, "compiled_dens_compute", None)
        # Maps cache_key -> task_buf_order for this instance so forward()
        # knows which buffers to pass and in what order.
        object.__setattr__(self, "_task_buf_order_cache", {})

        # Training follows `use_compile`. Evaluation/inference samples env
        # policy at init time so path and precision stay fixed per model.
        compile_infer_env = os.environ.get("DP_COMPILE_INFER")
        if compile_infer_env is None:
            self._env_use_compile_infer: bool | None = None
        else:
            compile_infer_env = compile_infer_env.strip().lower()
            if compile_infer_env not in _ENV_BOOL_CHOICES:
                choices = "/".join(_ENV_BOOL_CHOICES)
                raise ValueError(
                    f"DP_COMPILE_INFER must be one of {choices}, "
                    f"got {compile_infer_env!r}"
                )
            self._env_use_compile_infer = _ENV_BOOL_CHOICES[compile_infer_env]

        tf32_infer_env = os.environ.get("DP_TF32_INFER", "0").strip().lower()
        if tf32_infer_env not in _TF32_INFER_PRECISION_CHOICES:
            raise ValueError(
                f"DP_TF32_INFER must be one of 0/1/2, got {tf32_infer_env!r}"
            )
        self._tf32_infer_precision = _TF32_INFER_PRECISION_CHOICES[tf32_infer_env]
        if self.use_compile or self._env_use_compile_infer is True:
            check_compile_torch_version()

        # === Bridging (optional short-range zone bridging) ===
        self.bridging_method: str = str(bridging_method).upper()
        self.bridging_r_inner = float(bridging_r_inner)
        self.bridging_r_outer = float(bridging_r_outer)
        self.inter_potential: InterPotential | None = (
            InterPotential(type_map=self.get_type_map(), mode=self.bridging_method)
            if self.bridging_method != "NONE"
            else None
        )

    # =========================================================================
    # Forward Methods
    # =========================================================================

    def forward(
        self,
        coord: Float[Tensor, "nf nloc 3"] | Float[Tensor, "nf nloc_x3"],
        atype: Int[Tensor, "nf nloc"],
        box: Float[Tensor, "nf 9"] | None = None,
        fparam: Float[Tensor, "nf ndf"] | None = None,
        aparam: Float[Tensor, "nf nloc nda"] | None = None,
        do_atomic_virial: bool = False,
        force_input: Float[Tensor, "nf nloc 3"] | None = None,
        noise_mask: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass using standard neighbor list.

        Parameters
        ----------
        coord
            Coordinates with shape (nf, nloc*3) or (nf, nloc, 3) in Å.
        atype
            Atom types with shape (nf, nloc).
        box
            Box tensor with shape (nf, 9) in Å, or None.
        fparam
            Frame parameters with shape (nf, ndf) or None.
        aparam
            Atomic parameters with shape (nf, nloc, nda) or None.
        do_atomic_virial
            Whether to compute atomic virial.
        force_input
            Optional atom-wise force input tensor with shape `(nf, nloc, 3)`.
            It stays optional at the public model boundary because validation /
            inference and clean `dens` batches may not provide force labels.
        noise_mask
            Optional corruption mask with shape `(nf, nloc)`. It stays optional
            at the public model boundary because validation / inference and
            clean `dens` batches may not provide corruption masks.
        charge_spin
            Frame-level charge and spin conditions with shape `(nf, 2)`.

        Returns
        -------
        dict[str, torch.Tensor]
            Model predictions including atom_energy, energy, force, virial,
            atom_virial, and mask.
        """
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            force_input=force_input,
            noise_mask=noise_mask,
            charge_spin=charge_spin,
        )
        if self.get_fitting_net() is not None:
            model_predict: dict[str, torch.Tensor] = {}

            # === Step 1. Energy ===
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]

            # === Step 2. Force (independent branch) ===
            if self.do_grad_r("energy"):
                model_predict["force"] = rearrange(
                    model_ret["energy_derv_r"],
                    "nf nloc 1 three -> nf nloc three",
                    three=3,
                )
            else:
                model_predict["force"] = model_ret["dforce"]

            if self.get_active_mode() == "dens":
                if "energy_norm" in model_ret:
                    model_predict["energy_norm"] = model_ret["energy_norm"]
                if "atom_energy_norm" in model_ret:
                    model_predict["atom_energy_norm"] = model_ret["atom_energy_norm"]
                if "dforce_norm" in model_ret:
                    model_predict["force_norm"] = model_ret["dforce_norm"]
                if "clean_dforce_norm" in model_ret:
                    model_predict["clean_force_norm"] = model_ret["clean_dforce_norm"]
                if "denoising_dforce_norm" in model_ret:
                    model_predict["denoising_force_norm"] = model_ret[
                        "denoising_dforce_norm"
                    ]

            # === Step 3. Virial ===
            if self.do_grad_c("energy"):
                model_predict["virial"] = rearrange(
                    model_ret["energy_derv_c_redu"], "nf 1 nine -> nf nine", nine=9
                )
                if do_atomic_virial:
                    model_predict["atom_virial"] = rearrange(
                        model_ret["energy_derv_c"],
                        "nf nloc 1 nine -> nf nloc nine",
                        nine=9,
                    )

            # === Step 4. Mask ===
            if "mask" in model_ret:
                model_predict["mask"] = model_ret["mask"]

        else:
            model_predict = model_ret
            model_predict["updated_coord"] += coord
        return model_predict

    def forward_common(
        self,
        coord: Float[Tensor, "nf nloc 3"] | Float[Tensor, "nf nloc_x3"],
        atype: Int[Tensor, "nf nloc"],
        box: Float[Tensor, "nf 9"] | None = None,
        fparam: Float[Tensor, "nf ndf"] | None = None,
        aparam: Float[Tensor, "nf nloc nda"] | None = None,
        do_atomic_virial: bool = False,
        force_input: Float[Tensor, "nf nloc 3"] | None = None,
        noise_mask: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Return model prediction using standard neighbor list.

        Parameters
        ----------
        coord
            Coordinates with shape (nf, nloc*3) or (nf, nloc, 3) in Å.
        atype
            Atom types with shape (nf, nloc).
        box
            Box tensor with shape (nf, 9) in Å, or None.
        fparam
            Frame parameters with shape (nf, ndf) or None.
        aparam
            Atomic parameters with shape (nf, nloc, nda) or None.
        do_atomic_virial
            Whether to compute atomic virial.
        force_input
            Optional atom-wise force input tensor with shape `(nf, nloc, 3)`.
            It stays optional at the public model boundary because validation /
            inference and clean `dens` batches may not provide force labels.
        noise_mask
            Optional corruption mask with shape `(nf, nloc)`. It stays optional
            at the public model boundary because validation / inference and
            clean `dens` batches may not provide corruption masks.
        charge_spin
            Frame-level charge and spin conditions with shape `(nf, 2)`.

        Returns
        -------
        dict[str, torch.Tensor]
            Model predictions including energy, forces, etc.
        """
        with nvtx_range("SeZM/forward_common"):
            # === Step 1. Cast inputs to correct dtype ===
            with nvtx_range("SeZM/input_type_cast"):
                cc, bb, fp, ap, input_prec = self._input_type_cast(
                    coord, box=box, fparam=fparam, aparam=aparam
                )
                del coord, box, fparam, aparam
                nf, nloc = atype.shape[:2]
                if cc.ndim == 2:
                    cc = cc.view(nf, nloc, 3)

            # === Step 2. Build neighbor list ===
            with nvtx_range("SeZM/build_neighbor_list"):
                # extended_coord: (nf, nall, 3), extended_atype: (nf, nall)
                # mapping: (nf, nall), nlist: (nf, nloc, nsel)
                extended_coord, extended_atype, mapping, nlist = (
                    self.build_neighbor_list(cc, atype, bb)
                )

            # === Step 3. Run the shared extended-input path ===
            return self.forward_common_after_nlist(
                extended_coord,
                extended_atype,
                mapping,
                nlist,
                atype,
                fp,
                ap,
                input_prec,
                do_atomic_virial=do_atomic_virial,
                force_input=force_input,
                noise_mask=noise_mask,
                charge_spin=charge_spin,
            )

    def forward_common_after_nlist(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        mapping: torch.Tensor,
        nlist: torch.Tensor,
        atype: torch.Tensor,
        fp: torch.Tensor | None,
        ap: torch.Tensor | None,
        input_prec: torch.dtype,
        *,
        do_atomic_virial: bool = False,
        force_input: torch.Tensor | None = None,
        noise_mask: torch.Tensor | None = None,
        extended_coord_corr: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Run SeZM from already-built extended inputs.

        Parameters
        ----------
        extended_coord
            Coordinates in extended region with shape (nf, nall, 3).
        extended_atype
            Atom types in extended region with shape (nf, nall).
        mapping
            Extended-to-local mapping with shape (nf, nall).
        nlist
            Neighbor list with shape (nf, nloc, nsel).
        atype
            Local atom types with shape (nf, nloc).
        fp
            Cast frame parameters with shape (nf, ndf), or None.
        ap
            Cast atomic parameters with shape (nf, nloc, nda), or None.
        input_prec
            Original input precision used for output casting.
        do_atomic_virial
            Whether to compute per-atom virial.
        force_input
            Optional atom-wise force input for the ``dens`` path with shape
            (nf, nloc, 3).
        noise_mask
            Optional atom-wise corruption mask for the ``dens`` path with
            shape (nf, nloc).
        extended_coord_corr
            Coordinate correction for virial with shape (nf, nall, 3), or None.
        charge_spin
            Frame-level charge and spin conditions with shape `(nf, 2)`.

        Returns
        -------
        dict[str, torch.Tensor]
            Model predictions with the standard SeZM internal keys.
        """
        nf, nloc = atype.shape[:2]
        charge_spin = self.convert_charge_spin(
            charge_spin,
            nf=nf,
            dtype=extended_coord.dtype,
            device=extended_coord.device,
        )
        active_mode = self.get_active_mode()
        if active_mode == "dens":
            # === Step 1. `dens` path (no coordinate gradients needed) ===
            extended_coord = extended_coord.detach()
            force_input, noise_mask = self.canonicalize_dens_inputs(
                force_input,
                noise_mask,
                nf=nf,
                nloc=nloc,
                dtype=extended_coord.dtype,
                device=extended_coord.device,
            )

            with self.tf32_precision_ctx():
                if self.should_use_compile():
                    fp, ap = self.convert_fp_ap(
                        fp,
                        ap,
                        nf=nf,
                        nloc=nloc,
                        dtype=extended_coord.dtype,
                        device=extended_coord.device,
                    )
                    if self.compiled_dens_compute is None or not self._dens_compiled:
                        self.compile_dens()
                    with nvtx_range("SeZM/core_compute_dens"):
                        compute_ret = self.compiled_dens_compute(
                            extended_coord,
                            extended_atype,
                            nlist,
                            mapping,
                            force_input=force_input,
                            noise_mask=noise_mask,
                            fparam=fp,
                            aparam=ap,
                            charge_spin=charge_spin,
                        )
                    if self._dens_pending_compile_t0 is not None:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        log.info(
                            "SeZM: finished compiling dens path in %.2fs",
                            time.perf_counter() - self._dens_pending_compile_t0,
                        )
                        self._dens_pending_compile_t0 = None
                else:
                    with nvtx_range("SeZM/core_compute_dens"):
                        compute_ret = self.core_compute_dens(
                            extended_coord,
                            extended_atype,
                            nlist,
                            mapping,
                            force_input=force_input,
                            noise_mask=noise_mask,
                            fparam=fp,
                            aparam=ap,
                            charge_spin=charge_spin,
                        )
            with nvtx_range("SeZM/post_process"):
                model_predict = self.post_process_output_dens(
                    compute_ret,
                    atype,
                    noise_mask=noise_mask,
                )
        else:
            # === Step 1. `ener` path (edges built inside core_compute) ===
            # NOTE: Rebind the extended coordinates to a fresh leaf
            # tensor before entering either ``core_compute`` or the
            # compiled callable.  ``detach()`` breaks any upstream
            # autograd graph carried by the batch (data pipeline
            # artefacts, neighbor-list ops) and
            # ``requires_grad_(True)`` reinstates a grad-endpoint
            # owned exclusively by this forward.  The inner
            # ``autograd.grad`` inside ``fit_output_to_model_output``
            # will then compute ``dE/dx`` against a graph of known
            # shape and ownership -- the essential precondition for
            # symbolic make_fx tracing.  In eval without coordinate
            # gradients a bare detach is enough.
            if self.do_grad_r() or self.do_grad_c():
                extended_coord = extended_coord.detach().requires_grad_(True)
            else:
                extended_coord = extended_coord.detach()

            with self.tf32_precision_ctx():
                if self.should_use_compile():
                    fp, ap = self.convert_fp_ap(
                        fp,
                        ap,
                        nf=nf,
                        nloc=nloc,
                        dtype=extended_coord.dtype,
                        device=extended_coord.device,
                    )
                    has_coord_corr = extended_coord_corr is not None
                    cache_key = (
                        bool(self.training),
                        bool(do_atomic_virial),
                        has_coord_corr,
                    )
                    if cache_key not in self.compiled_core_compute_cache:
                        self.trace_and_compile(
                            extended_coord,
                            extended_atype,
                            nlist,
                            mapping,
                            fp,
                            ap,
                            charge_spin,
                            do_atomic_virial,
                            extended_coord_corr=extended_coord_corr,
                        )
                    compiled_core_compute = self.compiled_core_compute_cache[cache_key]
                    # Read current values of per-task buffers (optimizer steps
                    # update them in-place; out-of-place replacements from
                    # model_change_out_bias are captured because we read fresh
                    # each call rather than caching the values at compile time).
                    _task_buf_vals = get_task_buffer_values(
                        self,
                        self._task_buf_order_cache[cache_key],
                    )
                    # NOTE: Inference needs no autograd tape -- the force
                    # (-dE/dx) is already materialised as forward ops in the
                    # traced graph, so keeping the tape would only make
                    # AOTAutograd save the full forward activation set for a
                    # backward eval never runs (see NOTE 13).  Training keeps it
                    # for the force-loss second derivative.
                    grad_ctx: Any = nullcontext() if self.training else torch.no_grad()
                    with nvtx_range("SeZM/core_compute"), grad_ctx:
                        if extended_coord_corr is None:
                            model_predict_lower = compiled_core_compute(
                                extended_coord,
                                extended_atype,
                                nlist,
                                mapping,
                                fp,
                                ap,
                                charge_spin,
                                *_task_buf_vals,
                            )
                        else:
                            model_predict_lower = compiled_core_compute(
                                extended_coord,
                                extended_atype,
                                nlist,
                                mapping,
                                fp,
                                ap,
                                charge_spin,
                                extended_coord_corr,
                                *_task_buf_vals,
                            )
                    if (
                        self._core_compute_pending_compile_t0 is not None
                        and self._core_compute_pending_compile_key == cache_key
                    ):
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        log.info(
                            "SeZM: finished compiling "
                            "(mode=%s, atomic_virial=%s, coord_corr=%s) "
                            "in %.2fs",
                            "train" if self.training else "eval",
                            do_atomic_virial,
                            has_coord_corr,
                            time.perf_counter() - self._core_compute_pending_compile_t0,
                        )
                        self._core_compute_pending_compile_t0 = None
                        self._core_compute_pending_compile_key = None
                else:
                    with nvtx_range("SeZM/core_compute"):
                        model_predict_lower = self.core_compute(
                            extended_coord,
                            extended_atype,
                            nlist,
                            mapping=mapping,
                            fparam=fp,
                            aparam=ap,
                            charge_spin=charge_spin,
                            do_atomic_virial=do_atomic_virial,
                            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
                            extended_coord_corr=extended_coord_corr,
                        )

            with nvtx_range("SeZM/communicate_output"):
                model_predict = communicate_extended_output(
                    model_predict_lower,
                    self.model_output_def(),
                    mapping,
                    do_atomic_virial=do_atomic_virial,
                )

        # === Step 2. Type cast output ===
        with nvtx_range("SeZM/output_type_cast"):
            model_predict = self._output_type_cast(model_predict, input_prec)
            return model_predict

    def core_compute(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        extra_nlist_sort: bool = False,
        extended_coord_corr: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute SeZM lower outputs from extended inputs.

        Builds compact sparse edges, runs descriptor and fitting evaluation,
        applies output masking and the optional analytical pair potential,
        then calls ``fit_output_to_model_output`` for force / virial.

        Parameters
        ----------
        extended_coord
            Coordinates in extended region with shape (nf, nall, 3).
        extended_atype
            Atom types in extended region with shape (nf, nall).
        nlist
            Neighbor list with shape (nf, nloc, nsel).
        mapping
            Extended-to-local mapping with shape (nf, nall), or ``None``.
        fparam
            Frame parameters with shape (nf, ndf), or ``None``.
        aparam
            Atomic parameters with shape (nf, nloc, nda), or ``None``.
        charge_spin
            Frame-level charge and spin conditions with shape `(nf, 2)`.
        do_atomic_virial
            Whether to compute per-atom virial.
        comm_dict
            Communication data for parallel inference. Currently unused.
        extra_nlist_sort
            Whether to forcibly sort the nlist.
        extended_coord_corr
            Coordinates correction for virial with shape (nf, nall, 3) or ``None``.

        Returns
        -------
        dict[str, torch.Tensor]
            DeePMD lower-style outputs (energy, energy_redu, energy_derv_r, ...).
        """
        del comm_dict
        nlist = self.format_nlist(
            extended_coord, extended_atype, nlist, extra_nlist_sort=extra_nlist_sort
        )
        _, nloc, _ = nlist.shape
        atype = extended_atype[:, :nloc]
        descriptor_model = self.atomic_model.descriptor

        # === Step 1. Build compact sparse edges ===
        edge_index, edge_vec, edge_mask = self.build_edge_list_from_nlist(
            extended_coord=extended_coord,
            nlist=nlist,
            mapping=mapping,
        )

        # === Step 2. Descriptor forward ===
        with nvtx_range("SeZM/descriptor"):
            descriptor, _ = descriptor_model.forward_with_edges(
                extended_coord=extended_coord[:, :nloc, :],
                extended_atype=atype,
                edge_index=edge_index,
                edge_vec=edge_vec,
                edge_mask=edge_mask,
                charge_spin=charge_spin,
            )
        if self.atomic_model.enable_eval_descriptor_hook:
            self.atomic_model.eval_descriptor_list.append(descriptor.detach())

        # === Step 3. Fitting net + output statistics ===
        with nvtx_range("SeZM/fitting_net"):
            fit_ret = self.atomic_model.fitting_net(
                descriptor,
                atype,
                fparam=fparam,
                aparam=aparam,
            )
        if self.atomic_model.enable_eval_fitting_last_layer_hook:
            assert "middle_output" in fit_ret, (
                "eval_fitting_last_layer not supported for this fitting net!"
            )
            self.atomic_model.eval_fitting_last_layer_list.append(
                fit_ret.pop("middle_output").detach()
            )
        with nvtx_range("SeZM/apply_out_stat"):
            fit_ret = self.atomic_model.apply_out_stat(fit_ret, atype)

        # === Step 4. Apply atom mask ===
        ext_atom_mask = self.atomic_model.make_atom_mask(extended_atype)
        atom_mask = ext_atom_mask[:, :nloc].to(torch.int32)
        if self.atomic_model.atom_excl is not None:
            atom_mask *= self.atomic_model.atom_excl(atype)
        for key in fit_ret.keys():
            out_shape = fit_ret[key].shape
            flat_dim = 1
            for axis_size in out_shape[2:]:
                flat_dim *= axis_size
            fit_ret[key] = (
                fit_ret[key].reshape([out_shape[0], out_shape[1], flat_dim])
                * atom_mask[:, :, None]
            ).view(out_shape)
        fit_ret["mask"] = atom_mask

        # === Step 5. Inject analytical pair potential ===
        if self.inter_potential is not None:
            fit_ret["energy"] = fit_ret["energy"] + self.inter_potential(
                extended_coord,
                extended_atype,
                nlist,
                nloc,
                real_type_count=self._get_inter_potential_real_type_count(),
            )

        # === Step 6. Force / virial via fit_output_to_model_output ===
        # NOTE: ``create_graph=self.training`` is the single toggle that
        # activates force-loss training.  Internally this calls
        # ``torch.autograd.grad(energy, extended_coord, create_graph=...)``
        # to produce ``force = -dE/dx``.  When ``True`` the autograd graph
        # over the first derivative is kept alive, so the outer
        # optimiser's ``.backward()`` can continue differentiating into
        # parameters -- that chain is the full
        # ``d^2 E / (dx dtheta)`` second derivative.  When ``False`` the
        # double-backward graph is never built, saving memory during
        # inference.  The entire reason this file exists -- make_fx,
        # detach stripping, graph rebuild -- is to keep that
        # second-derivative chain intact after ``torch.compile`` has
        # captured the whole thing.
        return fit_output_to_model_output(
            fit_ret,
            self.atomic_output_def(),
            extended_coord,
            do_atomic_virial=do_atomic_virial,
            create_graph=self.training,
            mask=fit_ret["mask"],
            extended_coord_corr=extended_coord_corr,
        )

    def core_compute_dens(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        *,
        force_input: torch.Tensor,
        noise_mask: torch.Tensor,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute SeZM ``dens`` energy/direct-force tensors from extended inputs.

        Parameters
        ----------
        extended_coord
            Extended coordinates with shape (nf, nall, 3).
        extended_atype
            Extended atom types with shape (nf, nall).
        nlist
            Neighbor list with shape (nf, nloc, nsel).
        mapping
            Extended-to-local mapping with shape (nf, nall), or ``None``.
        force_input
            Atom-wise force input tensor with shape ``(nf, nloc, 3)``.
        noise_mask
            Atom-wise corruption mask with shape ``(nf, nloc)``.
        fparam
            Frame parameters with shape ``(nf, ndf)``, or ``None``.
        aparam
            Atomic parameters with shape ``(nf, nloc, nda)``, or ``None``.
        charge_spin
            Frame-level charge and spin conditions with shape `(nf, 2)`.

        Returns
        -------
        torch.Tensor
            Concatenated local tensor with shape ``(nf, nloc, 7)`` and layout
            ``[atom_energy_norm | clean_dforce_norm | denoising_dforce_norm]``.
        """
        if self.inter_potential is not None:
            raise NotImplementedError(
                "SeZM `dens` path does not support analytical bridging potentials."
            )

        nlist = self.format_nlist(
            extended_coord,
            extended_atype,
            nlist,
            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
        )
        _, nloc, _ = nlist.shape
        atype = extended_atype[:, :nloc]
        descriptor_model = self.atomic_model.descriptor

        # === Step 1. Build compact sparse edges ===
        edge_index, edge_vec, edge_mask = self.build_edge_list_from_nlist(
            extended_coord=extended_coord,
            nlist=nlist,
            mapping=mapping,
        )

        # === Step 2. Force embedding ===
        dens_fitting = self.atomic_model.get_dens_fitting_net()
        force_embedding = dens_fitting.build_force_embedding(
            force_input,
            noise_mask=noise_mask,
        )

        # === Step 3. Descriptor forward with force embedding ===
        with nvtx_range("SeZM/descriptor_dens"):
            descriptor, latent = descriptor_model.forward_with_edges(
                extended_coord=extended_coord[:, :nloc, :],
                extended_atype=atype,
                edge_index=edge_index,
                edge_vec=edge_vec,
                edge_mask=edge_mask,
                force_embedding=force_embedding,
                charge_spin=charge_spin,
            )
        if self.atomic_model.enable_eval_descriptor_hook:
            self.atomic_model.eval_descriptor_list.append(descriptor.detach())

        # === Step 4. Dens fitting net ===
        with nvtx_range("SeZM/dens_fitting_net"):
            fit_ret = dens_fitting(
                descriptor,
                latent,
                atype,
                noise_mask=noise_mask,
                fparam=fparam,
                aparam=aparam,
                return_components=True,
            )
        if self.atomic_model.enable_eval_fitting_last_layer_hook:
            assert "middle_output" in fit_ret, (
                "eval_fitting_last_layer not supported for this fitting net!"
            )
            self.atomic_model.eval_fitting_last_layer_list.append(
                fit_ret.pop("middle_output").detach()
            )
        return torch.cat(
            [
                fit_ret["energy"],
                fit_ret["clean_dforce"],
                fit_ret["denoising_dforce"],
            ],
            dim=-1,
        )

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord: Float[Tensor, "nf nall_x3"] | Float[Tensor, "nf nall 3"],
        extended_atype: Int[Tensor, "nf nall"],
        nlist: Int[Tensor, "nf nloc nsel"],
        mapping: Int[Tensor, "nf nall"] | None = None,
        fparam: Float[Tensor, "nf ndf"] | None = None,
        aparam: Float[Tensor, "nf nall nda"] | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Lower-level public forward using the DeePMD lower-interface contract.

        Parameters
        ----------
        extended_coord
            Extended coordinates with shape (nf, nall*3) or (nf, nall, 3) in Å.
        extended_atype
            Extended atom types with shape (nf, nall).
        nlist
            Neighbor list with shape (nf, nloc, nsel).
        mapping
            Mapping indices with shape (nf, nall), or None.
        fparam
            Frame parameters with shape (nf, ndf) or None.
        aparam
            Atomic parameters with shape (nf, nall, nda) or None.
        do_atomic_virial
            Whether to compute atomic virial.
        comm_dict
            Communication dict forwarded to `forward_common_lower()`.
        charge_spin
            Frame-level charge and spin conditions with shape `(nf, 2)`.

        Returns
        -------
        dict[str, torch.Tensor]
            Lower-interface outputs.
            When a fitting net is present, this always includes:
            - `atom_energy`: atomic energy on local atoms with shape (nf, nloc, 1)
            - `energy`: reduced energy with shape (nf, 1)
            It additionally includes:
            - `extended_force`: force on extended coordinates with shape (nf, nall, 3)
              when `self.do_grad_r("energy")` is true
            - `dforce`: fitting-net direct force output when energy is not coordinate differentiable
            - `virial`: reduced virial with shape (nf, 9) when `self.do_grad_c("energy")` is true
            - `extended_virial`: per-extended-atom virial with shape (nf, nall, 9)
              only when both `self.do_grad_c("energy")` and `do_atomic_virial` are true
            If no fitting net is present, the raw result of `forward_common_lower()` is returned.
        """
        if self.get_active_mode() == "dens":
            raise NotImplementedError(
                "SeZM `forward_lower` only supports the conservative `ener` mode."
            )
        cc_ext, _, fp, ap, input_prec = self._input_type_cast(
            extended_coord, fparam=fparam, aparam=aparam
        )
        model_ret = self.forward_common_lower(
            cc_ext,
            extended_atype,
            nlist,
            mapping,
            fparam=fp,
            aparam=ap,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
            charge_spin=charge_spin,
        )
        model_ret = self._output_type_cast(model_ret, input_prec)
        if self.get_fitting_net() is not None:
            model_predict: dict[str, torch.Tensor] = {}

            # === Step 1. Energy ===
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]

            # === Step 2. Force (independent branch) ===
            if self.do_grad_r("energy"):
                model_predict["extended_force"] = rearrange(
                    model_ret["energy_derv_r"],
                    "nf nall 1 three -> nf nall three",
                    three=3,
                )
            else:
                assert model_ret["dforce"] is not None
                model_predict["dforce"] = model_ret["dforce"]

            # === Step 3. Virial ===
            if self.do_grad_c("energy"):
                model_predict["virial"] = rearrange(
                    model_ret["energy_derv_c_redu"], "nf 1 nine -> nf nine", nine=9
                )
                if do_atomic_virial:
                    model_predict["extended_virial"] = rearrange(
                        model_ret["energy_derv_c"],
                        "nf nall 1 nine -> nf nall nine",
                        nine=9,
                    )
        else:
            model_predict = model_ret
        return model_predict

    def forward_common_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        extra_nlist_sort: bool = False,
        extended_coord_corr: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Public lower interface with dtype casting around ``core_compute()``."""
        cc_ext, _, fp, ap, input_prec = self._input_type_cast(
            extended_coord, fparam=fparam, aparam=aparam
        )
        cc_ext = cc_ext.reshape(extended_atype.shape[0], -1, 3)
        if extended_coord_corr is not None and extended_coord_corr.ndim == 2:
            extended_coord_corr = extended_coord_corr.reshape(
                extended_atype.shape[0], -1, 3
            )
        if self.do_grad_r() or self.do_grad_c():
            cc_ext = cc_ext.detach().requires_grad_(True)
        nf = extended_atype.shape[0]
        charge_spin = self.convert_charge_spin(
            charge_spin,
            nf=nf,
            dtype=cc_ext.dtype,
            device=cc_ext.device,
        )
        model_predict = self.core_compute(
            cc_ext,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fp,
            aparam=ap,
            charge_spin=charge_spin,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=extra_nlist_sort,
            extended_coord_corr=extended_coord_corr,
        )
        return self._output_type_cast(model_predict, input_prec)

    # =========================================================================
    # Compile Utilities
    # =========================================================================

    def trace_and_compile(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor,
        fp: torch.Tensor,
        ap: torch.Tensor,
        charge_spin: torch.Tensor,
        do_atomic_virial: bool,
        extended_coord_corr: torch.Tensor | None = None,
    ) -> None:
        """Trace ``core_compute()`` with ``make_fx`` and cache the compiled callable.

        The full flow is: wrap ``core_compute`` in a tensor-only
        ``compute_fn`` that also owns the coordinate grad-endpoint, trace
        it with ``make_fx(tracing_mode="symbolic")`` so all shape axes
        become sympy symbols, strip autograd-inserted detach chains in
        training mode, rebuild the FX graph to flush stale linked-list
        pointers, and finally hand the clean ``GraphModule`` to
        ``torch.compile(backend="inductor", dynamic=True)``.  The
        compiled callable is stored outside the ``nn.Module`` tree so
        FSDP/DDP cannot see or shard its duplicated parameters.
        """
        from torch._decomp import (
            get_decompositions,
        )

        mode = "train" if self.training else "eval"
        has_coord_corr = extended_coord_corr is not None
        _compile_t0 = time.perf_counter()

        # --- Check module-level shared cache first ---
        # Tasks sharing the same descriptor+fitting structure (after share_params)
        # should share one compiled graph.  If a sibling task already compiled,
        # populate this instance's per-instance caches and return immediately.
        structure_key = _sezm_structure_key(self)
        cache_key = (bool(self.training), bool(do_atomic_virial), has_coord_corr)
        full_cache_key = structure_key + cache_key
        if full_cache_key in _SEZM_COMPILE_CACHE:
            self.compiled_core_compute_cache[cache_key] = _SEZM_COMPILE_CACHE[
                full_cache_key
            ]
            self._task_buf_order_cache[cache_key] = _SEZM_TASK_BUF_ORDER[structure_key]
            log.info(
                "SeZM: reusing shared compiled graph "
                "(mode=%s, atomic_virial=%s, coord_corr=%s)",
                mode,
                do_atomic_virial,
                has_coord_corr,
            )
            return

        log.info(
            "SeZM: start tracing and compiling "
            "(mode=%s, atomic_virial=%s, coord_corr=%s)",
            mode,
            do_atomic_virial,
            has_coord_corr,
        )

        # Promote the per-task buffers (see ``get_task_buffer_names``) to
        # explicit graph inputs so one compiled graph serves the whole task
        # group regardless of their per-task values.
        task_buf_names = get_task_buffer_names(self)
        task_buf_vals_trace = get_task_buffer_values(self, task_buf_names)

        # Resolve module references once for the buffer-patching closures.
        _am_patch = self.atomic_model
        _fitting_patch = _am_patch.fitting_net

        def _patch_task_bufs(
            vals: tuple[torch.Tensor, ...],
        ) -> dict[str, torch.Tensor]:
            """Temporarily replace task-local buffers with FX proxy tensors.

            Executed at trace time inside compute_fn.  make_fx records the
            proxy tensors as placeholder nodes, so the compiled graph reads them
            as live inputs rather than baked-in constants.  The ``finally``
            block in compute_fn always calls ``_restore_task_bufs`` to leave
            the model in its original state after tracing.
            """
            if len(vals) != len(task_buf_names):
                raise ValueError(
                    "SeZM task-buffer placeholder count mismatch: "
                    f"expected {len(task_buf_names)}, got {len(vals)}"
                )
            saved: dict[str, torch.Tensor] = {}
            try:
                for name, val in zip(task_buf_names, vals):
                    if name.startswith(AM_PREFIX):
                        actual = name[len(AM_PREFIX) :]
                        saved[name] = _am_patch._buffers[actual]
                        _am_patch._buffers[actual] = val
                    elif name.startswith(FIT_PREFIX):
                        actual = name[len(FIT_PREFIX) :]
                        saved[name] = _fitting_patch._buffers[actual]
                        _fitting_patch._buffers[actual] = val
            except Exception:
                _restore_task_bufs(saved)
                raise
            return saved

        def _restore_task_bufs(
            saved: dict[str, torch.Tensor],
        ) -> None:
            """Restore original task-local buffers after tracing."""
            for name, orig in saved.items():
                if name.startswith(AM_PREFIX):
                    actual = name[len(AM_PREFIX) :]
                    _am_patch._buffers[actual] = orig
                elif name.startswith(FIT_PREFIX):
                    actual = name[len(FIT_PREFIX) :]
                    _fitting_patch._buffers[actual] = orig

        need_coord_grad = self.do_grad_r() or self.do_grad_c()

        def _prepare_coord_for_trace(coord: torch.Tensor) -> torch.Tensor:
            """Restart the coordinate autograd graph for the traced compute.

            ``detach()`` severs any upstream graph carried by the trace
            inputs and ``requires_grad_(True)`` reinstates a fresh
            grad-endpoint owned by this compute.  The inner
            ``autograd.grad`` inside ``fit_output_to_model_output`` then
            differentiates against a graph of known shape and ownership --
            the essential precondition for make_fx symbolic tracing to
            capture dE/dx as ordinary FX nodes.  In the eval-only branch
            a bare detach keeps the traced graph free of backward sections.
            """
            if need_coord_grad:
                return coord.detach().requires_grad_(True)
            else:
                return coord.detach()

        # NOTE: compute_fn accepts *task_buf_vals after the fixed tensor args.
        # make_fx treats each element as a separate placeholder so the compiled
        # graph reads them as live inputs every call — not baked-in constants.
        # The buffer-patching trick: at trace time the proxy tensors are written
        # into _buffers so downstream code (apply_out_stat, fitting_net.forward)
        # reads the proxies and the ops are recorded in the FX graph. The
        # finally block restores original state unconditionally.
        if extended_coord_corr is None:

            def compute_fn(
                extended_coord: torch.Tensor,
                extended_atype: torch.Tensor,
                nlist: torch.Tensor,
                mapping: torch.Tensor,
                fp: torch.Tensor,
                ap: torch.Tensor,
                charge_spin: torch.Tensor,
                *task_buf_vals: torch.Tensor,
            ) -> dict[str, torch.Tensor]:
                _saved = _patch_task_bufs(task_buf_vals)
                try:
                    return self.core_compute(
                        _prepare_coord_for_trace(extended_coord),
                        extended_atype,
                        nlist,
                        mapping=mapping,
                        fparam=fp,
                        aparam=ap,
                        charge_spin=charge_spin,
                        do_atomic_virial=do_atomic_virial,
                        extra_nlist_sort=self.need_sorted_nlist_for_lower(),
                    )
                finally:
                    _restore_task_bufs(_saved)

        else:

            def compute_fn(  # type: ignore[misc]
                extended_coord: torch.Tensor,
                extended_atype: torch.Tensor,
                nlist: torch.Tensor,
                mapping: torch.Tensor,
                fp: torch.Tensor,
                ap: torch.Tensor,
                charge_spin: torch.Tensor,
                extended_coord_corr: torch.Tensor,
                *task_buf_vals: torch.Tensor,
            ) -> dict[str, torch.Tensor]:
                # NOTE: Spin virial uses a coordinate correction derived from the
                # virtual-atom displacement.  Keeping it as a tensor input lets the
                # compiled graph stay reusable across frames.
                _saved = _patch_task_bufs(task_buf_vals)
                try:
                    return self.core_compute(
                        _prepare_coord_for_trace(extended_coord),
                        extended_atype,
                        nlist,
                        mapping=mapping,
                        fparam=fp,
                        aparam=ap,
                        charge_spin=charge_spin,
                        do_atomic_virial=do_atomic_virial,
                        extra_nlist_sort=self.need_sorted_nlist_for_lower(),
                        extended_coord_corr=extended_coord_corr,
                    )
                finally:
                    _restore_task_bufs(_saved)

        # Trace dims are pairwise-distinct primes >= 5 so ``make_fx`` neither
        # unifies two axes onto one symbol (duck-shape) nor specializes an axis
        # on a literal; ``next_safe_prime`` documents why.  The forbidden set
        # adds the model-contracted dims (``nsel``, fparam / aparam widths,
        # charge_spin) and the promoted task-buffer dims so the chosen primes
        # never collide with them.
        _forbidden: set[int] = {1, 2, 3, 9}
        for _tbv in task_buf_vals_trace:
            for _d in _tbv.shape:
                if _d > 1:
                    _forbidden.add(int(_d))
        # Model-contracted dims kept at their real values (changing them
        # would break the model's own assertions about ``sel``, fparam /
        # aparam widths, charge_spin dim).  Add to forbidden so primes
        # picked for free dims do not collide.
        _nsel_real = int(nlist.shape[2])
        _dim_fp = int(fp.shape[1])
        _dim_ap = int(ap.shape[2])
        _dim_cs = int(charge_spin.shape[1])
        for _d in (_nsel_real, _dim_fp, _dim_ap, _dim_cs):
            if _d > 1:
                _forbidden.add(_d)
        # Pick primes in physical order ``nf < nloc < nall``.  The order
        # ``trace_nloc < trace_nall`` matters: the model slices
        # ``extended_atype[:, :nloc]`` to get local atoms; if
        # ``trace_nloc > trace_nall`` the slice silently truncates at
        # trace time, breaking the captured symbolic shape relation
        # ``atype.shape[1] == nloc``.
        trace_nf = next_safe_prime(5, _forbidden)
        _forbidden.add(trace_nf)
        trace_nloc = next_safe_prime(trace_nf + 1, _forbidden)
        _forbidden.add(trace_nloc)
        trace_nall = next_safe_prime(trace_nloc + 1, _forbidden)

        # Build trace inputs by padding/trimming real-data tensors into the
        # chosen prime shapes; ``trace_pad_dim`` documents how index-bearing
        # tensors keep valid values.
        coord_for_trace = trace_pad_dim(extended_coord[:1], 0, trace_nf)
        coord_for_trace = trace_pad_dim(coord_for_trace, 1, trace_nall)
        atype_for_trace = trace_pad_dim(extended_atype[:1], 0, trace_nf)
        atype_for_trace = trace_pad_dim(atype_for_trace, 1, trace_nall)
        nlist_for_trace = trace_pad_dim(nlist[:1], 0, trace_nf)
        nlist_for_trace = trace_pad_dim(nlist_for_trace, 1, trace_nloc)
        # Real nlist values are in ``[-1, real_nall)`` (``-1`` marks
        # padded slots, non-negative entries index into extended_coord).
        # After trimming ``nall`` down to ``trace_nall`` some of those
        # values can exceed ``trace_nall``, which would produce
        # out-of-range gather indices in ``coord_flat.index_select(0,
        # src_ext)`` during the trace pass.  Clamp the upper bound to
        # ``trace_nall - 1`` (the ``-1`` padding stays untouched since
        # clamp only caps the high side).
        nlist_for_trace = torch.clamp(nlist_for_trace, max=trace_nall - 1)
        mapping_for_trace = trace_pad_dim(mapping[:1], 0, trace_nf)
        mapping_for_trace = trace_pad_dim(mapping_for_trace, 1, trace_nall)
        # Real mapping values are in ``[0, real_nloc)``.  If
        # ``trace_nloc < real_nloc`` they can exceed ``trace_nloc`` and
        # silently propagate into ``src_local`` (used as a local-atom
        # index downstream).  Clamp to ``trace_nloc - 1``.
        mapping_for_trace = torch.clamp(mapping_for_trace, min=0, max=trace_nloc - 1)
        fp_for_trace = trace_pad_dim(fp[:1], 0, trace_nf)
        ap_for_trace = trace_pad_dim(ap[:1], 0, trace_nf)
        ap_for_trace = trace_pad_dim(ap_for_trace, 1, trace_nloc)
        charge_spin_for_trace = trace_pad_dim(charge_spin[:1], 0, trace_nf)

        trace_args = [
            coord_for_trace,
            atype_for_trace,
            nlist_for_trace,
            mapping_for_trace,
            fp_for_trace,
            ap_for_trace,
            charge_spin_for_trace,
        ]
        if extended_coord_corr is not None:
            corr_for_trace = trace_pad_dim(extended_coord_corr[:1], 0, trace_nf)
            corr_for_trace = trace_pad_dim(corr_for_trace, 1, trace_nall)
            trace_args.append(corr_for_trace)
        # Append task-buffer values last so they map to the *task_buf_vals
        # varargs in compute_fn.  Their shapes are static (they don't vary
        # batch-to-batch), so passing the actual tensors is correct; make_fx
        # will create one placeholder per element.
        trace_args.extend(task_buf_vals_trace)

        # NOTE: Decompose ``silu_backward`` into primitive ops.
        # PyTorch ships forward and first-order backward for SiLU but no
        # symbolic higher-order derivative.  Without this decomposition
        # make_fx would emit ``aten.silu_backward.default`` opaquely
        # inside the first-derivative graph; when Inductor later has to
        # differentiate that op again for the optimiser step, it refuses
        # because silu_backward is not differentiable in its registered
        # form.  Lowering to ``sigmoid + pointwise mul + ...`` gives
        # every pointwise piece a well-defined higher derivative.
        decomp_table = get_decompositions([torch.ops.aten.silu_backward.default])

        # NOTE: ``tracing_mode="symbolic"`` makes every shape a sympy
        # symbol so the compiled graph can later accept any
        # (nframes, nall, n_edges, ...) at runtime.
        # ``_allow_non_fake_inputs=True`` lets us feed real tensors to
        # the trace -- the edge compactor contains data-dependent ops
        # (``torch.nonzero``, ``index_select``) that cannot execute on
        # FakeTensors, so we need concrete values to resolve their
        # control flow exactly once; shapes become symbolic immediately
        # afterwards.
        traced = make_fx(
            compute_fn,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            decomposition_table=decomp_table,
        )(*trace_args)

        if self.training:
            # Only the training trace runs with ``create_graph=True``, so only
            # it carries the autograd-inserted detach chains that the FX repair
            # targets; ``strip_saved_tensor_detach`` and ``rebuild_graph_module``
            # document why each step is required.
            #
            # Eval/inference must not take this path: ``create_graph=False``
            # inserts no detach chains to strip, and the eval graph carries
            # data-dependent ``nonzero`` sizes from sparse edge compaction whose
            # unbacked symbols fail Dynamo's shape-guard generation once the
            # graph is copied.  The original eval GraphModule is kept so Inductor
            # sees the traced metadata it needs.
            strip_saved_tensor_detach(traced)
            traced = rebuild_graph_module(traced)

        # The conservative Inductor option set that keeps the dynamic edge
        # graph lowerable is centralised in ``deepmd.pt.utils.compile_compat``.
        compile_options = build_inductor_compile_options()

        # NOTE: Store the compiled callable inside the plain-``dict``
        # cache ``compiled_core_compute_cache``.  The dict itself was installed
        # via ``object.__setattr__`` at __init__ time so that
        # ``nn.Module.__setattr__`` never saw any of this; mutating the
        # dict in place afterwards keeps the compile wrappers hidden
        # from parameter discovery (FSDP2/DDP would otherwise shard or
        # synchronise the wrapper's duplicated flat parameter views and
        # silently corrupt training).  The cache is keyed on
        # ``(training, do_atomic_virial, has_coord_corr)`` so that distinct
        # graph topologies coexist without evicting each other on every
        # ``model.eval()`` / ``model.train()`` switch.
        # NOTE: ``dynamic=True`` emits a single kernel per traced
        # shape symbol, so changes in ``nframes``, ``nall`` or edge
        # count do not trigger recompiles; and the option dict above
        # disables every Inductor/Triton feature that has ever
        # interacted badly with ``make_fx`` + double backward in
        # this project.
        if self.training:
            compiled = torch.compile(
                traced,
                backend="inductor",
                dynamic=True,
                options=compile_options,
            )
        else:
            # NOTE 13: Eval lowers the symbolic make_fx graph through
            # AOTAutograd's inference path rather than torch.compile.  Re-tracing
            # with Dynamo re-runs shape-guard production, which on the
            # forward-only graph leaves an intermediate view's extended-atom
            # (nall) axis without an input source and aborts ("sources must not
            # be empty for symbol s...").  AOTAutograd skips Dynamo but still
            # functionalizes the graph, letting Inductor reuse buffers (a plain
            # torch._inductor.compile does not, at ~3x peak memory).  no_grad
            # selects the forward-only partition; the fake placeholder values
            # keep the lowering dynamic.
            from torch._functorch.aot_autograd import (
                aot_module_simplified,
            )
            from torch._inductor import config as _ind_cfg
            from torch._inductor.compile_fx import (
                compile_fx_inner,
            )
            from torch._inductor.decomposition import (
                select_decomp_table,
            )

            example_inputs = [
                node.meta["val"]
                for node in traced.graph.nodes
                if node.op == "placeholder"
            ]

            # AOTAutograd's flat-output contract rejects the dict that
            # ``core_compute`` returns.  Rewrite the graph's output to a tuple
            # of the dict values, remember the keys, and re-pack the dict around
            # the compiled callable.  Insertion order makes keys and values line
            # up.
            _output_node = next(
                node for node in reversed(traced.graph.nodes) if node.op == "output"
            )
            _out_struct = _output_node.args[0]
            if isinstance(_out_struct, dict):
                _out_keys: list[str] | None = list(_out_struct.keys())
                _output_node.args = (tuple(_out_struct.values()),)
                traced.recompile()
            else:
                _out_keys = None

            def _inductor_inference_compiler(
                fx_gm: torch.fx.GraphModule, fx_inputs: list[Any]
            ) -> Any:
                # max_tiles=1 keeps pointwise grids 1D so the data-dependent
                # edge axis stays on Triton's x grid (limit 2**31); the default
                # tiling places it on the y/z grid (limit 65535), which
                # overflows for large systems.
                with _ind_cfg.patch({**compile_options, "triton.max_tiles": 1}):
                    return compile_fx_inner(fx_gm, fx_inputs)

            # select_decomp_table keeps the decomposition set aligned with
            # Inductor's fallback set (a mismatch raises an aten._to_copy
            # decomp/fallback clash).  The torch.device context pins the model's
            # device: AOTAutograd's PhiloxStateTracker allocates an RNG-state
            # tensor without an explicit device, which otherwise lands on a
            # stray default device and raises "invalid device ordinal".
            with torch.no_grad(), torch.device(extended_coord.device):
                _compiled_flat = aot_module_simplified(
                    traced,
                    example_inputs,
                    fw_compiler=_inductor_inference_compiler,
                    inference_compiler=_inductor_inference_compiler,
                    decompositions=select_decomp_table(),
                )

            if _out_keys is None:
                compiled = _compiled_flat
            else:
                _keys = _out_keys

                def compiled(*args: Any, _fn: Any = _compiled_flat) -> dict[str, Any]:
                    return dict(zip(_keys, _fn(*args)))

        # Populate both per-instance and module-level shared caches.
        # The shared cache (_SEZM_COMPILE_CACHE) lets a second task with the
        # same structure key skip re-tracing and re-compiling entirely.
        self.compiled_core_compute_cache[cache_key] = compiled
        self._task_buf_order_cache[cache_key] = task_buf_names
        _SEZM_COMPILE_CACHE[full_cache_key] = compiled
        _SEZM_TASK_BUF_ORDER[structure_key] = task_buf_names
        # NOTE: No dist.barrier() here.
        # The barrier premise is that all ranks reach trace_and_compile
        # simultaneously.  That is FALSE in several trainer code paths:
        #
        # 1. compute_or_load_stat (training.py:417) runs on rank 0 only.
        #    Rank 0 compiles → calls barrier → the other N-1 ranks are not
        #    inside trace_and_compile at that moment → deadlock.
        #
        # 2. Validation at disp_freq is rank-0-only inside the rank guard;
        #    if DP_COMPILE_INFER is set, same deadlock.
        #
        # Instead we rely on compilation being symmetric during the DDP
        # training loop itself: all ranks pick the same task per step (same
        # random seed), so they all hit trace_and_compile for the same task
        # at the same step.  The compile-time gap between ranks is on the
        # order of seconds while the NCCL default timeout is 30 minutes,
        # so no barrier is necessary for the training-loop case.
        # torch.compile is lazy; the "finished" log is emitted after the
        # first call triggers Inductor lowering (see forward_common).
        # ``pending_key`` pairs with ``pending_t0`` so the log is only
        # printed once, by the forward that actually triggers lowering
        # for *this* cache slot -- other slots may still be pending.
        self._core_compute_pending_compile_t0 = _compile_t0
        self._core_compute_pending_compile_key = cache_key

    def compile_dens(self) -> None:
        """Compile the direct-force `dens` path."""
        from torch._inductor import config as inductor_config

        log.info("SeZM: start compiling dens path")
        _compile_t0 = time.perf_counter()

        inductor_config.max_autotune_report_choices_stats = False
        inductor_config.autotune_num_choices_displayed = 0

        object.__setattr__(
            self,
            "compiled_dens_compute",
            torch.compile(
                self.core_compute_dens,
                backend="inductor",
                dynamic=True,
                options={
                    "max_autotune": False,
                    "epilogue_fusion": False,
                    "triton.cudagraphs": False,
                    "shape_padding": True,
                    "max_fusion_size": 64,
                },
            ),
        )
        self._dens_compiled = True
        # torch.compile is lazy; the "finished" log is emitted after the
        # first call triggers Inductor lowering (see forward_common).
        self._dens_pending_compile_t0 = _compile_t0

    def should_use_compile(self) -> bool:
        """Return whether the current forward should use the compile path."""
        if self.training:
            return self.use_compile
        return bool(self._env_use_compile_infer)

    # =========================================================================
    # Export Utilities
    # =========================================================================

    def _trace_lower_exportable(
        self,
        fn: Any,
        *sample_inputs: torch.Tensor | None,
    ) -> torch.nn.Module:
        """Trace a lower-interface closure into an exportable FX graph."""
        from torch._decomp import (
            get_decompositions,
        )

        return make_fx(
            fn,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            decomposition_table=get_decompositions(
                [torch.ops.aten.silu_backward.default]
            ),
        )(*sample_inputs)

    def forward_common_lower_exportable(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
        *,
        do_atomic_virial: bool = False,
    ) -> torch.nn.Module:
        """Trace ``forward_common_lower`` into an exportable FX ``GraphModule``.

        ``make_fx`` unfolds the inner ``autograd.grad`` that
        ``fit_output_to_model_output`` performs for force and virial, so
        the returned module can be handed to :func:`torch.export.export`
        directly.  ``silu_backward`` is decomposed to primitive ops so
        Inductor never sees an opaque higher-order derivative — the same
        decomposition the training compile path uses.

        Only the conservative ``ener`` mode is supported: ``dens``
        emits a direct-force tensor that has no ``DeepPotPTExpt`` consumer.
        """
        if self.get_active_mode() == "dens":
            raise NotImplementedError(
                "SeZM export supports only the conservative `ener` path."
            )

        model = self
        extra_sort = self.need_sorted_nlist_for_lower()

        def lower_fn(
            ext_coord: torch.Tensor,
            ext_atype: torch.Tensor,
            nlist_: torch.Tensor,
            mapping_: torch.Tensor | None,
            fparam_: torch.Tensor | None,
            aparam_: torch.Tensor | None,
            charge_spin_: torch.Tensor | None,
        ) -> dict[str, torch.Tensor]:
            # detach + requires_grad_ must live INSIDE the traced closure:
            # LAMMPS feeds a plain fp64 non-leaf tensor, and the exported
            # graph needs its own grad endpoint for the inner autograd.grad
            # that fit_output_to_model_output performs.
            ext_coord = ext_coord.detach().requires_grad_(True)
            return model.forward_common_lower(
                ext_coord,
                ext_atype,
                nlist_,
                mapping_,
                fparam=fparam_,
                aparam=aparam_,
                do_atomic_virial=do_atomic_virial,
                extra_nlist_sort=extra_sort,
                charge_spin=charge_spin_,
            )

        def fn(
            ext_coord: torch.Tensor,
            ext_atype: torch.Tensor,
            nlist_: torch.Tensor,
            mapping_: torch.Tensor | None,
            fparam_: torch.Tensor | None,
            aparam_: torch.Tensor | None,
            charge_spin_: torch.Tensor | None,
        ) -> dict[str, torch.Tensor]:
            return lower_fn(
                ext_coord,
                ext_atype,
                nlist_,
                mapping_,
                fparam_,
                aparam_,
                charge_spin_,
            )

        if self.get_dim_chg_spin() > 0:
            charge_spin = self.convert_charge_spin(
                charge_spin,
                nf=extended_atype.shape[0],
                dtype=extended_coord.dtype,
                device=extended_coord.device,
            )
        # Always include the charge_spin slot (possibly None) so the traced
        # module's forward signature matches the 7-tuple the freeze pipeline
        # passes at runtime, regardless of whether the model is conditioned.
        trace_inputs = (
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam,
            aparam,
            charge_spin,
        )

        return self._trace_lower_exportable(
            fn,
            *trace_inputs,
        )

    # =========================================================================
    # Neighbor List Construction
    # =========================================================================

    def use_self_built_nlist(self) -> bool:
        """Whether inference should keep this model's own neighbor-list path.

        SeZM builds an O(N) Toolkit-Ops neighbor list in
        :meth:`build_neighbor_list` and runs the compiled ``forward`` path, so
        the inference driver must not substitute an external ``NeighborList``
        strategy -- that would route through the eager lower interface and bypass
        the compiled graph.
        """
        return True

    def build_neighbor_list(
        self,
        coord: Float[Tensor, "nf nloc 3"] | Float[Tensor, "nf nloc_x3"],
        atype: Int[Tensor, "nf nloc"],
        box: Float[Tensor, "nf 9"] | None,
    ) -> tuple[
        Float[Tensor, "nf nall 3"],
        Int[Tensor, "nf nall"],
        Int[Tensor, "nf nall"],
        Int[Tensor, "nf nloc nsel"],
    ]:
        """
        Build extended inputs and the neighbor list for the ``forward`` entry.

        Used when the model constructs its own neighbor list from ``coord`` /
        ``box``, as opposed to ``forward_lower`` which receives an externally
        built nlist (e.g. from LAMMPS or an inference ``NeighborList`` strategy).
        Large CUDA systems use the Toolkit-Ops neighbor list
        (:class:`NvNeighborList`); all other cases use the dense all-pairs
        builder. The non-periodic Toolkit-Ops path uses a larger threshold
        because the dense builder is still faster at small sizes.

        Parameters
        ----------
        coord
            Coordinates with shape (nf, nloc, 3) in Å.
        atype
            Atom types with shape (nf, nloc).
        box
            Box tensor with shape (nf, 9) in Å, or None.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Extended coordinates, extended atom types, mapping, and neighbor list.
        """
        nloc = atype.shape[1]
        nv_threshold = (
            SEZM_NV_NLIST_THRESHOLD
            if box is not None
            else SEZM_NV_NONPERIODIC_NLIST_THRESHOLD
        )
        if coord.is_cuda and nloc >= nv_threshold and is_nv_available():
            # Large systems: the device-resident Toolkit-Ops neighbor list avoids
            # the dense all-pairs ghost expansion. It already keeps
            # the nearest sum(sel) neighbors (fixed width, like the standard
            # builder); only its (nlist, mapping) order is swapped to this
            # method's (mapping, nlist) contract.
            extended_coord, extended_atype, nlist, mapping = NvNeighborList().build(
                coord.view(atype.shape[0], nloc, 3),
                atype,
                box,
                self.get_rcut(),
                self.get_sel(),
            )
            return extended_coord, extended_atype, mapping, nlist
        return extend_input_and_build_neighbor_list(
            coord,
            atype,
            self.get_rcut(),
            self.get_sel(),
            mixed_types=True,
            box=box,
        )

    def build_edge_list_from_nlist(
        self,
        *,
        extended_coord: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build a compact edge list from DeePMD padded neighbor list.

        Edge vectors are computed via ``index_select`` on ``extended_coord``
        so they remain differentiable w.r.t. the input coordinates.  Two
        masked dummy edges are always appended to avoid data-dependent
        empty-edge branches that ``make_fx`` cannot trace and singular
        edge-axis guards in Inductor's batched matmul lowering.

        Parameters
        ----------
        extended_coord
            Extended coordinates with shape (nf, nall, 3).
        nlist
            DeePMD padded neighbor list with shape (nf, nloc, nsel).
        mapping
            Extended-to-local mapping with shape (nf, nall), or ``None``.

        Returns
        -------
        edge_index
            Edge indices with shape (2, E+2) where E is valid edge count.
        edge_vec
            Edge vectors with shape (E+2, 3).
        edge_mask
            Boolean mask with shape (E+2,).  The two trailing elements are ``False``.
        """
        nf, nloc, nsel = nlist.shape
        device = extended_coord.device
        nall = extended_coord.shape[1]
        descriptor_model = self.atomic_model.descriptor
        coord_for_diff = extended_coord.to(dtype=descriptor_model.compute_dtype)

        # === Step 1. Build per-edge geometry via index_select (differentiable) ===
        # NOTE: Edge vectors come from ``coord_flat.index_select(0, ...)``
        # rather than advanced indexing ``coord_flat[...]``.
        # ``index_select`` has an explicit, well-defined backward that
        # routes gradient cleanly back to the original extended
        # coordinate tensor.  Advanced indexing combined with make_fx
        # symbolic shapes has previously produced silent gradient
        # truncation in this project -- the second-derivative gradient
        # over coordinates was effectively zero, with no error raised.
        # ``torch.where(valid_flat, neighbor_flat, 0)`` sanitises padded
        # ``-1`` entries before indexing so we never hit an out-of-range
        # gather; the corresponding edges are filtered out below anyway.
        neighbor_flat = nlist.reshape(-1)
        # ``dst_actual = arange(N*K) // K`` produces the same value
        # sequence as ``arange(N).repeat_interleave(K)`` but its length
        # is derived from ``neighbor_flat.shape[0]`` -- a single symbolic
        # source shared with the ``torch.where`` below.  The previous
        # ``arange(nf*nloc).repeat_interleave(nsel)`` chain could
        # decouple from ``nlist.numel()`` in the FX graph if any
        # upstream code path ever specialized ``nloc`` at trace time;
        # deriving from ``neighbor_flat.shape[0]`` makes the equality
        # structural and survives any future change in trace-shape
        # selection in ``trace_and_compile``.
        dst_actual = (
            torch.arange(neighbor_flat.shape[0], device=device, dtype=torch.long)
            // nsel
        )
        f_idx = dst_actual // nloc
        dst_local = dst_actual % nloc
        valid_flat = neighbor_flat >= 0
        neighbor_safe = torch.where(
            valid_flat, neighbor_flat, torch.zeros_like(neighbor_flat)
        )
        # Gather coordinates within each frame instead of flattening
        # ``(nf, nall)`` into one index space.  The flattened form is
        # mathematically correct, but Inductor may lower
        # ``coord_flat.index_select(0, f_idx * nall + local_idx)`` to a kernel
        # that asserts the composite index against ``nall`` rather than
        # ``nf * nall`` when ``nf > 1``.  Frame-local gather keeps the same
        # differentiable path to coordinates while making every indirect index
        # visibly bounded by the atom axis.
        neighbor_safe_2d = neighbor_safe.to(dtype=torch.long).view(nf, nloc * nsel)
        nei_coord = torch.gather(
            coord_for_diff,
            1,
            neighbor_safe_2d.unsqueeze(-1).expand(-1, -1, 3),
        ).reshape(-1, 3)
        dst_coord = torch.gather(
            coord_for_diff[:, :nloc, :],
            1,
            dst_local.view(nf, -1).unsqueeze(-1).expand(-1, -1, 3),
        ).reshape(-1, 3)
        diff = nei_coord - dst_coord
        edge_len2 = torch.sum(diff * diff, dim=-1)

        # === Step 2. Build compact src/dst (local indices) ===
        if mapping is None:
            src_local = neighbor_safe.to(dtype=torch.long)
        else:
            src_local = torch.gather(mapping, 1, neighbor_safe_2d).reshape(-1)
        src_actual = f_idx * nloc + src_local.to(dtype=torch.long)

        # Filter: valid nlist entry AND src in [0, nloc) AND non-zero distance.
        src_local_valid = (src_local >= 0) & (src_local < nloc)
        len_positive = edge_len2 > 1e-10
        edge_mask_actual = valid_flat & src_local_valid & len_positive

        valid_idx = torch.nonzero(edge_mask_actual, as_tuple=False).flatten()

        # === Step 3. Compact edges + append masked dummies ===
        # NOTE: Always append two masked dummy edges.
        # ``torch.nonzero(edge_mask_actual)`` produces a data-dependent
        # number of valid edges, which can be zero on sparse or
        # single-type systems (e.g. isolated-atom reference frames).
        # make_fx cannot trace an ``if n_edges == 0: skip`` branch
        # symbolically; without the dummies it would fall back to
        # concrete shape specialisation and break
        # ``torch.compile(dynamic=True)`` for later batches.  Two dummy
        # slots also give Inductor's batched matmul lowering a static
        # ``E >= 2`` edge-axis bound, avoiding data-dependent layout
        # guards on ``E == 1`` that would otherwise trigger an extra
        # recompile when the first batch contains only a single edge.
        # Each dummy copies entry 0 (any in-range index is fine) and
        # carries ``edge_mask=False`` so every downstream sum, gather
        # or scatter ignores it.
        dummy_count = 2
        padded_idx = torch.cat(
            [valid_idx, torch.zeros(dummy_count, dtype=torch.long, device=device)]
        )
        src_sel = src_actual.index_select(0, padded_idx)
        dst_sel = dst_actual.index_select(0, padded_idx)
        edge_vec_sel = diff.index_select(0, padded_idx)
        edge_index = torch.stack([src_sel, dst_sel], dim=0)
        edge_mask = torch.cat(
            [
                torch.ones(valid_idx.shape[0], dtype=torch.bool, device=device),
                torch.zeros(dummy_count, dtype=torch.bool, device=device),
            ]
        )
        return edge_index, edge_vec_sel, edge_mask

    # =========================================================================
    # Input Canonicalization
    # =========================================================================

    def convert_fp_ap(
        self,
        fp: torch.Tensor | None,
        ap: torch.Tensor | None,
        nf: int,
        nloc: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert optional fitting inputs to tensor-only compile inputs."""
        dim_fparam = self.get_dim_fparam()
        dim_aparam = self.get_dim_aparam()

        # === Step 1. Canonicalize frame parameters ===
        if dim_fparam == 0:
            fp = torch.empty((nf, 0), dtype=dtype, device=device)
        elif fp is None:
            default_fparam = self.get_default_fparam()
            if default_fparam is None:
                raise ValueError(
                    "fparam is required because fitting net dim_fparam > 0"
                )
            fp = default_fparam.to(device=device, dtype=dtype).view(1, dim_fparam)
            fp = fp.expand(nf, -1)
        else:
            if fp.numel() != nf * dim_fparam:
                raise ValueError(
                    f"input fparam: cannot reshape {list(fp.shape)} "
                    f"into ({nf}, {dim_fparam})."
                )
            fp = fp.to(device=device, dtype=dtype).view(nf, dim_fparam)

        # === Step 2. Canonicalize atomic parameters ===
        if dim_aparam == 0:
            ap = torch.empty((nf, nloc, 0), dtype=dtype, device=device)
        elif ap is None:
            if dim_aparam > 0:
                raise ValueError(
                    "aparam is required because fitting net dim_aparam > 0"
                )
        else:
            if ap.numel() != nf * nloc * dim_aparam:
                raise ValueError(
                    f"input aparam: cannot reshape {list(ap.shape)} "
                    f"into ({nf}, {nloc}, {dim_aparam})."
                )
            ap = ap.to(device=device, dtype=dtype).view(nf, nloc, dim_aparam)

        return fp, ap

    def convert_charge_spin(
        self,
        charge_spin: torch.Tensor | None,
        nf: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Canonicalize optional charge/spin conditions for internal compute paths.

        Parameters
        ----------
        charge_spin
            Optional frame-level charge and spin conditions.
        nf
            Number of frames.
        dtype
            Target floating-point dtype.
        device
            Target device.

        Returns
        -------
        torch.Tensor
            Tensor with shape `(nf, 2)` when enabled, otherwise `(nf, 0)`.
        """
        dim_chg_spin = self.atomic_model.get_dim_chg_spin()
        if dim_chg_spin == 0:
            return torch.empty((nf, 0), dtype=dtype, device=device)

        if charge_spin is None:
            default_chg_spin = self.atomic_model.get_default_chg_spin()
            if default_chg_spin is None:
                raise ValueError("charge_spin is required for this SeZM model")
            charge_spin = default_chg_spin.to(device=device, dtype=dtype).view(1, 2)
        else:
            charge_spin = charge_spin.to(device=device, dtype=dtype)

        if charge_spin.ndim == 1:
            if charge_spin.numel() != dim_chg_spin:
                raise ValueError("charge_spin must contain [charge, spin]")
            charge_spin = charge_spin.view(1, dim_chg_spin)
        elif charge_spin.ndim != 2 or charge_spin.shape[-1] != dim_chg_spin:
            raise ValueError("charge_spin must have shape (nf, 2)")

        if charge_spin.shape[0] == 1 and nf != 1:
            charge_spin = charge_spin.expand(nf, -1)
        elif charge_spin.shape[0] != nf:
            raise ValueError("charge_spin first dimension must match nframes")
        return charge_spin

    def canonicalize_dens_inputs(
        self,
        force_input: torch.Tensor | None,
        noise_mask: torch.Tensor | None,
        nf: int,
        nloc: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Canonicalize optional public `dens` inputs to concrete tensors.

        Parameters
        ----------
        force_input
            Optional atom-wise force input tensor.
        noise_mask
            Optional atom-wise corruption mask.
        nf
            Number of frames.
        nloc
            Number of local atoms per frame.
        dtype
            Target floating-point dtype.
        device
            Target device.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Canonicalized force tensor with shape `(nf, nloc, 3)` and mask with
            shape `(nf, nloc)`.

        Notes
        -----
        `force_input` and `noise_mask` remain optional only at the outer model
        API. Internal `dens` compute functions always receive concrete tensors.
        """
        if force_input is None:
            force_input = torch.zeros((nf, nloc, 3), dtype=dtype, device=device)
        else:
            if force_input.ndim == 2:
                force_input = force_input.view(nf, nloc, 3)
            elif force_input.ndim != 3:
                raise ValueError(
                    "`force_input` must have shape (nf, nloc, 3) or (nf, nloc*3)."
                )
            force_input = force_input.to(device=device, dtype=dtype)

        if noise_mask is None:
            noise_mask = torch.zeros((nf, nloc), dtype=torch.bool, device=device)
        else:
            if noise_mask.ndim != 2:
                raise ValueError("`noise_mask` must have shape (nf, nloc).")
            noise_mask = noise_mask.to(device=device, dtype=torch.bool)

        return force_input, noise_mask

    # =========================================================================
    # Output Post-Processing
    # =========================================================================

    def post_process_output_dens(
        self,
        compute_ret: torch.Tensor,
        atype: torch.Tensor,
        *,
        noise_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Convert the concatenated `dens` output to DeePMD model outputs.

        Parameters
        ----------
        compute_ret
            Concatenated tensor with shape `(nf, nloc, 7)` or `(1, n_node, 7)`.
        atype
            Local atom types with shape `(nf, nloc)`.
        noise_mask
            Corruption mask with shape `(nf, nloc)`.

        Returns
        -------
        dict[str, torch.Tensor]
            Standard DeePMD model predictions for `dens` mode.
        """
        nf, nloc = atype.shape[:2]
        n_actual = nf * nloc
        dens_ret = {
            "energy": compute_ret[:, :n_actual, 0:1].view(nf, nloc, 1),
            "clean_dforce": compute_ret[:, :n_actual, 1:4].view(nf, nloc, 3),
            "denoising_dforce": compute_ret[:, :n_actual, 4:7].view(nf, nloc, 3),
        }
        return self.atomic_model.apply_out_stat_dens(
            dens_ret,
            atype,
            noise_mask=noise_mask,
            energy_redu_dtype=self.redu_prec,
        )

    # =========================================================================
    # Metadata
    # =========================================================================

    def has_chg_spin_ebd(self) -> bool:
        """Return whether charge/spin condition embedding is enabled."""
        return self.atomic_model.has_chg_spin_ebd()

    def get_dim_chg_spin(self) -> int:
        """Return charge/spin condition width."""
        return self.atomic_model.get_dim_chg_spin()

    def has_default_chg_spin(self) -> bool:
        """Return whether default charge/spin conditions are configured."""
        return self.atomic_model.has_default_chg_spin()

    def get_default_chg_spin(self) -> torch.Tensor | None:
        """Return default charge/spin conditions as a tensor."""
        return self.atomic_model.get_default_chg_spin()

    def has_message_passing(self) -> bool:
        """Return whether the descriptor performs message passing."""
        return self.atomic_model.has_message_passing()

    # =========================================================================
    # Mode Management
    # =========================================================================

    def get_active_mode(self) -> str:
        """Return the current SeZM execution mode."""
        return self.atomic_model.get_active_mode()

    def set_active_mode(self, mode: str) -> None:
        """
        Switch the active SeZM execution mode.

        Parameters
        ----------
        mode
            Target mode. Must be `ener` or `dens`.
        """
        self.atomic_model.set_active_mode(mode)

    def set_active_mode_from_loss(self, loss_type: str) -> None:
        """
        Select the active SeZM path from `loss.type`.

        Parameters
        ----------
        loss_type
            Loss type name.
        """
        normalized = str(loss_type).lower()
        if normalized in {"ener", "dens"}:
            self.set_active_mode(normalized)

    def reset_head_for_mode(self, mode: str) -> None:
        """
        Reinitialize one SeZM fitting head and reset mode-specific compile state.

        Parameters
        ----------
        mode
            Target mode to reset.
        """
        self.atomic_model.reset_head_for_mode(mode)
        if mode == "dens":
            self._dens_compiled = False
            self._dens_pending_compile_t0 = None
            object.__setattr__(self, "compiled_dens_compute", None)
        else:
            self._core_compute_pending_compile_t0 = None
            self._core_compute_pending_compile_key = None
            # Drop every compile slot so the next forward retraces against the
            # reinitialised fitting head.
            self.compiled_core_compute_cache.clear()

    # =========================================================================
    # Bridging Helpers
    # =========================================================================

    def _get_inter_potential_real_type_count(self) -> int:
        """Return the real-type count used to mask analytical pair potentials."""
        return len(self.get_type_map())

    # =========================================================================
    # Type and Output Metadata
    # =========================================================================

    def translated_output_def(self) -> dict[str, Any]:
        """
        Translate model output definition to a dictionary format.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping output names to their corresponding output definitions.
        """
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": out_def_data["energy"],
            "energy": out_def_data["energy_redu"],
        }
        if "dforce" in out_def_data:
            output_def["force"] = out_def_data["dforce"]
        elif self.do_grad_r("energy"):
            output_def["force"] = out_def_data["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy"):
            output_def["virial"] = out_def_data["energy_derv_c_redu"].squeeze(-2)
            output_def["atom_virial"] = out_def_data["energy_derv_c"].squeeze(-2)
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]

        return output_def

    def get_observed_type_list(self) -> list[str]:
        """
        Get observed types (elements) of the model during data statistics.

        Returns
        -------
        list[str]
            A list of the observed types in this model.
        """
        type_map = self.get_type_map()
        out_bias = self.atomic_model.get_out_bias()[0]

        assert out_bias is not None, "No out_bias found in the model."
        assert out_bias.dim() == 2, "The supported out_bias should be a 2D tensor."
        assert out_bias.size(0) == len(type_map), (
            "The out_bias shape does not match the type_map length."
        )
        bias_mask = (
            torch.gt(torch.abs(out_bias), 1e-6).any(dim=-1).detach().cpu()
        )  # 1e-6 for stability

        # TorchScript does not support list comprehension with if clause
        result: list[str] = []
        for t, m in zip(type_map, bias_mask.tolist()):
            if m:
                result.append(t)
        return result

    # =========================================================================
    # Serialization
    # =========================================================================

    def serialize(self) -> dict[str, Any]:
        """
        Serialize the SeZM model including model-level bridging state.

        Returns
        -------
        dict[str, Any]
            Serialized SeZM model data.
        """
        return {
            "@class": "Model",
            "@version": 1,
            "type": self.model_type,
            "atomic_model": self.atomic_model.serialize(),
            "bridging_method": self.bridging_method,
            "bridging_r_inner": self.bridging_r_inner,
            "bridging_r_outer": self.bridging_r_outer,
            "lora": self.lora_config,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SeZMModel:
        """
        Deserialize the SeZM model including model-level bridging state.

        Parameters
        ----------
        data
            Serialized SeZM model data.

        Returns
        -------
        SeZMModel
            Deserialized SeZM model.
        """
        data = data.copy()
        version = int(data.pop("@version", 1))
        check_version_compatibility(version, 1, 1)
        data.pop("@class", None)
        data.pop("type", None)
        atomic_model = SeZMAtomicModel.deserialize(data.pop("atomic_model"))
        return cls(atomic_model_=atomic_model, **data)

    # =========================================================================
    # Context Managers
    # =========================================================================

    @contextmanager
    def tf32_precision_ctx(self) -> Generator[None, None, None]:
        """Context manager to temporarily set TF32 matmul precision.

        Training follows ``enable_tf32``. Eval/inference follows
        ``DP_TF32_INFER``: 0 keeps ``highest`` precision, 1 selects
        ``high``, and 2 selects ``medium``.
        """
        if not torch.cuda.is_available():
            yield
            return
        prev_precision = torch.get_float32_matmul_precision()
        try:
            if self.training:
                precision = "high" if self.enable_tf32 else "highest"
            else:
                precision = self._tf32_infer_precision
            torch.set_float32_matmul_precision(precision)
            yield
        finally:
            torch.set_float32_matmul_precision(prev_precision)


# =============================================================================
# InterPotential: analytical pair potentials for bridging
# =============================================================================

# fmt: off
ELEMENT_TO_Z: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
    "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99,
    "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105,
    "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111,
    "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117,
    "Og": 118,
}
# fmt: on

# ZBL screening function coefficients
_ZBL_A_COEFF = (0.18175, 0.50986, 0.28022, 0.028171)
_ZBL_B_COEFF = (3.1998, 0.94229, 0.4029, 0.20162)

# Physical constants
_KE_EV_A = 14.3996  # Coulomb constant in eV·Å
_A_BOHR = 0.5291772109  # Bohr radius in Å


class InterPotential(torch.nn.Module):
    """
    Analytical pair potential module for Zone bridging.

    Supports the Ziegler-Biersack-Littmark (ZBL) screened nuclear repulsion
    potential. Designed to be extensible to other analytical forms (LJ, Morse,
    etc.) through the ``mode`` parameter.

    Each pair (i, j) contributes ``V_ZBL(r_ij) / 2`` to both atom i and atom j,
    avoiding double-counting from the symmetric neighbor list.

    Parameters
    ----------
    type_map : list[str]
        Element symbols (e.g. ``["O", "H"]``). Index in this list corresponds
        to the ``atype`` integer values.
    mode : str
        Potential formula. Currently only ``"zbl"`` is supported.

    Raises
    ------
    ValueError
        If ``mode`` is not recognized, or if any element in ``type_map`` is
        not found in the periodic table.
    """

    def __init__(self, type_map: list[str], mode: str = "zbl") -> None:
        super().__init__()
        mode = mode.upper()
        if mode != "ZBL":
            raise ValueError(f"Unknown InterPotential mode: {mode}")
        self.mode = mode

        atomic_numbers = []
        for elem in type_map:
            z = ELEMENT_TO_Z.get(elem)
            if z is None:
                raise ValueError(f"Unknown element symbol: {elem}")
            atomic_numbers.append(z)
        self.register_buffer(
            "atomic_numbers",
            torch.tensor(atomic_numbers, dtype=torch.float64, device=env.DEVICE),
        )

    def _zbl_pair_energy(
        self,
        r: torch.Tensor,
        zi: torch.Tensor,
        zj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ZBL pair energy for given distances and nuclear charges.

        Parameters
        ----------
        r : torch.Tensor
            Pair distances with shape (...) in Å.
        zi : torch.Tensor
            Nuclear charge of atom i with shape (...).
        zj : torch.Tensor
            Nuclear charge of atom j with shape (...).

        Returns
        -------
        torch.Tensor
            Pair energies with shape (...) in eV.
        """
        a_screen = 0.88534 * _A_BOHR / (zi.pow(0.23) + zj.pow(0.23))
        x = r / a_screen
        phi = sum(a * torch.exp(-b * x) for a, b in zip(_ZBL_A_COEFF, _ZBL_B_COEFF))
        return _KE_EV_A * zi * zj / r * phi

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        nloc: int,
        real_type_count: int | None = None,
    ) -> torch.Tensor:
        """
        Compute per-atom pair energy from the standard neighbor list path.

        Parameters
        ----------
        extended_coord
            Coordinates in extended region with shape (nf, nall, 3) in Å.
        extended_atype
            Atom types in extended region with shape (nf, nall).
        nlist
            Neighbor list with shape (nf, nloc, nsel).
        nloc : int
            Number of local atoms.
        real_type_count
            Number of real atom types. Types with index greater than or equal to
            this value are virtual spin types and are masked out of the
            analytical potential. If omitted, all configured types are real.

        Returns
        -------
        torch.Tensor
            Per-atom pair energy with shape (nf, nloc, 1) in eV.
        """
        if real_type_count is None:
            real_type_count = int(self.atomic_numbers.numel())
        nf = extended_coord.shape[0]
        coord64 = extended_coord.to(dtype=torch.float64)
        atype_for_z = extended_atype.clamp(min=0)
        atype_for_z = torch.where(
            atype_for_z >= real_type_count,
            atype_for_z - real_type_count,
            atype_for_z,
        )
        z_all = self.atomic_numbers[atype_for_z]  # (nf, nall)

        # === Step 1. Gather neighbor coordinates and types ===
        nsel = nlist.shape[2]
        nlist_clamp = nlist.clamp(min=0)  # (nf, nloc, nsel)
        nei_coord = torch.gather(
            coord64, 1, nlist_clamp.unsqueeze(-1).expand(-1, -1, -1, 3).view(nf, -1, 3)
        ).view(nf, nloc, nsel, 3)
        atom_coord = coord64[:, :nloc].unsqueeze(2)  # (nf, nloc, 1, 3)
        diff = nei_coord - atom_coord  # (nf, nloc, nsel, 3)
        r = diff.norm(dim=-1).clamp(min=1e-10)  # (nf, nloc, nsel)

        zi = z_all[:, :nloc].unsqueeze(2).expand_as(r)  # (nf, nloc, nsel)
        zj_idx = nlist_clamp
        zj = torch.gather(z_all, 1, zj_idx.view(nf, -1)).view(nf, nloc, nsel)

        # === Step 2. Compute pair energies ===
        pair_e = self._zbl_pair_energy(r, zi, zj)  # (nf, nloc, nsel)

        # Mask padding entries (nlist == -1)
        valid = (nlist >= 0).to(dtype=pair_e.dtype)
        center_is_real = (extended_atype[:, :nloc] < real_type_count).unsqueeze(2)
        neighbor_atype = torch.gather(extended_atype, 1, nlist_clamp.view(nf, -1)).view(
            nf, nloc, nsel
        )
        neighbor_is_real = neighbor_atype < real_type_count
        valid = valid * (center_is_real & neighbor_is_real).to(dtype=pair_e.dtype)
        pair_e = pair_e * valid

        # Half contribution to avoid double-counting
        atom_pair_energy = (pair_e * 0.5).sum(dim=-1, keepdim=True)  # (nf, nloc, 1)
        return atom_pair_energy.to(dtype=extended_coord.dtype)

    def forward_from_edges(
        self,
        edge_vec: torch.Tensor,
        edge_index: torch.Tensor,
        atype_flat: torch.Tensor,
        edge_mask: torch.Tensor,
        n_node: int,
    ) -> torch.Tensor:
        """
        Compute per-atom pair energy from the compile-path edge list.

        Parameters
        ----------
        edge_vec
            Edge vectors with shape (E, 3) in Å.
        edge_index
            Edge source/destination indices with shape (2, E).
        atype_flat
            Flat atom types with shape (N,).
        edge_mask
            Boolean mask with shape (E,). True means valid edge.
        n_node : int
            Number of flattened local nodes.

        Returns
        -------
        torch.Tensor
            Per-atom pair energy with shape (1, N, 1) in eV.
        """
        src = edge_index[0].to(dtype=torch.long)
        dst = edge_index[1].to(dtype=torch.long)

        r = edge_vec.to(dtype=torch.float64).norm(dim=-1).clamp(min=1e-10)  # (E,)
        z_all = self.atomic_numbers[atype_flat.clamp(min=0)]  # (N,)
        zi = z_all[src]  # (E,)
        zj = z_all[dst]  # (E,)

        pair_e = self._zbl_pair_energy(r, zi, zj)  # (E,)
        pair_e = pair_e * edge_mask.to(dtype=pair_e.dtype)

        # Half contribution to each destination atom
        atom_energy = torch.zeros(n_node, dtype=pair_e.dtype, device=pair_e.device)
        atom_energy.index_add_(0, dst, pair_e * 0.5)

        return atom_energy.to(dtype=edge_vec.dtype).view(1, n_node, 1)
