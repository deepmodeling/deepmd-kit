# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared infrastructure for the cuTile inference kernels.

This module holds what every cuTile kernel in :mod:`deepmd.pt_expt.kernels.cutile`
needs: the availability probe, the array annotation that keeps edge-scaled
indexing in 64 bits, the split-compensated fp16 representation of an fp32
operand, the launch-hint cache, and the source generator the tile model forces
on any kernel whose block structure is not a power of two.

Working notes for kernel authors
--------------------------------
The following properties of ``cuda.tile`` are not obvious from its reference
documentation and each one cost a measurable amount of performance or a wrong
result before it was understood.

``mma`` in fp32 is not usable.
    On ``sm_120`` the fp32 tensor-free multiply-accumulate lowers to separate
    ``FMUL`` and ``FADD`` instructions -- a disassembled square GEMM contains no
    ``FFMA`` at all -- and reaches roughly 15 TFLOPS against 74 for cuBLAS and 68
    for Triton. The fp16 tensor-core path runs at the hardware rate. Every
    contraction here is therefore evaluated in fp16 with split compensation
    (:func:`split_fp16`), which recovers fp32 accuracy at three tensor-core
    products per fp32 product.

Prefer few large ``mma`` calls over many exact small ones.
    The block-diagonal structure invites a contraction split into ``Cf``-wide
    blocks, which is exact and needs no padding. Measured, that formulation is
    three to seven times slower than padding the degree count to a power of two
    and issuing one wide contraction: the per-call weight load and its latency
    dominate below roughly 10^5 multiply-adds per call, whatever the arithmetic
    saving. Pad the output axis; skip the padded slabs on the contraction axis,
    where the weight rows are exact zeros and skipping is free.

Kernel variants must be cached.
    ``kernel.replace_hints()`` returns an object with its own JIT cache.
    Constructing one inside a launch wrapper costs a cache miss on every call and
    was measured at fifteen to twenty times the kernel's own runtime. Use
    :func:`kernel_variant`.

The ``occupancy`` hint matters and the others generally do not.
    Left to itself the compiler will spend the entire shared-memory budget on one
    block. ``occupancy=2`` was worth 1.4x on the mixing-stack forward.
    ``num_worker_warps`` was never better than the automatic choice and is
    frequently much worse, and thread-block clusters (``num_ctas``) were seven to
    sixty times slower on every kernel tried here.

Loop bounds must share one integer type.
    A ``range`` whose bounds come from an int64 array and whose step is an int32
    literal fails verification in the tile compiler. CSR offset arrays passed to
    a kernel are therefore int32; the edge counts they index stay well inside
    that range, while the *element* offsets derived from them do not, which is
    what :data:`BigArray` is for.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

try:
    import cuda.tile as ct

    CUTILE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without cuda.tile
    CUTILE_AVAILABLE = False

__all__ = [
    "CUTILE_AVAILABLE",
    "TAIL_SCALE",
    "generated_module",
    "kernel_variant",
    "next_pow2",
    "split_fp16",
]

#: Power-of-two scale carried by the tail of a split fp32 operand. An unscaled
#: tail is ``x * 2^-11``, which is subnormal in fp16 for any element below 0.125
#: and flushes to zero below 1.2e-4; the element then silently degrades to plain
#: fp16 accuracy. Only the tail is scaled, so the head represents the operand
#: unmodified and the representation stays valid up to the fp16 maximum.
TAIL_SCALE = 2048.0

if CUTILE_AVAILABLE:
    #: Element offsets on edge-scaled arrays pass 2^31 near 10^7 edges, which is
    #: within the production range at molecular-dynamics scale.
    BigArray = Annotated[ct.Array, ct.ArrayAnnotation(index_dtype=ct.int64)]
else:  # pragma: no cover - exercised only without cuda.tile
    BigArray = Any


def next_pow2(value: int) -> int:
    """Return the smallest power of two greater than or equal to ``value``."""
    return 1 << (value - 1).bit_length()


def split_fp16(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an fp32 tensor into an fp16 head and its ``TAIL_SCALE``-scaled tail.

    Parameters
    ----------
    tensor : torch.Tensor
        Operand in fp32.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The fp16 head ``fp16(x)`` and the fp16 tail ``(x - fp16(x)) * TAIL_SCALE``,
        both contiguous. Reconstructing ``head + tail / TAIL_SCALE`` recovers the
        operand to about 2^-22 relative.

    Notes
    -----
    The narrowing round trip must happen where a compiler cannot see through it.
    Expressed as tracer-visible tensor operations, Inductor is free to keep the
    head in fp32 and elide the rounding, which makes the tail identically zero
    and silently degrades the contraction to plain fp16. Callers therefore invoke
    this from inside an opaque operator body, never from the traced graph.
    """
    head = tensor.half()
    tail = ((tensor - head.float()) * TAIL_SCALE).half()
    return head.contiguous(), tail.contiguous()


_VARIANTS: dict[tuple[int, tuple[tuple[str, int], ...]], Any] = {}


def kernel_variant(kernel: Any, **hints: int) -> Any:
    """Return the cached variant of ``kernel`` carrying the given compiler hints.

    Parameters
    ----------
    kernel
        A ``cuda.tile`` kernel object.
    **hints
        Compiler hints accepted by ``cuda.tile.kernel``, typically ``occupancy``.

    Returns
    -------
    Any
        The hinted kernel, or ``kernel`` itself when no hint is given.

    Notes
    -----
    ``replace_hints`` produces an object with its own JIT cache, so a variant
    built per launch misses that cache on every call.
    """
    if not hints:
        return kernel
    key = (id(kernel), tuple(sorted(hints.items())))
    variant = _VARIANTS.get(key)
    if variant is None:
        variant = kernel.replace_hints(**hints)
        _VARIANTS[key] = variant
    return variant


def _cache_dir() -> str:
    """Return the directory holding generated kernel modules."""
    override = os.environ.get("DP_CUTILE_CACHE_DIR")
    if override:
        return override
    return os.path.join(tempfile.gettempdir(), f"deepmd_cutile_{os.getuid()}")


_MODULES: dict[str, Any] = {}


def generated_module(stem: str, source: str) -> Any:
    """Write generated kernel source to the cache directory and import it.

    Parameters
    ----------
    stem : str
        Configuration-identifying prefix of the module file name.
    source : str
        Complete module source.

    Returns
    -------
    Any
        The imported module.

    Notes
    -----
    The tile compiler reads a kernel from its Python source, so generated
    kernels must exist as importable files rather than as objects built by
    ``exec``. The file name carries a digest of the source, which makes the cache
    self-invalidating when a generator changes and lets concurrent processes
    share compiled artifacts. ``DP_CUTILE_CACHE_DIR`` overrides the location.
    """
    digest = hashlib.sha1(source.encode()).hexdigest()[:12]
    name = f"{stem}_{digest}"
    module = _MODULES.get(name)
    if module is not None:
        return module
    directory = _cache_dir()
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.py")
    if not os.path.exists(path):
        temporary = f"{path}.{os.getpid()}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(source)
        os.replace(temporary, path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _MODULES[name] = module
    return module


class Emitter:
    """Collect generated kernel statements at a fixed indentation.

    The tile model has no list type and requires every tile extent to be a power
    of two, so a kernel that keeps one tile per spherical-harmonic degree cannot
    be written generically: the degrees can neither live in an indexable
    container nor become a tile axis. Emitting one named tile per degree removes
    both limits and leaves the arithmetic exact.
    """

    def __init__(self, indent: str = "    ") -> None:
        self.lines: list[str] = []
        self.indent = indent

    def __call__(self, statement: str = "") -> None:
        self.lines.append(self.indent + statement if statement else "")

    def extend(self, statements: Sequence[str]) -> None:
        for statement in statements:
            self(statement)

    def concat(self, blocks: Sequence[str], width: int, target: str, tag: str) -> None:
        """Emit a balanced concatenation of ``blocks`` into a ``width``-block tile.

        ``ct.cat`` takes exactly two operands of equal shape, so a wide tile is
        assembled as a binary tree over power-of-two widths. Positions past the
        real block count are filled with exact zeros, which is what allows the
        padded contraction to equal the unpadded one.
        """
        items = list(blocks)
        for index in range(width - len(blocks)):
            self(f"_pad_{tag}{index} = ct.zeros((BE, CF), dtype=ct.float32)")
            items.append(f"_pad_{tag}{index}")
        level = 0
        while len(items) > 1:
            merged = []
            for position in range(0, len(items), 2):
                name = f"_cat_{tag}{level}_{position // 2}"
                self(f"{name} = ct.cat(({items[position]}, {items[position + 1]}), 1)")
                merged.append(name)
            items, level = merged, level + 1
        self(f"{target} = {items[0]}")

    def render(self, header: str, signature: Sequence[str]) -> str:
        """Return the complete module source."""
        return "\n".join([header, *signature, *self.lines]) + "\n"
