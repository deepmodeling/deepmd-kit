# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared masked per-frame reduction idioms for the loss modules.

These helpers factor out the three per-frame reduction patterns that the
mixed_type padding mask (PR #5738) introduced into every loss term (issue
#5768). They are written with ``array_api_compat`` so both the dpmodel
(numpy/jax/...) loss backend and the PyTorch loss backend can call them: the
PyTorch backend passes torch tensors and ``array_api_compat`` dispatches to the
torch namespace, preserving autograd and producing bit-identical results to the
previous hand-inlined torch code.

Each helper implements ONLY the masked branch. Callers keep the original
non-masked expression in the ``else`` branch verbatim, so the "bit-identical
for non-mixed batches" guarantee from PR #5738 is preserved (defaulting the
mask to all-ones would change the reduction order at the ULP level and is
deliberately NOT done here).
"""

from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)


def masked_atom_mean(elem: Array, maskf: Array, ncomp: int) -> Array:
    """Idiom 1: per-atom masked mean over ``ncomp`` components, averaged over frames.

    Parameters
    ----------
    elem : Array
        Non-negative per-element contribution of shape ``[nf, nloc, ncomp]``
        (already squared or abs'd, and pre-multiplied by any per-atom weight
        such as ``atom_pref``). NOT yet multiplied by the mask.
    maskf : Array
        Per-atom real/ghost mask of shape ``[nf, nloc]`` (1.0 real, 0.0 ghost).
    ncomp : int
        Number of components per atom (force: 3, atom energy: 1,
        dos: ``numb_dos``, tensor: ``tensor_size``).

    Returns
    -------
    Array
        ``mean_over_frames( sum(elem * mask) / (real_natoms * ncomp) )``.
        An all-padding frame (zero real atoms) contributes a neutral ``0``
        instead of ``0/0 = NaN``.
    """
    xp = array_api_compat.array_namespace(elem, maskf)
    nf = elem.shape[0]
    masked = elem * maskf[:, :, None]
    per_frame_sum = xp.sum(xp.reshape(masked, (nf, -1)), axis=-1)
    per_frame_dof = xp.sum(maskf, axis=-1) * ncomp
    # An all-padding frame has zero real atoms, so ``per_frame_dof`` is 0 and
    # the ratio would be 0/0 = NaN -- poisoning the frame mean and, under
    # autograd, its gradient. Divide by a safe denominator and map those frames
    # to a neutral per-frame value of 0. Frames with real atoms are untouched,
    # preserving the bit-identical guarantee.
    has_dof = per_frame_dof > 0
    safe_dof = xp.where(has_dof, per_frame_dof, xp.ones_like(per_frame_dof))
    per_frame = xp.where(
        has_dof, per_frame_sum / safe_dof, xp.zeros_like(per_frame_sum)
    )
    return xp.mean(per_frame)


def per_frame_component_mean(err: Array) -> Array:
    """Idiom 2 primitive: per-frame mean over the flattened component axis.

    Parameters
    ----------
    err : Array
        Per-frame error term of shape ``[nf, k]`` (already squared or abs'd).

    Returns
    -------
    Array
        Shape ``[nf]``: the mean over components for each frame. Callers apply
        the extensive ``inv**exp`` weighting for both the loss term and the
        RMSE display (which use different exponents), so ``err`` is reduced
        once here and reused.
    """
    xp = array_api_compat.array_namespace(err)
    nf = err.shape[0]
    return xp.mean(xp.reshape(err, (nf, -1)), axis=-1)


def masked_atom_num(mask: Array | None, natoms: Any, dtype: Any) -> Any:
    """Idiom 3 companion: the display-only divisor for already-reduced globals.

    The global loss itself is a plain ``mean`` regardless of masking (global
    quantities are padding-invariant); only the reported RMSE is divided by an
    atom count. This returns that divisor.

    Parameters
    ----------
    mask : Array or None
        Per-atom mask of shape ``[nf, nloc]``, or ``None`` when not mixed_type.
    natoms
        Fallback atom count used when ``mask`` is ``None``.
    dtype
        Target dtype for the summed atom count (each backend passes the dtype
        it currently uses: the diff's dtype for dpmodel, float32 for pt).

    Returns
    -------
    Array or int
        ``mean_over_frames(astype(sum(mask, axis=-1), dtype))`` when ``mask``
        is given, else ``natoms``.
    """
    if mask is None:
        return natoms
    xp = array_api_compat.array_namespace(mask)
    return xp.mean(xp.astype(xp.sum(mask, axis=-1), dtype))
