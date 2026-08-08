# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared masked reduction idioms for the loss modules.

These helpers factor out the reduction patterns every loss term needs once a
per-atom mask marks which rows of a batch are real. They are written with
``array_api_compat`` so both the dpmodel (numpy/jax/...) loss backend and the
PyTorch loss backend can call them: the PyTorch backend passes torch tensors
and ``array_api_compat`` dispatches to the torch namespace, preserving
autograd.

Reduction convention
--------------------
A batch may hold frames of unequal atom count, padded to a common width. Each
term must therefore decide how one frame's aggregate error weighs against
another's, and the answer differs by term because frames do not carry equally
many labels:

- **Per-atom terms** (force, atomic energy, atomic prefactor force, dos,
  tensor) carry a number of labels proportional to the frame's atom count.
  :func:`masked_atom_mean` pools them: it divides the summed contribution of
  the whole batch by the batch's total label count, so every real label counts
  once and a frame's weight is proportional to its atom count.
- **Frame-level terms** (energy, virial) carry a fixed number of labels per
  frame. Pooling and averaging over frames coincide there, so
  :func:`per_frame_component_mean` reduces per frame and leaves the frame axis
  to the caller, which applies the extensive ``1 / natoms`` weighting.

Pooling is what keeps a frame's weight independent of the company it keeps.
The alternative -- averaging each frame's own per-label mean -- gives every
frame the same weight whatever its size, which makes a label in a small frame
count for more than one in a large frame, by the ratio of their atom counts.

Writing ``S_f`` for the summed contribution of frame ``f``, ``k`` for the
number of frames and ``d`` for the labels each of them carries, the two
coincide whenever that count is common to the batch:

    sum_f(S_f) / (k * d)  ==  (1 / k) * sum_f(S_f / d)

so a batch of uniform atom count needs no special case anywhere in this
module, and the choice between the two is unobservable there. They part
company only where a batch holds frames of differing real atom count, which
arises in exactly two places:

- ``mix:N`` LMDB batching, which packs frames of differing atom count by
  construction.
- ``mixed_type`` npy data whose ``real_atom_types.npy`` spends a different
  number of ``-1`` rows on different frames of one system. The format permits
  this and the documentation describes it as the way to merge frames of
  unequal atom count, but a system written by dpdata pads every frame equally
  and is therefore unaffected.

The TensorFlow backend reaches neither case: it drops ``real_natoms_vec``
before the feed dict and normalizes by the padded width throughout.

Each helper implements ONLY the masked branch. The unmasked branch of each
caller is a plain mean over the whole batch, which pools by construction, so
both branches express the same convention.
"""

from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)


def masked_atom_mean(elem: Array, maskf: Array, ncomp: int) -> Array:
    """Idiom 1: mean of a per-atom contribution over the batch's real labels.

    The contribution of every real atom is pooled across frames before the
    division, so the reduction is a mean over labels rather than a mean over
    frames of per-frame means. See the module docstring for why the per-atom
    terms weigh frames by their label count, and for the identity that makes
    this reduce to the per-frame mean on a uniform-atom-count batch.

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
        ``sum(elem * mask) / (ncomp * sum(mask))`` over the whole batch.
        A batch holding no real atom contributes a neutral ``0`` instead of
        ``0/0 = NaN``.
    """
    xp = array_api_compat.array_namespace(elem, maskf)
    total = xp.sum(elem * maskf[:, :, None])
    total_dof = xp.sum(maskf) * ncomp
    # A batch of nothing but padding has no label to average over, and the
    # ratio would be 0/0 = NaN -- poisoning the whole batch loss and, under
    # autograd, its gradient. The division still runs on a safe denominator so
    # that the discarded branch stays differentiable.
    has_dof = total_dof > 0
    safe_dof = xp.where(has_dof, total_dof, xp.ones_like(total_dof))
    return xp.where(has_dof, total / safe_dof, xp.zeros_like(total))


def masked_pair_mean(elem: Array, maskf: Array, ncomp: int) -> Array:
    """Return a per-frame mean over valid atom-pair components.

    Parameters
    ----------
    elem : Array
        Non-negative pair contribution of shape
        ``[nf, nloc * ncomp, nloc * ncomp]``. The contribution has already
        been squared or converted to an absolute value, but is not masked.
    maskf : Array
        Per-atom real/placeholder mask of shape ``[nf, nloc]``.
    ncomp : int
        Number of components per atom on each pair axis. A Cartesian Hessian
        uses three components on both axes.

    Returns
    -------
    Array
        ``mean_over_frames(sum(valid_pair_elem) / (real_natoms*ncomp)**2)``.
        A pair is valid only when both atom indices are real. An all-padding
        frame contributes a neutral zero.
    """
    xp = array_api_compat.array_namespace(elem, maskf)
    nf, nloc = maskf.shape
    component_mask = xp.reshape(
        xp.broadcast_to(maskf[:, :, None], (nf, nloc, ncomp)),
        (nf, nloc * ncomp),
    )
    masked = elem * component_mask[:, :, None] * component_mask[:, None, :]
    per_frame_sum = xp.sum(xp.reshape(masked, (nf, -1)), axis=-1)
    per_frame_dof = xp.square(xp.sum(component_mask, axis=-1))
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
