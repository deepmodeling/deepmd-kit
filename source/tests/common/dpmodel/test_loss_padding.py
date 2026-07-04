# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reusable grad-accumulation invariant harness for dpmodel loss tests.

This module provides ``assert_grad_accum_invariant`` for Tasks 2-5 that
verify the loss on a padded multi-frame batch equals mean(per_frame_loss).

The dpmodel losses accept numpy arrays (via the array_api_compat backend).

Constants
---------
NA = 3   # real atoms in the short frame
NB = 5   # real atoms in the full-width frame (NB == NP)
NP = 5   # padded width (nloc)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constants used by the multi-frame test harness (Tasks 2-5)
# ---------------------------------------------------------------------------

NA = 3  # real atoms in frame A
NB = 5  # real atoms in frame B (== NP, so frame B is fully real)
NP = 5  # padded width (nloc)


# ---------------------------------------------------------------------------
# Reusable harness (used by Tasks 2-5; imported from here)
# ---------------------------------------------------------------------------


def assert_grad_accum_invariant(
    loss_fn,
    make_batch_A,
    make_batch_B,
    make_padded_batch,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Assert that padded-batch loss == mean(per_frame_loss) for two frames.

    The grad-accumulation invariant: a padded batch of [frame_A (NA real atoms
    padded to NP) + frame_B (NB==NP real atoms)] must yield the same loss as
    processing each frame separately and averaging.

    Parameters
    ----------
    loss_fn : callable
        Signature ``(model_pred, label, natoms) -> float``.  The function
        receives numpy-dict inputs and returns a scalar float.
    make_batch_A : callable
        Returns ``(model_pred, label, natoms)`` for frame A alone (1 frame, NA atoms).
    make_batch_B : callable
        Returns ``(model_pred, label, natoms)`` for frame B alone (1 frame, NB atoms).
    make_padded_batch : callable
        Returns ``(model_pred, label, natoms)`` for the 2-frame padded batch
        (nf=2, nloc=NP; frame A is padded with NP-NA ghost rows).
    rtol : float
        Relative tolerance for ``np.isclose``.
    atol : float
        Absolute tolerance for ``np.isclose``.
    """
    pred_A, label_A, natoms_A = make_batch_A()
    pred_B, label_B, natoms_B = make_batch_B()
    pred_pad, label_pad, natoms_pad = make_padded_batch()

    loss_A = float(loss_fn(pred_A, label_A, natoms_A))
    loss_B = float(loss_fn(pred_B, label_B, natoms_B))
    ref = 0.5 * (loss_A + loss_B)

    loss_pad = float(loss_fn(pred_pad, label_pad, natoms_pad))

    assert np.isclose(loss_pad, ref, rtol=rtol, atol=atol), (
        f"Grad-accum invariant violated: padded_loss={loss_pad:.8f}, "
        f"ref={ref:.8f}, diff={abs(loss_pad - ref):.2e}"
    )
