# SPDX-License-Identifier: LGPL-3.0-or-later
# dpa_adapt/_validation.py
#
# Small shared argument validators for the fine-tuning entry points
# (DPATrainer, MFTFineTuner, DPAFineTuner) so the same checks are not
# copy-pasted across constructors.

from __future__ import annotations


def validate_fparam_dim(fparam_dim: int) -> None:
    """Raise ``ValueError`` unless *fparam_dim* is a non-negative int.

    ``0`` means "no fparam conditioning"; any positive value is the width of
    the per-frame ``fparam.npy`` arrays.
    """
    if not isinstance(fparam_dim, int) or fparam_dim < 0:
        raise ValueError(f"fparam_dim must be a non-negative int; got {fparam_dim!r}.")
