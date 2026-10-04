# SPDX-License-Identifier: LGPL-3.0-or-later
"""Torch materialization of the canonical DPA4 coefficient layout."""

from __future__ import (
    annotations,
)

import torch

from deepmd.dpmodel.descriptor.dpa4_nn.indexing import (
    build_m_major_index as _build_m_major_index,
)


def build_m_major_index(
    lmax: int,
    mmax: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """
    Build the canonical m-major coefficient index on a Torch device.

    Parameters
    ----------
    lmax : int
        Maximum spherical-harmonic degree.
    mmax : int
        Maximum absolute order retained in the reduced layout.
    device : torch.device or str
        Device for the returned index tensor.

    Returns
    -------
    torch.Tensor
        Integer coefficient indices with shape (D_m,).
    """
    return torch.as_tensor(_build_m_major_index(lmax, mmax), device=device)
