# SPDX-License-Identifier: LGPL-3.0-or-later
"""Output statistics."""
from typing import (
    Optional,
)

import numpy as np


def compute_output_stat(
    output_redu: np.ndarray,
    natoms: np.ndarray,
    assigned_bias: Optional[np.ndarray] = None,
    rcond: Optional[float] = None,
) -> np.ndarray:
    """Compute the output statistics.

    Given the reduced output value and the number of atoms for each atom,
    compute the least-squares solution as the atomic output bais.

    Parameters
    ----------
    output_redu
        The reduced output value, shape is [nframes, ndim].
    natoms
        The number of atoms for each atom, shape is [nframes, ntypes].
    assigned_bias
        The assigned output bias, shape is [ntypes, ndim]. Set to nan
        if not assigned.
    rcond
        Cut-off ratio for small singular values of a.

    Returns
    -------
    np.ndarray
        The computed output bias, shape is [ntypes, ndim].
    """
    output_redu = np.array(output_redu)
    natoms = np.array(natoms)
    # check shape
    assert output_redu.ndim == 2
    assert natoms.ndim == 2
    assert output_redu.shape[0] == natoms.shape[0]  # nframes
    if assigned_bias is not None:
        assigned_bias = np.array(assigned_bias)
        assert assigned_bias.ndim == 2
        assert assigned_bias.shape[0] == natoms.shape[1]  # ntypes
        assert assigned_bias.shape[1] == output_redu.shape[1]  # ndim
    # compute output bias
    if assigned_bias is not None:
        # Atomic energies stats are incorrect if atomic energies are assigned.
        # In this situation, we directly use these assigned energies instead of computing stats.
        # This will make the loss decrease quickly
        assigned_bias_atom_mask = ~np.isnan(assigned_bias).any(axis=1)
        # assigned_bias_masked: nmask, ndim
        assigned_bias_masked = assigned_bias[assigned_bias_atom_mask]
        # assigned_bias_natoms: nframes, nmask
        assigned_bias_natoms = natoms[:, assigned_bias_atom_mask]
        # output_redu: nframes, ndim
        output_redu -= np.einsum(
            "ij,jk->ik", assigned_bias_natoms, assigned_bias_masked
        )
        # remove assigned atom
        natoms[:, assigned_bias_atom_mask] = 0

    # computed_output_bias: ntypes, ndim
    computed_output_bias, _, _, _ = np.linalg.lstsq(natoms, output_redu, rcond=rcond)
    if assigned_bias is not None:
        # add back assigned atom; this might not be required
        computed_output_bias[assigned_bias_atom_mask] = assigned_bias_masked
    return computed_output_bias
