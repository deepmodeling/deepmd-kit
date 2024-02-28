# SPDX-License-Identifier: LGPL-3.0-or-later
"""Output statistics."""
from typing import (
    Optional,
    Tuple,
)

import numpy as np


def compute_stats_from_redu(
    output_redu: np.ndarray,
    natoms: np.ndarray,
    assigned_bias: Optional[np.ndarray] = None,
    rcond: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the output statistics.

    Given the reduced output value and the number of atoms for each atom,
    compute the least-squares solution as the atomic output bais and std.

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
    np.ndarray
        The computed output std, shape is [ntypes, ndim].
    """
    output_redu = np.array(output_redu)
    natoms = np.array(natoms)
    # check shape
    assert output_redu.ndim == 2
    assert natoms.ndim == 2
    assert output_redu.shape[0] == natoms.shape[0]  # nframes
    if assigned_bias is not None:
        assigned_bias = np.array(assigned_bias).reshape(
            natoms.shape[1], output_redu.shape[1]
        )
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
    # rest_redu: nframes, ndim
    rest_redu = output_redu - np.einsum("ij,jk->ik", natoms, computed_output_bias)
    output_std = rest_redu.std(axis=0)
    return computed_output_bias, output_std


def compute_stats_from_atomic(
    output: np.ndarray,
    atype: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the output statistics.

    Given the output value and the type of atoms,
    compute the atomic output bais and std.

    Parameters
    ----------
    output
        The output value, shape is [nframes, nloc, ndim].
    atype
        The type of atoms, shape is [nframes, nloc].

    Returns
    -------
    np.ndarray
        The computed output bias, shape is [ntypes, ndim].
    np.ndarray
        The computed output std, shape is [ntypes, ndim].
    """
    output = np.array(output)
    atype = np.array(atype)
    # check shape
    assert output.ndim == 3
    assert atype.ndim == 2
    assert output.shape[:2] == atype.shape
    # compute output bias
    nframes, nloc, ndim = output.shape
    ntypes = atype.max() + 1
    output_bias = np.zeros((ntypes, ndim))
    output_std = np.zeros((ntypes, ndim))
    for type_i in range(ntypes):
        mask = atype == type_i
        output_bias[type_i] = output[mask].mean(axis=0)
        output_std[type_i] = output[mask].std(axis=0)
    return output_bias, output_std
