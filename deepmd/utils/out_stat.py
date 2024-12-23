# SPDX-License-Identifier: LGPL-3.0-or-later
"""Output statistics."""

from typing import (
    Optional,
)

import numpy as np

from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)


def compute_stats_from_redu(
    output_redu: np.ndarray,
    natoms: np.ndarray,
    assigned_bias: Optional[np.ndarray] = None,
    rcond: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the output statistics.

    Given the reduced output value and the number of atoms for each atom,
    compute the least-squares solution as the atomic output bias and std.

    Parameters
    ----------
    output_redu
        The reduced output value, shape is [nframes, *(odim0, odim1, ...)].
    natoms
        The number of atoms for each atom, shape is [nframes, ntypes].
    assigned_bias
        The assigned output bias, shape is [ntypes, *(odim0, odim1, ...)].
        Set to a tensor of shape (odim0, odim1, ...) filled with nan if the bias
        of the type is not assigned.
    rcond
        Cut-off ratio for small singular values of a.

    Returns
    -------
    np.ndarray
        The computed output bias, shape is [ntypes, *(odim0, odim1, ...)].
    np.ndarray
        The computed output std, shape is [*(odim0, odim1, ...)].
    """
    natoms = np.array(natoms)
    nf, _ = natoms.shape
    output_redu = np.array(output_redu)
    var_shape = list(output_redu.shape[1:])
    output_redu = output_redu.reshape(nf, -1)
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
    computed_output_bias = computed_output_bias.reshape([natoms.shape[1]] + var_shape)  # noqa: RUF005
    output_std = output_std.reshape(var_shape)
    return computed_output_bias, output_std


def compute_stats_from_atomic(
    output: np.ndarray,
    atype: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the output statistics.

    Given the output value and the type of atoms,
    compute the atomic output bias and std.

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
    output_bias = np.zeros((ntypes, ndim), dtype=GLOBAL_NP_FLOAT_PRECISION)
    output_std = np.zeros((ntypes, ndim), dtype=GLOBAL_NP_FLOAT_PRECISION)
    for type_i in range(ntypes):
        mask = atype == type_i
        output_bias[type_i] = (
            output[mask].mean(axis=0) if output[mask].size > 0 else np.nan
        )
        output_std[type_i] = (
            output[mask].std(axis=0) if output[mask].size > 0 else np.nan
        )
    return output_bias, output_std


def compute_stats_do_not_distinguish_types(
    output_redu: np.ndarray,
    natoms: np.ndarray,
    assigned_bias: Optional[np.ndarray] = None,
    intensive: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute element-independent statistics for property fitting.

    Computes mean and standard deviation of the output, treating all elements equally.
    For extensive properties, the output is normalized by the total number of atoms
    before computing statistics.

    Parameters
    ----------
    output_redu
        The reduced output value, shape is [nframes, *(odim0, odim1, ...)].
    natoms
        The number of atoms for each atom, shape is [nframes, ntypes].
        Used for normalization of extensive properties and generating uniform bias.
    assigned_bias
        The assigned output bias, shape is [ntypes, *(odim0, odim1, ...)].
        Set to a tensor of shape (odim0, odim1, ...) filled with nan if the bias
        of the type is not assigned.
    intensive
        Whether the output is intensive or extensive.
        If False, the output will be normalized by the total number of atoms before computing statistics.

    Returns
    -------
    np.ndarray
        The computed output mean(fake bias), shape is [ntypes, *(odim0, odim1, ...)].
        The same bias is used for all atom types.
    np.ndarray
        The computed output standard deviation, shape is [ntypes, *(odim0, odim1, ...)].
        The same standard deviation is used for all atom types.
    """
    natoms = np.array(natoms)  # [nf, ntypes]
    nf, ntypes = natoms.shape
    output_redu = np.array(output_redu)
    var_shape = list(output_redu.shape[1:])
    output_redu = output_redu.reshape(nf, -1)
    if not intensive:
        total_atoms = natoms.sum(axis=1)
        output_redu = output_redu / total_atoms[:, np.newaxis]
    # check shape
    assert output_redu.ndim == 2
    assert natoms.ndim == 2
    assert output_redu.shape[0] == natoms.shape[0]  # [nf,1]

    computed_output_bias = np.repeat(
        np.mean(output_redu, axis=0)[np.newaxis, :], ntypes, axis=0
    )
    output_std = np.std(output_redu, axis=0)

    computed_output_bias = computed_output_bias.reshape([natoms.shape[1]] + var_shape)  # noqa: RUF005
    output_std = output_std.reshape(var_shape)
    output_std = np.tile(output_std, (computed_output_bias.shape[0], 1))

    return computed_output_bias, output_std
