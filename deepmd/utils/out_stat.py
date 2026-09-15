# SPDX-License-Identifier: LGPL-3.0-or-later
"""Output statistics."""

from collections.abc import (
    Callable,
    Sequence,
)
from dataclasses import (
    dataclass,
)

import numpy as np

from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)


def compute_stats_from_redu(
    output_redu: np.ndarray,
    natoms: np.ndarray,
    assigned_bias: np.ndarray | None = None,
    rcond: float | None = None,
    intensive: bool = False,
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
    intensive
        Whether the output is intensive or extensive.

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
    if intensive:
        natoms = natoms / np.sum(natoms, axis=1, keepdims=True)
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
    assigned_bias: np.ndarray | None = None,
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


class ReduStatAccumulator:
    """Streaming, exact form of :func:`compute_stats_from_redu`.

    Frames are folded into a running QR factor of the augmented design matrix
    ``[natoms | output_redu | 1]``. Because ``R.T @ R == A.T @ A``, the least
    squares solution and the residual std are identical to the ones a single
    call on all frames would return, while memory stays at
    ``O((ntypes + ndim + 1) ** 2)`` regardless of the number of frames.

    Parameters
    ----------
    ntypes
        The number of atom types.
    ndim
        The flattened output dimension.
    var_shape
        The unflattened output shape, ``(ndim,)`` when not given.
    intensive
        Whether the output is intensive or extensive.
    """

    def __init__(
        self,
        ntypes: int,
        ndim: int,
        var_shape: list[int] | None = None,
        intensive: bool = False,
    ) -> None:
        self.ntypes = ntypes
        self.ndim = ndim
        self.var_shape = [ndim] if var_shape is None else list(var_shape)
        self.intensive = intensive
        self.nframes = 0
        # total occurrences of each type over every accumulated frame
        self.natoms_total = np.zeros(ntypes, dtype=np.int64)
        self._ncols = ntypes + ndim + 1
        self._r_factor = np.zeros((0, self._ncols), dtype=np.float64)
        # buffer whole blocks so that one QR amortizes over many small batches
        self._pending: list[np.ndarray] = []
        self._pending_rows = 0
        self._compress_every = max(1024, 8 * self._ncols)

    def add(self, output_redu: np.ndarray, natoms: np.ndarray) -> None:
        """Accumulate one chunk of frames.

        Parameters
        ----------
        output_redu
            The reduced output value, shape is [nframes, *(odim0, odim1, ...)].
        natoms
            The number of atoms of each type, shape is [nframes, ntypes].
        """
        natoms = np.asarray(natoms, dtype=np.float64).reshape(-1, self.ntypes)
        nf = natoms.shape[0]
        if nf == 0:
            return
        output_redu = np.asarray(output_redu, dtype=np.float64).reshape(nf, self.ndim)
        self.natoms_total += np.rint(natoms.sum(axis=0)).astype(np.int64)
        self.nframes += nf
        if self.intensive:
            natoms = natoms / np.sum(natoms, axis=1, keepdims=True)
        self._pending.append(
            np.concatenate(
                [natoms, output_redu, np.ones((nf, 1), dtype=np.float64)], axis=1
            )
        )
        self._pending_rows += nf
        if self._pending_rows >= self._compress_every:
            self._compress()

    def _compress(self) -> None:
        """Fold the buffered blocks into the running QR factor."""
        if not self._pending:
            return
        self._r_factor = np.linalg.qr(
            np.concatenate([self._r_factor, *self._pending], axis=0), mode="r"
        )
        self._pending.clear()
        self._pending_rows = 0

    def solve(
        self,
        assigned_bias: np.ndarray | None = None,
        rcond: float | None = None,
        type_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve the accumulated regression.

        Parameters
        ----------
        assigned_bias
            The assigned output bias, shape is [ntypes, *(odim0, odim1, ...)].
            Set to a tensor of shape (odim0, odim1, ...) filled with nan if the
            bias of the type is not assigned.
        rcond
            Cut-off ratio for small singular values of a. ``None`` reproduces
            the numpy default of the equivalent uncompressed problem.
        type_mask
            Excluded types, shape is [ntypes]. Types with a zero entry do not
            contribute to the regression.

        Returns
        -------
        np.ndarray
            The computed output bias, shape is [ntypes, *(odim0, odim1, ...)].
        np.ndarray
            The computed output std, shape is [*(odim0, odim1, ...)].

        Raises
        ------
        ValueError
            If no frame has been accumulated.
        """
        if self.nframes == 0:
            raise ValueError("No frame has been accumulated.")
        self._compress()
        r_factor = self._r_factor.copy()
        design = r_factor[:, : self.ntypes]
        redu = r_factor[:, self.ntypes : self.ntypes + self.ndim]
        constant = r_factor[:, -1]

        if type_mask is not None:
            design *= np.asarray(type_mask, dtype=np.float64).reshape(1, self.ntypes)

        assigned_mask = None
        if assigned_bias is not None:
            assigned_bias = np.asarray(assigned_bias, dtype=np.float64).reshape(
                self.ntypes, self.ndim
            )
            assigned_mask = ~np.isnan(assigned_bias).any(axis=1)
            # the same column operations compute_stats_from_redu applies to the
            # frames; they are linear, so applying them to R is equivalent
            redu -= design[:, assigned_mask] @ assigned_bias[assigned_mask]
            design[:, assigned_mask] = 0.0

        if rcond is None:
            # np.linalg.lstsq scales its default cut-off with the number of
            # rows, which compression changed; restore the uncompressed one
            rcond = np.finfo(np.float64).eps * max(self.nframes, self.ntypes)
        bias, _, _, _ = np.linalg.lstsq(design, redu, rcond=rcond)
        if assigned_mask is not None:
            bias[assigned_mask] = assigned_bias[assigned_mask]

        residual = redu - design @ bias
        residual_mean = (constant @ residual) / self.nframes
        centered = residual - np.outer(constant, residual_mean)
        variance = np.sum(centered * centered, axis=0) / self.nframes
        std = np.sqrt(np.maximum(variance, 0.0))
        return (
            bias.reshape([self.ntypes] + self.var_shape),  # noqa: RUF005
            std.reshape(self.var_shape),
        )


@dataclass
class ReduScanResult:
    """Exact statistics collected by one full pass over the training data.

    Parameters
    ----------
    stats
        One accumulator per output key that carries a global label.
    natoms_total
        Total occurrences of each type over every scanned frame, shape [ntypes].
    nframes
        The number of scanned frames.
    """

    stats: dict[str, ReduStatAccumulator]
    natoms_total: np.ndarray
    nframes: int


class ReduStatScanner:
    """Cache full passes over the training data for a single training run.

    The trainer attaches an instance to the stat sampler; the consumers of that
    sampler pick it up and use it instead of estimating the output statistics
    from a handful of sampled batches.

    Parameters
    ----------
    scan_fn
        Backend function performing one pass, called as
        ``scan_fn(ntypes, keys, intensive)``.
    """

    def __init__(self, scan_fn: Callable[..., ReduScanResult]) -> None:
        self._scan_fn = scan_fn
        self._cache: dict[tuple, ReduScanResult] = {}

    def scan(
        self, ntypes: int, keys: Sequence[str], intensive: bool = False
    ) -> ReduScanResult:
        """Return the statistics for *keys*, scanning the data once per request."""
        cache_key = (ntypes, tuple(keys), bool(intensive))
        if cache_key not in self._cache:
            self._cache[cache_key] = self._scan_fn(ntypes, tuple(keys), bool(intensive))
        return self._cache[cache_key]

    def natoms_total(self, ntypes: int) -> np.ndarray:
        """Return the per-type atom counts, reusing any scan already performed."""
        if self._cache:
            return next(iter(self._cache.values())).natoms_total
        return self.scan(ntypes, ()).natoms_total


def get_redu_stat_scanner(sampler: object) -> ReduStatScanner | None:
    """Return the full-data scanner a statistics sampler carries, if any.

    The scanner rides on the sampler as an attribute so that it reaches the
    statistics consumers without a new argument on every atomic model. The type
    check keeps that loose contract from picking up an unrelated attribute.
    """
    scanner = getattr(sampler, "redu_stat_scanner", None)
    return scanner if isinstance(scanner, ReduStatScanner) else None
