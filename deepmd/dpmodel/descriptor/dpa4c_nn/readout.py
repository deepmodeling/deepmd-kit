# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fixed invariant readout for degree-wise DPA4C moments."""

from __future__ import (
    annotations,
)

import math
from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    xp_asarray_nodetach,
)
from deepmd.dpmodel.utils.network import (
    NativeLayer,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .bispectrum import (
    build_bispectrum_layout,
    derive_bispectrum_ranks,
)
from .geometry import (
    degree_offsets,
    derive_degree_channels,
    packed_l2_to_stf,
)


class InvariantReadout(NativeOP):
    """Contract degree-wise moments into fixed O(3)-invariant features.

    The readout carries no learned nonlinearity. Degrees one and two first
    pass through full-width residual channel maps, after which three fixed
    contractions are emitted: the exact channel Gram of every non-scalar
    degree, a Cartesian bispectrum over every O(3)-even degree triple, and the
    projected quartic ``|Q_b v_a|^2``. The scalar moments are prepended
    unchanged.

    Parameters
    ----------
    channels
        Scalar degree-zero channel width.
    lmax
        Maximum angular degree.
    precision
        Parameter precision.
    trainable
        Whether the channel-alignment and probe projections are trainable.
    seed
        Random seed reserved for the readout.
    """

    def __init__(
        self,
        channels: int,
        lmax: int,
        *,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        self.degree_channels = derive_degree_channels(channels, lmax)
        self.bispectrum_ranks = derive_bispectrum_ranks(self.degree_channels)
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.precision = str(precision)
        self.trainable = bool(trainable)
        self.degree_offsets = degree_offsets(self.degree_channels)

        # === Step 1. Degree-local full channel alignment ===
        # Only degrees one and two are aligned. Higher degrees carry a single
        # channel, for which a residual full-width map is the identity.
        alignment_seed = child_seed(seed, 0)
        self.channel_alignment = []
        for degree in range(1, 3):
            width = self.degree_channels[degree]
            self.channel_alignment.append(
                NativeLayer(
                    width,
                    width,
                    bias=False,
                    resnet=True,
                    precision=self.precision,
                    seed=child_seed(alignment_seed, degree),
                    trainable=self.trainable,
                )
            )

        # === Step 2. Exact Gram layout ===
        # Only the upper triangle is emitted. Scaling the strict off-diagonal
        # by sqrt(2) makes the half-vectorization Frobenius isometric.
        gram_index_parts = []
        gram_scale_parts = []
        gram_offsets = [0]
        for width in self.degree_channels[1:]:
            row, column = np.triu_indices(width)
            gram_index_parts.append((row * width + column).astype(np.int64))
            gram_scale_parts.append(np.where(row == column, 1.0, math.sqrt(2.0)))
            gram_offsets.append(gram_offsets[-1] + row.size)
        self.gram_index = np.concatenate(gram_index_parts)
        self.gram_scale = np.concatenate(gram_scale_parts).astype(
            PRECISION_DICT[self.precision]
        )
        self.gram_offsets = tuple(gram_offsets)

        # === Step 3. Bispectrum layout and low-rank probes ===
        layout = build_bispectrum_layout(
            self.lmax,
            self.bispectrum_ranks,
        )
        self.degree_triples = layout.degree_triples
        self.coupling_offsets = layout.coupling_offsets
        self.probe_offsets = layout.probe_offsets
        self.bispectrum_coupling = layout.coupling.astype(
            PRECISION_DICT[self.precision]
        )
        self.probe_index = layout.probe_index
        self.probe_scale = layout.probe_scale.astype(PRECISION_DICT[self.precision])

        # A full-rank degree needs no projection. The others start from an
        # orthonormal basis with a deterministic sign, so the probes are an
        # isometry of a random rank-`K` subspace at initialization.
        probe_seed = child_seed(seed, 1)
        self.probe_projections = []
        for degree, (width, rank) in enumerate(
            zip(self.degree_channels[1:], self.bispectrum_ranks, strict=True),
            start=1,
        ):
            if rank == width:
                self.probe_projections.append(None)
                continue
            degree_seed = child_seed(probe_seed, degree)
            projection = NativeLayer(
                width,
                rank,
                bias=False,
                precision=self.precision,
                seed=degree_seed,
                trainable=self.trainable,
            )
            # The projection weights are overwritten below, so the basis draw
            # reuses the same child seed to stay reproducible.
            rng = np.random.default_rng(degree_seed)
            orthogonal, triangular = np.linalg.qr(
                rng.normal(size=(width, rank)),
                mode="reduced",
            )
            sign = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
            projection.w = (orthogonal * sign[None, :]).astype(
                PRECISION_DICT[self.precision]
            )
            self.probe_projections.append(projection)

        self.bispectrum_dim = int(self.probe_index.shape[0])
        self.quartic_dim = self.bispectrum_ranks[0] * self.bispectrum_ranks[1]

    def call(self, moments: Any) -> Any:
        """Build invariant features from flat degree-wise moments.

        The output concatenates the scalar moments, the exact aligned Grams in
        degree order, the bispectrum in degree-triple order, and the projected
        quartic.

        Parameters
        ----------
        moments
            Flat moment tensor with shape ``(N, S)``, where
            ``S = sum((2 * l + 1) * degree_channels[l])``.

        Returns
        -------
        Any
            Invariant features with shape ``(N, get_dim_out())``.
        """
        xp = array_api_compat.array_namespace(moments)
        blocks = [
            xp.reshape(
                moments[
                    :,
                    self.degree_offsets[degree] : self.degree_offsets[degree + 1],
                ],
                (moments.shape[0], 2 * degree + 1, self.degree_channels[degree]),
            )
            for degree in range(self.lmax + 1)
        ]
        aligned = list(blocks)
        for degree, projection in enumerate(self.channel_alignment, start=1):
            aligned[degree] = projection.call(blocks[degree])
        projected = [
            block if projection is None else projection.call(block)
            for projection, block in zip(
                self.probe_projections,
                aligned[1:],
                strict=True,
            )
        ]
        bispectrum, quartic = self.build_bispectrum(projected, xp)
        return xp.concat(
            [
                blocks[0][:, 0, :],
                *self.build_grams(aligned, xp),
                *bispectrum,
                quartic,
            ],
            axis=-1,
        )

    def build_grams(self, blocks: list[Any], xp: Any) -> list[Any]:
        """Build Frobenius-isometric exact channel Grams.

        Parameters
        ----------
        blocks
            Aligned degree blocks. Entry ``l`` has shape
            ``(N, 2 * l + 1, degree_channels[l])``.
        xp
            Array namespace associated with ``blocks``.

        Returns
        -------
        list[Any]
            Upper-triangular Gram blocks for degrees one through ``lmax``.
        """
        device = array_api_compat.device(blocks[0])
        gram_index = xp_asarray_nodetach(
            xp,
            self.gram_index,
            device=device,
        )
        gram_scale = xp_asarray_nodetach(
            xp,
            self.gram_scale,
            device=device,
        )
        parts = []
        for degree, block in enumerate(blocks[1:]):
            gram = xp.matmul(
                xp.permute_dims(block, (0, 2, 1)),
                block,
            )
            flat = xp.reshape(
                gram,
                (block.shape[0], self.degree_channels[degree + 1] ** 2),
            )
            start, end = self.gram_offsets[degree : degree + 2]
            parts.append(
                xp.take(flat, gram_index[start:end], axis=1)
                * gram_scale[None, start:end]
            )
        return parts

    def build_bispectrum(
        self,
        projected: list[Any],
        xp: Any,
    ) -> tuple[list[Any], Any]:
        """Contract projected moments with fixed Cartesian Gaunt tensors.

        Parameters
        ----------
        projected
            Probe-projected degree blocks. Entry ``l - 1`` has shape
            ``(N, 2 * l + 1, bispectrum_ranks[l - 1])``.
        xp
            Array namespace associated with ``projected``.

        Returns
        -------
        bispectrum
            Independent cubic contractions grouped by degree triple.
        quartic
            Projected ``|Q_b v_a|^2`` values with shape ``(N, K_2 * K_1)``.
        """
        device = array_api_compat.device(projected[0])
        coupling = xp_asarray_nodetach(
            xp,
            self.bispectrum_coupling,
            device=device,
        )
        probe_index = xp_asarray_nodetach(
            xp,
            self.probe_index,
            device=device,
        )
        probe_scale = xp_asarray_nodetach(
            xp,
            self.probe_scale,
            device=device,
        )
        # The 112 triple shares its matrix-vector intermediate with the
        # quartic, so it is contracted in closed form rather than through the
        # generic Gaunt path.
        parts = []
        bispectrum_112, quartic = self.contract_vector_tensor(
            projected[0],
            projected[1],
            xp,
        )
        for triple_index, degrees in enumerate(self.degree_triples):
            degree_1, degree_2, degree_3 = degrees
            if degrees == (1, 1, 2):
                full = bispectrum_112
            else:
                coupling_start, coupling_end = self.coupling_offsets[
                    triple_index : triple_index + 2
                ]
                full = self.contract_bispectrum(
                    xp.reshape(
                        coupling[coupling_start:coupling_end],
                        (2 * degree_1 + 1, 2 * degree_2 + 1, 2 * degree_3 + 1),
                    ),
                    projected[degree_1 - 1],
                    projected[degree_2 - 1],
                    projected[degree_3 - 1],
                    xp,
                )
            probe_start, probe_end = self.probe_offsets[triple_index : triple_index + 2]
            parts.append(
                xp.take(full, probe_index[probe_start:probe_end], axis=1)
                * probe_scale[None, probe_start:probe_end]
            )
        return parts, quartic

    def contract_vector_tensor(
        self,
        vector: Any,
        packed_tensor: Any,
        xp: Any,
    ) -> tuple[Any, Any]:
        r"""Contract the ``112`` triple and reuse ``Q_b v_a`` for the quartic.

        Both outputs are built from the same matrix-vector intermediate
        :math:`Q_bv_a`, so the quartic costs one extra reduction rather than a
        second contraction.

        Parameters
        ----------
        vector
            Degree-one probes with shape ``(N, 3, K_1)``.
        packed_tensor
            Degree-two probes with shape ``(N, 5, K_2)``.
        xp
            Array namespace associated with the probes.

        Returns
        -------
        bispectrum_112
            Full ordered cubic contractions with shape ``(N, K_1 * K_1 * K_2)``.
        quartic
            Values :math:`|Q_bv_a|^2` with shape ``(N, K_2 * K_1)``.
        """
        n_nodes = vector.shape[0]
        vector_rank = vector.shape[-1]
        tensor_rank = packed_tensor.shape[-1]
        vectors = xp.permute_dims(vector, (0, 2, 1))
        tensors = packed_l2_to_stf(xp.permute_dims(packed_tensor, (0, 2, 1)))
        tensor_vector = xp.reshape(
            xp.matmul(
                tensors[:, :, None, :, :],
                vectors[:, None, :, :, None],
            ),
            (n_nodes, tensor_rank, vector_rank, 3),
        )
        full = xp.matmul(
            vectors[:, None, :, :],
            xp.permute_dims(tensor_vector, (0, 1, 3, 2)),
        )
        full = xp.reshape(
            xp.permute_dims(full, (0, 2, 3, 1)),
            (n_nodes, vector_rank * vector_rank * tensor_rank),
        )
        # The unit-Frobenius 112 Gaunt tensor in this Cartesian convention is
        # exactly -v_left^T Q v_right / sqrt(5).
        full = full * (-1.0 / math.sqrt(5.0))
        quartic = xp.reshape(
            xp.sum(tensor_vector * tensor_vector, axis=-1),
            (n_nodes, tensor_rank * vector_rank),
        )
        return full, quartic

    def contract_bispectrum(
        self,
        coupling: Any,
        value_1: Any,
        value_2: Any,
        value_3: Any,
        xp: Any,
    ) -> Any:
        """Contract one angular coupling without backend-specific ``einsum``.

        Parameters
        ----------
        coupling
            Cartesian Gaunt tensor with shape
            ``(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)``.
        value_1
            First degree block with shape ``(N, 2 * l1 + 1, K_1)``.
        value_2
            Second degree block with shape ``(N, 2 * l2 + 1, K_2)``.
        value_3
            Third degree block with shape ``(N, 2 * l3 + 1, K_3)``.
        xp
            Array namespace associated with the degree blocks.

        Returns
        -------
        Any
            Ordered contractions with shape ``(N, K_1 * K_2 * K_3)``.
        """
        n_nodes = value_1.shape[0]
        rank_1 = value_1.shape[-1]
        rank_2 = value_2.shape[-1]
        rank_3 = value_3.shape[-1]
        dim_1, dim_2, dim_3 = coupling.shape

        first = xp.matmul(
            xp.permute_dims(value_1, (0, 2, 1)),
            xp.reshape(coupling, (dim_1, dim_2 * dim_3)),
        )
        first = xp.reshape(first, (n_nodes, rank_1, dim_2, dim_3))
        first = xp.reshape(
            xp.permute_dims(first, (0, 1, 3, 2)),
            (n_nodes, rank_1 * dim_3, dim_2),
        )
        second = xp.matmul(first, value_2)
        second = xp.reshape(
            second,
            (n_nodes, rank_1, dim_3, rank_2),
        )
        second = xp.reshape(
            xp.permute_dims(second, (0, 1, 3, 2)),
            (n_nodes, rank_1 * rank_2, dim_3),
        )
        return xp.reshape(
            xp.matmul(second, value_3),
            (n_nodes, rank_1 * rank_2 * rank_3),
        )

    def get_dim_out(self) -> int:
        """Return the geometric output width.

        Returns
        -------
        int
            Scalar moments, aligned exact Grams, bispectrum probes, and the
            projected quartic.
        """
        return (
            self.channels
            + int(self.gram_index.shape[0])
            + self.bispectrum_dim
            + self.quartic_dim
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize the invariant readout.

        Returns
        -------
        dict[str, Any]
            Versioned readout configuration and trainable projections.
        """
        return {
            "@class": "InvariantReadout",
            "@version": 1,
            "channels": self.channels,
            "lmax": self.lmax,
            "precision": self.precision,
            "trainable": self.trainable,
            "channel_alignment": [
                projection.serialize() for projection in self.channel_alignment
            ],
            "probe_projections": [
                None if projection is None else projection.serialize()
                for projection in self.probe_projections
            ],
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> InvariantReadout:
        """Deserialize an :class:`InvariantReadout`.

        Parameters
        ----------
        data
            Versioned dictionary produced by :meth:`serialize`.

        Returns
        -------
        InvariantReadout
            Reconstructed readout with restored projections.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        if data.pop("@class") != "InvariantReadout":
            raise ValueError("Invalid serialized class for InvariantReadout")
        alignment = data.pop("channel_alignment")
        probes = data.pop("probe_projections")
        obj = cls(**data)
        if len(alignment) != len(obj.channel_alignment):
            raise ValueError("Serialized alignment projection count is invalid.")
        if len(probes) != len(obj.probe_projections):
            raise ValueError("Serialized probe projection count is invalid.")
        obj.channel_alignment = [
            NativeLayer.deserialize(projection) for projection in alignment
        ]
        obj.probe_projections = [
            None if projection is None else NativeLayer.deserialize(projection)
            for projection in probes
        ]
        return obj
