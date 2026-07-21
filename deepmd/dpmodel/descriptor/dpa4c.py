# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Compact and Compressible degree-wise DPA4 descriptor.

DPA4C is the compact and compressible degree-wise member of the DPA4 family, a
graph-native local descriptor for lightweight training and for distillation
from DPA4. It reuses the DPA4 radial basis, bias-free radial network, type
embedding, and C³ cutoff envelope, and replaces equivariant message passing
with center-local moment reductions.

Degree :math:`\ell` retains :math:`C_\ell` channels from a profile derived
entirely from the scalar width, for :math:`2\le L\le 4`. Exact degree Grams
preserve the quadratic channel information, and low-rank Cartesian bispectrum
probes couple every non-scalar degree triple allowed by the O(3) triangle and
parity rules.

Evaluation is a single destination-local scan: one payload carries both
envelope masses and every degree-wise moment. The descriptor therefore
constructs no neighbor pairs, Wigner rotations, or source-node features, and
reduces the edge axis exactly once.
"""

from __future__ import (
    annotations,
)

import dataclasses
import math
from typing import (
    TYPE_CHECKING,
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
from deepmd.dpmodel.common import (
    cast_precision,
    get_xp_precision,
    to_numpy_array,
)
from deepmd.dpmodel.utils import (
    PairExcludeMask,
)
from deepmd.dpmodel.utils.network import (
    NativeLayer,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.dpmodel.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .dpa4_nn import (
    C3CutoffEnvelope,
    RadialBasis,
    SeZMTypeEmbedding,
    SwiGLUMLP,
    resolve_swiglu_hidden_width,
)
from .dpa4c_nn import (
    InvariantReadout,
    OrderedPairFiLM,
    build_angular_basis,
    build_moment_indices,
    derive_bispectrum_ranks,
    derive_degree_channels,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from deepmd.dpmodel.array_api import (
        Array,
    )
    from deepmd.utils.data_system import (
        DeepmdDataSystem,
    )
    from deepmd.utils.path import (
        DPPath,
    )


@BaseDescriptor.register("dpa4c")
class DescrptDPA4C(NativeOP, BaseDescriptor):
    r"""Construct the Compact and Compressible degree-wise DPA4 descriptor.

    Let :math:`\chi(r)` denote the fixed DPA4 C³ envelope and
    :math:`\phi_{ijc}` the ordered type-conditioned radial amplitude, which
    already carries one envelope factor. Degree zero is an additive reduction
    under one envelope, while every non-scalar degree carries a second
    envelope factor and its own matched normalizer:

    .. math::

       d^{(0)}_i=\sum_j\chi_{ij}^2,\qquad
       d^{(+)}_i=\sum_j\chi_{ij}^4,\qquad
       n^{(\bullet)}_i=\bigl(d^{(\bullet)}_i+\tfrac14\bigr)^{-1/2},\\
       X^{(0)}_{ic}=n^{(0)}_i\sum_j\phi_{ijc},\qquad
       X^{(\ell)}_{ic}
       =n^{(+)}_i\sum_j\chi_{ij}\phi^{(\ell)}_{ijc}
        B^{(\ell)}(\hat{\mathbf r}_{ij}).

    Degree :math:`\ell` reads the leading :math:`C_\ell` channels of the one
    shared radial map, so the tabulated edge width equals :math:`C_0`. The
    invariant readout then contracts these moments into exact channel Grams, a
    fixed O(3)-even Cartesian bispectrum over low-rank probes, and the
    projected quartic.

    Two further blocks close the output. The divisors
    :math:`1/n^{(0)}` and :math:`1/n^{(+)}` are emitted alongside the
    invariants, because normalization is otherwise irreversible and the
    effective coordination they encode would reach neither the readout nor the
    fitting network. The center type embedding is concatenated last as an
    independent block.

    Parameters
    ----------
    rcut
        Outer cutoff radius in Å.
    ntypes
        Number of atom types.
    channels
        Scalar degree-zero and edge-amplitude width. Supported values are 8,
        16, 32, 64, and 128.
    lmax
        Maximum angular degree. Supported values are 2, 3, and 4.
    basis_type
        DPA4 radial basis type: ``"bessel"`` or ``"gaussian"``.
    n_radial
        Number of DPA4 radial basis functions forming the fixed analytic
        radial input.
    radial_modes
        Number :math:`R` of shared radial mode profiles that every ordered
        atom-type pair mixes with its own coefficients. Zero leaves each pair
        with a rescaled copy of one shared radial function.
    use_amp
        Whether the per-edge stage runs under bfloat16 automatic mixed
        precision on CUDA during training. This is an execution policy rather
        than model state: it is never serialized, the backend-neutral
        equations only record it, and the autocast region itself is a backend
        concern. Evaluation and inference follow ``DP_AMP_INFER`` instead.
    exclude_types
        Ordered atom-type pairs excluded from the descriptor.
    precision
        Floating-point precision of descriptor parameters.
    trainable
        Whether descriptor parameters are trainable.
    type_map
        Atom-type names.
    seed
        Random seed.
    spin
        Reserved for descriptor API compatibility; only ``None`` is supported.

    Raises
    ------
    TypeError
        If ``channels`` or ``lmax`` is not an integer.
    ValueError
        If ``rcut``, ``ntypes``, or ``n_radial`` is not positive, if
        ``radial_modes`` is negative, or if ``channels`` or ``lmax`` is
        outside its supported set.
    NotImplementedError
        If ``spin`` is given.
    """

    _update_sel_cls = UpdateSel
    _ENVELOPE_EXPONENT = 5
    _DEGREE_NORM_FLOOR = 0.25
    _EPS = 1.0e-7
    _STAT_EPS = 1.0e-12
    # Frames drawn per sampled system for the calibration. The count trades
    # start-up time, linear in it, against the spread of a sample mean. On a
    # variable-size store the available frames follow the training batch-size
    # specification and ``data_stat_nbatch``, which for OMat24 leave the pool
    # far larger than this count.
    _STAT_FRAMES_PER_SAMPLE = 64
    _COMPRESSION_BUFFER_NAMES = (
        "data",
        "info",
        "pair_film",
        "pair_mixing",
        "type_embedding",
        "readout_matrices",
        "coupling_meta",
        "coupling_entry",
        "coupling_value",
        "output_mean",
        "output_inv_std",
    )

    def __init__(
        self,
        rcut: float,
        ntypes: int,
        channels: int = 32,
        lmax: int = 2,
        basis_type: str = "bessel",
        n_radial: int = 16,
        radial_modes: int = 0,
        use_amp: bool = False,
        exclude_types: list[tuple[int, int]] = [],
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        type_map: list[str] | None = None,
        seed: int | list[int] | None = None,
        spin: None = None,
    ) -> None:
        # === Step 1. Validate the public architecture contract ===
        if spin is not None:
            raise NotImplementedError("DPA4C does not support spin inputs.")
        if rcut <= 0.0:
            raise ValueError(f"`rcut` must be positive, got {rcut}")
        if ntypes <= 0:
            raise ValueError(f"`ntypes` must be positive, got {ntypes}")
        if n_radial <= 0:
            raise ValueError(f"`n_radial` must be positive, got {n_radial}")
        if (
            not isinstance(radial_modes, int)
            or isinstance(radial_modes, bool)
            or radial_modes < 0
        ):
            raise ValueError("`radial_modes` must be a non-negative integer.")
        # `channels` and `lmax` are validated inside the profile derivation,
        # which owns their supported sets.
        degree_channels = derive_degree_channels(channels, lmax)
        bispectrum_ranks = derive_bispectrum_ranks(degree_channels)

        # === Step 2. Resolve the scalar configuration ===
        self.rcut = float(rcut)
        self.ntypes = int(ntypes)
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.degree_channels = degree_channels
        self.bispectrum_ranks = bispectrum_ranks
        self.basis_type = str(basis_type).lower()
        self.n_radial = int(n_radial)
        self.radial_modes = int(radial_modes)
        self.use_amp = bool(use_amp)
        self.precision = precision
        self.trainable = bool(trainable)
        self.type_map = type_map
        self.seed = seed
        radial_hidden = resolve_swiglu_hidden_width(self.channels)

        # === Step 3. Build the shared DPA4 edge representation ===
        # The radial basis is raw. One p=5 DPA4 envelope gates the complete
        # role-conditioned amplitude, preserving a single C³ cutoff factor.
        #
        # Child seeds are numbered contiguously in construction order, so each
        # trainable component owns one slot and no two components can collide.
        self.type_embedding = SeZMTypeEmbedding(
            ntypes=self.ntypes,
            embed_dim=self.channels,
            precision=self.precision,
            seed=child_seed(seed, 0),
            trainable=self.trainable,
            padding=True,
        )
        self.radial_basis = RadialBasis(
            rcut=self.rcut,
            basis_type=self.basis_type,
            n_radial=self.n_radial,
            precision=self.precision,
            exponent=self._ENVELOPE_EXPONENT,
            apply_envelope=False,
        )
        self.radial_embedding = SwiGLUMLP(
            [self.n_radial, radial_hidden, self.channels],
            precision=self.precision,
            trainable=self.trainable,
            seed=child_seed(seed, 1),
        )
        # The mode profiles branch off the shared radial hidden state, so the
        # residual costs one linear head rather than a second trunk.
        self.radial_mode_head = (
            NativeLayer(
                radial_hidden,
                self.radial_modes,
                bias=False,
                precision=self.precision,
                trainable=self.trainable,
                seed=child_seed(seed, 2),
            )
            if self.radial_modes > 0
            else None
        )
        self.edge_envelope = C3CutoffEnvelope(
            rcut=self.rcut,
            exponent=self._ENVELOPE_EXPONENT,
            precision=self.precision,
        )
        self.pair_film = OrderedPairFiLM(
            self.channels,
            radial_modes=self.radial_modes,
            precision=self.precision,
            trainable=self.trainable,
            seed=child_seed(seed, 3),
        )
        self.readout = InvariantReadout(
            self.channels,
            self.lmax,
            precision=self.precision,
            trainable=self.trainable,
            seed=child_seed(seed, 4),
        )

        # === Step 4. Lay out the flat moment payload ===
        # Degree zero owns the leading `channels` entries of the flat layout,
        # so the non-scalar block is exactly its complement and needs no
        # separate degree index.
        channel_index, harmonic_index = build_moment_indices(self.degree_channels)
        self.angular_channel_index = channel_index[self.channels :]
        self.angular_harmonic_index = harmonic_index[self.channels :]

        # === Step 5. Initialize the output calibration state ===
        mean = np.zeros(
            self.get_dim_out(),
            dtype=PRECISION_DICT[self.precision],
        )
        self.mean = mean
        self.stddev = np.ones_like(mean)
        self.compress = False
        self.reinit_exclude(exclude_types)

    # === Descriptor evaluation ===

    def call_graph(
        self,
        graph: Any,
        atype: Array,
        type_embedding: Array | None = None,
        comm_dict: dict | None = None,
    ) -> tuple[Array, None]:
        """Evaluate DPA4C on a flat neighbor graph.

        Parameters
        ----------
        graph
            Neighbor graph containing ``edge_index`` with shape ``(2, E)``,
            ``edge_vec`` with shape ``(E, 3)`` in Å, and ``edge_mask`` with
            shape ``(E,)``.
        atype
            Flat node types with shape ``(N,)``. Padding nodes use type index
            ``ntypes`` and therefore gather the zero type-embedding row.
        type_embedding
            Optional precomputed DPA4 type table with shape
            ``(ntypes + 1, channels)``. If omitted, the descriptor
            evaluates its type-embedding module.
        comm_dict
            Communication metadata accepted by the common graph ABI. DPA4C
            does not read source-node features, so no halo-feature exchange is
            required and this argument is unused.

        Returns
        -------
        descriptor
            Rotation- and permutation-invariant node features with shape
            ``(N, get_dim_out())`` and the same floating dtype as ``edge_vec``.
        rot_mat
            ``None``. DPA4C does not expose an equivariant fitting input.
        """
        del comm_dict
        # === Step 1. Resolve type features and compute precision ===
        if type_embedding is None:
            type_embedding = self.type_embedding.call()
        xp = array_api_compat.array_namespace(graph.edge_vec)
        in_dtype = graph.edge_vec.dtype
        compute_dtype = get_xp_precision(xp, self.precision)
        if in_dtype != compute_dtype:
            graph = dataclasses.replace(
                graph,
                edge_vec=xp.astype(graph.edge_vec, compute_dtype),
            )

        # === Step 2. Evaluate the graph-native equations ===
        descriptor, _ = self.evaluate_graph(graph, atype, type_embedding)

        # === Step 3. Restore the graph input dtype ===
        if descriptor.dtype != in_dtype:
            descriptor = xp.astype(descriptor, in_dtype)
        return descriptor, None

    @cast_precision
    def call(
        self,
        coord_ext: Array,
        atype_ext: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        comm_dict: dict | None = None,
        charge_spin: Array | None = None,
    ) -> tuple[Array, None, None, None, Array]:
        """Adapt a bounded dense neighbor list to the graph-native equations.

        This method exists for the common descriptor ABI and numerical
        reference tests. Production DPA4C execution uses :meth:`call_graph`
        with a carry-all graph. A rectangular list at the internal compatibility
        capacity is rejected because its completeness cannot be established.

        Parameters
        ----------
        coord_ext
            Extended coordinates with shape ``(F, N_all, 3)`` or
            ``(F, 3 * N_all)`` in Å.
        atype_ext
            Extended atom types with shape ``(F, N_all)``.
        nlist
            Bounded neighbor list with shape ``(F, N_local, N_slot)``. Negative
            indices denote padding.
        mapping
            Extended-to-local owner mapping with shape ``(F, N_all)``. ``None``
            denotes the identity mapping.
        fparam
            Frame parameters accepted by the common descriptor ABI; unused.
        comm_dict
            Communication metadata accepted by the common descriptor ABI;
            unused.
        charge_spin
            Charge/spin conditioning accepted by the common descriptor ABI;
            unsupported and unused.

        Returns
        -------
        descriptor
            Invariant features with shape
            ``(F, N_local, get_dim_out())``.
        rot_mat
            ``None``.
        g2
            ``None``.
        h2
            ``None``.
        envelope
            Per-slot C³ envelope with shape
            ``(F, N_local, N_slot, 1)``.
        """
        from deepmd.dpmodel.utils.neighbor_graph import (
            graph_from_dense_quartet,
        )

        del fparam, comm_dict, charge_spin
        xp = array_api_compat.array_namespace(coord_ext, atype_ext, nlist)
        nf, nloc, nnei = nlist.shape

        # === Step 1. Convert the dense quartet without compacting its edge axis ===
        graph, atype_local = graph_from_dense_quartet(
            coord_ext,
            atype_ext,
            nlist,
            mapping,
        )

        # === Step 2. Evaluate the same graph-native equations ===
        descriptor, envelope = self.evaluate_graph(
            graph,
            atype_local,
            self.type_embedding.call(),
        )

        # === Step 3. Restore the common dense descriptor ABI ===
        descriptor = xp.reshape(
            descriptor,
            (nf, nloc, descriptor.shape[-1]),
        )
        envelope = xp.reshape(envelope, (nf, nloc, nnei, 1))
        return descriptor, None, None, None, envelope

    def evaluate_graph(
        self,
        graph: Any,
        atype: Array,
        type_embedding: Array,
    ) -> tuple[Array, Array]:
        """Evaluate the graph-native descriptor equations.

        The pipeline is edge amplitudes, one destination reduction into
        degree-wise moments, and the fixed invariant readout.

        Parameters
        ----------
        graph
            Neighbor graph in descriptor compute precision.
        atype
            Flat node types with shape ``(N,)``.
        type_embedding
            Complete type table with shape ``(ntypes + 1, channels)``.

        Returns
        -------
        descriptor
            Invariant node features with shape ``(N, get_dim_out())``.
        envelope
            Masked per-edge C³ envelope with shape ``(E, 1)``.
        """
        xp = array_api_compat.array_namespace(graph.edge_vec)

        # === Step 1. Place the precomputed type table in the graph namespace ===
        # A converted dpmodel may already store the table in the active
        # namespace. Conversion is required only for a direct NumPy-defined
        # descriptor evaluated with JAX or another array backend.
        type_namespace = array_api_compat.array_namespace(type_embedding)
        if type_namespace is not xp:
            type_embedding = xp.asarray(
                type_embedding,
                dtype=graph.edge_vec.dtype,
                device=array_api_compat.device(graph.edge_vec),
            )
        dst = graph.edge_index[1]
        n_total = atype.shape[0]
        center_type_embedding = self.gather_rows(type_embedding, atype, xp)
        pair_tables = self.pair_film.call(type_embedding)

        # === Step 2. Build the masked edge amplitudes and harmonics ===
        amplitude, basis, envelope = self.build_edge_features(
            graph,
            atype,
            *pair_tables,
        )

        # === Step 3. Reduce the degree-wise moments ===
        moments, divisors = self.aggregate_moments(
            amplitude,
            basis,
            envelope,
            dst,
            n_total,
        )

        # === Step 4. Build calibrated invariant features ===
        return (
            self.build_invariant_descriptor(
                moments,
                center_type_embedding,
                divisors,
            ),
            envelope[:, None],
        )

    def build_edge_features(
        self,
        graph: Any,
        atype: Array,
        pair_scale: Array,
        pair_shift: Array,
        pair_mixing: Array | None,
    ) -> tuple[Array, Array, Array]:
        r"""Build the enveloped edge amplitudes and the masked harmonics.

        The ordered type pair :math:`(a,b)` rescales the one shared radial
        function :math:`g` and mixes the :math:`R` shared mode profiles
        :math:`q_\mu` with its own coefficients:

        .. math::

           \phi_{ijc}
           =\chi_{ij}\Bigl(
             \gamma_{ab,c}g_c(\rho_{ij})+\beta_{ab,c}
             +\sum_\mu U_{ab,c\mu}q_\mu(\rho_{ij})
            \Bigr).

        Parameters
        ----------
        graph
            Neighbor graph in descriptor compute precision.
        atype
            Flat node types with shape ``(N,)``.
        pair_scale
            Ordered radial scales with shape
            ``((ntypes + 1) ** 2, channels)``.
        pair_shift
            Ordered radial shifts with shape
            ``((ntypes + 1) ** 2, channels)``.
        pair_mixing
            Ordered mode-mixing table with shape
            ``((ntypes + 1) ** 2, channels, radial_modes)``, or ``None`` when
            ``radial_modes`` is zero.

        Returns
        -------
        amplitude
            Masked edge amplitudes with shape ``(E, channels)``.
        basis
            Masked Cartesian harmonics with shape ``(E, (lmax + 1) ** 2)``.
        envelope
            Masked C³ envelope with shape ``(E,)``.
        """
        from deepmd.dpmodel.utils.neighbor_graph import (
            apply_pair_exclusion,
        )

        # === Step 1. Merge graph and descriptor-level exclusion masks ===
        graph = apply_pair_exclusion(graph, atype, self.emask)
        xp = array_api_compat.array_namespace(graph.edge_vec)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        center_type = self.gather_rows(atype, dst, xp)
        neighbor_type = self.gather_rows(atype, src, xp)

        # === Step 2. Build regularized distances and directions ===
        # sqrt(r^2 + eps^2) keeps the direction finite for coincident or guard
        # edges. Valid physical edges are unaffected above the 1e-7 Å scale.
        distance_squared = xp.sum(
            graph.edge_vec * graph.edge_vec,
            axis=-1,
            keepdims=True,
        )
        distance = xp.sqrt(distance_squared + self._EPS * self._EPS)
        direction = graph.edge_vec / distance
        real_type = (center_type < self.ntypes) & (neighbor_type < self.ntypes)
        edge_mask = graph.edge_mask & real_type
        mask = xp.astype(edge_mask[:, None], graph.edge_vec.dtype)

        # === Step 3. Evaluate the shared DPA4 radial representation ===
        # DPA4C requests the raw radial basis. One explicit C³ envelope gates
        # the combined radial and type feature, so the scalar edge amplitude
        # contains exactly one cutoff factor and vanishes smoothly at rcut.
        envelope = self.evaluate_cutoff_envelope(distance) * mask
        radial_basis = self.radial_basis.call(distance)
        radial_hidden = self.radial_embedding.call_hidden(radial_basis)
        radial = self.radial_embedding.call_output(radial_hidden)
        pair_index = center_type * (self.ntypes + 1) + neighbor_type
        scale = self.gather_rows(pair_scale, pair_index, xp)  # (E, C)
        shift = self.gather_rows(pair_shift, pair_index, xp)  # (E, C)
        amplitude = radial * scale + shift

        # === Step 4. Add the pair-conditioned radial mode residual ===
        # Each ordered pair selects its own combination of the R shared mode
        # profiles. The contraction is written as a broadcast product reduced
        # over the mode axis rather than as a batched matrix-vector product:
        # one GEMV per edge leaves the tiny C-by-R operands far short of
        # memory bandwidth, whereas the reduction is a plain streaming pass.
        # Expanding the ordered table per edge dominates the cost either way.
        if pair_mixing is not None:
            mixing = self.gather_rows(pair_mixing, pair_index, xp)  # (E, C, R)
            modes = self.radial_mode_head(radial_hidden)  # (E, R)
            amplitude = amplitude + xp.sum(mixing * modes[:, None, :], axis=-1)

        # === Step 5. Gate the amplitude and build the masked harmonics ===
        return (
            amplitude * envelope,
            self.build_angular_basis(direction) * mask,
            envelope[:, 0],
        )

    def aggregate_moments(
        self,
        amplitude: Array,
        basis: Array,
        envelope: Array,
        dst: Array,
        n_total: int,
    ) -> tuple[Array, Array]:
        r"""Aggregate every degree-wise moment in one segment reduction.

        Degree zero is additive under the single envelope already carried by
        the amplitude, whereas every non-scalar degree carries a second
        envelope factor and its own matched normalizer:

        .. math::

           d^{(0)}_i=\sum_j\chi_{ij}^2,\qquad
           d^{(+)}_i=\sum_j\chi_{ij}^4,\qquad
           n^{(\bullet)}_i=\bigl(d^{(\bullet)}_i+\tfrac14\bigr)^{-1/2},\\
           X^{(0)}_{ic}=n^{(0)}_i\sum_j\phi_{ijc},\qquad
           X^{(\ell)}_{icm}
           =n^{(+)}_i\sum_j\chi_{ij}\phi_{ijc}B^{(\ell)}_m(\hat u_{ij}).

        Both envelope masses and both moment blocks share one payload, so the
        descriptor reduces the edge axis exactly once.

        Parameters
        ----------
        amplitude
            Masked edge amplitudes with shape ``(E, channels)``.
        basis
            Masked Cartesian harmonics with shape ``(E, (lmax + 1) ** 2)``.
        envelope
            Masked C³ envelope with shape ``(E,)``.
        dst
            Destination node indices with shape ``(E,)``.
        n_total
            Number of output nodes ``N``.

        Returns
        -------
        moments
            Flat normalized moments with shape ``(N, S)``, where
            ``S = sum((2 * l + 1) * degree_channels[l])``.
        divisors
            The two divisors :math:`1/n^{(0)}` and :math:`1/n^{(+)}` with shape
            ``(N, 2)``. They are retained because normalization is otherwise
            irreversible: the readout sees only scaled moments and can neither
            recover the unnormalized ones nor read the effective coordination
            they encode.
        """
        from deepmd.dpmodel.utils.neighbor_graph import (
            segment_sum,
        )

        xp = array_api_compat.array_namespace(amplitude)
        device = array_api_compat.device(amplitude)
        channel_index = xp_asarray_nodetach(
            xp,
            self.angular_channel_index,
            device=device,
        )
        harmonic_index = xp_asarray_nodetach(
            xp,
            self.angular_harmonic_index,
            device=device,
        )

        # Payload layout: [chi^2, chi^4, degree zero (C_0), degrees one and
        # above (S - C_0)]. Degree zero is the whole edge amplitude under its
        # single envelope, while the non-scalar block gathers an amplitude and
        # a harmonic per flat moment coordinate and carries a second envelope.
        envelope_squared = envelope * envelope
        payload = xp.concat(
            [
                envelope_squared[:, None],
                (envelope_squared * envelope_squared)[:, None],
                amplitude,
                xp.take(amplitude, channel_index, axis=1)
                * xp.take(basis, harmonic_index, axis=1)
                * envelope[:, None],
            ],
            axis=1,
        )
        reduced = segment_sum(payload, dst, n_total)

        scalar_end = 2 + self.channels
        floor = self._DEGREE_NORM_FLOOR
        divisors = xp.sqrt(reduced[:, :2] + floor)
        return (
            xp.concat(
                [
                    reduced[:, 2:scalar_end] / divisors[:, :1],
                    reduced[:, scalar_end:] / divisors[:, 1:],
                ],
                axis=1,
            ),
            divisors,
        )

    def build_invariant_descriptor(
        self,
        moments: Array,
        center_type_embedding: Array,
        divisors: Array,
    ) -> Array:
        """Build and calibrate the geometric and center-type output blocks.

        The calibration is the fixed diagonal preconditioner established by
        :meth:`compute_input_stats`, not a running normalization.

        Parameters
        ----------
        moments
            Flat degree-wise moments with shape ``(N, S)``.
        center_type_embedding
            Center type embeddings with shape ``(N, channels)``.
        divisors
            The two moment divisors with shape ``(N, 2)``. They close the
            geometric block, so the calibration treats them like any other
            invariant and the center-type tail keeps its trailing position.

        Returns
        -------
        Array
            Descriptor with shape ``(N, get_dim_out())``.
        """
        xp = array_api_compat.array_namespace(moments)
        device = array_api_compat.device(moments)
        descriptor = xp.concat(
            [self.readout.call(moments), divisors, center_type_embedding],
            axis=-1,
        )
        mean = xp_asarray_nodetach(xp, self.mean, device=device)
        stddev = xp_asarray_nodetach(xp, self.stddev, device=device)
        return (descriptor - mean[None, :]) / stddev[None, :]

    # === Backend primitives ===
    # A backend wrapper overrides these to reach native kernels; the equations
    # above stay array-API neutral.

    def gather_rows(
        self,
        values: Array,
        index: Array,
        xp: Any | None = None,
    ) -> Array:
        """Gather rows along the leading axis.

        Parameters
        ----------
        values
            Source array with shape ``(N, ...)``.
        index
            Row indices with arbitrary shape.
        xp
            Optional array namespace for ``values``.

        Returns
        -------
        Array
            Gathered values with shape ``index.shape + values.shape[1:]``.
        """
        if xp is None:
            xp = array_api_compat.array_namespace(values)
        return xp.take(values, index, axis=0)

    def evaluate_cutoff_envelope(self, distance: Array) -> Array:
        """Evaluate the fixed C³ cutoff envelope.

        Parameters
        ----------
        distance
            Regularized edge distances with shape ``(E, 1)`` in Å.

        Returns
        -------
        Array
            Envelope values with shape ``(E, 1)``.
        """
        return self.edge_envelope.call(distance)

    def build_angular_basis(self, direction: Array) -> Array:
        """Build real Cartesian harmonics through ``lmax``.

        Parameters
        ----------
        direction
            Regularized edge directions with shape ``(E, 3)``.

        Returns
        -------
        Array
            Packed harmonics with shape ``(E, (lmax + 1) ** 2)``.
        """
        return build_angular_basis(direction, self.lmax)

    # === Parameter sharing and statistics ===

    def share_params(
        self,
        base_class: Any,
        shared_level: int,
        model_prob: float = 1.0,
        resume: bool = False,
    ) -> None:
        """Share all descriptor parameters for multitask training.

        Parameters
        ----------
        base_class
            DPA4C descriptor that owns the shared parameters.
        shared_level
            Sharing level. DPA4C supports only level ``0``, which shares the
            complete descriptor.
        model_prob
            Model sampling probability accepted by the common multitask ABI;
            unused because DPA4C has no mergeable input statistics.
        resume
            Whether sharing occurs during checkpoint restoration; unused.
        """
        del model_prob, resume
        if self.__class__ != base_class.__class__:
            raise TypeError("Only DPA4C descriptors can share parameters.")
        signature = self.structure_signature()
        base_signature = base_class.structure_signature()
        if signature != base_signature:
            raise ValueError(
                "DPA4C parameter sharing requires identical structural "
                f"parameters, got {signature} and {base_signature}"
            )
        if shared_level != 0:
            raise NotImplementedError("DPA4C supports only shared_level=0.")
        for name in (
            "type_embedding",
            "radial_basis",
            "radial_embedding",
            "radial_mode_head",
            "pair_film",
            "readout",
        ):
            setattr(self, name, getattr(base_class, name))
        self.mean = base_class.mean
        self.stddev = base_class.stddev

    def structure_signature(self) -> tuple:
        """Return the configuration that must agree between sharing replicas.

        The signature compares two live descriptors and is never persisted,
        so it may reference execution policy as well as persisted structure.

        It covers every field that fixes the shape or the meaning
        of a shared module. ``rcut``, ``basis_type``, and ``n_radial`` define
        the radial basis; ``ntypes`` defines the type table and the ordered
        pair index space; ``channels``, ``lmax``, and ``radial_modes`` define
        every remaining width; ``use_amp`` selects the precision policy that a
        backend attaches to the shared layers, so a replica that autocasts
        against layers configured without it would silently lose the effect;
        ``trainable`` decides whether
        those layers carry gradients at all; ``type_map`` fixes what the rows
        of the shared type table mean. Precision itself enters through its
        resolved dtype so that equivalent spellings agree.

        Branch-local state is deliberately absent. ``exclude_types`` is the
        only such field: it configures the pair-exclusion mask, which each
        replica keeps for itself.

        Returns
        -------
        tuple
            Structural configuration of the shareable modules.
        """
        return (
            self.rcut,
            self.ntypes,
            self.channels,
            self.lmax,
            self.basis_type,
            self.n_radial,
            self.radial_modes,
            self.use_amp,
            self.trainable,
            None if self.type_map is None else tuple(self.type_map),
            np.dtype(PRECISION_DICT[self.precision]).name,
        )

    def change_type_map(
        self,
        type_map: list[str],
        model_with_new_type_stat: Any | None = None,
    ) -> None:
        """Reject unsupported atom-type remapping.

        Parameters
        ----------
        type_map
            Requested atom-type map.
        model_with_new_type_stat
            Optional descriptor carrying statistics for newly introduced
            types. DPA4C does not use descriptor input statistics.
        """
        del type_map, model_with_new_type_stat
        raise NotImplementedError("DPA4C does not support changing `type_map`.")

    def set_stat_mean_and_stddev(self, mean: Array, stddev: Array) -> None:
        """Store fixed output calibration arrays.

        Parameters
        ----------
        mean
            Output shift with shape ``(get_dim_out(),)``.
        stddev
            Positive output scale with shape ``(get_dim_out(),)``.
        """
        expected = (self.get_dim_out(),)
        if mean.shape != expected or stddev.shape != expected:
            raise ValueError(
                "DPA4C output statistics must both have shape "
                f"{expected}, got {mean.shape} and {stddev.shape}"
            )
        if np.any(to_numpy_array(stddev) <= 0.0):
            raise ValueError("DPA4C output scales must be positive.")
        self.mean = mean
        self.stddev = stddev

    def get_stat_mean_and_stddev(self) -> tuple[Array, Array]:
        """Return interface-compatible descriptor statistics.

        Returns
        -------
        mean
            Stored mean array.
        stddev
            Stored standard-deviation array.
        """
        return self.mean, self.stddev

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        path: DPPath | None = None,
    ) -> None:
        """Calibrate polynomial output families against the type-embedding RMS.

        The calibration is a fixed initialization preconditioner. Every
        geometric coordinate is measured independently, while the center type
        tail is left unchanged. No sample-dependent normalization is evaluated
        during training or inference.

        Parameters
        ----------
        merged
            Sampled training systems or a callable returning them.
        path
            Optional statistics path. Model-dependent calibration is always
            recomputed and therefore does not consume this path.
        """
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_neighbor_graph,
        )

        del path
        sampled = merged() if callable(merged) else merged
        if not sampled:
            return

        xp = array_api_compat.array_namespace(self.stddev)
        device = array_api_compat.device(self.stddev)
        dtype = self.stddev.dtype
        mean_backup, stddev_backup = self.mean, self.stddev
        self.mean = xp.zeros_like(self.mean)
        self.stddev = xp.ones_like(self.stddev)
        geometry_dim = self.get_dim_out() - self.channels
        square_sum = np.zeros(geometry_dim, dtype=np.float64)
        value_sum = np.zeros(geometry_dim, dtype=np.float64)
        value_count = 0

        try:
            for system in sampled:
                coord_np = to_numpy_array(system["coord"])
                nframes = coord_np.shape[0]
                coord_np = np.reshape(coord_np, (nframes, -1, 3))
                atype_np = np.reshape(
                    to_numpy_array(system["atype"]),
                    (nframes, -1),
                )
                box_value = system.get("box", None)
                box_np = (
                    None
                    if box_value is None
                    else np.reshape(to_numpy_array(box_value), (nframes, -1))
                )
                nstat_frames = min(nframes, self._STAT_FRAMES_PER_SAMPLE)
                frame_indices = np.linspace(
                    0,
                    nframes - 1,
                    num=nstat_frames,
                    dtype=np.int64,
                )
                for frame_index in frame_indices:
                    coord = xp.asarray(
                        coord_np[frame_index : frame_index + 1],
                        dtype=dtype,
                        device=device,
                    )
                    atype = xp.asarray(
                        atype_np[frame_index : frame_index + 1],
                        device=device,
                    )
                    box = (
                        None
                        if box_np is None
                        else xp.asarray(
                            box_np[frame_index : frame_index + 1],
                            dtype=dtype,
                            device=device,
                        )
                    )
                    graph = build_neighbor_graph(
                        coord,
                        atype,
                        box,
                        self.get_rcut(),
                    )
                    output, _ = self.call_graph(graph, xp.reshape(atype, (-1,)))
                    output_np = to_numpy_array(output).reshape(
                        -1,
                        self.get_dim_out(),
                    )
                    if output_np.shape[0] == 0:
                        continue
                    square_sum += np.sum(
                        np.square(
                            output_np[:, :geometry_dim],
                            dtype=np.float64,
                        ),
                        axis=0,
                        dtype=np.float64,
                    )
                    value_sum += np.sum(
                        output_np[:, :geometry_dim],
                        axis=0,
                        dtype=np.float64,
                    )
                    value_count += output_np.shape[0]
        finally:
            self.mean, self.stddev = mean_backup, stddev_backup

        if value_count == 0:
            return
        feature_rms = np.sqrt(square_sum / float(value_count))
        if np.any(~np.isfinite(feature_rms)) or np.any(feature_rms <= self._STAT_EPS):
            raise ValueError(
                "DPA4C output calibration requires non-degenerate finite "
                f"features, got RMS values {feature_rms.tolist()}"
            )
        type_table = to_numpy_array(self.type_embedding.call())[: self.ntypes]
        target_rms = float(np.sqrt(np.mean(np.square(type_table, dtype=np.float64))))
        if not math.isfinite(target_rms) or target_rms <= self._STAT_EPS:
            raise ValueError(
                f"DPA4C type embedding has a degenerate calibration RMS {target_rms}"
            )
        geometry_stddev = feature_rms / target_rms
        geometry_mean = np.zeros(geometry_dim, dtype=np.float64)

        # The two moment divisors are the only outputs carrying their
        # information on a large offset rather than around zero: their RMS
        # exceeds a typical invariant by two orders of magnitude, so scaling
        # alone would leave them at one plus a small fluctuation. They are
        # standardized; every other coordinate keeps the shared RMS
        # preconditioner, whose zero mean the readout construction justifies.
        mass = slice(geometry_dim - 2, geometry_dim)
        mass_mean = value_sum[mass] / float(value_count)
        mass_stddev = np.sqrt(
            np.maximum(
                square_sum[mass] / float(value_count) - np.square(mass_mean), 0.0
            )
        )
        if np.any(mass_stddev <= self._STAT_EPS):
            raise ValueError(
                "DPA4C neighborhood masses are constant over the calibration "
                f"sample, got standard deviations {mass_stddev.tolist()}"
            )
        geometry_mean[mass] = mass_mean
        geometry_stddev[mass] = mass_stddev / target_rms

        tail = np.zeros(self.channels, dtype=np.float64)
        self.mean = np.concatenate([geometry_mean, tail]).astype(
            PRECISION_DICT[self.precision]
        )
        self.stddev = np.concatenate([geometry_stddev, tail + 1.0]).astype(
            PRECISION_DICT[self.precision]
        )

    # === Serialization and neighbor statistics ===

    def serialize(self) -> dict:
        """Serialize the descriptor.

        ``use_amp`` is deliberately absent. It selects an execution policy
        rather than any part of the learned function, so it is supplied by the
        training configuration and by ``DP_AMP_INFER``, never restored from a
        checkpoint.

        Returns
        -------
        dict
            Versioned descriptor configuration, nested DPA4 components, and
            interface statistics.
        """
        data = {
            "@class": "Descriptor",
            "type": "dpa4c",
            "@version": 1,
            "rcut": self.rcut,
            "ntypes": self.ntypes,
            "channels": self.channels,
            "lmax": self.lmax,
            "basis_type": self.basis_type,
            "n_radial": self.n_radial,
            "radial_modes": self.radial_modes,
            "exclude_types": self.exclude_types,
            "precision": np.dtype(PRECISION_DICT[self.precision]).name,
            "trainable": self.trainable,
            "type_map": self.type_map,
            "seed": self.seed,
            "spin": None,
            "type_embedding": self.type_embedding.serialize(),
            "radial_basis": self.radial_basis.serialize(),
            "radial_embedding": self.radial_embedding.serialize(),
            "radial_mode_head": (
                None
                if self.radial_mode_head is None
                else self.radial_mode_head.serialize()
            ),
            "pair_film": self.pair_film.serialize(),
            "readout": self.readout.serialize(),
            "@variables": {
                "mean": to_numpy_array(self.mean),
                "stddev": to_numpy_array(self.stddev),
            },
        }
        if self.compress:
            data["compress"] = {
                "@variables": {
                    name: to_numpy_array(getattr(self, f"compress_{name}"))
                    for name in self._COMPRESSION_BUFFER_NAMES
                }
            }
        return data

    @classmethod
    def deserialize(cls, data: dict) -> DescrptDPA4C:
        """Deserialize a DPA4C descriptor.

        Parameters
        ----------
        data
            Versioned descriptor dictionary produced by :meth:`serialize`.

        Returns
        -------
        DescrptDPA4C
            Reconstructed descriptor with restored trainable components.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        data.pop("type")
        compression = data.pop("compress", None)
        variables = data.pop("@variables")
        type_embedding = data.pop("type_embedding")
        radial_basis = data.pop("radial_basis")
        radial_embedding = data.pop("radial_embedding")
        radial_mode_head = data.pop("radial_mode_head")
        pair_film = data.pop("pair_film")
        readout = data.pop("readout")

        obj = cls(**data)
        obj.type_embedding = SeZMTypeEmbedding.deserialize(type_embedding)
        obj.radial_basis = RadialBasis.deserialize(radial_basis)
        obj.radial_embedding = SwiGLUMLP.deserialize(radial_embedding)
        obj.radial_mode_head = (
            None
            if radial_mode_head is None
            else NativeLayer.deserialize(radial_mode_head)
        )
        obj.pair_film = OrderedPairFiLM.deserialize(pair_film)
        obj.readout = InvariantReadout.deserialize(readout)
        obj.set_stat_mean_and_stddev(
            variables["mean"],
            variables["stddev"],
        )
        if compression is not None:
            obj._load_compression(compression)
        return obj

    def _load_compression(self, compression: dict) -> None:
        """Restore the compressed-inference artifact payload."""
        variables = compression["@variables"]
        for name in self._COMPRESSION_BUFFER_NAMES:
            setattr(self, f"compress_{name}", variables[name])
        self.compress = True

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float]:
        """Report the minimum neighbor distance without introducing a ``sel``.

        The descriptor is graph-native, so no neighbor capacity has to be
        derived from the data and the returned configuration is unchanged.
        Only the minimum neighbor distance is measured, which downstream
        consumers use to bound tabulated radial ranges.

        Parameters
        ----------
        train_data
            Training dataset used for neighbor statistics.
        type_map
            Ordered atom-type names.
        local_jdata
            DPA4C descriptor configuration.

        Returns
        -------
        local_jdata
            Unmodified descriptor configuration.
        min_nbor_dist
            Minimum observed neighbor distance in Å.
        """
        del type_map
        return local_jdata.copy(), cls._update_sel_cls().get_min_nbor_dist(train_data)

    # === Common descriptor ABI ===

    @property
    def dim_out(self) -> int:
        """Return the invariant descriptor width."""
        return self.get_dim_out()

    def get_rcut(self) -> float:
        """Return the outer cutoff radius in Å."""
        return self.rcut

    def get_rcut_smth(self) -> float:
        """Return the outer cutoff used as the common smoothing radius."""
        return self.rcut

    def get_sel(self) -> list[int]:
        """Return an effectively unbounded neighbor capacity.

        The descriptor is graph-native and carries every neighbor within the
        cutoff, so it imposes no capacity. The common ``sel`` ABI still
        requires a number; a value no environment can reach reports that
        absence of a bound.
        """
        return [999999]

    def get_ntypes(self) -> int:
        """Return the number of real atom types."""
        return self.ntypes

    def get_type_map(self) -> list[str] | None:
        """Return the ordered atom-type names."""
        return self.type_map

    def get_dim_out(self) -> int:
        r"""Return the complete invariant descriptor width ``D``.

        The flat equivariant moment state has width

        .. math::

           S=\sum_{\ell=0}^{L}(2\ell+1)C_\ell,

        where ``C_l = degree_channels[l]`` is derived from the scalar
        ``channels`` parameter. ``S`` controls edge-reduction work and
        moment-state memory, but these equivariant coefficients are not
        exposed directly to the fitting network.

        The invariant output width is

        .. math::

           D=2C_0
             +\sum_{\ell=1}^{L}\frac{C_\ell(C_\ell+1)}{2}
             +D_{\mathrm{bispectrum}}
             +K_1K_2
             +2.

        The two ``C_0`` blocks are the scalar moments and the center-type
        embedding. The summation contains the exact upper-triangular Gram for
        each non-scalar degree, ``K_l`` denotes the derived bispectrum rank,
        the quartic term is the projected ``|Q_b v_a|^2``, and the trailing
        pair is the two neighborhood masses. The mixing rank does not enter
        ``D``.

        ``D_bispectrum`` is the sum over allowed O(3)-even degree triples:

        - distinct degrees contribute ``K_l1 * K_l2 * K_l3``;
        - two equal degrees contribute one symmetric pair count
          ``K * (K + 1) // 2`` times the remaining rank;
        - three equal degrees contribute ``K * (K + 1) * (K + 2) // 6``.

        Returns
        -------
        int
            Width of the invariant descriptor consumed by the fitting network.
        """
        return self.readout.get_dim_out() + self.channels + 2

    def get_dim_emb(self) -> int:
        """Return zero because fitting receives no equivariant channels."""
        return 0

    def mixed_types(self) -> bool:
        """Return whether the descriptor consumes a mixed-type neighbor list."""
        return True

    def has_message_passing(self) -> bool:
        """Return whether source-node features are exchanged."""
        return False

    def has_message_passing_across_ranks(self) -> bool:
        """Return whether intermediate halo communication is required."""
        return False

    def need_sorted_nlist_for_lower(self) -> bool:
        """Return whether graph-lower edges must be destination sorted."""
        return False

    def get_env_protection(self) -> float:
        """Return the direction regularization scale in Å."""
        return self._EPS

    def uses_graph_lower(self) -> bool:
        """Return whether graph-native lowering is supported."""
        return True

    def graph_edge_dtype(self) -> str:
        """Return the edge-geometry dtype accepted by graph deployment.

        Returns
        -------
        str
            ``"float32"`` for compressed float32 inference, otherwise
            ``"float64"``.
        """
        precision = np.dtype(PRECISION_DICT[self.precision]).name
        return "float32" if self.compress and precision == "float32" else "float64"

    def reinit_exclude(
        self,
        exclude_types: list[tuple[int, int]] | None = None,
    ) -> None:
        """Rebuild the ordered pair-exclusion mask."""
        if exclude_types is None:
            exclude_types = []
        self.exclude_types = list(exclude_types)
        self.emask = PairExcludeMask(
            self.ntypes,
            exclude_types=self.exclude_types,
        )
