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
    ChargeStateEmbedding,
    InvariantReadout,
    OrderedPairFiLM,
    SpinChannels,
    build_angular_basis,
    build_moment_indices,
    canonicalize_charge_spin,
    degree_offsets,
    derive_bispectrum_ranks,
    derive_degree_channels,
    derive_spin_channels,
    validate_charge_state,
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
    use_spin
        Per-type flags marking which atom types carry a magnetic moment. When
        given, the descriptor conditions on a per-node spin vector and
        declares :meth:`supports_native_spin`. ``None`` reproduces the
        spin-free descriptor exactly.
    add_chg_spin_ebd
        Whether to condition on the frame-level total charge and spin
        multiplicity. This is unrelated to ``use_spin``, which carries a
        per-atom magnetic moment.
    default_chg_spin
        Fallback ``[charge, multiplicity]`` used when a caller supplies no
        explicit condition. Compression bakes this value, so a deployed
        artifact requires it.
    spin
        Reserved for descriptor API compatibility; only ``None`` is supported.
        Native spin is configured through ``use_spin``.

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
        "spin_pair",
        "spin_type",
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
        use_spin: list[bool] | None = None,
        add_chg_spin_ebd: bool = False,
        default_chg_spin: list[float] | None = None,
        spin: None = None,
    ) -> None:
        # === Step 1. Validate the public architecture contract ===
        if spin is not None:
            raise NotImplementedError(
                "DPA4C configures native spin through `use_spin`; the `spin` "
                "argument of the common descriptor ABI is not supported."
            )
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
        default_chg_spin = (
            None
            if default_chg_spin is None
            else validate_charge_state(default_chg_spin)
        )
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
        self.use_spin = None if use_spin is None else [bool(flag) for flag in use_spin]
        self.add_chg_spin_ebd = bool(add_chg_spin_ebd)
        self.default_chg_spin = default_chg_spin
        # The spin branch reads the leading channels of the shared radial map,
        # so its width is derived rather than exposed.
        self.spin_channels = (
            0 if self.use_spin is None else derive_spin_channels(degree_channels)
        )
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
            spin_channels=self.spin_channels,
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
        self.spin = (
            None
            if self.use_spin is None
            else SpinChannels(
                self.ntypes,
                self.degree_channels,
                self.use_spin,
                precision=self.precision,
                trainable=self.trainable,
                seed=child_seed(seed, 5),
            )
        )
        self.charge_spin_embedding = (
            ChargeStateEmbedding(
                self.channels,
                self.pair_film.pair_hidden_width,
                precision=self.precision,
                trainable=self.trainable,
                seed=child_seed(seed, 6),
            )
            if self.add_chg_spin_ebd
            else None
        )

        # === Step 4. Lay out the flat moment payload ===
        # Degree zero owns the leading `channels` entries of the flat layout,
        # so the non-scalar block is exactly its complement and needs no
        # separate degree index. The spin families, when present, are appended
        # after the geometric degrees, leaving every geometric offset intact.
        channel_index, harmonic_index = build_moment_indices(self.degree_channels)
        self.angular_channel_index = channel_index[self.channels :]
        self.angular_harmonic_index = harmonic_index[self.channels :]
        self.degree_offsets = degree_offsets(self.degree_channels)

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
        spin: Array | None = None,
        charge_spin: Array | None = None,
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
        spin
            Per-node spin vectors with shape ``(N, 3)`` on the same flat node
            axis as ``atype``, including ghost and padding rows. Mandatory
            when the descriptor is configured with ``use_spin`` and ignored
            otherwise.
        charge_spin
            Frame-level total charge and spin multiplicity with shape
            ``(nf, 2)``, or a single pair broadcast over the frames. Read
            only when the descriptor is configured with ``add_chg_spin_ebd``,
            which then falls back to ``default_chg_spin`` if this is absent.

        Returns
        -------
        descriptor
            Rotation- and permutation-invariant node features with shape
            ``(N, get_dim_out())`` and the same floating dtype as ``edge_vec``.
        rot_mat
            ``None``. DPA4C does not expose an equivariant fitting input.

        Raises
        ------
        ValueError
            If the descriptor is spin conditioned and ``spin`` is absent, or
            charge conditioned with neither ``charge_spin`` nor a default.
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
        descriptor, _ = self.evaluate_graph(
            graph,
            atype,
            type_embedding,
            spin,
            self.require_charge_spin(
                charge_spin, graph.n_node.shape[0], graph.edge_vec
            ),
        )

        # === Step 3. Restore the graph input dtype ===
        if descriptor.dtype != in_dtype:
            descriptor = xp.astype(descriptor, in_dtype)
        return descriptor, None

    def require_spin(self, spin: Array | None) -> Array:
        """Return the per-node moment a spin-conditioned descriptor must receive.

        A missing moment is an error rather than a vanishing one. Substituting
        zeros would report an identically zero magnetic force, which in
        molecular dynamics is indistinguishable from frozen moments under a
        plausible energy. A corpus that is only partially labelled is admitted
        through ``model.spin.allow_missing_label``, which relaxes the ``spin``
        data requirement to optional with a zero default, so the data pipeline
        supplies an explicit zero moment and this contract still holds.

        Parameters
        ----------
        spin
            Per-node spin vectors with shape ``(N, 3)``, or ``None``.

        Returns
        -------
        Array
            The same moments, unchanged.

        Raises
        ------
        ValueError
            If ``spin`` is ``None``.
        """
        if spin is None:
            raise ValueError(
                "A spin-conditioned DPA4C requires a per-node magnetic "
                "moment. Set `model.spin.allow_missing_label` to admit "
                "systems that carry no spin label; the data pipeline then "
                "supplies an explicit zero moment."
            )
        return spin

    def require_charge_spin(
        self,
        charge_spin: Array | None,
        nf: int,
        ref: Array,
    ) -> Array | None:
        """Resolve the frame condition a charge-conditioned descriptor needs.

        Parameters
        ----------
        charge_spin
            Frame conditions supplied by the caller, or ``None``.
        nf
            Number of frames on the flat node axis.
        ref
            Reference array supplying the compute namespace, dtype and device.

        Returns
        -------
        Array or None
            Frame conditions with shape ``(nf, 2)``, or ``None`` for a
            descriptor without charge conditioning.
        """
        if self.charge_spin_embedding is None:
            return None
        return canonicalize_charge_spin(
            charge_spin,
            self.default_chg_spin,
            nf=nf,
            ref=ref,
        )

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
            Frame-level total charge and spin multiplicity with shape
            ``(F, 2)``. Read only when the descriptor is configured with
            ``add_chg_spin_ebd``.

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

        del fparam, comm_dict
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
            None,
            self.require_charge_spin(charge_spin, nf, graph.edge_vec),
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
        spin: Array | None = None,
        charge_spin: Array | None = None,
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
        spin
            Per-node spin vectors with shape ``(N, 3)``, mandatory for a
            spin-conditioned descriptor and ignored otherwise.
        charge_spin
            Canonical frame conditions with shape ``(nf, 2)``, mandatory for
            a charge-conditioned descriptor and ignored otherwise.

        Returns
        -------
        descriptor
            Invariant node features with shape ``(N, get_dim_out())``.
        envelope
            Masked per-edge C³ envelope with shape ``(E, 1)``.

        Raises
        ------
        ValueError
            If the descriptor is spin conditioned and ``spin`` is absent.
        """
        from deepmd.dpmodel.utils.neighbor_graph import (
            frame_id_from_n_node,
        )

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

        # === Step 2. Embed the frame condition ===
        # The two emitted vectors reach the two places the type embedding
        # enters: the centre type tail and the ordered pair encoder. Both
        # frames and edges address them through the node-to-frame map, an
        # edge inheriting the frame of the centre it reduces onto.
        type_shift, pair_hidden_bias, edge_hidden_bias = None, None, None
        if self.charge_spin_embedding is not None:
            type_shift, pair_hidden_bias = self.charge_spin_embedding.call(charge_spin)
            frame_index = frame_id_from_n_node(graph.n_node, n_total)
            edge_hidden_bias = self.gather_rows(
                pair_hidden_bias,
                self.gather_rows(frame_index, dst, xp),
                xp,
            )
            type_shift = self.gather_rows(type_shift, frame_index, xp)

        center_type_embedding = self.build_center_type_features(
            type_embedding,
            atype,
            type_shift,
        )
        pair_latent = self.pair_film.pair_latent(type_embedding)

        # === Step 3. Condition the per-node spin ===
        # The mask and the reference magnitude are applied once, so every
        # downstream spin route inherits them and the magnetic force of a
        # non-magnetic type vanishes identically rather than numerically.
        conditioned_spin = (
            None
            if self.spin is None
            else self.spin.conditioned_spin(self.require_spin(spin), atype)
        )

        # === Step 4. Build the masked edge amplitudes and harmonics ===
        amplitude, basis, envelope, spin_payload = self.build_edge_features(
            graph,
            atype,
            pair_latent,
            edge_hidden_bias,
            conditioned_spin,
        )

        # === Step 5. Reduce the degree-wise moments ===
        moments, divisors = self.aggregate_moments(
            amplitude,
            basis,
            envelope,
            spin_payload,
            None
            if conditioned_spin is None
            else self.spin.onsite_payload(conditioned_spin, atype),
            dst,
            n_total,
        )

        # === Step 6. Build calibrated invariant features ===
        return (
            self.build_invariant_descriptor(
                moments,
                center_type_embedding,
                divisors,
            ),
            envelope[:, None],
        )

    def build_center_type_features(
        self,
        type_embedding: Array,
        atype: Array,
        type_shift: Array | None,
    ) -> Array:
        """Gather the centre type embedding and add the frame condition.

        The padding type keeps its zero row. Its output row is discarded, but
        compressed inference conditions the real rows of a frozen type table
        and leaves the padding row untouched, so shifting it here would break
        the parity between the two paths.

        Parameters
        ----------
        type_embedding
            Complete type table with shape ``(ntypes + 1, channels)``.
        atype
            Flat node types with shape ``(N,)``.
        type_shift
            Per-node condition shift with shape ``(N, channels)``, or
            ``None`` for a descriptor without charge conditioning.

        Returns
        -------
        Array
            Centre type features with shape ``(N, channels)``.
        """
        xp = array_api_compat.array_namespace(type_embedding)
        features = self.gather_rows(type_embedding, atype, xp)
        if type_shift is None:
            return features
        real_type = xp.astype(atype < self.ntypes, features.dtype)
        return features + type_shift * real_type[:, None]

    def build_pair_conditioning(
        self,
        pair_latent: tuple[Array, Array],
        pair_index: Array,
        edge_hidden_bias: Array | None,
    ) -> tuple[Array, Array, Array | None, Array | None, Array | None]:
        """Evaluate the ordered pair conditioning of every edge.

        The heads are applied on the coarsest axis over which their argument
        is constant. Without a frame condition that axis is the ordered type
        pair, so the finite cache of :math:`(T+1)^2` rows is built once and
        gathered. With one, the argument additionally depends on the frame,
        and the product axis is larger than the edge count for the molecular
        systems a charge state describes, so the heads move to the edge axis.
        Both routes evaluate the same function.

        Parameters
        ----------
        pair_latent
            Condition-independent ordered-pair state from
            :meth:`~deepmd.dpmodel.descriptor.dpa4c_nn.pair_film.OrderedPairFiLM.pair_latent`.
        pair_index
            Ordered type-pair index of each edge with shape ``(E,)``.
        edge_hidden_bias
            Per-edge condition bias with shape ``(E, pair_hidden_width)``, or
            ``None`` for a descriptor without charge conditioning.

        Returns
        -------
        tuple
            Radial scale and shift with shape ``(E, channels)``, the
            mode-mixing matrices with shape ``(E, channels, radial_modes)``,
            and the ordered spin scale and shift with shape
            ``(E, spin_channels)``. The trailing three are ``None`` when
            their mechanism is disabled.
        """
        pre_activation, base_shift = pair_latent
        if edge_hidden_bias is None:
            return tuple(
                None if table is None else self.gather_rows(table, pair_index)
                for table in self.pair_film.heads(pre_activation, base_shift)
            )
        return self.pair_film.heads(
            self.gather_rows(pre_activation, pair_index) + edge_hidden_bias,
            self.gather_rows(base_shift, pair_index),
        )

    def build_edge_features(
        self,
        graph: Any,
        atype: Array,
        pair_latent: tuple[Array, Array],
        edge_hidden_bias: Array | None = None,
        conditioned_spin: Array | None = None,
    ) -> tuple[Array, Array, Array, Array | None]:
        r"""Build the enveloped edge amplitudes, harmonics, and spin payload.

        The ordered type pair :math:`(a,b)` rescales the one shared radial
        function :math:`g` and mixes the :math:`R` shared mode profiles
        :math:`q_\mu` with its own coefficients:

        .. math::

           \phi_{ijc}
           =\chi_{ij}\Bigl(
             \gamma_{ab,c}g_c(\rho_{ij})+\beta_{ab,c}
             +\sum_\mu U_{ab,c\mu}q_\mu(\rho_{ij})
            \Bigr).

        The spin payload reuses the radial map and the ordered pair index of
        this same stage, so the whole per-edge computation reads the radial
        table exactly once.

        Parameters
        ----------
        graph
            Neighbor graph in descriptor compute precision.
        atype
            Flat node types with shape ``(N,)``.
        pair_latent
            Condition-independent ordered-pair state from
            :meth:`~deepmd.dpmodel.descriptor.dpa4c_nn.pair_film.OrderedPairFiLM.pair_latent`.
        edge_hidden_bias
            Per-edge frame-condition bias with shape
            ``(E, pair_hidden_width)``, or ``None`` for a descriptor without
            charge conditioning.
        conditioned_spin
            Conditioned per-node spin with shape ``(N, 3)``, or
            ``None`` for a spin-free descriptor.

        Returns
        -------
        amplitude
            Masked edge amplitudes with shape ``(E, channels)``.
        basis
            Masked Cartesian harmonics with shape ``(E, (lmax + 1) ** 2)``.
        envelope
            Masked C³ envelope with shape ``(E,)``.
        spin_payload
            Masked per-edge spin payload with shape
            ``(E, spin.edge_width)``, or ``None`` for a spin-free descriptor.
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
        scale, shift, mixing, spin_scale, spin_shift = self.build_pair_conditioning(
            pair_latent,
            pair_index,
            edge_hidden_bias,
        )
        amplitude = radial * scale + shift

        # === Step 4. Add the pair-conditioned radial mode residual ===
        # Each ordered pair selects its own combination of the R shared mode
        # profiles. The contraction is written as a broadcast product reduced
        # over the mode axis rather than as a batched matrix-vector product:
        # one GEMV per edge leaves the tiny C-by-R operands far short of
        # memory bandwidth, whereas the reduction is a plain streaming pass.
        # Expanding the ordered table per edge dominates the cost either way.
        if mixing is not None:
            modes = self.radial_mode_head(radial_hidden)  # (E, R)
            amplitude = amplitude + xp.sum(mixing * modes[:, None, :], axis=-1)

        # === Step 5. Build the spin payload on the same radial evaluation ===
        # The bond-projected family reads the same regularized direction as the
        # harmonics, so the spin branch contributes an angular term to the
        # coordinate gradient alongside the radial one.
        spin_payload = (
            None
            if conditioned_spin is None
            else self.spin.edge_payload(
                conditioned_spin,
                atype,
                src,
                direction,
                radial,
                envelope[:, 0],
                spin_scale,
                spin_shift,
            )
        )

        # === Step 6. Gate the amplitude and build the masked harmonics ===
        return (
            amplitude * envelope,
            self.build_angular_basis(direction) * mask,
            envelope[:, 0],
            spin_payload,
        )

    def aggregate_moments(
        self,
        amplitude: Array,
        basis: Array,
        envelope: Array,
        spin_payload: Array | None,
        spin_onsite: Array | None,
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
        spin_payload
            Masked per-edge spin payload with shape ``(E, spin.edge_width)``,
            or ``None``. It carries the same squared envelope as every
            non-scalar geometric moment and therefore shares the normalizer
            :math:`n^{(+)}`.
        spin_onsite
            Node-local on-site spin payload with shape
            ``(N, spin.node_width)``, or ``None``. It is appended after the
            division so that the invariants it enters carry exactly one
            neighborhood normalizer, contributed by its neighbour partner.
        dst
            Destination node indices with shape ``(E,)``.
        n_total
            Number of output nodes ``N``.

        Returns
        -------
        moments
            Flat normalized moments with shape ``(N, S)``, where
            ``S = sum((2 * l + 1) * degree_channels[l])`` plus the spin
            moment width when the descriptor is spin conditioned.
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
        parts = [
            envelope_squared[:, None],
            (envelope_squared * envelope_squared)[:, None],
            amplitude,
            xp.take(amplitude, channel_index, axis=1)
            * xp.take(basis, harmonic_index, axis=1)
            * envelope[:, None],
        ]
        if spin_payload is not None:
            parts.append(spin_payload)
        reduced = segment_sum(xp.concat(parts, axis=1), dst, n_total)

        scalar_end = 2 + self.channels
        floor = self._DEGREE_NORM_FLOOR
        divisors = xp.sqrt(reduced[:, :2] + floor)
        normalized = [
            reduced[:, 2:scalar_end] / divisors[:, :1],
            reduced[:, scalar_end:] / divisors[:, 1:],
        ]
        if spin_onsite is not None:
            normalized.append(spin_onsite)
        return xp.concat(normalized, axis=1), divisors

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
        blocks = [self.readout.call(moments)]
        if self.spin is not None:
            geometric_width = self.degree_offsets[-1]
            blocks.append(
                self.spin.call(
                    moments[:, geometric_width:],
                    xp.reshape(
                        moments[:, self.degree_offsets[2] : self.degree_offsets[3]],
                        (moments.shape[0], 5, self.degree_channels[2]),
                    ),
                )
            )
        # The two divisors close the geometric block, so the spin invariants
        # precede them and the center-type tail keeps its trailing position.
        blocks.extend([divisors, center_type_embedding])
        descriptor = xp.concat(blocks, axis=-1)
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
            "spin",
            "charge_spin_embedding",
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
        every remaining width;         ``use_amp`` selects the precision policy that a
        backend attaches to the shared layers, so a replica that autocasts
        against layers configured without it would silently lose the effect;
        ``trainable`` decides whether
        those layers carry gradients at all; ``type_map`` fixes what the rows
        of the shared type table mean; ``use_spin`` fixes both the presence
        and the row meaning of the shared spin tables; ``add_chg_spin_ebd``
        fixes the presence of the condition module and the width of the pair
        encoder head it drives. Precision itself enters through its resolved
        dtype so that equivalent spellings agree.

        ``default_chg_spin`` is deliberately absent: it is a fallback for a
        missing input rather than a property of the shared parameters, so two
        branches may legitimately default to different charge states.

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
            None if self.use_spin is None else tuple(self.use_spin),
            self.add_chg_spin_ebd,
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
            Sampled training systems or a callable returning them. A
            spin-conditioned descriptor additionally requires the per-atom
            moment on every system, under either ``model_spin`` or ``spin``.
        path
            Optional statistics path. Model-dependent calibration is always
            recomputed and therefore does not consume this path.

        Raises
        ------
        ValueError
            If a geometric coordinate is degenerate over the sample, or if a
            spin-conditioned descriptor is calibrated on a system that carries
            no moment.
        """
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_neighbor_graph,
        )

        del path
        sampled = merged() if callable(merged) else merged
        if not sampled:
            return

        # The reference magnitudes rescale the spin before it reaches the
        # descriptor, so they have to be fixed before the output coordinates
        # are measured.
        if self.spin is not None:
            self.spin.set_spin_reference(self._measure_spin_reference(sampled))

        xp = array_api_compat.array_namespace(self.stddev)
        device = array_api_compat.device(self.stddev)
        dtype = self.stddev.dtype
        mean_backup, stddev_backup = self.mean, self.stddev
        self.mean = xp.zeros_like(self.mean)
        self.stddev = xp.ones_like(self.stddev)
        geometry_dim = self.get_dim_out() - self.channels
        square_sum = np.zeros(geometry_dim, dtype=np.float64)
        value_sum = np.zeros(geometry_dim, dtype=np.float64)
        # A spin coordinate is exactly zero on every node that carries no
        # magnetic information, so pooling it over all nodes would scale the
        # preconditioner with the magnetic fraction of the sample, that is
        # with the stoichiometry rather than with the physics. The spin block
        # is therefore averaged over the nodes on which it is active.
        #
        # Activity is a property of the coordinate block, not of the value. A
        # geometric coordinate is legitimately zero on an atom with no
        # neighbour inside the cutoff, and counting only nonzero values would
        # scale the geometric preconditioner with the vacuum fraction of the
        # sample -- the same bias, transposed. The geometric block therefore
        # keeps the plain node count.
        spin_block = slice(self.readout.get_dim_out(), geometry_dim - 2)
        active_count = np.zeros(geometry_dim, dtype=np.float64)

        try:
            for system in sampled:
                for frame in self._calibration_frames(system):
                    coord = xp.asarray(frame["coord"], dtype=dtype, device=device)
                    atype = xp.asarray(frame["atype"], device=device)
                    box = (
                        None
                        if frame["box"] is None
                        else xp.asarray(frame["box"], dtype=dtype, device=device)
                    )
                    spin = (
                        None
                        if frame["spin"] is None
                        else xp.asarray(frame["spin"], dtype=dtype, device=device)
                    )
                    charge_spin = (
                        None
                        if frame["charge_spin"] is None
                        else xp.asarray(
                            frame["charge_spin"], dtype=dtype, device=device
                        )
                    )
                    graph = build_neighbor_graph(
                        coord,
                        atype,
                        box,
                        self.get_rcut(),
                    )
                    output, _ = self.call_graph(
                        graph,
                        xp.reshape(atype, (-1,)),
                        spin=None if spin is None else xp.reshape(spin, (-1, 3)),
                        charge_spin=charge_spin,
                    )
                    output_np = to_numpy_array(output).reshape(
                        -1,
                        self.get_dim_out(),
                    )
                    if output_np.shape[0] == 0:
                        continue
                    geometry = output_np[:, :geometry_dim]
                    square_sum += np.sum(np.square(geometry, dtype=np.float64), axis=0)
                    value_sum += np.sum(geometry, axis=0, dtype=np.float64)
                    active_count += geometry.shape[0]
                    active_count[spin_block] += (
                        np.count_nonzero(geometry[:, spin_block], axis=0)
                        - geometry.shape[0]
                    )
        finally:
            self.mean, self.stddev = mean_backup, stddev_backup

        if not np.any(active_count > 0.0):
            return
        # A coordinate that never activates carries no information to
        # precondition. That is the normal state of every spin coordinate on a
        # demagnetized calibration corpus, which is exactly the corpus used to
        # pretrain a model that is later fine-tuned on magnetic data, so it
        # takes the identity preconditioner rather than an error.
        measured = active_count > 0.0
        feature_rms = np.ones(geometry_dim, dtype=np.float64)
        feature_rms[measured] = np.sqrt(
            square_sum[measured] / active_count[measured],
        )
        geometric = np.zeros(geometry_dim, dtype=bool)
        geometric[: self.readout.get_dim_out()] = True
        geometric[geometry_dim - 2 :] = True
        degenerate = geometric & (
            ~np.isfinite(feature_rms) | (feature_rms <= self._STAT_EPS)
        )
        if np.any(degenerate):
            raise ValueError(
                "DPA4C output calibration requires non-degenerate finite "
                f"geometric features, got RMS values {feature_rms.tolist()}"
            )
        type_table = to_numpy_array(self.type_embedding.call())[: self.ntypes]
        target_rms = float(np.sqrt(np.mean(np.square(type_table, dtype=np.float64))))
        if not math.isfinite(target_rms) or target_rms <= self._STAT_EPS:
            raise ValueError(
                f"DPA4C type embedding has a degenerate calibration RMS {target_rms}"
            )
        # A coordinate earns a preconditioner only where its measured scale is
        # meaningful. The geometric block is already required to be
        # non-degenerate above; the spin block is not, because a corpus whose
        # moments are uniformly weak drives the quartic spin coordinates to a
        # vanishing root mean square, and dividing by it would hand them an
        # unbounded gain. Those coordinates keep the identity instead, which is
        # the same treatment a coordinate that never activates receives.
        conditioned = measured & np.isfinite(feature_rms)
        conditioned &= feature_rms > self._STAT_EPS
        geometry_stddev = np.ones(geometry_dim, dtype=np.float64)
        geometry_stddev[conditioned] = feature_rms[conditioned] / target_rms
        geometry_mean = np.zeros(geometry_dim, dtype=np.float64)

        # The two moment divisors are the only outputs carrying their
        # information on a large offset rather than around zero: their RMS
        # exceeds a typical invariant by two orders of magnitude, so scaling
        # alone would leave them at one plus a small fluctuation. They are
        # standardized; every other coordinate keeps the shared RMS
        # preconditioner, whose zero mean the readout construction justifies.
        mass = slice(geometry_dim - 2, geometry_dim)
        mass_count = active_count[mass]
        mass_mean = value_sum[mass] / mass_count
        mass_stddev = np.sqrt(
            np.maximum(square_sum[mass] / mass_count - np.square(mass_mean), 0.0)
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

    def _calibration_frames(self, system: dict) -> list[dict]:
        """Draw the calibration frames of one sampled system.

        The frames are taken on a linear index grid, so the draw is
        deterministic and spreads over the whole system rather than over its
        leading frames.

        Parameters
        ----------
        system
            Sampled system carrying ``coord``, ``atype``, an optional ``box``,
            for a spin-conditioned descriptor the per-atom moment under either
            ``model_spin`` or ``spin``, and for a charge-conditioned
            descriptor the frame condition under ``charge_spin``.

        Returns
        -------
        list[dict]
            One entry per drawn frame, each with a leading frame axis of
            length one and ``spin`` and ``charge_spin`` entries that are
            ``None`` when their mechanism is disabled or, for the frame
            condition, when the descriptor falls back to its default.

        Raises
        ------
        ValueError
            If a spin-conditioned descriptor is calibrated on a system that
            carries no moment under either key.
        """
        coord = to_numpy_array(system["coord"])
        nframes = coord.shape[0]
        coord = np.reshape(coord, (nframes, -1, 3))
        atype = np.reshape(to_numpy_array(system["atype"]), (nframes, -1))
        box = system.get("box")
        box = None if box is None else np.reshape(to_numpy_array(box), (nframes, -1))
        # The two keys are the two packings the training pipelines use: the
        # native-spin route hands the moment through as ``spin``, while the
        # virtual-atom route packs the model-facing arrays under a ``model_``
        # prefix so they survive next to the physical ones.
        spin = (
            None
            if self.spin is None
            else self.require_spin(system.get("model_spin", system.get("spin")))
        )
        spin = (
            None if spin is None else np.reshape(to_numpy_array(spin), (nframes, -1, 3))
        )
        # The calibration must see the sampled distribution of charge states,
        # because the fixed diagonal preconditioner it freezes has to hold for
        # every one of them. A system without the key falls back to the
        # configured default at the descriptor boundary. A system that states
        # one condition for all of its frames is broadcast here, so that the
        # calibration accepts exactly the shapes evaluation accepts.
        charge_spin = None if not self.add_chg_spin_ebd else system.get("charge_spin")
        charge_spin = (
            None
            if charge_spin is None
            else np.broadcast_to(
                np.reshape(to_numpy_array(charge_spin), (-1, 2)),
                (nframes, 2),
            )
        )
        indices = np.linspace(
            0,
            nframes - 1,
            num=min(nframes, self._STAT_FRAMES_PER_SAMPLE),
            dtype=np.int64,
        )
        return [
            {
                "coord": coord[index : index + 1],
                "atype": atype[index : index + 1],
                "box": None if box is None else box[index : index + 1],
                "spin": None if spin is None else spin[index : index + 1],
                "charge_spin": (
                    None if charge_spin is None else charge_spin[index : index + 1]
                ),
            }
            for index in indices
        ]

    def _measure_spin_reference(self, sampled: list[dict]) -> np.ndarray:
        """Measure the per-type root-mean-square magnetic moment.

        The spin invariants are quadratic and quartic in the spin, so a
        chemistry whose moments differ by a factor of three spreads the
        quartic coordinates by two orders of magnitude. Rescaling the spin by
        a per-type reference collapses that spread before the fixed diagonal
        preconditioner sees it. The reference is a constant of the type, so
        the rescaled spin stays linear in the input and every smoothness
        property is preserved.

        Parameters
        ----------
        sampled
            Sampled training systems, each carrying a per-atom moment.

        Returns
        -------
        numpy.ndarray
            Strictly positive reference magnitudes with shape
            ``(ntypes + 1,)``. A type that the sample never observes with a
            finite moment keeps a unit reference, which leaves its spin
            untouched; the estimator therefore never emits a zero.
        """
        square_sum = np.zeros(self.ntypes + 1, dtype=np.float64)
        count = np.zeros(self.ntypes + 1, dtype=np.float64)
        for system in sampled:
            for frame in self._calibration_frames(system):
                atype = np.reshape(frame["atype"], (-1,))
                spin = np.reshape(frame["spin"], (-1, 3))
                magnitude = np.sum(np.square(spin, dtype=np.float64), axis=-1)
                np.add.at(square_sum, atype, magnitude)
                np.add.at(count, atype, 1.0)
        reference = np.ones(self.ntypes + 1, dtype=np.float64)
        observed = (count > 0.0) & (square_sum > count * self._STAT_EPS)
        reference[observed] = np.sqrt(square_sum[observed] / count[observed])
        return reference

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
            "use_spin": self.use_spin,
            "add_chg_spin_ebd": self.add_chg_spin_ebd,
            "default_chg_spin": self.default_chg_spin,
            "spin": None,
            "spin_channels": (None if self.spin is None else self.spin.serialize()),
            "charge_spin_embedding": (
                None
                if self.charge_spin_embedding is None
                else self.charge_spin_embedding.serialize()
            ),
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
        spin_channels = data.pop("spin_channels")
        charge_spin_embedding = data.pop("charge_spin_embedding")

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
        obj.spin = (
            None if spin_channels is None else SpinChannels.deserialize(spin_channels)
        )
        obj.charge_spin_embedding = (
            None
            if charge_spin_embedding is None
            else ChargeStateEmbedding.deserialize(charge_spin_embedding)
        )
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

        A spin-conditioned descriptor appends the invariants of the spin
        channels to the geometric block, ahead of the two divisors.

        Returns
        -------
        int
            Width of the invariant descriptor consumed by the fitting network.
        """
        spin_dim = 0 if self.spin is None else self.spin.get_dim_out()
        return self.readout.get_dim_out() + spin_dim + self.channels + 2

    def get_dim_emb(self) -> int:
        """Return zero because fitting receives no equivariant channels."""
        return 0

    def mixed_types(self) -> bool:
        """Return whether the descriptor consumes a mixed-type neighbor list."""
        return True

    def has_message_passing(self) -> bool:
        """Return whether source-node features are exchanged.

        The spin branch reads the raw source spin, which is a node input
        rather than a derived feature, so a spin-conditioned descriptor
        remains message-passing free.
        """
        return False

    def supports_native_spin(self) -> bool:
        """Return whether ``call_graph`` conditions on a per-node spin."""
        return self.spin is not None

    def supports_charge_spin(self) -> bool:
        """Return whether ``call_graph`` conditions on a frame charge state."""
        return self.charge_spin_embedding is not None

    def get_dim_chg_spin(self) -> int:
        """Return the runtime width of the frame condition.

        Compression folds one charge state into the frozen type and ordered
        pair tables, so the resulting snapshot evaluates that state and
        consumes no runtime condition. Reporting zero is what routes such a
        model onto the compact canonical lower, whose argument list carries
        no conditioning slot.
        """
        return 0 if self.compress or self.charge_spin_embedding is None else 2

    def has_default_chg_spin(self) -> bool:
        """Return whether a fallback frame condition is configured."""
        return self.default_chg_spin is not None

    def get_default_chg_spin(self) -> list[float] | None:
        """Return the fallback ``[charge, multiplicity]``, if configured."""
        return self.default_chg_spin

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
