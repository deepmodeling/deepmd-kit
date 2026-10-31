# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Callable,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.neighbor_graph import (
        NeighborGraph,
    )

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
    xp_take_first_n,
)
from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.dpmodel.long_range import (
    BaseLongRange,
    check_long_range_support,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.vacuum_reference import (
    reference_charge_spin,
    reference_spin,
    resolve_vacuum_ref,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_atomic_model import (
    BaseAtomicModel,
)


def _extend_graph_aparam(
    aparam: Array,
    n_node: Array,
    n_local: Array,
    n_total: int,
) -> Array:
    """Expand frame-local atomic parameters onto a local-plus-halo node axis."""
    import array_api_compat

    from deepmd.dpmodel.utils.neighbor_graph import (
        frame_id_from_n_node,
        node_ownership_mask,
    )

    xp = array_api_compat.array_namespace(aparam, n_node, n_local)
    frame_id = frame_id_from_n_node(n_node, n_total=n_total)
    frame_end = xp.cumulative_sum(n_node)
    frame_start = frame_end - n_node
    node_index = xp.arange(
        n_total,
        dtype=n_node.dtype,
        device=array_api_compat.device(n_node),
    )
    index_in_frame = node_index - xp.take(frame_start, frame_id, axis=0)
    local_capacity = aparam.shape[1]
    sentinel = xp.zeros(
        (aparam.shape[0], 1, aparam.shape[2]),
        dtype=aparam.dtype,
        device=array_api_compat.device(aparam),
    )
    padded_aparam = xp.concat([aparam, sentinel], axis=1)
    padded_capacity = local_capacity + 1
    local_index = index_in_frame % padded_capacity
    flat_index = frame_id * padded_capacity + local_index
    flat_aparam = xp.reshape(padded_aparam, (-1, aparam.shape[-1]))
    gathered = xp.take(flat_aparam, flat_index, axis=0)
    ownership = node_ownership_mask(n_node, n_local, n_total)
    return xp.where(ownership[:, None], gathered, xp.zeros_like(gathered))


@BaseAtomicModel.register("standard")
class DPAtomicModel(BaseAtomicModel):
    r"""Model give atomic prediction of some physical property.

    The atomic model computes atomic properties by first extracting a descriptor
    from the atomic environment, then passing it through a fitting network:

    .. math::
        \mathcal{D}^i = \mathcal{D}(\mathbf{R}^i, \mathbf{R}_j, \alpha_j),

    .. math::
        \mathbf{y}^i = \mathcal{F}(\mathcal{D}^i),

    where :math:`\mathcal{D}^i` is the descriptor for atom :math:`i`,
    :math:`\alpha_j` is the atom type of neighbor :math:`j`,
    :math:`\mathcal{F}` is the fitting network, and
    :math:`\mathbf{y}^i` is the predicted atomic property (energy, dipole, etc.).

    An energy model may carry a long-range module beside its fitting: it reads
    the same descriptor and adds its own atomic energy to the fitting output,
    so the atomic energy :math:`E^i` is the complete potential, including the
    interactions without a cutoff. The module is evaluated on the
    NeighborGraph route.

    Parameters
    ----------
    descriptor
            Descriptor
    fitting_net
            Fitting net
    type_map
            Mapping atom type to the name (str) of the type.
            For example `type_map[1]` gives the name of the type 1.
    long_range
            Long-range module added to the energy output of the fitting.

    """

    CONFIG_DERIVED_ARRAYS = ("vacuum_charge_spin", "vacuum_spin")

    def __init__(
        self,
        descriptor: BaseDescriptor,
        fitting: BaseFitting,
        type_map: list[str],
        long_range: BaseLongRange | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(type_map, **kwargs)
        self.descriptor = descriptor
        self.fitting_net = fitting
        self.fitting_net.reinit_exclude(self.atom_exclude_types)
        self.long_range = long_range
        self.type_map = type_map
        self.add_chg_spin_ebd: bool = self.descriptor.has_chg_spin_ebd()
        # Structural capability: only descriptors with a native spin
        # conditioning mechanism (currently DPA4) accept a ``spin`` kwarg on
        # ``call_graph`` at all -- unlike ``charge_spin``, which every
        # descriptor's dense ``call()`` accepts (and ignores) for interface
        # stability, the graph-native ``call_graph`` signature is
        # per-descriptor, so ``forward_atomic_graph`` must not pass the
        # keyword to a descriptor whose ``call_graph`` does not declare it
        # (that would be a ``TypeError``, not a no-op). Queried via the
        # ``supports_native_spin`` capability method declared on
        # ``BaseDescriptor`` (concrete default ``False``; DPA4 overrides).
        self._supports_native_spin: bool = self.descriptor.supports_native_spin()
        # Same capability method as ``supports_native_spin`` above, for the
        # frame-level ``charge_spin`` FiLM kwarg: only DPA4's ``call_graph``
        # declares it; other descriptors' ``call_graph`` would ``TypeError``
        # on an unconditional ``charge_spin=`` kwarg.
        self.supports_charge_spin: bool = self.descriptor.supports_charge_spin()
        self.add_spin_ebd: bool = (
            self._supports_native_spin and self.descriptor.use_spin is not None
        )
        if self.long_range is not None:
            self._check_long_range()
        super().init_out_stat()
        # === Reference of the isolated atoms ===
        # The fitting references its output only when the preset fixes the
        # bias of that output; the conditions of the reference atoms follow.
        resolve_vacuum_ref(self.fitting_net, self.preset_out_bias)
        self.init_vacuum_conditions()

    def _check_long_range(self) -> None:
        """Validate that the long-range module can join this descriptor and fitting."""
        from deepmd.dpmodel.utils.neighbor_graph import (
            ALL_PAIRS,
        )

        check_long_range_support(self.descriptor)
        if not self.descriptor.uses_graph_lower():
            raise NotImplementedError(
                "a long-range module is evaluated on the NeighborGraph lower, "
                "which this descriptor does not provide"
            )
        if "energy" not in self.fitting_output_def().keys():
            raise ValueError(
                "a long-range module adds to the energy output, which this "
                "fitting does not have"
            )
        if self.atom_exclude_types:
            raise NotImplementedError(
                "`atom_exclude_types` is not supported with a long-range module, "
                "which attributes each pair energy to both atoms of the pair"
            )
        if self.pair_exclude_types and any(
            name != ALL_PAIRS for name in self.long_range.required_topologies()
        ):
            raise NotImplementedError(
                "`pair_exclude_types` is not supported with a long-range module "
                "that serves periodic systems, whose reciprocal-space sum is "
                "global and excludes no pair"
            )
        if self.long_range.uses_charge_spin() and not self.add_chg_spin_ebd:
            raise ValueError(
                "the long-range module reads the total charge from "
                "`charge_spin`, but the descriptor is not configured with the "
                "charge/spin condition (`add_chg_spin_ebd`)"
            )

    def init_vacuum_conditions(self) -> None:
        """Build the conditioning inputs of the reference atoms for the type map.

        A table exists for every condition the descriptor takes when the
        fitting references the isolated atoms; the type names must then be
        element symbols.
        """
        vacuum_ref = self.fitting_net.vacuum_ref
        self.vacuum_charge_spin = (
            reference_charge_spin(self.type_map)
            if vacuum_ref and self.add_chg_spin_ebd
            else None
        )
        self.vacuum_spin = (
            reference_spin(self.type_map) if vacuum_ref and self.add_spin_ebd else None
        )

    def vacuum_conditions(self) -> dict[str, Array]:
        """Conditioning inputs of one isolated neutral ground-state atom per type.

        Returns
        -------
        dict[str, Array]
            The charge/spin conditions with shape (ntypes, 2) under
            ``charge_spin`` and the spin vectors with shape (ntypes, 3) under
            ``spin``, for the conditions the descriptor takes.
        """
        conditions = {}
        if self.vacuum_charge_spin is not None:
            conditions["charge_spin"] = self.vacuum_charge_spin
        if self.vacuum_spin is not None:
            conditions["spin"] = self.vacuum_spin
        return conditions

    def adam_route_patterns(self) -> list[str]:
        """Prefix the descriptor's AdamW patterns with its parameter path."""
        return [f"descriptor.{p}" for p in self.descriptor.adam_route_patterns()]

    def has_chg_spin_ebd(self) -> bool:
        """Check if the model has charge spin embedding."""
        return self.add_chg_spin_ebd

    def get_dim_chg_spin(self) -> int:
        """Get the dimension of charge_spin input."""
        if self.add_chg_spin_ebd:
            return self.descriptor.get_dim_chg_spin()
        return 0

    def get_default_chg_spin(self) -> list[float] | None:
        """Get the default charge_spin values."""
        if self.add_chg_spin_ebd:
            return self.descriptor.get_default_chg_spin()
        return None

    def uses_graph_lower(self) -> bool:
        """Delegates to this model's own descriptor."""
        return bool(self.descriptor.uses_graph_lower())

    def fused_decomposition(self) -> "tuple[DPAtomicModel, None]":
        """A descriptor-fitting model is its own learned part.

        The learned part carries the long-range module of the model, if any;
        a fused pipeline that does not evaluate long-range modules declines a
        learned part with one.
        """
        return self, None

    def has_long_range(self) -> bool:
        """Whether this model carries a long-range module."""
        return self.long_range is not None

    def required_topologies(self) -> tuple[str, ...]:
        """The topologies of the long-range module."""
        if self.long_range is None:
            return ()
        return self.long_range.required_topologies()

    def topology_params(self) -> dict[str, dict[str, Any]]:
        """The build parameters of the topologies of the long-range module."""
        if self.long_range is None:
            return {}
        return self.long_range.topology_params()

    def has_message_passing_across_ranks(self) -> bool:
        """Delegates to this model's own descriptor."""
        return bool(self.descriptor.has_message_passing_across_ranks())

    def supports_edge_parallel(self) -> bool:
        """Delegates to this model's own descriptor.

        A long-range module sums over complete systems, so a model carrying
        one does not run under domain decomposition.
        """
        return self.long_range is None and bool(
            self.descriptor.supports_edge_parallel()
        )

    def dense_lower_supports_comm(self) -> bool:
        """Delegates to this model's own descriptor; a long-range module has no dense lower."""
        return self.long_range is None and bool(
            self.descriptor.dense_lower_supports_comm()
        )

    def uses_compact_edge_pairs(self) -> bool:
        """Delegates to this model's own descriptor."""
        return bool(self.descriptor.uses_compact_edge_pairs())

    def graph_edge_dtype(self) -> str:
        """Delegates to this model's own descriptor."""
        return str(self.descriptor.graph_edge_dtype())

    def supports_graph_export(self) -> bool:
        """Delegates to this model's own descriptor."""
        return bool(self.descriptor.supports_graph_export())

    def compression_needs_min_nbor_dist(self) -> bool:
        """Delegates to this model's own descriptor."""
        return bool(self.descriptor.compression_needs_min_nbor_dist())

    def supports_native_spin(self) -> bool:
        """Delegates to this model's own descriptor (cached at construction)."""
        return self._supports_native_spin

    def fitting_output_def(self) -> FittingOutputDef:
        """Get the output def of the fitting net."""
        return self.fitting_net.output_def()

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.descriptor.get_rcut()

    def get_sel(self) -> list[int]:
        """Get the neighbor selection."""
        return self.descriptor.get_sel()

    def set_case_embd(self, case_idx: int) -> None:
        """
        Set the case embedding of this atomic model by the given case_idx,
        typically concatenated with the output of the descriptor and fed into the fitting net.
        """
        self.fitting_net.set_case_embd(case_idx)
        if self.long_range is not None:
            self.long_range.set_case_embd(case_idx)

    def mixed_types(self) -> bool:
        """If true, the model
        1. assumes total number of atoms aligned across frames;
        2. uses a neighbor list that does not distinguish different atomic types.

        If false, the model
        1. assumes total number of atoms of each atom type aligned across frames;
        2. uses a neighbor list that distinguishes different atomic types.

        """
        return self.descriptor.mixed_types()

    def has_message_passing(self) -> bool:
        """Returns whether the atomic model has message passing."""
        return self.descriptor.has_message_passing()

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the atomic model needs sorted nlist when using `forward_lower`."""
        return self.descriptor.need_sorted_nlist_for_lower()

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Call descriptor enable_compression().

        A long-range module keeps its networks: it reads the output of the
        compressed descriptor like that of the uncompressed one. A compressed
        descriptor folds the charge/spin condition into its tables, so a
        constraint that reads the total charge from the condition is folded
        to the same state, the default condition of the descriptor.

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between atoms
        table_extrapolate
            The scale of model extrapolation
        table_stride_1
            The uniform stride of the first table
        table_stride_2
            The uniform stride of the second table
        check_frequency
            The overflow check frequency
        """
        self.descriptor.enable_compression(
            min_nbor_dist,
            table_extrapolate,
            table_stride_1,
            table_stride_2,
            check_frequency,
        )
        if self.long_range is not None and self.long_range.uses_charge_spin():
            self.long_range.fold_charge_state(self.descriptor.get_default_chg_spin())

    def forward_atomic(
        self,
        extended_coord: Array,
        extended_atype: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        comm_dict: dict | None = None,
        charge_spin: Array | None = None,
    ) -> dict[str, Array]:
        """Models' atomic predictions.

        Parameters
        ----------
        extended_coord
            coordinates in extended region
        extended_atype
            atomic type in extended region
        nlist
            neighbor list. nf x nloc x nsel
        mapping
            mapps the extended indices to local indices. nf x nall
        fparam
            frame parameter. nf x ndf
        aparam
            atomic parameter. nf x nloc x nda
        comm_dict
            MPI communication metadata for parallel inference. ``None`` for
            non-parallel inference (default). Forwarded to the descriptor.
        charge_spin
            charge and spin parameter for descriptor. nf x 2

        Returns
        -------
        result_dict
            the result dict, defined by the `FittingOutputDef`.

        """
        if self.long_range is not None:
            raise NotImplementedError(
                "a long-range module is evaluated on the NeighborGraph lower "
                "only; the dense (nlist) lower cannot evaluate this model"
            )
        nframes, nloc, nnei = nlist.shape
        atype = xp_take_first_n(extended_atype, 1, nloc)

        # Handle default charge_spin if descriptor supports it
        if self.add_chg_spin_ebd and charge_spin is None:
            default_cs = self.descriptor.get_default_chg_spin()
            if default_cs is not None:
                xp = array_api_compat.array_namespace(extended_coord)
                cs_array = xp.asarray(
                    default_cs,
                    dtype=extended_coord.dtype,
                    device=array_api_compat.device(extended_coord),
                )
                charge_spin = xp.tile(xp.reshape(cs_array, (1, -1)), (nframes, 1))

        # A bridged descriptor returns a sixth element, the per-atom source
        # gate of its bridging window; every other descriptor returns five.
        outputs = self.descriptor(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            comm_dict=comm_dict,
            charge_spin=charge_spin if self.add_chg_spin_ebd else None,
        )
        descriptor, rot_mat, g2, h2 = outputs[:4]
        node_gate = outputs[5] if len(outputs) > 5 else None
        # The vacuum descriptor of every type is handed to a fitting that
        # references its atoms; other fittings do not take the keyword. The
        # source gate fades the learned part of the fitting output: an atom
        # of a frozen pair keeps only the bias of its type.
        vacuum_kwargs = {}
        if self.fitting_net.needs_vacuum_descriptor():
            vacuum_kwargs["vacuum_descriptor"] = self.vacuum_descriptor()
        gate_kwargs = {} if node_gate is None else {"node_gate": node_gate}
        ret = self.fitting_net(
            descriptor,
            atype,
            gr=rot_mat,
            g2=g2,
            h2=h2,
            fparam=fparam,
            aparam=aparam,
            **vacuum_kwargs,
            **gate_kwargs,
        )
        return ret

    def vacuum_descriptor(self) -> Array:
        """Descriptor of an isolated atom of every type.

        Every type is evaluated as a single-atom frame without neighbors,
        conditioned as the neutral ground-state atom: zero charge with the
        ground-state multiplicity when the descriptor takes the charge/spin
        condition, and a spin vector of one Bohr magneton per unpaired
        electron when it takes the native spin.

        Descriptors that opt into graph vacuum evaluation use isolated nodes,
        independently of their advertised neighbor capacity; other descriptors
        retain their configured all-masked dense neighbor list.

        Returns
        -------
        Array
            The vacuum descriptor with shape (ntypes, dim_descrpt), in the
            array namespace, precision and device of the fitting bias.
        """
        bias = self.fitting_net.bias_atom_e
        xp = array_api_compat.array_namespace(bias)
        device = array_api_compat.device(bias)
        ntypes = self.get_ntypes()
        atype = xp.arange(ntypes, dtype=xp.int64, device=device)
        conditions = {
            name: xp.asarray(table, dtype=bias.dtype, device=device)
            for name, table in self.vacuum_conditions().items()
        }
        if self.uses_graph_lower() and self.descriptor.supports_graph_vacuum():
            from deepmd.dpmodel.utils.neighbor_graph import (
                NeighborGraph,
            )

            # One isolated node per frame, with the same two masked guard
            # edges as the graph builders. Never size this graph from sel:
            # graph-only descriptors may report an unbounded sentinel there.
            # The graph interface also carries native spin, which the dense
            # compatibility adapters need not support.
            graph = NeighborGraph(
                n_node=xp.ones((ntypes,), dtype=xp.int64, device=device),
                edge_index=xp.zeros((2, 2), dtype=xp.int64, device=device),
                edge_vec=xp.zeros((2, 3), dtype=bias.dtype, device=device),
                edge_mask=xp.zeros((2,), dtype=xp.bool, device=device),
            )
            return self.descriptor.call_graph(
                graph,
                atype,
                type_embedding=self.descriptor.graph_type_embedding_table(),
                **conditions,
            )[0]

        coord = xp.zeros((ntypes, 1, 3), dtype=bias.dtype, device=device)
        atype = xp.reshape(atype, (ntypes, 1))
        nlist = xp.full(
            (ntypes, 1, self.descriptor.get_nnei()),
            -1,
            dtype=xp.int64,
            device=device,
        )
        mapping = xp.zeros((ntypes, 1), dtype=xp.int64, device=device)
        if "spin" in conditions:
            conditions["spin"] = conditions["spin"][:, None, :]
        descriptor = self.descriptor(
            coord, atype, nlist, mapping=mapping, **conditions
        )[0]
        return xp.reshape(descriptor, (ntypes, -1))

    def fold_vacuum_reference(self) -> None:
        """Fold the vacuum reference of the fitting so the forward carries no reference atoms.

        The vacuum descriptor of every type is evaluated once with the current
        parameters and handed to the fitting, which folds the reference into
        its bias or stores the table (see its ``fold_vacuum_reference``).
        """
        fitting = self.fitting_net
        if fitting.needs_vacuum_descriptor():
            fitting.fold_vacuum_reference(self.vacuum_descriptor())

    def append_vacuum_frames(
        self,
        graph: "NeighborGraph",
        atype: Array,
        charge_spin: Array | None,
        spin: Array | None,
    ) -> tuple["NeighborGraph", Array, Array | None, Array | None]:
        """Append one isolated atom of every type as a single-node frame.

        The reference nodes follow the node axis of the real frames, padding
        included, as single-node frames: one node of every type without
        neighbors, conditioned as the neutral ground-state atom (zero charge
        and the ground-state multiplicity for the charge/spin conditioning, a
        spin vector of as many Bohr magnetons as unpaired electrons for the
        native spin). Node-wise operations of the descriptor leave the real
        nodes unaffected, so the descriptor rows of the reference nodes are the
        vacuum descriptor of every type.

        Parameters
        ----------
        graph : NeighborGraph
            Neighbor graph of the real frames.
        atype : Array
            Flat node types with shape (N,).
        charge_spin : Array, optional
            Charge/spin condition of the real frames with shape (nf, 2) or
            (1, 2); None takes the descriptor default.
        spin : Array, optional
            Per-node spin vectors with shape (N, 3), or None.

        Returns
        -------
        tuple
            The extended graph, node types, charge/spin conditions and spin
            vectors, each with the reference entries appended.
        """
        from deepmd.dpmodel.utils.neighbor_graph import (
            append_isolated_frames,
        )

        xp = array_api_compat.array_namespace(atype, graph.edge_vec)
        device = array_api_compat.device(atype)
        dtype = graph.edge_vec.dtype
        ntypes = self.get_ntypes()
        nf = graph.n_node.shape[0]
        conditions = self.vacuum_conditions()
        graph = append_isolated_frames(graph, atype.shape[0], ntypes)
        atype = xp.concat(
            [atype, xp.arange(ntypes, dtype=atype.dtype, device=device)], axis=0
        )
        if self.add_chg_spin_ebd:
            if charge_spin is None:
                default_cs = self.descriptor.get_default_chg_spin()
                if default_cs is None:
                    raise ValueError("`charge_spin` is required for this descriptor.")
                charge_spin = xp.reshape(
                    xp.asarray(default_cs, dtype=dtype, device=device), (1, 2)
                )
            # one row per real frame, one for the padding frame, one per type
            charge_spin = xp.concat(
                [
                    xp.broadcast_to(
                        xp.astype(xp.reshape(charge_spin, (-1, 2)), dtype), (nf, 2)
                    ),
                    xp.zeros((1, 2), dtype=dtype, device=device),
                    xp.asarray(conditions["charge_spin"], dtype=dtype, device=device),
                ],
                axis=0,
            )
        if self.add_spin_ebd and spin is not None:
            spin = xp.concat(
                [
                    xp.reshape(spin, (-1, 3)),
                    xp.asarray(conditions["spin"], dtype=spin.dtype, device=device),
                ],
                axis=0,
            )
        return graph, atype, charge_spin, spin

    def forward_atomic_graph(
        self,
        graph: "NeighborGraph",
        atype: Array,
        fparam: Array | None = None,
        aparam: Array | None = None,
        charge_spin: Array | None = None,
        spin: Array | None = None,
        comm_dict: dict | None = None,
        topologies: dict[str, Any] | None = None,
        node_mask: Array | None = None,
    ) -> dict[str, Array]:
        """Graph analogue of :meth:`forward_atomic` on the flat node axis.

        Runs the descriptor ``call_graph`` then the fitting ``call_graph`` PER NODE
        and returns the raw fitting dict on the flat ``(N, *)`` axis (no reduction
        or masking; the wrapper handles those). ``fparam`` is gathered to nodes by
        ``frame_id`` so each node sees its frame's parameter. The atomic energy
        of the long-range module is added to the energy output of the fitting.

        Parameters
        ----------
        graph
            neighbor graph for the local atoms (ghost-free)
        atype
            flat local atom types. N
        fparam
            frame parameter. nf x ndf
        aparam
            atomic parameter. N x nda
        charge_spin
            frame-level charge/spin conditioning, forwarded to the
            descriptor's ``call_graph`` only when
            ``self.supports_charge_spin`` (currently DPA4 only); ignored (not
            forwarded, never a ``TypeError``) for descriptors without that
            capability, keeping the interface stable for all of them.
        spin
            flat (N, 3) per-node spin, forwarded to the descriptor's
            ``call_graph``; None for spin-less models.
        comm_dict
            MPI communication metadata forwarded to the descriptor's
            ``call_graph`` (the message-passing part). ``None`` for
            non-parallel inference (default). Mirrors :meth:`forward_atomic`'s
            ``comm_dict`` on the dense route.
        topologies
            the topologies of :meth:`required_topologies` by name.
        node_mask
            flat mask of the real and owned nodes, which carry the fields of
            the long-range module.

        Returns
        -------
        result_dict
            the result dict on the flat node axis, defined by the `FittingOutputDef`.

        """
        from deepmd.dpmodel.utils.neighbor_graph import (
            frame_id_from_n_node,
        )

        xp = array_api_compat.array_namespace(graph.edge_vec)
        n_real = atype.shape[0]
        # === Step 1. Conditioning of the real nodes ===
        fparam_node = None
        if fparam is not None:
            # Pass the STATIC flat node count (``atype.shape[0] == N``) so the
            # helper does not fall back to ``int(sum(n_node))``: that int() on a
            # traced tensor breaks make_fx / torch.export
            # (``GuardOnDataDependentSymNode``) for the graph .pt2 export and
            # compiled-training paths when ``numb_fparam > 0``.
            frame_id = frame_id_from_n_node(graph.n_node, n_total=atype.shape[0])
            fparam_node = xp.take(fparam, frame_id, axis=0)  # (N, ndf)
        aparam_node = aparam
        if aparam is not None and graph.n_local is not None and aparam.ndim == 3:
            aparam_node = _extend_graph_aparam(
                aparam,
                graph.n_node,
                graph.n_local,
                atype.shape[0],
            )
        # The long-range module reads the frame conditions of the real frames,
        # with the defaults the descriptor and fitting fall back on.
        n_node = graph.n_node
        if self.long_range is not None:
            # The constraint and a compressed descriptor serve one charge
            # state; compressing the atomic model folds both to it.
            if self.long_range.uses_charge_spin() and self.get_dim_chg_spin() == 0:
                raise ValueError(
                    "the descriptor folds the charge/spin condition into its "
                    "compressed tables, while the long-range charge constraint "
                    "still reads it; compress the atomic model, which folds "
                    "both to one state"
                )
            fparam, charge_spin_frame = self._frame_conditions(
                fparam, charge_spin, n_node.shape[0], graph.edge_vec
            )
        # === Step 2. Vacuum reference nodes ===
        # Opted-in descriptors evaluate the reference in the same graph call.
        # Legacy descriptors must retain the dense reference used by folding.
        vacuum_ref = self.fitting_net.needs_vacuum_descriptor()
        graph_vacuum = vacuum_ref and self.descriptor.supports_graph_vacuum()
        if graph_vacuum:
            graph, atype, charge_spin, spin = self.append_vacuum_frames(
                graph, atype, charge_spin, spin
            )
        # === Step 3. Descriptor ===
        # Descriptor-owned: dpa1/dpa2 hand out their full tebd table; DPA4
        # embeds types internally from ``atype`` and returns None.
        type_embedding = self.descriptor.graph_type_embedding_table()
        # Only forward the ``spin``/``charge_spin`` keyword to descriptors
        # whose ``call_graph`` declares it.  Queried through the public
        # capability -- an override must reach THIS call site too, so the
        # cached field stays behind the method.
        spin_kwargs = {"spin": spin} if self.supports_native_spin() else {}
        charge_spin_kwargs = (
            {"charge_spin": charge_spin} if self.supports_charge_spin else {}
        )
        # A bridged descriptor returns a third element, the per-node source
        # gate of its bridging window; every other descriptor returns two.
        outputs = self.descriptor.call_graph(
            graph,
            atype,
            type_embedding=type_embedding,
            comm_dict=comm_dict,
            **spin_kwargs,
            **charge_spin_kwargs,
        )
        gg, rot_mat = outputs[0], outputs[1]
        node_gate = outputs[2] if len(outputs) > 2 else None
        vacuum_kwargs = {}
        if graph_vacuum:
            vacuum_kwargs["vacuum_descriptor"] = gg[n_real:]
            gg = gg[:n_real]
            atype = atype[:n_real]
            rot_mat = None if rot_mat is None else rot_mat[:n_real]
            node_gate = None if node_gate is None else node_gate[:n_real]
        elif vacuum_ref:
            vacuum_kwargs["vacuum_descriptor"] = self.vacuum_descriptor()
        # === Step 4. Fitting ===
        # The source gate of a bridged descriptor fades the learned part of
        # the fitting output: an atom of a frozen pair keeps only the bias of
        # its type, so along the closing of a pair the learned energy follows
        # the switch linearly while the analytical pair term takes over.
        gate_kwargs = {} if node_gate is None else {"node_gate": node_gate}
        ret = self.fitting_net.call_graph(
            gg,
            atype,
            gr=rot_mat,
            g2=None,
            h2=None,
            fparam=fparam_node,
            aparam=aparam_node,
            **vacuum_kwargs,
            **gate_kwargs,
        )
        # === Step 5. Long-range module ===
        # The module reads the descriptor of the real nodes; the reference
        # nodes appended above take no part in it.
        if self.long_range is not None:
            if topologies is None or node_mask is None:
                raise ValueError(
                    "the long-range module of this model reads the topologies "
                    f"{self.required_topologies()} and the node mask, which were "
                    "not given"
                )
            ret["energy"] = ret["energy"] + self.long_range.call_graph(
                gg,
                atype,
                node_mask,
                n_node,
                topologies,
                fparam=fparam,
                aparam=aparam_node,
                charge_spin=charge_spin_frame,
            )
        return ret

    def _frame_conditions(
        self,
        fparam: Array | None,
        charge_spin: Array | None,
        nf: int,
        reference: Array,
    ) -> tuple[Array | None, Array | None]:
        """Frame parameters and charge/spin condition with their defaults resolved.

        Parameters
        ----------
        fparam
            Frame parameters with shape (nf, numb_fparam), or None.
        charge_spin
            Charge/spin condition with shape (nf, 2) or (1, 2), or None.
        nf
            The number of frames.
        reference
            An array giving the namespace, dtype and device of the defaults.

        Returns
        -------
        tuple
            The frame parameters with shape (nf, numb_fparam) and the
            charge/spin condition with shape (nf, 2), each None when the model
            takes no such input or it has no default.
        """
        xp = array_api_compat.array_namespace(reference)
        device = array_api_compat.device(reference)
        if fparam is None and self.has_default_fparam():
            default_fparam = xp.asarray(
                self.get_default_fparam(), dtype=reference.dtype, device=device
            )
            fparam = xp.tile(xp.reshape(default_fparam, (1, -1)), (nf, 1))
        if not self.add_chg_spin_ebd:
            return fparam, None
        if charge_spin is None:
            default_cs = self.descriptor.get_default_chg_spin()
            if default_cs is None:
                return fparam, None
            charge_spin = xp.asarray(default_cs, dtype=reference.dtype, device=device)
        charge_spin = xp.broadcast_to(xp.reshape(charge_spin, (-1, 2)), (nf, 2))
        return fparam, charge_spin

    def compute_or_load_stat(
        self,
        sampled_func: Callable[[], list[dict]],
        stat_file_path: DPPath | None = None,
        compute_or_load_out_stat: bool = True,
        preset_observed_type: list[str] | None = None,
    ) -> None:
        """Compute or load the statistics parameters of the model,
        such as mean and standard deviation of descriptors or the energy bias of the fitting net.

        Parameters
        ----------
        sampled_func
            The lazy sampled function to get data frames from different data systems.
        stat_file_path
            The path to the stat file.
        compute_or_load_out_stat : bool
            Whether to compute the output statistics.
            If False, it will only compute the input statistics
            (e.g. mean and standard deviation of descriptors).
        """
        if stat_file_path is not None and self.type_map is not None:
            stat_file_path /= " ".join(self.type_map)

        wrapped_sampler = self._make_wrapped_sampler(sampled_func)
        self.descriptor.compute_input_stats(wrapped_sampler, stat_file_path)
        self.fitting_net.compute_input_stats(
            wrapped_sampler, stat_file_path=stat_file_path
        )
        if self.long_range is not None:
            self.long_range.compute_input_stats(
                wrapped_sampler, stat_file_path=stat_file_path
            )
        self._collect_and_set_observed_type(
            wrapped_sampler, stat_file_path, preset_observed_type
        )
        if compute_or_load_out_stat:
            self.compute_or_load_out_stat(wrapped_sampler, stat_file_path)

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any | None = None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        super().change_type_map(
            type_map=type_map, model_with_new_type_stat=model_with_new_type_stat
        )
        self.type_map = type_map
        self.descriptor.change_type_map(
            type_map=type_map,
            model_with_new_type_stat=model_with_new_type_stat.descriptor
            if model_with_new_type_stat is not None
            else None,
        )
        self.fitting_net.change_type_map(type_map=type_map)
        if self.long_range is not None:
            self.long_range.change_type_map(type_map=type_map)
        self.init_vacuum_conditions()

    def compute_fitting_input_stat(
        self,
        sample_merged: Callable[[], list[dict]] | list[dict],
        stat_file_path: DPPath | None = None,
    ) -> None:
        """Compute the input statistics (e.g. mean and stddev) for the fittings from packed data.

        The networks of the long-range module take the inputs of the fitting
        and receive their statistics as well.

        Parameters
        ----------
        sample_merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, ``merged[i]``, is a data dictionary containing
                ``keys``: ``np.ndarray`` originating from the ``i``-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples
                in the above format only when needed.
        stat_file_path : Optional[DPPath]
            The path to the stat file.
        """
        self.fitting_net.compute_input_stats(
            sample_merged,
            protection=self.data_stat_protect,
            stat_file_path=stat_file_path,
        )
        if self.long_range is not None:
            self.long_range.compute_input_stats(
                sample_merged,
                protection=self.data_stat_protect,
                stat_file_path=stat_file_path,
            )

    def compute_long_range_input_stat(
        self,
        sampled_func: Callable[[], list[dict]],
        stat_file_path: DPPath | None = None,
    ) -> None:
        """Compute the input statistics of the long-range module alone.

        A fine-tuning run that adds a long-range module to a pretrained model
        calls it for the new module, whose statistics, unlike those of the
        inherited descriptor and fitting, the pretrained model cannot provide.

        Parameters
        ----------
        sampled_func
            The lazy sampled function to get data frames from different data
            systems.
        stat_file_path
            The path to the stat file.
        """
        self.long_range.compute_input_stats(
            self._make_wrapped_sampler(sampled_func),
            protection=self.data_stat_protect,
            stat_file_path=stat_file_path,
        )

    def serialize(self) -> dict:
        dd = super().serialize()
        dd.update(
            {
                "@class": "Model",
                "type": "standard",
                "@version": 2,
                "type_map": self.type_map,
                "descriptor": self.descriptor.serialize(),
                "fitting": self.fitting_net.serialize(),
            }
        )
        # A record with a long-range module is version 3, which a reader that
        # cannot evaluate the module rejects; every other record keeps
        # version 2.
        if self.long_range is not None:
            dd["@version"] = 3
            dd["long_range"] = self.long_range.serialize()
        return dd

    # for subclass overridden
    base_descriptor_cls = BaseDescriptor
    """The base descriptor class."""
    base_fitting_cls = BaseFitting
    """The base fitting class."""
    base_long_range_cls: type | None = BaseLongRange
    """The base long-range module class; ``None`` for a backend without one."""

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "DPAtomicModel":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 3, 2)
        data.pop("@class")
        data.pop("type")
        descriptor_obj = cls.base_descriptor_cls.deserialize(data.pop("descriptor"))
        fitting_obj = cls.base_fitting_cls.deserialize(data.pop("fitting"))
        data["descriptor"] = descriptor_obj
        data["fitting"] = fitting_obj
        long_range = data.pop("long_range", None)
        if long_range is not None:
            if cls.base_long_range_cls is None:
                raise NotImplementedError(
                    "this backend cannot evaluate the long-range module of the "
                    "model record"
                )
            data["long_range"] = cls.base_long_range_cls.deserialize(long_range)
        obj = super().deserialize(data)
        return obj

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.fitting_net.get_dim_fparam()

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.fitting_net.get_dim_aparam()

    def has_default_fparam(self) -> bool:
        """Check if the model has default frame parameters."""
        return self.fitting_net.has_default_fparam()

    def get_default_fparam(self) -> list[float] | None:
        """Get the default frame parameters."""
        return self.fitting_net.get_default_fparam()

    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return self.fitting_net.get_sel_type()

    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return False
