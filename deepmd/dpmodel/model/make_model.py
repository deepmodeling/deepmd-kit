# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Callable,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.atomic_model.dp_atomic_model import (
        DPAtomicModel,
    )
    from deepmd.dpmodel.utils.exclude_mask import (
        PairExcludeMask,
    )

import array_api_compat
import numpy as np

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.atomic_model.base_atomic_model import (
    BaseAtomicModel,
)
from deepmd.dpmodel.common import (
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_NP_FLOAT_PRECISION,
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
    get_xp_precision,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableCategory,
    OutputVariableOperation,
    check_operation_applied,
)
from deepmd.dpmodel.utils import (
    DefaultNeighborList,
    NeighborList,
    format_nlist,
    nlist_distinguish_types,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
    build_neighbor_graph,
    build_neighbor_graph_ase,
)
from deepmd.utils.path import (
    DPPath,
)

from .edge_transform_output import (
    fit_output_to_model_output_graph,
)
from .transform_output import (
    communicate_extended_output,
    fit_output_to_model_output,
)


def model_call_from_call_lower(
    *,  # enforce keyword-only arguments
    call_lower: Callable[
        [
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
            np.ndarray | None,
            bool,
        ],
        dict[str, Array],
    ],
    rcut: float,
    sel: list[int],
    mixed_types: bool,
    model_output_def: ModelOutputDef,
    coord: Array,
    atype: Array,
    box: Array | None = None,
    fparam: Array | None = None,
    aparam: Array | None = None,
    do_atomic_virial: bool = False,
    coord_corr_for_virial: Array | None = None,
    charge_spin: Array | None = None,
    neighbor_list: NeighborList | None = None,
    pair_excl: "PairExcludeMask | None" = None,
) -> dict[str, Array]:
    """Return model prediction from lower interface.

    Parameters
    ----------
    coord
        The coordinates of the atoms.
        shape: nf x (nloc x 3)
    atype
        The type of atoms. shape: nf x nloc
    box
        The simulation box. shape: nf x 9
    fparam
        frame parameter. nf x ndf
    aparam
        atomic parameter. nf x nloc x nda
    do_atomic_virial
        If calculate the atomic virial.
    neighbor_list
        The neighbor-list construction strategy.  ``None`` uses the default
        all-pairs builder (:class:`DefaultNeighborList`), reproducing the
        historical behavior.  An alternative strategy (e.g. an O(N) cell list)
        may be injected to speed up neighbor-list construction; it returns the
        same extended representation, so model outputs are unchanged.
    pair_excl
        Model-level pair-type exclusion mask. Exclusion is a nlist-BUILD
        transform (decision #18/A4): it is folded into the nlist here, at the
        build seam, and ``call_lower`` consumes a pre-excluded nlist without
        re-applying it.

    Returns
    -------
    ret_dict
        The result dict of type dict[str,np.ndarray].
        The keys are defined by the `ModelOutputDef`.

    """
    nframes, nloc = atype.shape[:2]
    cc, bb, fp, ap = coord, box, fparam, aparam
    del coord, box, fparam, aparam
    builder = neighbor_list if neighbor_list is not None else DefaultNeighborList()
    # Model-level pair exclusion is a nlist-BUILD transform (decision #18/A4):
    # the BUILDER owns it (mirroring build_neighbor_graph on the graph path), so
    # the lower always consumes a pre-excluded nlist. ``pair_excl`` is part of
    # the NeighborList.build() contract; a custom strategy predating it fails
    # loudly (TypeError) instead of silently including excluded pairs.
    extended_coord, extended_atype, nlist, mapping = builder.build(
        cc, atype, bb, rcut, sel, pair_excl=pair_excl
    )
    extended_coord = extended_coord.reshape(nframes, -1, 3)
    if coord_corr_for_virial is not None:
        xp = array_api_compat.array_namespace(coord_corr_for_virial)
        # mapping: nf x nall -> nf x nall x 1, then tile to nf x nall x 3
        mapping_idx = xp.tile(
            xp.reshape(mapping, (nframes, -1, 1)),
            (1, 1, 3),
        )
        extended_coord_corr = xp.take_along_axis(
            coord_corr_for_virial,
            mapping_idx,
            axis=1,
        )
    else:
        extended_coord_corr = None
    call_lower_kwargs: dict[str, Any] = {
        "fparam": fp,
        "aparam": ap,
        "do_atomic_virial": do_atomic_virial,
        "charge_spin": charge_spin,
    }
    if extended_coord_corr is not None:
        call_lower_kwargs["extended_coord_corr"] = extended_coord_corr
    model_predict_lower = call_lower(
        extended_coord,
        extended_atype,
        nlist,
        mapping,
        **call_lower_kwargs,
    )
    model_predict = communicate_extended_output(
        model_predict_lower,
        model_output_def,
        mapping,
        do_atomic_virial=do_atomic_virial,
    )
    return model_predict


def make_model(
    T_AtomicModel: type[BaseAtomicModel],
    T_Bases: tuple[type, ...] = (),
) -> type:
    """Make a model as a derived class of an atomic model.

    The model provide two interfaces.

    1. the `call_lower`, that takes extended coordinates, atyps and neighbor list,
    and outputs the atomic and property and derivatives (if required) on the extended region.

    2. the `call`, that takes coordinates, atypes and cell and predicts
    the atomic and reduced property, and derivatives (if required) on the local region.

    Parameters
    ----------
    T_AtomicModel
        The atomic model.
    T_Bases
        Additional base classes for the returned model class.
        Defaults to ``()``.  For example, dpmodel passes ``(NativeOP,)``.

    Returns
    -------
    CM
        The model.

    """

    class CM(*T_Bases):
        def __init__(
            self,
            *args: Any,
            # underscore to prevent conflict with normal inputs
            atomic_model_: T_AtomicModel | None = None,
            **kwargs: Any,
        ) -> None:
            self.model_def_script = ""
            self.min_nbor_dist = None
            if atomic_model_ is not None:
                self.atomic_model: T_AtomicModel = atomic_model_
            else:
                self.atomic_model: T_AtomicModel = T_AtomicModel(*args, **kwargs)
            self.precision_dict = PRECISION_DICT
            # not supported by flax
            # self.reverse_precision_dict = RESERVED_PRECISION_DICT
            self.global_np_float_precision = GLOBAL_NP_FLOAT_PRECISION
            self.global_ener_float_precision = GLOBAL_ENER_FLOAT_PRECISION

        def model_output_def(self) -> ModelOutputDef:
            """Get the output def for the model."""
            return ModelOutputDef(self.atomic_output_def())

        def model_output_type(self) -> list[str]:
            """Get the output type for the model."""
            output_def = self.model_output_def()
            var_defs = output_def.var_defs
            vars = [
                kk
                for kk, vv in var_defs.items()
                if vv.category == OutputVariableCategory.OUT
            ]
            return vars

        def enable_compression(
            self,
            table_extrapolate: float = 5,
            table_stride_1: float = 0.01,
            table_stride_2: float = 0.1,
            check_frequency: int = -1,
        ) -> None:
            """Call atomic_model enable_compression().

            Parameters
            ----------
            table_extrapolate
                The scale of model extrapolation
            table_stride_1
                The uniform stride of the first table
            table_stride_2
                The uniform stride of the second table
            check_frequency
                The overflow check frequency
            """
            self.atomic_model.enable_compression(
                self.get_min_nbor_dist(),
                table_extrapolate,
                table_stride_1,
                table_stride_2,
                check_frequency,
            )

        def call_common(
            self,
            coord: Array,
            atype: Array,
            box: Array | None = None,
            fparam: Array | None = None,
            aparam: Array | None = None,
            do_atomic_virial: bool = False,
            coord_corr_for_virial: Array | None = None,
            charge_spin: Array | None = None,
            neighbor_list: NeighborList | None = None,
            neighbor_graph_method: str | None = None,
        ) -> dict[str, Array]:
            """Return model prediction.

            Parameters
            ----------
            coord
                The coordinates of the atoms.
                shape: nf x (nloc x 3)

            atype
                The type of atoms. shape: nf x nloc

            box
                The simulation box. shape: nf x 9

            fparam
                frame parameter. nf x ndf

            aparam
                atomic parameter. nf x nloc x nda

            do_atomic_virial
                If calculate the atomic virial.

            coord_corr_for_virial
                The coordinates correction for virial.
                shape: nf x (nloc x 3)

            neighbor_list
                Neighbor-list construction strategy for the DENSE-nlist path
                only.  ``None`` uses the default all-pairs builder; an
                alternative strategy (e.g. an O(N) cell list) may be injected to
                speed up nlist construction without changing model outputs.  It
                is consumed by the dense lower; supplying it forces the dense
                route (see below) and it is rejected together with an explicit
                ``neighbor_graph_method``.

            neighbor_graph_method
                Selects the lower the model routes through.  The option strings
                refer to the neighbor-GRAPH builder, NOT the legacy dense nlist:

                - ``None`` -- default.  dpmodel/jax keep the dense nlist path;
                  pt_expt default-flips graph-eligible mixed_types descriptors to
                  the carry-all graph (decision #17).
                - ``"legacy"`` -- force the dense nlist path (opt out of the
                  default-flip).
                - ``"dense"`` -- build a carry-all :class:`NeighborGraph` with the
                  in-tree O(N^2) ALL-PAIRS search (this is NOT the dense nlist
                  lower; "dense" = the all-pairs graph builder).
                - ``"ase"`` -- build the carry-all graph with the O(N) ASE cell
                  list.

                The graph routes (``"dense"``/``"ase"``, and the pt_expt
                default-flip) require a ``mixed_types`` descriptor with a graph
                lower (dpa1/se_atten with concat type embedding; attention layers included).
                At non-binding ``sel`` the graph matches the dense path exactly for the
                non-smooth branch; at binding ``sel`` the carry-all graph keeps
                neighbors the dense path truncates, and for
                ``smooth_type_embedding=True`` the graph drops the dense
                layout's sel-padding softmax terms, so the energy intentionally
                differs (sel-independent graph semantics).

            Returns
            -------
            ret_dict
                The result dict of type dict[str,np.ndarray].
                The keys are defined by the `ModelOutputDef`.

            """
            cc, bb, fp, ap, cs, input_prec = self._input_type_cast(
                coord, box=box, fparam=fparam, aparam=aparam, charge_spin=charge_spin
            )
            del coord, box, fparam, aparam, charge_spin
            graph_method = self._resolve_graph_method(neighbor_graph_method)
            # ``neighbor_list`` is a DENSE-nlist strategy; the graph path cannot
            # consume it. Reject an explicit graph+nlist combination, and
            # otherwise honor the supplied nlist by taking the dense route
            # (don't let the pt_expt default-flip silently ignore it).
            if neighbor_list is not None:
                if neighbor_graph_method not in (None, "legacy"):
                    raise ValueError(
                        "neighbor_list is a dense-nlist strategy and cannot be "
                        f"combined with neighbor_graph_method={neighbor_graph_method!r}; "
                        "pass one or the other"
                    )
                graph_method = None
            # the graph lower does not consume charge_spin yet -> keep those
            # models on dense (a None check, so it stays jit/export-safe)
            if cs is not None:
                graph_method = None
            if graph_method is not None:
                # carry-all NeighborGraph energy forward (Option B / decision #17)
                model_predict = self._call_common_graph(
                    cc,
                    atype,
                    bb,
                    fp,
                    ap,
                    graph_method,
                    do_atomic_virial,
                )
            else:
                # legacy dense-nlist path (builds the extended quartet)
                model_predict = model_call_from_call_lower(
                    call_lower=self.call_common_lower,
                    rcut=self.get_rcut(),
                    sel=self.get_sel(),
                    mixed_types=self.mixed_types(),
                    model_output_def=self.model_output_def(),
                    coord=cc,
                    atype=atype,
                    box=bb,
                    fparam=fp,
                    aparam=ap,
                    do_atomic_virial=do_atomic_virial,
                    coord_corr_for_virial=coord_corr_for_virial,
                    charge_spin=cs,
                    neighbor_list=neighbor_list,
                    # exclusion is a nlist-BUILD transform (decision #18/A4)
                    pair_excl=getattr(self.atomic_model, "pair_excl", None),
                )
            model_predict = self._output_type_cast(model_predict, input_prec)
            return model_predict

        def _resolve_graph_method(
            self, neighbor_graph_method: str | None
        ) -> str | None:
            """Resolve the neighbor-graph method.

            Base (dpmodel/jax): ``None`` => the dense path. These backends compute
            force/virial ANALYTICALLY inside ``call_common`` (``energy_derv_r`` in
            the output); the carry-all graph lower here is ENERGY-only, so it is
            NOT used by default (it would drop force). ``"legacy"`` => dense;
            explicit ``"dense"``/``"ase"`` => opt into the (energy-only) graph.

            pt_expt OVERRIDES this so ``None`` defaults graph-eligible mixed_types
            descriptors to the carry-all graph (decision #17) -- pt_expt has the
            autograd ``forward_common_lower_graph`` that produces force/virial.

            Parameters
            ----------
            neighbor_graph_method
                The user-requested method: ``None`` (default), ``"legacy"``
                (force dense), or ``"dense"``/``"ase"`` (force the graph builder).

            Returns
            -------
            method
                The resolved method passed to :meth:`_call_common_graph`, or
                ``None`` to take the dense path.
            """
            if neighbor_graph_method == "legacy":
                return None
            return neighbor_graph_method

        def _call_common_graph(
            self,
            cc: Array,
            atype: Array,
            bb: Array | None,
            fp: Array | None,
            ap: Array | None,
            method: str,
            do_atomic_virial: bool = False,
        ) -> dict[str, Array]:
            """Carry-all graph forward (opt-in, Option B).

            Builds a carry-all :class:`NeighborGraph` from ``cc``/``atype``/``bb``
            and routes the forward through the OUTPUT-AGNOSTIC
            :meth:`call_lower_graph`. Input/output type-casting is done by the
            caller.

            Parameters
            ----------
            cc
                coordinates. nf x nloc x 3 (or nf x (nloc x 3))
            atype
                the atom types. nf x nloc
            bb
                the simulation cell. nf x 3 x 3, or ``None`` for non-periodic.
            fp
                the frame parameter. nf x ndf
            ap
                the atomic parameter. nf x nloc x nda
            method
                the carry-all builder, ``"dense"`` or ``"ase"``.
            do_atomic_virial
                whether to calculate the atomic virial.

            Returns
            -------
            model_predict
                the standard model dict mirroring the dense ``call_common`` keys
                (``<var>`` per-atom, ``<var>_redu`` reduced, derivative
                name-holders ``None``, plus the int ``mask``).
            """
            descriptor = getattr(self.atomic_model, "descriptor", None)
            uses_graph_lower = getattr(descriptor, "uses_graph_lower", lambda: False)
            if not (self.mixed_types() and uses_graph_lower()):
                raise NotImplementedError(
                    "neighbor_graph_method requires a mixed_types descriptor with a "
                    "graph lower (e.g. dpa1 attn_layer=0)"
                )
            # Model-level ``pair_exclude_types`` is a graph-BUILD transform
            # (decision #18): apply it here, at the seam where the NeighborGraph
            # is constructed, so the graph lower / exported ``.pt2`` consumes an
            # already-excluded ``edge_mask`` and never re-applies it. Mirrors the
            # pt_expt eager path and the C++ ``applyPairExclusion`` at build.
            pair_excl = getattr(self.atomic_model, "pair_excl", None)
            if method == "dense":
                ng = build_neighbor_graph(
                    cc, atype, bb, self.get_rcut(), pair_excl=pair_excl
                )
            elif method == "ase":
                ng = build_neighbor_graph_ase(
                    cc, atype, bb, self.get_rcut(), pair_excl=pair_excl
                )
            else:
                raise ValueError(
                    f"unknown neighbor_graph_method {method!r}; the dpmodel/jax backend "
                    "supports 'dense'/'ase' only ('vesin'/'nv' require the pt_expt backend)."
                )
            xp = array_api_compat.array_namespace(atype)
            nf, nloc = atype.shape[:2]
            # OUTPUT-AGNOSTIC standard model dict (``<var>``, ``<var>_redu``,
            # derivative name-holders ``None``, plus int ``mask``), like the
            # dense ``call_common``.  ``call_lower_graph`` masks virtual atoms
            # (atype<0) and sets the real int mask.
            model_predict = self.call_lower_graph(
                atype=xp.reshape(atype, (nf * nloc,)),
                n_node=ng.n_node,
                edge_index=ng.edge_index,
                edge_vec=ng.edge_vec,
                edge_mask=ng.edge_mask,
                fparam=fp,
                aparam=ap,
            )
            # Public ABI is rectangular (nf, nloc, *); the lower is flat
            # (N=nf*nloc, *).  Unravel per-atom keys here at the boundary.
            # public call_common always passes rectangular (nf,nloc) coord/atype (N == nf*nloc), so this unravel always applies; ragged graphs reach call_lower_graph/forward_common_lower_graph directly (no unravel) and stay flat (N,*).
            for k in list(model_predict.keys()):
                v = model_predict[k]
                # per-frame reduced keys (..._redu) keep their (nf, *) shape; only node-level (N,*) keys unravel — guards the nloc==1 case where N == nf.
                if (
                    v is not None
                    and not k.endswith("_redu")
                    and v.shape[:1] == (nf * nloc,)
                ):
                    model_predict[k] = xp.reshape(v, (nf, nloc, *v.shape[1:]))
            return model_predict

        def call_common_lower(
            self,
            extended_coord: Array,
            extended_atype: Array,
            nlist: Array,
            mapping: Array | None = None,
            fparam: Array | None = None,
            aparam: Array | None = None,
            do_atomic_virial: bool = False,
            extended_coord_corr: Array | None = None,
            comm_dict: dict | None = None,
            charge_spin: Array | None = None,
        ) -> dict[str, Array]:
            """Return model prediction. Lower interface that takes
            extended atomic coordinates and types, nlist, and mapping
            as input, and returns the predictions on the extended region.
            The predictions are not reduced.

            Parameters
            ----------
            extended_coord
                coordinates in extended region. nf x (nall x 3).
            extended_atype
                atomic type in extended region. nf x nall.
            nlist
                neighbor list. nf x nloc x nsel.
            mapping
                mapps the extended indices to local indices. nf x nall.
            fparam
                frame parameter. nf x ndf
            aparam
                atomic parameter. nf x nloc x nda
            do_atomic_virial
                whether calculate atomic virial
            extended_coord_corr
                coordinates correction for virial in extended region.
                nf x (nall x 3)
            comm_dict
                MPI communication metadata for parallel inference (e.g.
                LAMMPS multi-rank). Carries send/recv lists, processor IDs,
                the MPI communicator handle, and per-rank nlocal/nghost.
                ``None`` for non-parallel inference (default).

            Returns
            -------
            result_dict
                the result dict, defined by the `FittingOutputDef`.

            """
            nframes, nall = extended_atype.shape[:2]
            extended_coord = extended_coord.reshape(nframes, -1, 3)
            nlist = self.format_nlist(
                extended_coord,
                extended_atype,
                nlist,
                extra_nlist_sort=self.need_sorted_nlist_for_lower(),
            )
            cc_ext, _, fp, ap, cs, input_prec = self._input_type_cast(
                extended_coord, fparam=fparam, aparam=aparam, charge_spin=charge_spin
            )
            del extended_coord, fparam, aparam, charge_spin
            model_predict = self.forward_common_atomic(
                cc_ext,
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fp,
                aparam=ap,
                do_atomic_virial=do_atomic_virial,
                extended_coord_corr=extended_coord_corr,
                comm_dict=comm_dict,
                charge_spin=cs,
            )
            model_predict = self._output_type_cast(model_predict, input_prec)
            return model_predict

        def forward_common_atomic(
            self,
            extended_coord: Array,
            extended_atype: Array,
            nlist: Array,
            mapping: Array | None = None,
            fparam: Array | None = None,
            aparam: Array | None = None,
            do_atomic_virial: bool = False,
            extended_coord_corr: Array | None = None,
            comm_dict: dict | None = None,
            charge_spin: Array | None = None,
        ) -> dict[str, Array]:
            atomic_ret = self.atomic_model.forward_common_atomic(
                extended_coord,
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fparam,
                aparam=aparam,
                comm_dict=comm_dict,
                charge_spin=charge_spin,
            )
            return fit_output_to_model_output(
                atomic_ret,
                self.atomic_output_def(),
                extended_coord,
                do_atomic_virial=do_atomic_virial,
                mask=atomic_ret["mask"] if "mask" in atomic_ret else None,
            )

        def forward_common_atomic_graph(
            self,
            atype: Array,
            n_node: Array,
            edge_index: Array,
            edge_vec: Array,
            edge_mask: Array,
            n_local: Array | None = None,
            fparam: Array | None = None,
            aparam: Array | None = None,
            comm_dict: dict | None = None,
            charge_spin: Array | None = None,
        ) -> dict[str, Array]:
            """Model-level graph forward (no type cast). Analogue of the dense
            :meth:`forward_common_atomic`.

            Builds a :class:`NeighborGraph` from the flat edge fields, runs the
            atomic model's :meth:`forward_common_atomic_graph` (flat ``(N, *)``
            per-node output), then the flat-N output transform (per-frame
            ``segment_sum`` reduction; derivative name-holders ``None`` --
            force/virial come from the pt_expt autograd lower). The
            ``(nf, nloc)`` unravel for the public ABI happens in the caller
            (:meth:`_call_common_graph`).

            Parameters
            ----------
            atype
                (N,) flat LOCAL atom types, ``N == sum(n_node)``.
            n_node
                (nf,) per-frame local atom counts.
            edge_index
                (2, E) ``[src, dst]`` edge endpoints (flat local indices).
            edge_vec
                (E, 3) neighbor-minus-center edge vectors.
            edge_mask
                (E,) boolean/0-1 valid-edge mask.
            n_local
                Per-rank local (owned) atom counts for multi-rank inference,
                ``(nf,)``. When given, halo rows (index ``>= n_local[frame]``)
                are excluded from ``<var>_redu`` (see
                :func:`fit_output_to_model_output_graph`); ``None`` (default)
                is the single-rank/all-owned behavior.
            fparam
                Frame parameter, ``(nf, ndf)``.
            aparam
                Atomic parameter, ``(N, nda)``.
            comm_dict
                MPI communication metadata. Ignored in PR-A; accepted for ABI
                stability.
            charge_spin
                charge/spin conditioning. Ignored in PR-A; accepted for ABI
                stability with charge/spin-conditioned descriptors.

            Returns
            -------
            dict
                The standard model dict (``<var>`` per-node, ``<var>_redu``
                reduced, derivative name-holders ``None``), matching
                :func:`fit_output_to_model_output_graph`.
            """
            graph = NeighborGraph(
                n_node=n_node,
                edge_index=edge_index,
                edge_vec=edge_vec,
                edge_mask=edge_mask,
            )
            atomic_ret = self.atomic_model.forward_common_atomic_graph(
                graph, atype, fparam=fparam, aparam=aparam, charge_spin=charge_spin
            )
            return fit_output_to_model_output_graph(
                atomic_ret,
                self.atomic_output_def(),
                graph,
                mask=atomic_ret["mask"] if "mask" in atomic_ret else None,
                n_local=n_local,
            )

        def call_common_lower_graph(
            self,
            atype: Array,
            n_node: Array,
            edge_index: Array,
            edge_vec: Array,
            edge_mask: Array,
            n_local: Array | None = None,
            fparam: Array | None = None,
            aparam: Array | None = None,
            comm_dict: dict | None = None,
            charge_spin: Array | None = None,
        ) -> dict[str, Array]:
            """Graph-native PUBLIC lower (dpa1/se_atten concat-tebd, attention included).

            The PRIMARY directly-callable graph interface (spec decision #14).
            Casts inputs/outputs to/from the model precision exactly like the
            dense :meth:`call_common_lower` (``edge_vec`` is the geometry, in
            place of ``coord``), then runs :meth:`forward_common_atomic_graph`.
            OUTPUT-AGNOSTIC: every fitting (energy/dos/dipole/polar/property/...)
            flows through with no change on the fitting side; force/virial are
            produced by the pt_expt autograd lower. Must match the dense
            :meth:`call_common_lower` reduction on the SAME neighbor set.

            Parameters
            ----------
            atype
                (N,) flat LOCAL atom types, ``N == sum(n_node)``.
            n_node
                (nf,) per-frame local atom counts.
            edge_index
                (2, E) ``[src, dst]`` edge endpoints (flat local indices).
            edge_vec
                (E, 3) neighbor-minus-center edge vectors.
            edge_mask
                (E,) boolean/0-1 valid-edge mask.
            n_local
                Per-rank local (owned) atom counts for multi-rank inference,
                ``(nf,)``. When given, halo rows (index ``>= n_local[frame]``)
                are excluded from ``<var>_redu``; ``None`` (default) is the
                single-rank/all-owned behavior.
            fparam
                Frame parameter, ``(nf, ndf)``.
            aparam
                Atomic parameter, ``(N, nda)``.
            comm_dict
                MPI communication metadata. Ignored in PR-A; accepted for ABI
                stability.
            charge_spin
                charge/spin conditioning. Ignored in PR-A; accepted for ABI
                stability with charge/spin-conditioned descriptors.

            Returns
            -------
            dict
                The standard model dict in the INPUT precision.
            """
            edge_vec, _, fparam, aparam, cs, input_prec = self._input_type_cast(
                edge_vec, fparam=fparam, aparam=aparam, charge_spin=charge_spin
            )
            model_predict = self.forward_common_atomic_graph(
                atype,
                n_node,
                edge_index,
                edge_vec,
                edge_mask,
                n_local=n_local,
                fparam=fparam,
                aparam=aparam,
                comm_dict=comm_dict,
                charge_spin=cs,
            )
            model_predict = self._output_type_cast(model_predict, input_prec)
            return model_predict

        # backward-compat alias (mirrors ``call_lower = call_common_lower``)
        call_lower_graph = call_common_lower_graph

        call = call_common
        call_lower = call_common_lower

        def get_out_bias(self) -> Array:
            """Get the output bias."""
            return self.atomic_model.get_out_bias()

        def get_observed_type_list(self) -> list[str]:
            """Get observed types (elements) of the model during data statistics.

            Bias-based fallback for old models without metadata.

            Returns
            -------
            list[str]
                A list of the observed type names in this model.
            """
            type_map = self.get_type_map()
            out_bias = self.get_out_bias()[0]
            assert out_bias is not None, "No out_bias found in the model."
            assert out_bias.ndim == 2, "The supported out_bias should be a 2D array."
            assert out_bias.shape[0] == len(type_map), (
                "The out_bias shape does not match the type_map length."
            )
            xp = array_api_compat.array_namespace(out_bias)
            bias_mask = xp.any(xp.abs(out_bias) > 1e-6, axis=-1)
            return [type_map[i] for i in range(len(type_map)) if bias_mask[i]]

        def set_out_bias(self, out_bias: Array) -> None:
            """Set the output bias."""
            self.atomic_model.set_out_bias(out_bias)

        def change_out_bias(
            self,
            merged: Any,
            bias_adjust_mode: str = "change-by-statistic",
        ) -> None:
            """Change the output bias according to the input data and the pretrained model.

            Parameters
            ----------
            merged
                The merged data samples.
            bias_adjust_mode : str
                The mode for changing output bias:
                'change-by-statistic' or 'set-by-statistic'.
            """
            self.atomic_model.change_out_bias(merged, bias_adjust_mode=bias_adjust_mode)

        def _input_type_cast(
            self,
            coord: Array,
            box: Array | None = None,
            fparam: Array | None = None,
            aparam: Array | None = None,
            charge_spin: Array | None = None,
        ) -> tuple[Array, Array | None, Array | None, Array | None, Array | None, Any]:
            """Cast the input data to global float type."""
            xp = array_api_compat.array_namespace(coord)
            input_dtype = coord.dtype
            global_dtype = get_xp_precision(
                xp, RESERVED_PRECISION_DICT[self.global_np_float_precision]
            )
            ###
            ### type checking would not pass jit, convert to coord prec anyway
            ###
            _lst: list[Array | None] = [
                xp.astype(vv, input_dtype) if vv is not None else None
                for vv in [box, fparam, aparam, charge_spin]
            ]
            box, fparam, aparam, charge_spin = _lst
            if input_dtype == global_dtype:
                return coord, box, fparam, aparam, charge_spin, input_dtype
            else:
                return (
                    xp.astype(coord, global_dtype),
                    xp.astype(box, global_dtype) if box is not None else None,
                    xp.astype(fparam, global_dtype) if fparam is not None else None,
                    xp.astype(aparam, global_dtype) if aparam is not None else None,
                    xp.astype(charge_spin, global_dtype)
                    if charge_spin is not None
                    else None,
                    input_dtype,
                )

        def _output_type_cast(
            self,
            model_ret: dict[str, Array],
            input_prec: Any,
        ) -> dict[str, Array]:
            """Convert the model output to the input prec.

            Parameters
            ----------
            model_ret
                The model output.
            input_prec
                The input dtype returned by ``_input_type_cast``.
            """
            model_ret_not_none = [vv for vv in model_ret.values() if vv is not None]
            if not model_ret_not_none:
                return model_ret
            xp = array_api_compat.array_namespace(model_ret_not_none[0])
            global_dtype = get_xp_precision(
                xp, RESERVED_PRECISION_DICT[self.global_np_float_precision]
            )
            ener_dtype = get_xp_precision(
                xp, RESERVED_PRECISION_DICT[self.global_ener_float_precision]
            )
            do_cast = input_prec != global_dtype
            odef = self.model_output_def()
            for kk in odef.keys():
                if kk not in model_ret.keys():
                    # do not return energy_derv_c if not do_atomic_virial
                    continue
                if check_operation_applied(odef[kk], OutputVariableOperation.REDU):
                    model_ret[kk] = (
                        xp.astype(model_ret[kk], ener_dtype)
                        if model_ret[kk] is not None
                        else None
                    )
                elif do_cast:
                    model_ret[kk] = (
                        xp.astype(model_ret[kk], input_prec)
                        if model_ret[kk] is not None
                        else None
                    )
            return model_ret

        def format_nlist(
            self,
            extended_coord: Array,
            extended_atype: Array,
            nlist: Array,
            extra_nlist_sort: bool = False,
        ) -> Array:
            """Format the neighbor list.

            1. If the number of neighbors in the `nlist` is equal to sum(self.sel),
            it does nothong

            2. If the number of neighbors in the `nlist` is smaller than sum(self.sel),
            the `nlist` is pad with -1.

            3. If the number of neighbors in the `nlist` is larger than sum(self.sel),
            the nearest sum(sel) neighbors will be preserved.

            Known limitations:

            In the case of not self.mixed_types, the nlist is always formatted.
            May have side effact on the efficiency.

            Parameters
            ----------
            extended_coord
                coordinates in extended region. nf x nall x 3
            extended_atype
                atomic type in extended region. nf x nall
            nlist
                neighbor list. nf x nloc x nsel
            extra_nlist_sort
                whether to forcibly sort the nlist.

            Returns
            -------
            formatted_nlist
                the formatted nlist.

            """
            n_nf, n_nloc, n_nnei = nlist.shape
            mixed_types = self.mixed_types()
            ret = self._format_nlist(
                extended_coord,
                nlist,
                sum(self.get_sel()),
                extra_nlist_sort=extra_nlist_sort,
            )
            if not mixed_types:
                ret = nlist_distinguish_types(ret, extended_atype, self.get_sel())
            return ret

        def _format_nlist(
            self,
            extended_coord: Array,
            nlist: Array,
            nnei: int,
            extra_nlist_sort: bool = False,
        ) -> Array:
            return format_nlist(
                extended_coord,
                nlist,
                nnei,
                self.get_rcut(),
                extra_nlist_sort=extra_nlist_sort,
            )

        def do_grad_r(
            self,
            var_name: str | None = None,
        ) -> bool:
            """Tell if the output variable `var_name` is r_differentiable.
            if var_name is None, returns if any of the variable is r_differentiable.
            """
            return self.atomic_model.do_grad_r(var_name)

        def do_grad_c(
            self,
            var_name: str | None = None,
        ) -> bool:
            """Tell if the output variable `var_name` is c_differentiable.
            if var_name is None, returns if any of the variable is c_differentiable.
            """
            return self.atomic_model.do_grad_c(var_name)

        def change_type_map(
            self, type_map: list[str], model_with_new_type_stat: Any | None = None
        ) -> None:
            """Change the type related params to new ones, according to `type_map` and the original one in the model.
            If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
            """
            self.atomic_model.change_type_map(
                type_map=type_map,
                model_with_new_type_stat=model_with_new_type_stat.atomic_model
                if model_with_new_type_stat is not None
                else None,
            )

        def serialize(self) -> dict:
            return self.atomic_model.serialize()

        @classmethod
        def deserialize(cls, data: dict) -> "CM":
            return cls(atomic_model_=T_AtomicModel.deserialize(data))

        def set_case_embd(self, case_idx: int) -> None:
            self.atomic_model.set_case_embd(case_idx)

        def get_dim_fparam(self) -> int:
            """Get the number (dimension) of frame parameters of this atomic model."""
            return self.atomic_model.get_dim_fparam()

        def get_dim_aparam(self) -> int:
            """Get the number (dimension) of atomic parameters of this atomic model."""
            return self.atomic_model.get_dim_aparam()

        def get_numb_dos(self) -> int:
            """Get the number of DOS. Zero for models without a DOS output."""
            return 0

        def has_default_fparam(self) -> bool:
            """Check if the model has default frame parameters."""
            return self.atomic_model.has_default_fparam()

        def get_default_fparam(self) -> list[float] | None:
            """Get the default frame parameters."""
            return self.atomic_model.get_default_fparam()

        def has_chg_spin_ebd(self) -> bool:
            """Check if the model has charge spin embedding."""
            return self.atomic_model.has_chg_spin_ebd()

        def get_dim_chg_spin(self) -> int:
            """Get the dimension of charge_spin input."""
            return self.atomic_model.get_dim_chg_spin()

        def has_default_chg_spin(self) -> bool:
            """Check if the model has default charge_spin values."""
            return self.atomic_model.has_default_chg_spin()

        def get_default_chg_spin(self) -> list[float] | None:
            """Get the default charge_spin values."""
            return self.atomic_model.get_default_chg_spin()

        def get_sel_type(self) -> list[int]:
            """Get the selected atom types of this model.

            Only atoms with selected atom types have atomic contribution
            to the result of the model.
            If returning an empty list, all atom types are selected.
            """
            return self.atomic_model.get_sel_type()

        def is_aparam_nall(self) -> bool:
            """Check whether the shape of atomic parameters is (nframes, nall, ndim).

            If False, the shape is (nframes, nloc, ndim).
            """
            return self.atomic_model.is_aparam_nall()

        def get_dp_atomic_model(self) -> "DPAtomicModel | None":
            """Get the underlying DPAtomicModel with descriptor and fitting_net.

            Returns the ``atomic_model`` if it is a ``DPAtomicModel`` instance
            (i.e. has both ``descriptor`` and ``fitting_net``).  Returns ``None``
            for composite atomic models such as ``LinearEnergyAtomicModel``.
            """
            from deepmd.dpmodel.atomic_model.dp_atomic_model import (
                DPAtomicModel,
            )

            if isinstance(self.atomic_model, DPAtomicModel):
                return self.atomic_model
            return None

        def get_rcut(self) -> float:
            """Get the cut-off radius."""
            return self.atomic_model.get_rcut()

        def get_type_map(self) -> list[str]:
            """Get the type map."""
            return self.atomic_model.get_type_map()

        def get_nsel(self) -> int:
            """Returns the total number of selected neighboring atoms in the cut-off radius."""
            return self.atomic_model.get_nsel()

        def get_nnei(self) -> int:
            """Returns the total number of selected neighboring atoms in the cut-off radius."""
            return self.atomic_model.get_nnei()

        def get_sel(self) -> list[int]:
            """Returns the number of selected atoms for each type."""
            return self.atomic_model.get_sel()

        def mixed_types(self) -> bool:
            """If true, the model
            1. assumes total number of atoms aligned across frames;
            2. uses a neighbor list that does not distinguish different atomic types.

            If false, the model
            1. assumes total number of atoms of each atom type aligned across frames;
            2. uses a neighbor list that distinguishes different atomic types.

            """
            return self.atomic_model.mixed_types()

        def has_message_passing(self) -> bool:
            """Returns whether the model has message passing."""
            return self.atomic_model.has_message_passing()

        def need_sorted_nlist_for_lower(self) -> bool:
            """Returns whether the model needs sorted nlist when using `forward_lower`."""
            return self.atomic_model.need_sorted_nlist_for_lower()

        def atomic_output_def(self) -> FittingOutputDef:
            """Get the output def of the atomic model."""
            return self.atomic_model.atomic_output_def()

        def compute_or_load_stat(
            self,
            sampled_func: Callable[[], Any],
            stat_file_path: DPPath | None = None,
            preset_observed_type: list[str] | None = None,
        ) -> None:
            """Compute or load the statistics parameters of the model.

            Parameters
            ----------
            sampled_func
                The lazy sampled function to get data frames from different
                data systems.
            stat_file_path
                The path to the stat file.
            preset_observed_type
                User-specified observed types that take highest priority.
            """
            self.atomic_model.compute_or_load_stat(
                sampled_func, stat_file_path, preset_observed_type=preset_observed_type
            )

        def get_model_def_script(self) -> str:
            """Get the model definition script."""
            return self.model_def_script

        def get_min_nbor_dist(self) -> float | None:
            """Get the minimum distance between two atoms."""
            return self.min_nbor_dist

        def get_ntypes(self) -> int:
            """Get the number of types."""
            return len(self.get_type_map())

    return CM
