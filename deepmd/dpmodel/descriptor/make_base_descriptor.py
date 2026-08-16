# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Callable,
)
from typing import (
    Any,
    NoReturn,
)

from deepmd.common import (
    j_get_type,
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
from deepmd.utils.plugin import (
    PluginVariant,
    make_plugin_registry,
)


def make_base_descriptor(
    t_tensor: type,
    fwd_method_name: str = "forward",
) -> type:
    """Make the base class for the descriptor.

    Parameters
    ----------
    t_tensor
        The type of the tensor. used in the type hint.
    fwd_method_name
        Name of the forward method. For dpmodels, it should be "call".
        For torch models, it should be "forward".

    """

    class BD(ABC, PluginVariant, make_plugin_registry("descriptor")):
        """Base descriptor provides the interfaces of descriptor."""

        # Stat-behavior flags with concrete defaults so stat machinery (e.g.
        # ``merge_env_stat``, which accepts either a ``Descriptor`` or a
        # ``DescriptorBlock``) can read them on any descriptor without
        # getattr probes; descriptors that configure them assign instance
        # attributes in __init__ (issue #5897). Mirrors the same defaults on
        # ``DescriptorBlock``.
        set_davg_zero: bool = False
        set_stddev_constant: bool = False

        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            if cls is BD:
                cls = cls.get_class_by_type(j_get_type(kwargs, cls.__name__))
            return object.__new__(cls)

        @abstractmethod
        def get_rcut(self) -> float:
            """Returns the cut-off radius."""
            pass

        @abstractmethod
        def get_rcut_smth(self) -> float:
            """Returns the radius where the neighbor information starts to smoothly decay to 0."""
            pass

        @abstractmethod
        def get_sel(self) -> list[int]:
            """Returns the number of selected neighboring atoms for each type."""
            pass

        def get_nsel(self) -> int:
            """Returns the total number of selected neighboring atoms in the cut-off radius."""
            return sum(self.get_sel())

        def get_nnei(self) -> int:
            """Returns the total number of selected neighboring atoms in the cut-off radius."""
            return self.get_nsel()

        @abstractmethod
        def get_ntypes(self) -> int:
            """Returns the number of element types."""
            pass

        @abstractmethod
        def get_type_map(self) -> list[str]:
            """Get the name to each type of atoms."""
            pass

        @abstractmethod
        def get_dim_out(self) -> int:
            """Returns the output descriptor dimension."""
            pass

        @abstractmethod
        def get_dim_emb(self) -> int:
            """Returns the embedding dimension of g2."""
            pass

        def get_dim_chg_spin(self) -> int:
            """Returns the dimension of charge_spin input (0 if not supported)."""
            return 0

        def get_default_chg_spin(self) -> Any:
            """Returns the default charge_spin value, or None."""
            return None

        def get_chg_spin_table_ranges(self) -> list[tuple[int, int]] | None:
            """Returns the row range each charge_spin value indexes, or None.

            A descriptor that embeds the condition by indexing tables reports
            one half-open range per value, which makes every acceptable state
            an integer tuple inside those ranges. ``None``, the default, means
            the condition enters as a continuous quantity and only its width
            is constrained.
            """
            return None

        def get_geo_compress(self) -> bool:
            """Return whether geometric tabulated compression is active.

            Concrete default ``False``; descriptor families with a
            geometric compression path override from their own state.
            """
            return False

        @abstractmethod
        def mixed_types(self) -> bool:
            """Returns if the descriptor requires a neighbor list that distinguish different
            atomic types or not.
            """
            pass

        @abstractmethod
        def has_message_passing(self) -> bool:
            """Returns whether the descriptor has message passing."""

        def has_message_passing_across_ranks(self) -> bool:
            """Returns whether the descriptor's message passing extends across rank
            boundaries — i.e. whether it requires cross-rank exchange of intermediate
            atomic features (per-layer node embeddings) during the forward pass.

            Distinct from generic ghost-coord/force exchange that every LAMMPS
            pair_style does. This question gates whether the pt_expt backend
            compiles a second "with-comm" AOTI artifact for multi-rank deployment.

            Concrete default ``False`` (non-GNN behavior) so pt and pd backend
            descriptors that subclass ``BaseDescriptor`` directly do not have
            to implement this method until they grow a multi-rank GNN path of
            their own. GNN descriptors that need MPI ghost-feature exchange
            (DPA2, DPA3 with ``use_loc_mapping=False``, hybrids wrapping such
            children) override to return ``True``.
            """
            return False

        def supports_edge_parallel(self) -> bool:
            """Whether this descriptor can run under MPI domain decomposition.

            Distinct from :meth:`has_message_passing_across_ranks` (whether
            multi-rank inference NEEDS a per-block ghost exchange): this asks
            whether any part of the computation folds state that a single
            rank cannot observe. Default ``True``: an ordinary descriptor
            reads only rank-local neighbourhoods.
            """
            return True

        def supports_native_spin(self) -> bool:
            """Returns whether the descriptor natively conditions on per-atom spin.

            Declaring ``True`` obliges the descriptor's ``call_graph`` to
            accept a per-node ``spin`` keyword; the atomic model only
            forwards the keyword to descriptors that declare the capability,
            since an unconditional ``spin=`` kwarg would be a ``TypeError``
            on a ``call_graph`` signature that does not declare it.

            Concrete default ``False`` so descriptors across all backends
            (pt/pd/tf subclass this same base) need no change until they grow
            a native spin mechanism of their own; such descriptors override
            this method to return ``True``.
            """
            return False

        def supports_charge_spin(self) -> bool:
            """Returns whether the descriptor conditions on a frame-level ``charge_spin`` input.

            Declaring ``True`` obliges the descriptor's ``call_graph`` to
            accept a frame-level ``charge_spin`` keyword; the atomic model
            only forwards the keyword to descriptors that declare the
            capability. Concrete default ``False`` (see
            ``supports_native_spin``); descriptors that condition on this
            input override this method to return ``True``.
            """
            return False

        def uses_graph_lower(self) -> bool:
            """Returns whether the descriptor supports the graph-native (NeighborGraph) lower.

            Declaring ``True`` obliges the descriptor to implement
            ``call_graph``; the model layer routes ``forward_lower`` through
            the NeighborGraph path only for descriptors that declare the
            capability, and falls back to the legacy dense (nlist) lower
            otherwise.

            Concrete default ``False`` so descriptors across all backends
            (which subclass this same base) stay on the dense lower until
            they implement a graph-native forward; such descriptors override
            this method (typically conditioning on their configuration and
            on :meth:`disable_graph_lower`).
            """
            return False

        def disable_graph_lower(self) -> None:
            """Force the legacy dense (nlist) lower for this descriptor.

            An explicit opt-out knob used by contexts where the graph-native
            lower is unsupported or undesirable. After calling this,
            :meth:`uses_graph_lower` must return ``False`` regardless of the
            descriptor configuration.

            Concrete default: a no-op, since a descriptor without a graph
            lower is already dense-only. Descriptors overriding
            :meth:`uses_graph_lower` must also override this to set their
            escape hatch.
            """
            return None

        def uses_compact_edge_pairs(self) -> bool:
            """Returns whether the descriptor's graph lower traces compact edge pairs.

            The compact ``center_edge_pairs`` realization uses
            unbacked-SymInt ``nonzero``/``repeat`` sizes when traced for
            export; ``check_graph_trace_torch_version`` keys its
            torch >= 2.6 requirement on this capability. Concrete default
            ``False``; only meaningful for descriptors whose
            :meth:`uses_graph_lower` can return ``True``.
            """
            return False

        def dense_lower_supports_comm(self) -> bool:
            """Whether the DENSE (nlist) lower implements comm_dict exchange.

            Default ``True`` — dense comm is the production multi-rank path
            for dpa2/dpa3 (previously this was probed by method absence in
            the freeze machinery); a descriptor whose dense adapter raises
            on ``comm_dict`` (DPA4) overrides to ``False``.
            """
            return True

        def graph_edge_dtype(self) -> str:
            """Edge-geometry dtype the graph deployment artifact accepts.

            ``"float64"`` is the model-agnostic ABI; geometrically
            compressed float32 descriptors override to ``"float32"``.
            """
            return "float64"

        def supports_graph_export(self) -> bool:
            """Whether an exportable graph-lower implementation exists.

            A compressed descriptor without its fused opaque operator cannot
            be traced through the reference tabulation kernel.
            """
            return True

        def graph_type_embedding_table(self) -> Any | None:
            """Full type-embedding table consumed by the graph-route forward.

            Returns
            -------
            Any | None
                The ``(ntypes + 1, tebd_dim)`` type-embedding table for
                descriptors whose graph lower consumes an external table, or
                ``None`` (the concrete default) for descriptors that embed
                types internally or have no graph lower.
            """
            return None

        @abstractmethod
        def need_sorted_nlist_for_lower(self) -> bool:
            """Returns whether the descriptor needs sorted nlist when using `forward_lower`."""

        @abstractmethod
        def get_env_protection(self) -> float:
            """Returns the protection of building environment matrix."""
            pass

        @abstractmethod
        def share_params(
            self, base_class: Any, shared_level: Any, resume: bool = False
        ) -> None:
            """
            Share the parameters of self to the base_class with shared_level during multitask training.
            If not start from checkpoint (resume is False),
            some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
            """
            pass

        @abstractmethod
        def change_type_map(
            self, type_map: list[str], model_with_new_type_stat: Any | None = None
        ) -> None:
            """Change the type related params to new ones, according to `type_map` and the original one in the model.
            If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
            """
            pass

        @abstractmethod
        def set_stat_mean_and_stddev(self, mean: Any, stddev: Any) -> None:
            """Update mean and stddev for descriptor."""
            pass

        @abstractmethod
        def get_stat_mean_and_stddev(self) -> Any:
            """Get mean and stddev for descriptor."""
            pass

        def compute_input_stats(
            self,
            merged: Callable[[], list[dict]] | list[dict],
            path: DPPath | None = None,
        ) -> NoReturn:
            """Update mean and stddev for descriptor elements."""
            raise NotImplementedError

        def enable_compression(
            self,
            min_nbor_dist: float,
            table_extrapolate: float = 5,
            table_stride_1: float = 0.01,
            table_stride_2: float = 0.1,
            check_frequency: int = -1,
        ) -> None:
            """Receive the statistics (distance, max_nbor_size and env_mat_range) of the training data.

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
            raise NotImplementedError("This descriptor doesn't support compression!")

        def compression_needs_min_nbor_dist(self) -> bool:
            """Whether :meth:`enable_compression` consumes ``min_nbor_dist``.

            Returns
            -------
            bool
                Concrete default ``True``: a tabulated embedding starts its
                table at the shortest distance the training data contains, so
                the caller must measure it first. ``False`` for descriptors
                whose table domain is fixed analytically; the caller may then
                skip the neighbor-statistics pass, which is a dense all-pairs
                computation over the training data.
            """
            return True

        @abstractmethod
        def fwd(
            self,
            extended_coord: Array,
            extended_atype: Array,
            nlist: Array,
            mapping: Array | None = None,
            fparam: Array | None = None,
            charge_spin: Array | None = None,
        ) -> Array:
            """Calculate descriptor."""
            pass

        @abstractmethod
        def serialize(self) -> dict:
            """Serialize the obj to dict."""
            pass

        @classmethod
        def deserialize(cls, data: dict) -> "BD":
            """Deserialize the model.

            Parameters
            ----------
            data : dict
                The serialized data

            Returns
            -------
            BD
                The deserialized descriptor
            """
            if cls is BD:
                return BD.get_class_by_type(data["type"]).deserialize(data)
            raise NotImplementedError(f"Not implemented in class {cls.__name__}")

        @classmethod
        @abstractmethod
        def update_sel(
            cls,
            train_data: DeepmdDataSystem,
            type_map: list[str] | None,
            local_jdata: dict,
        ) -> tuple[dict, float | None]:
            """Update the selection and perform neighbor statistics.

            Parameters
            ----------
            train_data : DeepmdDataSystem
                data used to do neighbor statistics
            type_map : list[str], optional
                The name of each type of atoms
            local_jdata : dict
                The local data refer to the current class

            Returns
            -------
            dict
                The updated local data
            float
                The minimum distance between two atoms
            """
            # call subprocess
            cls = cls.get_class_by_type(j_get_type(local_jdata, cls.__name__))
            return cls.update_sel(train_data, type_map, local_jdata)

    setattr(BD, fwd_method_name, BD.fwd)
    delattr(BD, "fwd")

    return BD
