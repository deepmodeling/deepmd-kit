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
)

import numpy as np

from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.plugin import (
    PluginVariant,
    make_plugin_registry,
)


def check_long_range_support(descriptor: Any) -> None:
    """Admit only the descriptors whose combination with long-range modules is validated.

    A long-range module reads nothing but the invariant node features every
    graph descriptor returns, so this gate is a statement of validation, not
    of implementation: a further descriptor is admitted by adding it here once
    its combination has been tested.

    Parameters
    ----------
    descriptor
        The descriptor of the atomic model receiving the long-range module.

    Raises
    ------
    NotImplementedError
        If the descriptor is not among the validated ones.
    """
    from deepmd.dpmodel.descriptor.dpa4c import (
        DescrptDPA4C,
    )

    if not isinstance(descriptor, DescrptDPA4C):
        raise NotImplementedError(
            "long-range modules are currently validated with the DPA4C "
            f"descriptor only, got {type(descriptor).__name__}"
        )


def make_base_long_range(t_tensor: Any) -> type:
    """Make the base class of the long-range modules of one backend.

    Each backend owns its registry, so a configuration builds the long-range
    module native to the backend that reads it.

    Parameters
    ----------
    t_tensor
        The type of the tensor, used in the type hints.
    """

    class BLR(ABC, PluginVariant, make_plugin_registry("long range")):
        """Long-range contribution to the atomic energy of a descriptor-fitting model.

        A long-range module reads the invariant node features of the
        descriptor and the topologies it declares, and returns an energy per
        node in eV. The atomic model adds it to the output of its energy
        fitting before the output statistics and masks, so every consumer of
        the atomic model sees one complete potential. The module owns the
        attribution of its energy to the nodes; the per-frame energy is their
        sum.
        """

        @abstractmethod
        def required_topologies(self) -> tuple[str, ...]:
            """Names of the topologies the module reads beside the neighbor graph."""

        def topology_params(self) -> dict[str, dict[str, Any]]:
            """The build parameters of the topologies, by name; none by default."""
            return {}

        @abstractmethod
        def uses_charge_spin(self) -> bool:
            """Whether the module reads the frame-level charge/spin condition."""

        @abstractmethod
        def call_graph(
            self,
            descriptor: t_tensor,
            atype: t_tensor,
            node_mask: t_tensor,
            n_node: t_tensor,
            topologies: dict[str, Any],
            fparam: t_tensor | None = None,
            aparam: t_tensor | None = None,
            charge_spin: t_tensor | None = None,
        ) -> t_tensor:
            """Evaluate the atomic energy of the module on the flat node axis.

            Parameters
            ----------
            descriptor
                Invariant node features with shape ``(N, dim_descrpt)``.
            atype
                Node types with shape ``(N,)``; invalid nodes carry any valid
                type.
            node_mask
                Nodes carrying fields with shape ``(N,)``: the real and owned
                nodes.
            n_node
                Per-frame node counts with shape ``(nf,)``.
            topologies
                The topologies of :meth:`required_topologies` by name.
            fparam
                Frame parameters with shape ``(nf, numb_fparam)``, defaults
                resolved.
            aparam
                Atomic parameters with shape ``(N, numb_aparam)``.
            charge_spin
                Frame charge/spin condition with shape ``(nf, 2)``, defaults
                resolved, when the model takes one.

            Returns
            -------
            t_tensor
                The atomic energy with shape ``(N, 1)``, in eV.
            """

        def compute_input_stats(
            self,
            merged: Callable[[], list[dict]] | list[dict],
            protection: float = 1e-2,
            stat_file_path: DPPath | None = None,
        ) -> None:
            """Compute the input statistics of the networks of the module."""

        def change_type_map(self, type_map: list[str]) -> None:
            """Remap the type-dependent state of the module to a new type map."""

        def set_case_embd(self, case_idx: int) -> None:
            """Select the case embedding of the networks of the module."""

        def fold_charge_state(self, charge_spin: list[float]) -> None:
            """Specialize the module to one frame charge/spin condition.

            Called when the descriptor of the model is compressed, which folds
            the condition into its tables, for a module whose
            :meth:`uses_charge_spin` holds.

            Raises
            ------
            NotImplementedError
                If the module cannot be specialized.
            """
            raise NotImplementedError(
                f"{type(self).__name__} cannot be specialized to a charge/spin "
                "condition, which the compression of the model requires"
            )

        def charge_state_constants(self) -> list[t_tensor]:
            """The tensors that carry the folded charge/spin condition.

            A deployment that serves another condition writes the outputs of
            :meth:`rebuild_charge_state` over them, in this order.
            """
            return []

        def rebuild_charge_state(self, charge_spin: t_tensor) -> list[t_tensor]:
            """Build the tensors of :meth:`charge_state_constants` for a condition.

            Parameters
            ----------
            charge_spin
                The condition with shape ``(1, 2)``.
            """
            return []

        @abstractmethod
        def serialize(self) -> dict:
            """Serialize the module to a dict."""

        @classmethod
        def deserialize(cls, data: dict) -> "BLR":
            """Deserialize a module, dispatching on its registered type."""
            if cls is BLR:
                return BLR.get_class_by_type(data["type"]).deserialize(data)
            raise NotImplementedError(f"Not implemented in class {cls.__name__}")

    return BLR


BaseLongRange = make_base_long_range(np.ndarray)
