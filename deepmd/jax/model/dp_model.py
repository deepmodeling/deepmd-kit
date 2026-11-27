# SPDX-License-Identifier: LGPL-3.0-or-later
from functools import (
    partial,
)
from typing import (
    Any,
)

from deepmd.dpmodel.atomic_model.base_atomic_model import (
    BaseAtomicModel,
)
from deepmd.dpmodel.model import (
    DPModelCommon,
)
from deepmd.dpmodel.model.make_model import (
    model_call_from_call_lower,
)
from deepmd.dpmodel.utils.neighbor_list import (
    NeighborList,
)
from deepmd.jax.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.jax.common import (
    flax_module,
)
from deepmd.jax.env import (
    jax,
    jnp,
)
from deepmd.jax.model.base_model import (
    forward_common_atomic,
)
from deepmd.jax.model.graph_model import forward_common_atomic as forward_graph_quartet
from deepmd.jax.model.graph_model import (
    forward_common_atomic_graph,
)


def validate_neighbor_contract(atomic_model: BaseAtomicModel) -> None:
    """Reject compositions that cannot retain the required physical edges."""
    if atomic_model.requires_full_neighbor_list() and not (
        atomic_model.mixed_types() and atomic_model.uses_graph_lower()
    ):
        raise NotImplementedError(
            "JAX models requiring a full neighbor list need a mixed-type graph "
            "lower for every atomic-model component. Dense-only components "
            "such as tabulated pair potentials cannot be composed with them."
        )


def make_jax_dp_model_from_dpmodel(
    dpmodel_model: type[DPModelCommon], jax_atomicmodel: type[DPAtomicModel]
) -> type[DPModelCommon]:
    """Make a JAX backend DP model from a DPModel backend DP model.

    Parameters
    ----------
    dpmodel_model : type[DPModelCommon]
        The DPModel backend DP model.
    jax_atomicmodel : type[DPAtomicModel]
        The JAX backend DP atomic model.

    Returns
    -------
    type[DPModelCommon]
        The JAX backend DP model.
    """

    @flax_module
    class jax_model(dpmodel_model):
        forward_common_atomic_graph = forward_common_atomic_graph

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            validate_neighbor_contract(self.atomic_model)

        @classmethod
        def deserialize(cls, data: dict) -> "jax_model":
            """Restore children through their backend descriptor and fitting registries."""
            return cls(atomic_model_=jax_atomicmodel.deserialize(data))

        def uses_full_neighbor_list(self) -> bool:
            """Whether the atomic model requires every neighbor in range."""
            return self.atomic_model.requires_full_neighbor_list()

        def call_common(
            self,
            coord: jnp.ndarray,
            atype: jnp.ndarray,
            box: jnp.ndarray | None = None,
            fparam: jnp.ndarray | None = None,
            aparam: jnp.ndarray | None = None,
            do_atomic_virial: bool = False,
            coord_corr_for_virial: jnp.ndarray | None = None,
            charge_spin: jnp.ndarray | None = None,
            neighbor_list: NeighborList | None = None,
            neighbor_graph_method: str | None = None,
            spin: jnp.ndarray | None = None,
        ) -> dict[str, jnp.ndarray]:
            """Build neighbors eagerly and evaluate the compiled lower contract."""
            if not self.uses_full_neighbor_list() or neighbor_graph_method not in (
                None,
                "dense",
                "legacy",
            ):
                return super().call_common(
                    coord,
                    atype,
                    box=box,
                    fparam=fparam,
                    aparam=aparam,
                    do_atomic_virial=do_atomic_virial,
                    coord_corr_for_virial=coord_corr_for_virial,
                    charge_spin=charge_spin,
                    neighbor_list=neighbor_list,
                    neighbor_graph_method=neighbor_graph_method,
                    spin=spin,
                )
            from deepmd.jax.utils.neighbor_list import (
                FullNeighborList,
            )

            if neighbor_list is not None and neighbor_graph_method == "dense":
                raise ValueError(
                    "neighbor_list is a dense-nlist strategy and cannot be "
                    "combined with neighbor_graph_method='dense'; pass one or "
                    "the other"
                )
            if self.has_spin() and spin is None:
                raise ValueError("A native-spin model requires per-atom spin vectors.")
            cc, bb, fp, ap, cs, sp, input_prec = self._input_type_cast(
                coord,
                box=box,
                fparam=fparam,
                aparam=aparam,
                charge_spin=charge_spin,
                spin=spin,
            )
            result = model_call_from_call_lower(
                call_lower=partial(self.call_common_lower, spin=sp),
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
                pair_excl=self.atomic_model.pair_excl,
                neighbor_list=(
                    FullNeighborList() if neighbor_list is None else neighbor_list
                ),
            )
            return self._output_type_cast(result, input_prec)

        def forward_common_atomic(
            self,
            extended_coord: jnp.ndarray,
            extended_atype: jnp.ndarray,
            nlist: jnp.ndarray,
            mapping: jnp.ndarray | None = None,
            fparam: jnp.ndarray | None = None,
            aparam: jnp.ndarray | None = None,
            do_atomic_virial: bool = False,
            extended_coord_corr: jnp.ndarray | None = None,
            comm_dict: dict | None = None,
            charge_spin: jnp.ndarray | None = None,
            spin: jnp.ndarray | None = None,
        ) -> dict[str, jnp.ndarray]:
            if self.uses_full_neighbor_list():
                if comm_dict is not None:
                    raise NotImplementedError(
                        "JAX graph evaluation does not implement MPI exchange."
                    )
                return forward_graph_quartet(
                    self,
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping=mapping,
                    fparam=fparam,
                    aparam=aparam,
                    charge_spin=charge_spin,
                    spin=spin,
                    extended_coord_corr=extended_coord_corr,
                )
            if spin is not None:
                raise NotImplementedError(
                    "Native spin requires a full-neighbor graph descriptor."
                )
            return forward_common_atomic(
                self,
                extended_coord,
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=do_atomic_virial,
                extended_coord_corr=extended_coord_corr,
                charge_spin=charge_spin,
            )

        def call_common_lower(
            self,
            extended_coord: jnp.ndarray,
            extended_atype: jnp.ndarray,
            nlist: jnp.ndarray,
            mapping: jnp.ndarray | None = None,
            fparam: jnp.ndarray | None = None,
            aparam: jnp.ndarray | None = None,
            do_atomic_virial: bool = False,
            extended_coord_corr: jnp.ndarray | None = None,
            comm_dict: dict | None = None,
            charge_spin: jnp.ndarray | None = None,
            spin: jnp.ndarray | None = None,
        ) -> dict[str, jnp.ndarray]:
            """Evaluate a fixed-capacity neighbor list, including native spin."""
            nf, nall = extended_atype.shape
            extended_coord = extended_coord.reshape(nf, nall, 3)
            nlist = self.format_nlist(
                extended_coord,
                extended_atype,
                nlist,
                extra_nlist_sort=self.need_sorted_nlist_for_lower(),
            )
            cc, _, fp, ap, cs, sp, input_prec = self._input_type_cast(
                extended_coord,
                fparam=fparam,
                aparam=aparam,
                charge_spin=charge_spin,
                spin=spin,
            )
            if sp is not None:
                sp = sp.reshape(nf, -1, 3)
                sp = jnp.pad(sp, ((0, 0), (0, nall - sp.shape[1]), (0, 0)))
            result = self.forward_common_atomic(
                cc,
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fp,
                aparam=ap,
                do_atomic_virial=do_atomic_virial,
                extended_coord_corr=extended_coord_corr,
                comm_dict=comm_dict,
                charge_spin=cs,
                spin=sp,
            )
            return self._output_type_cast(result, input_prec)

        def format_nlist(
            self,
            extended_coord: jnp.ndarray,
            extended_atype: jnp.ndarray,
            nlist: jnp.ndarray,
            extra_nlist_sort: bool = False,
        ) -> jnp.ndarray:
            if self.uses_full_neighbor_list():
                return nlist
            return dpmodel_model.format_nlist(
                self,
                jax.lax.stop_gradient(extended_coord),
                extended_atype,
                nlist,
                extra_nlist_sort=extra_nlist_sort,
            )

    return jax_model
