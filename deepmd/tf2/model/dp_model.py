# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.model import (
    DPModelCommon,
)
from deepmd.dpmodel.utils.neighbor_list import (
    NeighborList,
)
from deepmd.tf2.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.tf2.common import (
    tf2_module,
    to_tensorflow_array,
    to_tf_tensor,
)
from deepmd.tf2.env import (
    jnp,
    stop_gradient,
)
from deepmd.tf2.format_nlist import format_nlist as tf2_format_nlist
from deepmd.tf2.model.base_model import (
    forward_common_atomic,
)
from deepmd.tf2.nlist import (
    nlist_distinguish_types,
)


def make_tf2_dp_model_from_dpmodel(
    dpmodel_model: type[DPModelCommon], tf2_atomicmodel: type[DPAtomicModel]
) -> type[DPModelCommon]:
    """Make a tf2 backend DP model from a DPModel backend DP model.

    Parameters
    ----------
    dpmodel_model : type[DPModelCommon]
        The DPModel backend DP model.
    tf2_atomicmodel : type[DPAtomicModel]
        The tf2 backend DP atomic model.

    Returns
    -------
    type[DPModelCommon]
        The tf2 backend DP model.
    """

    @tf2_module
    class tf2_model(dpmodel_model):
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
        ) -> dict[str, jnp.ndarray]:
            return super().call_common(
                to_tensorflow_array(coord),
                to_tensorflow_array(atype),
                box=to_tensorflow_array(box),
                fparam=to_tensorflow_array(fparam),
                aparam=to_tensorflow_array(aparam),
                do_atomic_virial=do_atomic_virial,
                coord_corr_for_virial=to_tensorflow_array(coord_corr_for_virial),
                charge_spin=to_tensorflow_array(charge_spin),
                neighbor_list=neighbor_list,
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
        ) -> dict[str, jnp.ndarray]:
            return super().call_common_lower(
                to_tensorflow_array(extended_coord),
                to_tensorflow_array(extended_atype),
                to_tensorflow_array(nlist),
                mapping=to_tensorflow_array(mapping),
                fparam=to_tensorflow_array(fparam),
                aparam=to_tensorflow_array(aparam),
                do_atomic_virial=do_atomic_virial,
                extended_coord_corr=to_tensorflow_array(extended_coord_corr),
                comm_dict=comm_dict,
                charge_spin=to_tensorflow_array(charge_spin),
            )

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
        ) -> dict[str, jnp.ndarray]:
            del comm_dict  # tf2 path has no MPI ghost exchange
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

        def format_nlist(
            self,
            extended_coord: jnp.ndarray,
            extended_atype: jnp.ndarray,
            nlist: jnp.ndarray,
            extra_nlist_sort: bool = False,
        ) -> jnp.ndarray:
            del extra_nlist_sort
            ret = tf2_format_nlist(
                to_tf_tensor(stop_gradient(extended_coord)),
                to_tf_tensor(nlist),
                sum(self.get_sel()),
                self.get_rcut(),
            )
            if not self.mixed_types():
                ret = nlist_distinguish_types(
                    ret,
                    to_tf_tensor(extended_atype),
                    self.get_sel(),
                )
            return to_tensorflow_array(ret)

    return tf2_model
