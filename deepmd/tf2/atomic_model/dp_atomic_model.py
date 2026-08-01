# SPDX-License-Identifier: LGPL-3.0-or-later
import deepmd.tf2.descriptor as _tf2_descriptor  # noqa: F401
import deepmd.tf2.fitting.fitting as _tf2_fitting  # noqa: F401
import deepmd.tf2.utils.exclude_mask as _tf2_exclude_mask  # noqa: F401
from deepmd.dpmodel.atomic_model.dp_atomic_model import DPAtomicModel as DPAtomicModelDP
from deepmd.tf2.common import (
    tf2_module,
)
from deepmd.tf2.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.tf2.env import (
    stop_gradient,
    tf,
    xp,
)
from deepmd.tf2.fitting.base_fitting import (
    BaseFitting,
)


def make_tf2_dp_atomic_model_from_dpmodel(
    dpmodel_atomic_model: type[DPAtomicModelDP],
) -> type[DPAtomicModelDP]:
    """Make a tf2 backend DP atomic model from a DPModel backend DP atomic model.

    Parameters
    ----------
    dpmodel_atomic_model : type[DPAtomicModelDP]
        The DPModel backend DP atomic model.

    Returns
    -------
    type[DPAtomicModel]
        The tf2 backend DP atomic model.
    """

    @tf2_module
    class tf2_atomic_model(dpmodel_atomic_model):
        base_descriptor_cls = BaseDescriptor
        """The base descriptor class."""
        base_fitting_cls = BaseFitting
        """The base fitting class."""

        @tf.autograph.experimental.do_not_convert
        def make_atom_mask(
            self,
            atype: xp.ndarray,
        ) -> xp.ndarray:
            return atype >= 0

        def forward_common_atomic(
            self,
            extended_coord: xp.ndarray,
            extended_atype: xp.ndarray,
            nlist: xp.ndarray,
            mapping: xp.ndarray | None = None,
            fparam: xp.ndarray | None = None,
            aparam: xp.ndarray | None = None,
            comm_dict: dict | None = None,
            charge_spin: xp.ndarray | None = None,
        ) -> dict[str, xp.ndarray]:
            del comm_dict  # tf2 path has no MPI ghost exchange
            return super().forward_common_atomic(
                extended_coord,
                extended_atype,
                stop_gradient(nlist),
                mapping=mapping,
                fparam=fparam,
                aparam=aparam,
                charge_spin=charge_spin,
            )

    return tf2_atomic_model


class DPAtomicModel(make_tf2_dp_atomic_model_from_dpmodel(DPAtomicModelDP)):
    pass
