# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

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
    unwrap_value,
    wrap_value,
)
from deepmd.tf2.env import (
    stop_gradient,
    tf,
    xp,
)
from deepmd.tf2.make_model import (
    model_call_from_call_lower as tf2_model_call_from_call_lower,
)
from deepmd.tf2.model.base_model import (
    forward_common_atomic,
)
from deepmd.tf2.utils.jit import (
    default_jit_compile,
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
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.set_enable_compile(default_jit_compile())

        def set_enable_compile(self, enable_compile: bool) -> None:
            """Enable or disable XLA compilation for the formatted lower path."""
            if enable_compile:
                self._tf2_call_common_lower_formatted = tf.function(
                    self._call_common_lower_formatted,
                    reduce_retracing=True,
                    jit_compile=True,
                )
            else:
                self._tf2_call_common_lower_formatted = (
                    self._call_common_lower_formatted
                )

        def call_common(
            self,
            coord: xp.ndarray,
            atype: xp.ndarray,
            box: xp.ndarray | None = None,
            fparam: xp.ndarray | None = None,
            aparam: xp.ndarray | None = None,
            do_atomic_virial: bool = False,
            do_deriv_c: bool = True,
            coord_corr_for_virial: xp.ndarray | None = None,
            charge_spin: xp.ndarray | None = None,
            neighbor_list: NeighborList | None = None,
        ) -> dict[str, xp.ndarray]:
            cc, bb, fp, ap, cs, input_prec = self._input_type_cast(
                to_tensorflow_array(coord),
                box=to_tensorflow_array(box),
                fparam=to_tensorflow_array(fparam),
                aparam=to_tensorflow_array(aparam),
                charge_spin=to_tensorflow_array(charge_spin),
            )
            model_predict = tf2_model_call_from_call_lower(
                call_lower=self.call_common_lower,
                rcut=self.get_rcut(),
                sel=self.get_sel(),
                mixed_types=self.mixed_types(),
                model_output_def=self.model_output_def(),
                coord=cc,
                atype=to_tensorflow_array(atype),
                box=bb,
                fparam=fp,
                aparam=ap,
                do_atomic_virial=do_atomic_virial,
                do_deriv_c=do_deriv_c,
                coord_corr_for_virial=to_tensorflow_array(coord_corr_for_virial),
                charge_spin=cs,
                neighbor_list=neighbor_list,
                # Model-level pair exclusion is a nlist-BUILD transform
                # (decision #18/A4): fold it into the freshly built nlist here so
                # the live/eager TF2 upper path matches the SavedModel export and
                # the other backends. Identity when nothing is excluded.
                pair_excl=getattr(self.atomic_model, "pair_excl", None),
                pass_lower_kwargs=True,
            )
            return self._output_type_cast(model_predict, input_prec)

        def call_common_lower(
            self,
            extended_coord: xp.ndarray,
            extended_atype: xp.ndarray,
            nlist: xp.ndarray,
            mapping: xp.ndarray | None = None,
            fparam: xp.ndarray | None = None,
            aparam: xp.ndarray | None = None,
            do_atomic_virial: bool = False,
            do_deriv_c: bool = True,
            extended_coord_corr: xp.ndarray | None = None,
            comm_dict: dict | None = None,
            charge_spin: xp.ndarray | None = None,
            nlist_is_formatted: bool = False,
        ) -> dict[str, xp.ndarray]:
            if nlist_is_formatted:
                return wrap_value(
                    self._tf2_call_common_lower_formatted(
                        extended_coord,
                        extended_atype,
                        nlist,
                        mapping=mapping,
                        fparam=fparam,
                        aparam=aparam,
                        do_atomic_virial=do_atomic_virial,
                        do_deriv_c=do_deriv_c,
                        extended_coord_corr=extended_coord_corr,
                        comm_dict=comm_dict,
                        charge_spin=charge_spin,
                    )
                )
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

        def _call_common_lower_formatted(
            self,
            extended_coord: xp.ndarray,
            extended_atype: xp.ndarray,
            nlist: xp.ndarray,
            mapping: xp.ndarray | None = None,
            fparam: xp.ndarray | None = None,
            aparam: xp.ndarray | None = None,
            do_atomic_virial: bool = False,
            do_deriv_c: bool = True,
            extended_coord_corr: xp.ndarray | None = None,
            comm_dict: dict | None = None,
            charge_spin: xp.ndarray | None = None,
        ) -> dict[str, tf.Tensor]:
            del comm_dict  # tf2 path has no MPI ghost exchange
            extended_coord = to_tensorflow_array(extended_coord)
            extended_atype = to_tensorflow_array(extended_atype)
            nlist = to_tensorflow_array(nlist)
            nframes, _nall = extended_atype.shape[:2]
            extended_coord = xp.reshape(extended_coord, (nframes, -1, 3))
            cc_ext, _, fp, ap, cs, input_prec = self._input_type_cast(
                extended_coord,
                fparam=to_tensorflow_array(fparam),
                aparam=to_tensorflow_array(aparam),
                charge_spin=to_tensorflow_array(charge_spin),
            )
            model_predict = self.forward_common_atomic(
                cc_ext,
                extended_atype,
                nlist,
                mapping=to_tensorflow_array(mapping),
                fparam=fp,
                aparam=ap,
                do_atomic_virial=do_atomic_virial,
                do_deriv_c=do_deriv_c,
                extended_coord_corr=to_tensorflow_array(extended_coord_corr),
                charge_spin=cs,
            )
            return unwrap_value(self._output_type_cast(model_predict, input_prec))

        def forward_common_atomic(
            self,
            extended_coord: xp.ndarray,
            extended_atype: xp.ndarray,
            nlist: xp.ndarray,
            mapping: xp.ndarray | None = None,
            fparam: xp.ndarray | None = None,
            aparam: xp.ndarray | None = None,
            do_atomic_virial: bool = False,
            do_deriv_c: bool = True,
            extended_coord_corr: xp.ndarray | None = None,
            comm_dict: dict | None = None,
            charge_spin: xp.ndarray | None = None,
        ) -> dict[str, xp.ndarray]:
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
                do_deriv_c=do_deriv_c,
                extended_coord_corr=extended_coord_corr,
                charge_spin=charge_spin,
            )

        def format_nlist(
            self,
            extended_coord: xp.ndarray,
            extended_atype: xp.ndarray,
            nlist: xp.ndarray,
            extra_nlist_sort: bool = False,
        ) -> xp.ndarray:
            return dpmodel_model.format_nlist(
                self,
                stop_gradient(extended_coord),
                extended_atype,
                nlist,
                extra_nlist_sort=extra_nlist_sort,
            )

    return tf2_model
