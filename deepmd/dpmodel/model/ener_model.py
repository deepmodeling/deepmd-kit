# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Mapping,
)
from copy import (
    deepcopy,
)
from itertools import (
    pairwise,
)
from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.atomic_model import (
    DPEnergyAtomicModel,
)
from deepmd.dpmodel.common import (
    NativeOP,
    to_numpy_array,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
)
from deepmd.dpmodel.utils.neighbor_list import (
    NeighborList,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)

DPEnergyModel_ = make_model(DPEnergyAtomicModel, T_Bases=(NativeOP, BaseModel))


@BaseModel.register("ener")
@BaseModel.register("sezm_ener")
@BaseModel.register("dpa4_ener")
class EnergyModel(DPModelCommon, DPEnergyModel_):
    r"""Energy model that predicts total energy and derived quantities.

    The model takes atomic energies from the atomic model and computes
    global properties by reduction and differentiation:

    **Reduction** (total energy):

    .. math::
        E = \sum_{i=1}^{N} E^i,

    where :math:`E^i` is the atomic energy from the atomic model.

    **Differentiation** (forces and virials):

    .. math::
        \mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i},

    .. math::
        \boldsymbol{\Xi} = -\sum_{i=1}^{N} \frac{\partial E}{\partial \mathbf{r}_i} \otimes \mathbf{r}_i
        = \sum_{i=1}^{N} \mathbf{r}_i \otimes \mathbf{F}_i,

    where :math:`\mathbf{F}_i` is the force on atom :math:`i` and
    :math:`\boldsymbol{\Xi}` is the virial tensor.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        DPEnergyModel_.__init__(self, *args, **kwargs)
        self._enable_hessian = False
        self.hess_fitting_def = None

    def enable_hessian(self) -> None:
        self.hess_fitting_def = deepcopy(self.atomic_output_def())
        self.hess_fitting_def["energy"].r_hessian = True
        self._enable_hessian = True

    def atomic_output_def(self) -> FittingOutputDef:
        if self._enable_hessian:
            return self.hess_fitting_def
        return super().atomic_output_def()

    def call(
        self,
        coord: Array,
        atype: Array,
        box: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        do_atomic_virial: bool = False,
        mixed_batch: Mapping[str, Array] | None = None,
        charge_spin: Array | None = None,
        neighbor_list: NeighborList | None = None,
    ) -> dict[str, Array]:
        """Evaluate the energy model.

        Most arguments share the meaning of :meth:`call_common`.

        Parameters
        ----------
        neighbor_list
            The neighbor-list construction strategy forwarded to
            :meth:`call_common`.  ``None`` uses the default all-pairs builder
            (:class:`~deepmd.dpmodel.utils.neighbor_list.NeighborList`
            subclass :class:`~deepmd.dpmodel.utils.default_neighbor_list.DefaultNeighborList`),
            reproducing the historical behavior; an alternative strategy may be
            injected to accelerate neighbor-list construction without changing
            the model outputs.
        """
        if mixed_batch is not None:
            return self.call_flat(
                coord=coord,
                atype=atype,
                box=box,
                fparam=fparam,
                aparam=aparam,
                charge_spin=charge_spin,
                do_atomic_virial=do_atomic_virial,
                mixed_batch=mixed_batch,
                neighbor_list=neighbor_list,
            )

        model_ret = self.call_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
            do_atomic_virial=do_atomic_virial,
            neighbor_list=neighbor_list,
        )
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if self.do_grad_r("energy") and model_ret["energy_derv_r"] is not None:
            model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy") and model_ret["energy_derv_c_redu"] is not None:
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial and model_ret["energy_derv_c"] is not None:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-2)
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        if self._enable_hessian and model_ret.get("energy_derv_r_derv_r") is not None:
            model_predict["hessian"] = model_ret["energy_derv_r_derv_r"].squeeze(-3)
        return model_predict

    def call_flat(
        self,
        coord: Array,
        atype: Array,
        mixed_batch: Mapping[str, Array],
        box: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        charge_spin: Array | None = None,
        do_atomic_virial: bool = False,
        neighbor_list: NeighborList | None = None,
    ) -> dict[str, Array]:
        """Evaluate a flattened mixed-nloc batch with the dpmodel backend.

        The dpmodel backend reuses the regular one-frame call path for each
        segment described by ``ptr`` and merges the translated outputs back into
        the flat mixed-batch layout.
        """
        batch = mixed_batch.get("batch")
        ptr = mixed_batch.get("ptr")
        if batch is None or ptr is None:
            raise ValueError("mixed_batch must contain both batch and ptr.")
        if self._enable_hessian:
            raise NotImplementedError(
                "Hessian is not implemented for dpmodel mixed-batch flat calls."
            )

        xp = array_api_compat.array_namespace(coord, atype)
        ptr_np = to_numpy_array(ptr)
        if ptr_np is None:
            raise ValueError("ptr is required for mixed batches.")
        ptr_np = np.asarray(ptr_np, dtype=np.int64)
        if ptr_np.ndim != 1 or ptr_np.size < 2:
            raise ValueError("ptr must be a 1D array with at least two entries.")

        total_atoms = coord.shape[0]
        if ptr_np[0] != 0 or ptr_np[-1] != total_atoms:
            raise ValueError("ptr must start at 0 and end at the number of atoms.")
        if batch.shape[0] != total_atoms:
            raise ValueError("batch length must match the number of atoms.")

        frame_outputs = []
        for frame_idx, (start, end) in enumerate(pairwise(ptr_np)):
            nloc = int(end - start)
            frame_coord = xp.reshape(coord[start:end], (1, nloc * 3))
            frame_atype = xp.reshape(atype[start:end], (1, nloc))
            frame_box = box[frame_idx : frame_idx + 1] if box is not None else None
            frame_fparam = (
                fparam[frame_idx : frame_idx + 1] if fparam is not None else None
            )
            frame_aparam = (
                xp.reshape(aparam[start:end], (1, nloc, *aparam.shape[1:]))
                if aparam is not None
                else None
            )
            frame_charge_spin = (
                charge_spin[frame_idx : frame_idx + 1]
                if charge_spin is not None
                else None
            )
            frame_outputs.append(
                self.call(
                    frame_coord,
                    frame_atype,
                    box=frame_box,
                    fparam=frame_fparam,
                    aparam=frame_aparam,
                    charge_spin=frame_charge_spin,
                    do_atomic_virial=do_atomic_virial,
                    neighbor_list=neighbor_list,
                )
            )

        return self._merge_flat_frame_outputs(frame_outputs)

    @staticmethod
    def _merge_flat_frame_outputs(
        frame_outputs: list[dict[str, Array]],
    ) -> dict[str, Array]:
        if not frame_outputs:
            raise ValueError("mixed-batch input must contain at least one frame.")

        framewise_keys = {"energy", "virial"}
        result: dict[str, Array] = {}
        for key in frame_outputs[0]:
            values = [frame_output[key] for frame_output in frame_outputs]
            xp = array_api_compat.array_namespace(values[0])
            if key in framewise_keys:
                result[key] = xp.concat(values, axis=0)
            elif key == "mask":
                result[key] = xp.concat(
                    [xp.reshape(value, (-1,)) for value in values],
                    axis=0,
                )
            else:
                result[key] = xp.concat(
                    [
                        xp.reshape(value, (-1, *value.shape[2:]))
                        if value.ndim >= 3
                        else xp.reshape(value, (-1,))
                        for value in values
                    ],
                    axis=0,
                )
        return result

    def call_lower(
        self,
        extended_coord: Array,
        extended_atype: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        do_atomic_virial: bool = False,
        charge_spin: Array | None = None,
    ) -> dict[str, Array]:
        model_ret = self.call_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if self.do_grad_r("energy") and model_ret.get("energy_derv_r") is not None:
            model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy") and model_ret.get("energy_derv_c_redu") is not None:
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial and model_ret.get("energy_derv_c") is not None:
                model_predict["extended_virial"] = model_ret["energy_derv_c"].squeeze(
                    -2
                )
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    def translated_output_def(self) -> dict[str, Any]:
        """Get the translated output definition.

        Maps internal output names to user-facing names, e.g.
        ``energy_redu`` -> ``energy``, ``energy_derv_r`` -> ``force``.
        """
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": out_def_data["energy"],
            "energy": out_def_data["energy_redu"],
        }
        if self.do_grad_r("energy"):
            output_def["force"] = out_def_data["energy_derv_r"]
            output_def["force"].squeeze(-2)
        if self.do_grad_c("energy"):
            output_def["virial"] = out_def_data["energy_derv_c_redu"]
            output_def["virial"].squeeze(-2)
            output_def["atom_virial"] = out_def_data["energy_derv_c"]
            output_def["atom_virial"].squeeze(-2)
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        if self._enable_hessian:
            output_def["hessian"] = out_def_data["energy_derv_r_derv_r"]
        return output_def
