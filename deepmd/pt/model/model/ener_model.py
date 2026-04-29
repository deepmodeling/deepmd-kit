# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Any
import pickle
import torch
import os
from deepmd.pt.model.atomic_model import (
    DPEnergyAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)

from .dp_model import (
    DPModelCommon,
)
from .make_hessian_model import (
    make_hessian_model,
)
from .make_model import (
    make_model,
)

DPEnergyModel_ = make_model(DPEnergyAtomicModel)


@BaseModel.register("ener")
class EnergyModel(DPModelCommon, DPEnergyModel_):
    model_type = "ener"
    _FLAT_MIXED_BATCH_ATOMWISE_KEYS = frozenset(
        {"atom_energy", "atom_virial", "force", "mask"}
    )

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        DPEnergyModel_.__init__(self, *args, **kwargs)
        self._hessian_enabled = False

    def enable_hessian(self) -> None:
        self.__class__ = make_hessian_model(type(self))
        self.hess_fitting_def = super(type(self), self).atomic_output_def()
        self.requires_hessian("energy")
        self._hessian_enabled = True

    def translated_output_def(self) -> dict[str, Any]:
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
        if self._hessian_enabled:
            output_def["hessian"] = out_def_data["energy_derv_r_derv_r"]
        return output_def

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        batch: torch.Tensor | None = None,
        ptr: torch.Tensor | None = None,
        extended_atype: torch.Tensor | None = None,
        extended_batch: torch.Tensor | None = None,
        extended_image: torch.Tensor | None = None,
        extended_ptr: torch.Tensor | None = None,
        mapping: torch.Tensor | None = None,
        central_ext_index: torch.Tensor | None = None,
        nlist: torch.Tensor | None = None,
        nlist_ext: torch.Tensor | None = None,
        a_nlist: torch.Tensor | None = None,
        a_nlist_ext: torch.Tensor | None = None,
        nlist_mask: torch.Tensor | None = None,
        a_nlist_mask: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        angle_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Detect mixed batch format (flattened atoms with batch/ptr)

        if not os.path.exists("/aisi/mnt/data_nas/liwentao/devel_workspace/deepmd-kit-lmdb/pkl/debug.pkl"):
            with open("/aisi/mnt/data_nas/liwentao/devel_workspace/deepmd-kit-lmdb/pkl/debug.pkl", "wb") as f:
                pickle.dump({
                    "coord": coord,
                    "atype": atype,
                    "box": box,
                    "fparam": fparam,
                    "aparam": aparam,
                    "do_atomic_virial": do_atomic_virial,
                "batch": batch,
                "ptr": ptr,
                "extended_atype": extended_atype,
                "extended_batch": extended_batch,
                "extended_image": extended_image,
                "extended_ptr": extended_ptr,
                "mapping": mapping,
                "central_ext_index": central_ext_index,
                "nlist": nlist,
                "nlist_ext": nlist_ext,
                "a_nlist": a_nlist,
                "a_nlist_ext": a_nlist_ext,
                "nlist_mask": nlist_mask,
                "a_nlist_mask": a_nlist_mask,
                "edge_index": edge_index,
                "angle_index": angle_index,
                }, f)
        if batch is not None and ptr is not None:
            # Use new graph-native path with pack/unpack
            model_ret = self.forward_common_flat(
                coord=coord,
                atype=atype,
                batch=batch,
                ptr=ptr,
                box=box,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=do_atomic_virial,
                extended_atype=extended_atype,
                extended_batch=extended_batch,
                extended_image=extended_image,
                extended_ptr=extended_ptr,
                mapping=mapping,
                central_ext_index=central_ext_index,
                nlist=nlist,
                nlist_ext=nlist_ext,
                a_nlist=a_nlist,
                a_nlist_ext=a_nlist_ext,
                nlist_mask=nlist_mask,
                a_nlist_mask=a_nlist_mask,
                edge_index=edge_index,
                angle_index=angle_index,
            )
            # Reorganize output for mixed batch
            if self.get_fitting_net() is not None:
                model_predict = {}
                model_predict["atom_energy"] = model_ret["energy"]
                model_predict["energy"] = model_ret["energy_redu"]
                if self.do_grad_r("energy"):
                    # For flat batch, energy_derv_r is [total_atoms, 1, 3]
                    model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
                else:
                    # Fallback to dforce if gradient computation is disabled
                    if "dforce" in model_ret:
                        model_predict["force"] = model_ret["dforce"]
                if self.do_grad_c("energy"):
                    model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
                    if do_atomic_virial:
                        model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-2)
                if "mask" in model_ret:
                    model_predict["mask"] = model_ret["mask"]
                if self._hessian_enabled:
                    model_predict["hessian"] = model_ret["energy_derv_r_derv_r"].squeeze(-3)
            else:
                model_predict = model_ret
                model_predict["updated_coord"] += coord
            
            return model_predict

        # Regular dense batch path
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            if self.do_grad_r("energy"):
                model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
            else:
                # Fallback to dforce if gradient computation is disabled
                if "dforce" in model_ret:
                    model_predict["force"] = model_ret["dforce"]
            if self.do_grad_c("energy"):
                model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(
                        -2
                    )
            if "mask" in model_ret:
                model_predict["mask"] = model_ret["mask"]
            if self._hessian_enabled:
                model_predict["hessian"] = model_ret["energy_derv_r_derv_r"].squeeze(-3)
        else:
            model_predict = model_ret
            model_predict["updated_coord"] += coord

        return model_predict

    def forward_mixed_batch(
        self,
        coord: list[torch.Tensor],
        atype: list[torch.Tensor],
        box: list[torch.Tensor | None] | None = None,
        fparam: list[torch.Tensor | None] | None = None,
        aparam: list[torch.Tensor | None] | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, Any]:
        """Forward a mixed batch of frames with different ``nloc``.

        This is a correctness-first helper for sidecar raw-LMDB experiments.
        Each frame is forwarded independently through the existing dense path,
        then outputs are merged:

        - reducible outputs with identical non-batch shape (for example ``energy``
          and ``virial``) are concatenated along batch dimension
        - atom-wise outputs with frame-dependent shape (for example ``force`` and
          ``atom_energy``) are returned as ``list[Tensor]``

        The normal ``forward`` path is unchanged.
        """
        nframe = len(coord)
        if len(atype) != nframe:
            raise ValueError(
                f"coord and atype length mismatch: {nframe} vs {len(atype)}"
            )
        if box is None:
            box = [None] * nframe
        if fparam is None:
            fparam = [None] * nframe
        if aparam is None:
            aparam = [None] * nframe
        if not (len(box) == len(fparam) == len(aparam) == nframe):
            raise ValueError("mixed-batch input lists must have the same length")

        outputs = []
        for coord_i, atype_i, box_i, fparam_i, aparam_i in zip(
            coord, atype, box, fparam, aparam, strict=True
        ):
            coord_i = coord_i.unsqueeze(0) if coord_i.dim() == 2 else coord_i
            atype_i = atype_i.unsqueeze(0) if atype_i.dim() == 1 else atype_i
            if box_i is not None and box_i.dim() == 1:
                box_i = box_i.unsqueeze(0)
            if fparam_i is not None and fparam_i.dim() == 1:
                fparam_i = fparam_i.unsqueeze(0)
            if aparam_i is not None and aparam_i.dim() == 2:
                aparam_i = aparam_i.unsqueeze(0)
            outputs.append(
                self.forward(
                    coord_i,
                    atype_i,
                    box=box_i,
                    fparam=fparam_i,
                    aparam=aparam_i,
                    do_atomic_virial=do_atomic_virial,
                )
            )
        return self._merge_mixed_batch_outputs(outputs)

    @staticmethod
    def _merge_mixed_batch_outputs(
        outputs: list[dict[str, torch.Tensor]],
    ) -> dict[str, Any]:
        if not outputs:
            return {}
        merged: dict[str, Any] = {}
        for key in outputs[0]:
            values = [item[key] for item in outputs if key in item]
            if not values:
                continue
            same_shape = all(v.shape == values[0].shape for v in values)
            if same_shape:
                merged[key] = torch.cat(values, dim=0)
            else:
                merged[key] = values
        return merged

    def forward_mixed_batch_flat(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, Any]:
        """Forward a flattened mixed batch.

        Parameters
        ----------
        coord
            Flattened atomic coordinates with shape ``[N, 3]``.
        atype
            Flattened atomic types with shape ``[N]``.
        batch
            Atom-to-system assignment with shape ``[N]``.
        ptr
            Prefix-sum system offsets with shape ``[B + 1]``.
        box
            Optional frame-wise box tensor with shape ``[B, 9]``.
        fparam
            Optional frame-wise tensor with shape ``[B, df]``.
        aparam
            Optional atom-wise tensor with shape ``[N, da]``.

        Notes
        -----
        This is a correctness-first bridge from a PyG-style flattened batch to the
        existing dense per-frame forward path. Each system is sliced out using
        ``ptr`` and forwarded independently through ``forward_mixed_batch``.
        """
        if coord.dim() != 2 or coord.shape[-1] != 3:
            raise ValueError(f"coord must have shape [N, 3], got {tuple(coord.shape)}")
        if atype.dim() != 1:
            raise ValueError(f"atype must have shape [N], got {tuple(atype.shape)}")
        if batch.dim() != 1:
            raise ValueError(f"batch must have shape [N], got {tuple(batch.shape)}")
        if ptr.dim() != 1:
            raise ValueError(f"ptr must have shape [B + 1], got {tuple(ptr.shape)}")
        if not (coord.shape[0] == atype.shape[0] == batch.shape[0]):
            raise ValueError(
                "coord, atype and batch must agree on the flattened atom dimension"
            )
        if ptr.numel() < 2:
            raise ValueError("ptr must contain at least one system boundary")
        nframe = int(ptr.numel() - 1)
        if int(ptr[0].item()) != 0 or int(ptr[-1].item()) != int(coord.shape[0]):
            raise ValueError("ptr must start at 0 and end at N")
        counts = ptr[1:] - ptr[:-1]
        expected_batch = torch.repeat_interleave(
            torch.arange(nframe, device=batch.device, dtype=batch.dtype),
            counts.to(batch.device),
        )
        if not torch.equal(batch, expected_batch):
            raise ValueError("batch is inconsistent with ptr")
        if box is not None and box.shape[0] != nframe:
            raise ValueError("box first dimension must match the number of systems")
        if fparam is not None and fparam.shape[0] != nframe:
            raise ValueError("fparam first dimension must match the number of systems")

        coord_list: list[torch.Tensor] = []
        atype_list: list[torch.Tensor] = []
        box_list: list[torch.Tensor | None] = []
        fparam_list: list[torch.Tensor | None] = []
        aparam_list: list[torch.Tensor | None] = []

        for ii in range(nframe):
            start = int(ptr[ii].item())
            end = int(ptr[ii + 1].item())
            coord_list.append(coord[start:end])
            atype_list.append(atype[start:end])
            box_list.append(box[ii] if box is not None else None)
            fparam_list.append(fparam[ii] if fparam is not None else None)
            aparam_list.append(aparam[start:end] if aparam is not None else None)

        mixed_out = self.forward_mixed_batch(
            coord_list,
            atype_list,
            box=box_list,
            fparam=fparam_list,
            aparam=aparam_list,
            do_atomic_virial=do_atomic_virial,
        )
        for key in self._FLAT_MIXED_BATCH_ATOMWISE_KEYS:
            if key not in mixed_out:
                continue
            value = mixed_out[key]
            if isinstance(value, list):
                flat_chunks = []
                for chunk in value:
                    if chunk.dim() > 0 and chunk.shape[0] == 1:
                        flat_chunks.append(chunk.squeeze(0))
                    else:
                        flat_chunks.append(chunk)
                mixed_out[key] = torch.cat(flat_chunks, dim=0)
            elif torch.is_tensor(value) and value.dim() >= 2 and value.shape[0] == nframe:
                tail_shape = value.shape[2:]
                if tail_shape:
                    mixed_out[key] = value.reshape(-1, *tail_shape)
                else:
                    mixed_out[key] = value.reshape(-1)
        return mixed_out

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
        )
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            if self.do_grad_r("energy"):
                model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
            if self.do_grad_c("energy"):
                model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["extended_virial"] = model_ret[
                        "energy_derv_c"
                    ].squeeze(-2)
            else:
                assert model_ret["dforce"] is not None
                model_predict["dforce"] = model_ret["dforce"]
            if "mask" in model_ret:
                model_predict["mask"] = model_ret["mask"]
        else:
            model_predict = model_ret
        return model_predict
