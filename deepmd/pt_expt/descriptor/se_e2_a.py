# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.common import (
    cast_precision,
)
from deepmd.dpmodel.descriptor.se_e2_a import DescrptSeA as DescrptSeADP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.utils.update_sel import (
    UpdateSel,
)


@BaseDescriptor.register("se_e2_a")
@BaseDescriptor.register("se_a")
@torch_module
class DescrptSeA(DescrptSeADP):
    _update_sel_cls = UpdateSel

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        from deepmd.pt.utils.utils import (
            ActivationFn,
        )
        from deepmd.pt_expt.utils.tabulate import (
            DPTabulate,
        )

        if self.compress:
            raise ValueError("Compression is already enabled.")
        data = self.serialize()
        self.table = DPTabulate(
            self,
            data["neuron"],
            data["type_one_side"],
            data["exclude_types"],
            ActivationFn(data["activation_function"]),
        )
        self.table_config = [
            table_extrapolate,
            table_stride_1,
            table_stride_2,
            check_frequency,
        ]
        self.lower, self.upper = self.table.build(
            min_nbor_dist, table_extrapolate, table_stride_1, table_stride_2
        )
        self._store_compress_data()
        self.compress = True

    def _store_compress_data(self) -> None:
        """Store tabulated data as buffers for the compressed forward path."""
        table_data = self.table.data
        table_config = self.table_config
        lower = self.lower
        upper = self.upper
        prec = self.davg.dtype

        compress_data_list = []
        compress_info_list = []

        ndim = 1 if self.type_one_side else 2
        n_networks = self.ntypes**ndim
        for embedding_idx in range(n_networks):
            if self.type_one_side:
                ii = embedding_idx
                ti = -1
            else:
                ii = embedding_idx // self.ntypes
                ti = embedding_idx % self.ntypes
            if self.type_one_side:
                net = "filter_-1_net_" + str(ii)
            else:
                net = "filter_" + str(ti) + "_net_" + str(ii)
            info_ii = torch.as_tensor(
                [
                    lower[net],
                    upper[net],
                    upper[net] * table_config[0],
                    table_config[1],
                    table_config[2],
                    table_config[3],
                ],
                dtype=prec,
                device="cpu",
            )
            tensor_data_ii = table_data[net].to(dtype=prec)
            compress_data_list.append(
                torch.nn.Parameter(tensor_data_ii, requires_grad=False)
            )
            compress_info_list.append(torch.nn.Parameter(info_ii, requires_grad=False))
        self.compress_data = torch.nn.ParameterList(compress_data_list)
        self.compress_info = torch.nn.ParameterList(compress_info_list)

    @cast_precision
    def call(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
    ) -> Any:
        if not self.compress:
            return DescrptSeADP.call.__wrapped__(
                self, coord_ext, atype_ext, nlist, mapping
            )
        return self._call_compressed(coord_ext, atype_ext, nlist)

    def _call_compressed(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
    ) -> Any:
        """Compressed forward using tabulate_fusion_se_a custom op."""
        # env_mat: nf x nloc x nnei x 4
        rr, _diff, ww = self.env_mat.call(
            coord_ext,
            atype_ext,
            nlist,
            self.davg[...],
            self.dstd[...],
        )
        nf, nloc, nnei, _ = rr.shape
        sec = self.sel_cumsum
        ng = self.neuron[-1]

        nfnl = nf * nloc
        xyz_scatter = torch.zeros(
            [nfnl, 4, ng],
            dtype=coord_ext.dtype,
            device=coord_ext.device,
        )
        exclude_mask = self.emask.build_type_exclude_mask(nlist, atype_ext)
        exclude_mask = exclude_mask.view(nfnl, nnei)
        rr = rr.view(nfnl, nnei, 4)

        if self.type_one_side:
            for embedding_idx, (compress_data_ii, compress_info_ii) in enumerate(
                zip(self.compress_data, self.compress_info, strict=True)
            ):
                ii = embedding_idx
                mm = exclude_mask[:, sec[ii] : sec[ii + 1]]
                rr_i = rr[:, sec[ii] : sec[ii + 1], :]
                rr_i = rr_i * mm[:, :, None]
                ss = rr_i[:, :, :1]
                ss = ss.reshape(-1, 1)
                gr = torch.ops.deepmd.tabulate_fusion_se_a(
                    compress_data_ii.contiguous(),
                    compress_info_ii.cpu().contiguous(),
                    ss.contiguous(),
                    rr_i.contiguous(),
                    ng,
                )[0]
                xyz_scatter += gr
        else:
            atype_loc = atype_ext[:, :nloc].reshape(nfnl)
            for embedding_idx, (compress_data_ii, compress_info_ii) in enumerate(
                zip(self.compress_data, self.compress_info, strict=True)
            ):
                ii = embedding_idx // self.ntypes
                ti = embedding_idx % self.ntypes
                ti_mask = atype_loc.eq(ti)
                mm = exclude_mask[ti_mask, sec[ii] : sec[ii + 1]]
                rr_i = rr[ti_mask, sec[ii] : sec[ii + 1], :]
                rr_i = rr_i * mm[:, :, None]
                ss = rr_i[:, :, :1]
                ss = ss.reshape(-1, 1)
                gr = torch.ops.deepmd.tabulate_fusion_se_a(
                    compress_data_ii.contiguous(),
                    compress_info_ii.cpu().contiguous(),
                    ss.contiguous(),
                    rr_i.contiguous(),
                    ng,
                )[0]
                xyz_scatter[ti_mask] += gr

        xyz_scatter /= self.nnei
        xyz_scatter_1 = xyz_scatter.permute(0, 2, 1)
        rot_mat = xyz_scatter_1[:, :, 1:4]
        xyz_scatter_2 = xyz_scatter[:, :, 0 : self.axis_neuron]
        result = torch.matmul(xyz_scatter_1, xyz_scatter_2)
        result = result.view(nf, nloc, ng * self.axis_neuron)
        rot_mat = rot_mat.view(nf, nloc, ng, 3)
        return result, rot_mat, None, None, ww
