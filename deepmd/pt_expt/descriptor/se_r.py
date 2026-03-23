# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.common import (
    cast_precision,
)
from deepmd.dpmodel.descriptor.se_r import DescrptSeR as DescrptSeRDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.utils.update_sel import (
    UpdateSel,
)


@BaseDescriptor.register("se_e2_r")
@BaseDescriptor.register("se_r")
@torch_module
class DescrptSeR(DescrptSeRDP):
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

        for ii in range(self.ntypes):
            net = "filter_-1_net_" + str(ii)
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
            return DescrptSeRDP.call.__wrapped__(
                self, coord_ext, atype_ext, nlist, mapping
            )
        return self._call_compressed(coord_ext, atype_ext, nlist)

    def _call_compressed(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
    ) -> Any:
        """Compressed forward using tabulate_fusion_se_r custom op."""
        # env_mat: nf x nloc x nnei x 1 (radial only)
        rr, _diff, ww = self.env_mat.call(
            coord_ext,
            atype_ext,
            nlist,
            self.davg[...],
            self.dstd[...],
            True,
        )
        nf, nloc, nnei, _ = rr.shape
        sec = self.sel_cumsum
        ng = self.neuron[-1]
        nfnl = nf * nloc

        exclude_mask = self.emask.build_type_exclude_mask(nlist, atype_ext)
        exclude_mask = exclude_mask.view(nfnl, nnei)
        rr = rr.view(nfnl, nnei, 1)

        xyz_scatter_total = []
        for ii, (compress_data_ii, compress_info_ii) in enumerate(
            zip(self.compress_data, self.compress_info, strict=True)
        ):
            mm = exclude_mask[:, sec[ii] : sec[ii + 1]]
            ss = rr[:, sec[ii] : sec[ii + 1], :]
            ss = ss * mm[:, :, None]
            ss = ss.squeeze(-1)
            xyz_scatter = torch.ops.deepmd.tabulate_fusion_se_r(
                compress_data_ii.contiguous(),
                compress_info_ii.cpu().contiguous(),
                ss,
                ng,
            )[0]
            xyz_scatter_total.append(xyz_scatter)

        res_rescale = 1.0 / 5.0
        xyz_scatter = torch.cat(xyz_scatter_total, dim=1)
        result = torch.mean(xyz_scatter, dim=1) * res_rescale
        result = result.view(nf, nloc, ng)
        return result, None, None, None, ww
