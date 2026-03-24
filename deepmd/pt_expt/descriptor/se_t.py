# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.common import (
    cast_precision,
)
from deepmd.dpmodel.descriptor.se_t import DescrptSeT as DescrptSeTDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.utils.update_sel import (
    UpdateSel,
)


@BaseDescriptor.register("se_e3")
@BaseDescriptor.register("se_at")
@BaseDescriptor.register("se_a_3be")
@torch_module
class DescrptSeT(DescrptSeTDP):
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
            exclude_types=data["exclude_types"],
            activation_fn=ActivationFn(data["activation_function"]),
        )
        # SE_T scales strides by 10
        stride_1_scaled = table_stride_1 * 10
        stride_2_scaled = table_stride_2 * 10
        self.table_config = [
            table_extrapolate,
            stride_1_scaled,
            stride_2_scaled,
            check_frequency,
        ]
        self.lower, self.upper = self.table.build(
            min_nbor_dist, table_extrapolate, stride_1_scaled, stride_2_scaled
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

        n_networks = self.ntypes * self.ntypes
        for embedding_idx in range(n_networks):
            ti = embedding_idx % self.ntypes
            tj = embedding_idx // self.ntypes
            if ti <= tj:
                net = "filter_" + str(ti) + "_net_" + str(tj)
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
            else:
                # Placeholder for ti > tj (not used, but keeps indexing consistent)
                info_ii = torch.zeros(6, dtype=prec, device="cpu")
                tensor_data_ii = torch.zeros(0, dtype=prec, device="cpu")
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
            return DescrptSeTDP.call.__wrapped__(
                self, coord_ext, atype_ext, nlist, mapping
            )
        return self._call_compressed(coord_ext, atype_ext, nlist)

    def _call_compressed(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
    ) -> Any:
        """Compressed forward using tabulate_fusion_se_t custom op."""
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

        result = torch.zeros(
            [nfnl, ng],
            dtype=coord_ext.dtype,
            device=coord_ext.device,
        )
        exclude_mask = self.emask.build_type_exclude_mask(nlist, atype_ext)
        exclude_mask = exclude_mask.view(nfnl, nnei)
        rr = rr.view(nfnl, nnei, 4)

        for embedding_idx, (compress_data_ii, compress_info_ii) in enumerate(
            zip(self.compress_data, self.compress_info, strict=True)
        ):
            ti = embedding_idx % self.ntypes
            tj = embedding_idx // self.ntypes
            if ti <= tj:
                nei_type_i = self.sel[tj]
                nei_type_j = self.sel[ti]
                # nfnl x nt_i x 3
                rr_i = rr[:, sec[ti] : sec[ti + 1], 1:]
                mm_i = exclude_mask[:, sec[ti] : sec[ti + 1]]
                rr_i = rr_i * mm_i[:, :, None]
                # nfnl x nt_j x 3
                rr_j = rr[:, sec[tj] : sec[tj + 1], 1:]
                mm_j = exclude_mask[:, sec[tj] : sec[tj + 1]]
                rr_j = rr_j * mm_j[:, :, None]
                # nfnl x nt_i x nt_j
                env_ij = torch.einsum("ijm,ikm->ijk", rr_i, rr_j)
                ebd_env_ij = env_ij.view(-1, 1)
                res_ij = torch.ops.deepmd.tabulate_fusion_se_t(
                    compress_data_ii.contiguous(),
                    compress_info_ii.cpu().contiguous(),
                    ebd_env_ij.contiguous(),
                    env_ij.contiguous(),
                    ng,
                )[0]
                res_ij = res_ij * (1.0 / float(nei_type_i) / float(nei_type_j))
                result += res_ij

        result = result.view(nf, nloc, ng)
        return result, None, None, None, ww
