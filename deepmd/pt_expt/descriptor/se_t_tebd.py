# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.common import (
    cast_precision,
)
from deepmd.dpmodel.descriptor.se_t_tebd import DescrptSeTTebd as DescrptSeTTebdDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.utils.update_sel import (
    UpdateSel,
)


@BaseDescriptor.register("se_e3_tebd")
@torch_module
class DescrptSeTTebd(DescrptSeTTebdDP):
    _update_sel_cls = UpdateSel

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Enable compression for the SE_T_TEBD descriptor.

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between atoms
        table_extrapolate
            The scale of model extrapolation
        table_stride_1
            The uniform stride of the first table
        table_stride_2
            The uniform stride of the second table
        check_frequency
            The overflow check frequency
        """
        from deepmd.pt.utils.utils import (
            ActivationFn,
        )
        from deepmd.pt_expt.utils.tabulate import (
            DPTabulate,
        )

        if self.compress:
            raise ValueError("Compression is already enabled.")
        if self.se_ttebd.tebd_input_mode != "strip":
            raise RuntimeError("Cannot compress model when tebd_input_mode != 'strip'")

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

        # Store geometric embedding compress data
        self._store_compress_data()

        # Precompute type embedding data
        self._store_type_embd_data()

        self.compress = True

    def _store_compress_data(self) -> None:
        """Store tabulated data as buffers for the compressed forward path."""
        table_data = self.table.data
        table_config = self.table_config
        lower = self.lower
        upper = self.upper
        prec = self.se_ttebd.mean.dtype

        net_key = "filter_net"
        info = torch.as_tensor(
            [
                lower[net_key],
                upper[net_key],
                upper[net_key] * table_config[0],
                table_config[1],
                table_config[2],
                table_config[3],
            ],
            dtype=prec,
            device="cpu",
        )
        tensor_data = table_data[net_key].to(dtype=prec)
        self.compress_data = torch.nn.ParameterList(
            [torch.nn.Parameter(tensor_data, requires_grad=False)]
        )
        self.compress_info = torch.nn.ParameterList(
            [torch.nn.Parameter(info, requires_grad=False)]
        )

    def _store_type_embd_data(self) -> None:
        """Precompute type embedding pairs and store as a buffer."""
        with torch.no_grad():
            # type_embedding.call() returns (ntypes+1) x tebd_dim (with padding)
            full_embd = self.type_embedding.call()
            nt, t_dim = full_embd.shape
            # Build all (i, j) type embedding pairs
            type_embedding_i = full_embd.view(nt, 1, t_dim).expand(nt, nt, t_dim)
            type_embedding_j = full_embd.view(1, nt, t_dim).expand(nt, nt, t_dim)
            two_side = torch.cat([type_embedding_i, type_embedding_j], dim=-1).reshape(
                -1, t_dim * 2
            )
            # Run through the strip embedding network
            embd_tensor = self.se_ttebd.embeddings_strip[0].call(two_side).detach()
            torch.nn.Module.register_buffer(self, "type_embd_data", embd_tensor)

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
            return DescrptSeTTebdDP.call.__wrapped__(
                self, coord_ext, atype_ext, nlist, mapping
            )
        return self._call_compressed(coord_ext, atype_ext, nlist)

    def _call_compressed(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
    ) -> Any:
        """Compressed forward using tabulate_fusion_se_t_tebd custom op."""
        # env_mat: nf x nloc x nnei x 4
        rr, _diff, sw = self.se_ttebd.env_mat.call(
            coord_ext,
            atype_ext,
            nlist,
            self.se_ttebd.mean[...],
            self.se_ttebd.stddev[...],
        )
        nf, nloc, nnei, _ = rr.shape
        ng = self.se_ttebd.neuron[-1]
        nfnl = nf * nloc

        # Exclude mask and nlist processing
        exclude_mask = self.se_ttebd.emask.build_type_exclude_mask(nlist, atype_ext)
        exclude_mask = exclude_mask.view(nfnl, nnei)
        nlist = nlist.view(nfnl, nnei)
        exclude_mask = exclude_mask.to(torch.bool)
        nlist = torch.where(exclude_mask, nlist, torch.full_like(nlist, -1))
        nlist_mask = nlist != -1
        # nfnl x nnei x 1
        sw = torch.where(
            nlist_mask[:, :, None],
            sw.view(nfnl, nnei, 1),
            torch.zeros(nfnl, nnei, 1, dtype=sw.dtype, device=sw.device),
        )

        # nfnl x nnei x 4
        rr = rr.view(nfnl, nnei, 4)
        rr = rr * exclude_mask[:, :, None].to(rr.dtype)
        # nfnl x nnei x 3
        rr_i = rr[:, :, 1:]
        # nfnl x nnei x nnei
        env_ij = torch.einsum("ijm,ikm->ijk", rr_i, rr_i)

        # Geometric embedding via tabulation
        ebd_env_ij = env_ij.view(-1, 1)
        gg_s = torch.ops.deepmd.tabulate_fusion_se_t_tebd(
            self.compress_data[0].contiguous(),
            self.compress_info[0].cpu().contiguous(),
            ebd_env_ij.contiguous(),
            env_ij.contiguous(),
            ng,
        )[0]
        # nfnl x nnei x nnei x ng
        gg_s = gg_s.view(nfnl, nnei, nnei, ng)

        # Type embedding lookup from precomputed buffer
        nlist_masked = torch.where(nlist_mask, nlist, torch.zeros_like(nlist))
        type_embedding = self.type_embedding.call()
        ntypes_with_padding = type_embedding.shape[0]
        # nf x (nloc x nnei)
        nlist_index = nlist_masked.view(nf, nloc * nnei)
        # nf x (nloc x nnei)
        nei_type = torch.gather(atype_ext, dim=1, index=nlist_index)
        # nfnl x nnei
        nei_type = nei_type.view(nfnl, nnei)
        # nfnl x nnei x nnei
        nei_type_i = nei_type.unsqueeze(2).expand(-1, -1, nnei)
        nei_type_j = nei_type.unsqueeze(1).expand(-1, nnei, -1)
        idx = (nei_type_i * ntypes_with_padding + nei_type_j).reshape(-1).to(torch.long)

        # (nfnl x nnei x nnei) x ng
        gg_t = self.type_embd_data[idx]
        # nfnl x nnei x nnei x ng
        gg_t = gg_t.view(nfnl, nnei, nnei, ng)
        if self.se_ttebd.smooth:
            gg_t = gg_t * sw.view(nfnl, nnei, 1, 1) * sw.view(nfnl, 1, nnei, 1)

        # Combine geometric and type embeddings: gg_s * (1 + gg_t)
        gg = gg_s * gg_t + gg_s

        # Contract: nfnl x ng
        res_ij = torch.einsum("ijk,ijkm->im", env_ij, gg)
        res_ij = res_ij * (1.0 / float(nnei) / float(nnei))
        # nf x nloc x ng
        result = res_ij.view(nf, nloc, ng)

        # Concat type embedding at output if needed
        nall = coord_ext.view(nf, -1).shape[1] // 3
        if self.concat_output_tebd:
            atype_embd_ext = type_embedding[atype_ext.reshape(-1)].view(
                nf, nall, self.tebd_dim
            )
            atype_embd = atype_embd_ext[:, :nloc, :]
            result = torch.cat(
                [result, atype_embd.view(nf, nloc, self.tebd_dim)], dim=-1
            )

        return result, None, None, None, sw.view(nf, nloc, nnei, 1)
