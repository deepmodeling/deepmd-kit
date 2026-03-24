# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.common import (
    cast_precision,
)
from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1 as DescrptDPA1DP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.utils.update_sel import (
    UpdateSel,
)


@BaseDescriptor.register("se_atten")
@BaseDescriptor.register("dpa1")
@torch_module
class DescrptDPA1(DescrptDPA1DP):
    _update_sel_cls = UpdateSel

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Enable compression for the DPA1 descriptor.

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
        if self.se_atten.tebd_input_mode != "strip":
            raise RuntimeError("Type embedding compression only works in strip mode")
        if self.se_atten.resnet_dt:
            raise RuntimeError(
                "Model compression error: descriptor resnet_dt must be false!"
            )

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

        # Precompute type embedding data
        self._store_type_embd_data()

        if self.se_atten.attn_layer == 0:
            # Build geometric embedding table
            self.lower, self.upper = self.table.build(
                min_nbor_dist, table_extrapolate, table_stride_1, table_stride_2
            )
            self._store_compress_data()
            self.geo_compress = True
        else:
            self.geo_compress = False

        self.compress = True

    def _store_compress_data(self) -> None:
        """Store tabulated data as buffers for the compressed geometric embedding."""
        table_data = self.table.data
        table_config = self.table_config
        lower = self.lower
        upper = self.upper
        prec = self.se_atten.mean.dtype

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
        """Precompute type embedding outputs and store as a buffer."""
        with torch.no_grad():
            # type_embedding.call() returns (ntypes+1) x tebd_dim (with padding)
            full_embd = self.type_embedding.call()
            nt, t_dim = full_embd.shape

            if self.se_atten.type_one_side:
                # One-side: only neighbor types
                # (ntypes+1) x tebd_dim -> (ntypes+1) x ng
                embd_tensor = self.se_atten.embeddings_strip[0].call(full_embd).detach()
            else:
                # Two-side: all (ntypes+1)^2 type pair combinations
                # Build [neighbor, center] combinations
                embd_nei = full_embd.view(1, nt, t_dim).expand(nt, nt, t_dim)
                embd_center = full_embd.view(nt, 1, t_dim).expand(nt, nt, t_dim)
                two_side_embd = torch.cat([embd_nei, embd_center], dim=-1).reshape(
                    -1, t_dim * 2
                )
                # ((ntypes+1)^2) x ng
                embd_tensor = (
                    self.se_atten.embeddings_strip[0].call(two_side_embd).detach()
                )

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
            return DescrptDPA1DP.call.__wrapped__(
                self, coord_ext, atype_ext, nlist, mapping
            )
        return self._call_compressed(coord_ext, atype_ext, nlist)

    def _call_compressed(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
    ) -> Any:
        """Compressed forward for DPA1 descriptor."""
        # env_mat: nf x nloc x nnei x 4
        rr, _diff, sw = self.se_atten.env_mat.call(
            coord_ext,
            atype_ext,
            nlist,
            self.se_atten.mean[...],
            self.se_atten.stddev[...],
        )
        nf, nloc, nnei, _ = rr.shape
        ng = self.se_atten.neuron[-1]
        nfnl = nf * nloc

        # Exclude mask and nlist processing
        exclude_mask = self.se_atten.emask.build_type_exclude_mask(nlist, atype_ext)
        exclude_mask = exclude_mask.view(nfnl, nnei)
        nlist = nlist.view(nfnl, nnei)
        exclude_mask = exclude_mask.to(torch.bool)
        nlist = torch.where(exclude_mask, nlist, torch.full_like(nlist, -1))
        nlist_mask = nlist != -1
        nlist_masked = torch.where(nlist_mask, nlist, torch.zeros_like(nlist))
        # nfnl x nnei x 1
        sw = torch.where(
            nlist_mask[:, :, None],
            sw.view(nfnl, nnei, 1),
            torch.zeros(nfnl, nnei, 1, dtype=sw.dtype, device=sw.device),
        )

        # nfnl x nnei x 4
        rr = rr.view(nfnl, nnei, 4)
        rr = rr * exclude_mask[:, :, None].to(rr.dtype)
        # nfnl x nnei x 1
        ss = rr[:, :, :1]

        # Type embedding lookup from precomputed buffer
        type_embedding = self.type_embedding.call()
        ntypes_with_padding = type_embedding.shape[0]
        # nf x (nloc x nnei)
        nlist_index = nlist_masked.view(nf, nloc * nnei)
        # nf x (nloc x nnei)
        nei_type = torch.gather(atype_ext, dim=1, index=nlist_index)

        if self.se_atten.type_one_side:
            # (nf*nl*nnei,) -> (nf*nl*nnei, ng)
            gg_t = self.type_embd_data[nei_type.view(-1).to(torch.long)]
        else:
            atype = atype_ext[:, :nloc]
            idx_i = torch.tile(
                atype.reshape(-1, 1) * ntypes_with_padding, [1, nnei]
            ).view(-1)
            idx_j = nei_type.view(-1)
            idx = (idx_i + idx_j).to(torch.long)
            # (nf x nl x nnei) x ng
            gg_t = self.type_embd_data[idx]

        # (nf x nl) x nnei x ng
        gg_t = gg_t.view(nfnl, nnei, ng)
        if self.se_atten.smooth:
            gg_t = gg_t * sw.view(nfnl, self.se_atten.nnei, 1)

        if self.geo_compress:
            # Flatten for tabulate op
            ss_flat = ss.reshape(-1, 1)
            gg_t_flat = gg_t.reshape(-1, gg_t.size(-1))
            is_sorted = len(self.se_atten.exclude_types) == 0
            xyz_scatter = torch.ops.deepmd.tabulate_fusion_se_atten(
                self.compress_data[0].contiguous(),
                self.compress_info[0].cpu().contiguous(),
                ss_flat.contiguous(),
                rr.contiguous(),
                gg_t_flat.contiguous(),
                self.se_atten.neuron[-1],
                is_sorted,
            )[0]
        else:
            # No geometric compression, run embedding net + attention
            # nfnl x nnei x ng
            gg_s = self.se_atten.embeddings[0].call(ss)
            # nfnl x nnei x ng
            gg = gg_s * gg_t + gg_s
            input_r = torch.nn.functional.normalize(
                rr.view(-1, self.se_atten.nnei, 4)[:, :, 1:4], dim=-1
            )
            gg = self.se_atten.dpa1_attention(gg, nlist_mask, input_r=input_r, sw=sw)
            # nfnl x 4 x ng
            xyz_scatter = torch.matmul(rr.permute(0, 2, 1), gg)

        xyz_scatter = xyz_scatter / self.se_atten.nnei
        # nfnl x ng x 4
        xyz_scatter_1 = xyz_scatter.permute(0, 2, 1)
        # nfnl x ng x 3
        rot_mat = xyz_scatter_1[:, :, 1:4]
        # nfnl x 4 x axis_neuron
        xyz_scatter_2 = xyz_scatter[:, :, 0 : self.se_atten.axis_neuron]
        # nfnl x ng x axis_neuron
        result = torch.matmul(xyz_scatter_1, xyz_scatter_2)
        # nf x nloc x (ng x axis_neuron)
        result = result.view(nf, nloc, ng * self.se_atten.axis_neuron)
        # nf x nloc x ng x 3
        rot_mat = rot_mat.view(nf, nloc, ng, 3)

        # Concat type embedding at output if needed
        if self.concat_output_tebd:
            nall = coord_ext.view(nf, -1).shape[1] // 3
            atype_embd_ext = type_embedding[atype_ext.reshape(-1)].view(
                nf, nall, self.tebd_dim
            )
            atype_embd = atype_embd_ext[:, :nloc, :]
            result = torch.cat(
                [result, atype_embd.view(nf, nloc, self.tebd_dim)], dim=-1
            )

        return result, rot_mat, None, None, sw.view(nf, nloc, nnei, 1)
