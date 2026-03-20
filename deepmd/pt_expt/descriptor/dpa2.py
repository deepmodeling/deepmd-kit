# SPDX-License-Identifier: LGPL-3.0-or-later
import warnings
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.common import (
    cast_precision,
)
from deepmd.dpmodel.descriptor.dpa2 import DescrptDPA2 as DescrptDPA2DP
from deepmd.dpmodel.descriptor.dpa2 import (
    build_multiple_neighbor_list,
    get_multiple_nlist_key,
)
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.utils.update_sel import (
    UpdateSel,
)


@BaseDescriptor.register("dpa2")
@torch_module
class DescrptDPA2(DescrptDPA2DP):
    _update_sel_cls = UpdateSel

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Enable compression for the DPA2 descriptor.

        Compression applies to the repinit block (DescrptBlockSeAtten).
        When attn_layer == 0, the geometric embedding is tabulated.
        Type embedding outputs are always precomputed.

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
        if self.repinit.resnet_dt:
            raise RuntimeError(
                "Model compression error: repinit resnet_dt must be false!"
            )
        for tt in self.repinit.exclude_types:
            if (tt[0] not in range(self.repinit.ntypes)) or (
                tt[1] not in range(self.repinit.ntypes)
            ):
                raise RuntimeError(
                    "Repinit exclude types"
                    + str(tt)
                    + " must within the number of atomic types "
                    + str(self.repinit.ntypes)
                    + "!"
                )
        if (
            self.repinit.ntypes * self.repinit.ntypes - len(self.repinit.exclude_types)
            == 0
        ):
            raise RuntimeError(
                "Repinit empty embedding-nets are not supported in model compression!"
            )

        if self.repinit.tebd_input_mode != "strip":
            raise RuntimeError(
                "Cannot compress model when repinit tebd_input_mode != 'strip'"
            )

        # Precompute type embedding data for repinit
        self._store_type_embd_data()

        if self.repinit.attn_layer == 0:
            # Build geometric embedding table
            repinit_data = self.repinit.serialize()
            self.table = DPTabulate(
                self.repinit,
                repinit_data["neuron"],
                repinit_data.get("type_one_side", False),
                repinit_data.get("exclude_types", []),
                ActivationFn(repinit_data["activation_function"]),
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
            self.geo_compress = True
        else:
            warnings.warn(
                "Attention layer is not 0, only type embedding is compressed. "
                "Geometric part is not compressed.",
                UserWarning,
                stacklevel=2,
            )
            self.geo_compress = False

        self.compress = True

    def _store_compress_data(self) -> None:
        """Store tabulated data as buffers for the compressed geometric embedding."""
        table_data = self.table.data
        table_config = self.table_config
        lower = self.lower
        upper = self.upper
        prec = self.repinit.mean.dtype

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
        """Precompute type embedding outputs for repinit and store as a buffer."""
        with torch.no_grad():
            # type_embedding.call() returns (ntypes+1) x tebd_dim (with padding)
            full_embd = self.type_embedding.call()
            nt, t_dim = full_embd.shape

            if self.repinit.type_one_side:
                # One-side: only neighbor types
                # (ntypes+1) x tebd_dim -> (ntypes+1) x ng
                embd_tensor = self.repinit.embeddings_strip[0].call(full_embd).detach()
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
                    self.repinit.embeddings_strip[0].call(two_side_embd).detach()
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
            return DescrptDPA2DP.call.__wrapped__(
                self, coord_ext, atype_ext, nlist, mapping
            )
        return self._call_compressed(coord_ext, atype_ext, nlist, mapping)

    def _call_compressed(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
    ) -> Any:
        """Compressed forward for DPA2 descriptor.

        The repinit forward is done inline with compressed ops,
        then the rest (g1_shape_transform, repformers, etc.) proceeds normally.
        """
        use_three_body = self.use_three_body
        nframes, nloc, _nnei = nlist.shape
        nall = coord_ext.view(nframes, -1).shape[1] // 3

        # Build multiple neighbor lists
        nlist_dict = build_multiple_neighbor_list(
            coord_ext,
            nlist,
            self.rcut_list,
            self.nsel_list,
        )

        # Type embedding
        type_embedding = self.type_embedding.call()
        g1_ext = type_embedding[atype_ext.reshape(-1).to(torch.long)].reshape(
            nframes, nall, self.tebd_dim
        )
        g1_inp = g1_ext[:, :nloc, :]

        # Compressed repinit forward
        nlist_repinit = nlist_dict[
            get_multiple_nlist_key(self.repinit.get_rcut(), self.repinit.get_nsel())
        ]
        g1 = self._compressed_repinit_forward(
            coord_ext, atype_ext, nlist_repinit, nframes, nloc, nall, type_embedding
        )

        # Three-body (not compressed, call normally)
        if use_three_body:
            assert self.repinit_three_body is not None
            g1_three_body, __, __, __, __ = self.repinit_three_body(
                nlist_dict[
                    get_multiple_nlist_key(
                        self.repinit_three_body.get_rcut(),
                        self.repinit_three_body.get_nsel(),
                    )
                ],
                coord_ext,
                atype_ext,
                g1_ext,
                mapping,
                type_embedding=type_embedding,
            )
            g1 = torch.cat([g1, g1_three_body], dim=-1)

        # Linear to change shape
        g1 = self.g1_shape_tranform(g1)
        if self.add_tebd_to_repinit_out:
            assert self.tebd_transform is not None
            g1 = g1 + self.tebd_transform(g1_inp)

        # Mapping g1 to extended region for repformers
        assert mapping is not None
        mapping_ext = mapping.view(nframes, nall, 1).expand(-1, -1, g1.shape[-1])
        g1_ext = torch.gather(g1, 1, mapping_ext)

        # Repformers (not compressed)
        g1, g2, h2, rot_mat, sw = self.repformers(
            nlist_dict[
                get_multiple_nlist_key(
                    self.repformers.get_rcut(), self.repformers.get_nsel()
                )
            ],
            coord_ext,
            atype_ext,
            g1_ext,
            mapping,
        )

        # Concat type embedding at output if needed
        if self.concat_output_tebd:
            g1 = torch.cat([g1, g1_inp], dim=-1)

        return g1, rot_mat, g2, h2, sw

    def _compressed_repinit_forward(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
        nframes: int,
        nloc: int,
        nall: int,
        type_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compressed forward for the repinit block.

        Same logic as DPA1's _call_compressed but only produces the g1 output
        (nf x nloc x (ng x axis_neuron)), without rot_mat/sw returns.

        Parameters
        ----------
        coord_ext
            Extended coordinates. shape: nf x (nall x 3)
        atype_ext
            Extended atom types. shape: nf x nall
        nlist
            Neighbor list for repinit. shape: nf x nloc x nnei_repinit
        nframes
            Number of frames.
        nloc
            Number of local atoms.
        nall
            Number of all atoms (local + ghost).
        type_embedding
            Full type embedding. shape: (ntypes+1) x tebd_dim

        Returns
        -------
        torch.Tensor
            Repinit output. shape: nf x nloc x (ng x axis_neuron)
        """
        # env_mat: nf x nloc x nnei x 4
        rr, _diff, sw = self.repinit.env_mat.call(
            coord_ext,
            atype_ext,
            nlist,
            self.repinit.mean[...],
            self.repinit.stddev[...],
        )
        nf, nloc_r, nnei, _ = rr.shape
        ng = self.repinit.neuron[-1]
        nfnl = nf * nloc_r

        # Exclude mask and nlist processing
        exclude_mask = self.repinit.emask.build_type_exclude_mask(nlist, atype_ext)
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
        ntypes_with_padding = type_embedding.shape[0]
        # nf x (nloc x nnei)
        nlist_index = nlist_masked.view(nf, nloc_r * nnei)
        # nf x (nloc x nnei)
        nei_type = torch.gather(atype_ext, dim=1, index=nlist_index)

        if self.repinit.type_one_side:
            # (nf*nl*nnei,) -> (nf*nl*nnei, ng)
            gg_t = self.type_embd_data[nei_type.view(-1).to(torch.long)]
        else:
            atype = atype_ext[:, :nloc_r]
            idx_i = torch.tile(
                atype.reshape(-1, 1) * ntypes_with_padding, [1, nnei]
            ).view(-1)
            idx_j = nei_type.view(-1)
            idx = (idx_i + idx_j).to(torch.long)
            # (nf x nl x nnei) x ng
            gg_t = self.type_embd_data[idx]

        # (nf x nl) x nnei x ng
        gg_t = gg_t.view(nfnl, nnei, ng)
        if self.repinit.smooth:
            gg_t = gg_t * sw.view(nfnl, self.repinit.nnei, 1)

        if self.geo_compress:
            # Flatten for tabulate op
            ss_flat = ss.reshape(-1, 1)
            gg_t_flat = gg_t.reshape(-1, gg_t.size(-1))
            is_sorted = len(self.repinit.exclude_types) == 0
            xyz_scatter = torch.ops.deepmd.tabulate_fusion_se_atten(
                self.compress_data[0].contiguous(),
                self.compress_info[0].cpu().contiguous(),
                ss_flat.contiguous(),
                rr.contiguous(),
                gg_t_flat.contiguous(),
                self.repinit.neuron[-1],
                is_sorted,
            )[0]
        else:
            # No geometric compression, run embedding net + attention
            # nfnl x nnei x ng
            gg_s = self.repinit.embeddings[0].call(ss)
            # nfnl x nnei x ng
            gg = gg_s * gg_t + gg_s
            input_r = torch.nn.functional.normalize(
                rr.view(-1, self.repinit.nnei, 4)[:, :, 1:4], dim=-1
            )
            gg = self.repinit.dpa1_attention(gg, nlist_mask, input_r=input_r, sw=sw)
            # nfnl x 4 x ng
            xyz_scatter = torch.matmul(rr.permute(0, 2, 1), gg)

        xyz_scatter = xyz_scatter / self.repinit.nnei
        # nfnl x ng x 4
        xyz_scatter_1 = xyz_scatter.permute(0, 2, 1)
        # nfnl x 4 x axis_neuron
        xyz_scatter_2 = xyz_scatter[:, :, 0 : self.repinit.axis_neuron]
        # nfnl x ng x axis_neuron
        result = torch.matmul(xyz_scatter_1, xyz_scatter_2)
        # nf x nloc x (ng x axis_neuron)
        result = result.view(nf, nloc_r, ng * self.repinit.axis_neuron)

        return result
