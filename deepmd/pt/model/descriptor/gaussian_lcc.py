# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
)

import torch
import torch.nn as nn

from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt.model.network.network import (
    Evoformer3bEncoder,
    GaussianEmbedding,
    TypeEmbedNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.utils.path import (
    DPPath,
)


class DescrptGaussianLcc(torch.nn.Module, BaseDescriptor):
    def __init__(
        self,
        rcut,
        rcut_smth,
        sel: int,
        ntypes: int,
        num_pair: int,
        embed_dim: int = 768,
        kernel_num: int = 128,
        pair_embed_dim: int = 64,
        num_block: int = 1,
        layer_num: int = 12,
        attn_head: int = 48,
        pair_hidden_dim: int = 16,
        ffn_embedding_dim: int = 768,
        dropout: float = 0.0,
        droppath_prob: float = 0.1,
        pair_dropout: float = 0.25,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        pre_ln: bool = True,
        do_tag_embedding: bool = False,
        tag_ener_pref: bool = False,
        atomic_sum_gbf: bool = False,
        pre_add_seq: bool = True,
        tri_update: bool = True,
        **kwargs,
    ):
        """Construct a descriptor of Gaussian Based Local Cluster.

        Args:
        - rcut: Cut-off radius.
        - rcut_smth: Smooth hyper-parameter for pair force & energy. **Not used in this descriptor**.
        - sel: For each element type, how many atoms is selected as neighbors.
        - ntypes: Number of atom types.
        - num_pair: Number of atom type pairs. Default is 2 * ntypes.
        - kernel_num: Number of gaussian kernels.
        - embed_dim: Dimension of atomic representation.
        - pair_embed_dim: Dimension of pair representation.
        - num_block: Number of evoformer blocks.
        - layer_num: Number of attention layers.
        - attn_head: Number of attention heads.
        - pair_hidden_dim: Hidden dimension of pair representation during attention process.
        - ffn_embedding_dim: Dimension during feed forward network.
        - dropout: Dropout probability of atomic representation.
        - droppath_prob: If not zero, it will use drop paths (Stochastic Depth) per sample and ignore `dropout`.
        - pair_dropout: Dropout probability of pair representation during triangular update.
        - attention_dropout: Dropout probability during attetion process.
        - activation_dropout: Dropout probability of pair feed forward network.
        - pre_ln: Do previous layer norm or not.
        - do_tag_embedding: Add tag embedding to atomic and pair representations. (`tags`, `tags2`, `tags3` must exist)
        - atomic_sum_gbf: Add sum of gaussian outputs to atomic representation or not.
        - pre_add_seq: Add output of other descriptor (if has) to the atomic representation before attention.
        """
        super().__init__()
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.embed_dim = embed_dim
        self.num_pair = num_pair
        self.kernel_num = kernel_num
        self.pair_embed_dim = pair_embed_dim
        self.num_block = num_block
        self.layer_num = layer_num
        self.attention_heads = attn_head
        self.pair_hidden_dim = pair_hidden_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.dropout = dropout
        self.droppath_prob = droppath_prob
        self.pair_dropout = pair_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.pre_ln = pre_ln
        self.do_tag_embedding = do_tag_embedding
        self.tag_ener_pref = tag_ener_pref
        self.atomic_sum_gbf = atomic_sum_gbf
        self.local_cluster = True
        self.pre_add_seq = pre_add_seq
        self.tri_update = tri_update

        if isinstance(sel, int):
            sel = [sel]

        self.ntypes = ntypes
        self.sec = torch.tensor(sel)
        self.nnei = sum(sel)

        if self.do_tag_embedding:
            self.tag_encoder = nn.Embedding(3, self.embed_dim)
            self.tag_encoder2 = nn.Embedding(2, self.embed_dim)
            self.tag_type_embedding = TypeEmbedNet(10, pair_embed_dim)
        self.edge_type_embedding = nn.Embedding(
            (ntypes + 1) * (ntypes + 1),
            pair_embed_dim,
            padding_idx=(ntypes + 1) * (ntypes + 1) - 1,
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
        )
        self.gaussian_encoder = GaussianEmbedding(
            rcut,
            kernel_num,
            num_pair,
            embed_dim,
            pair_embed_dim,
            sel,
            ntypes,
            atomic_sum_gbf,
        )
        self.backbone = Evoformer3bEncoder(
            self.nnei,
            layer_num=self.layer_num,
            attn_head=self.attention_heads,
            atomic_dim=self.embed_dim,
            pair_dim=self.pair_embed_dim,
            pair_hidden_dim=self.pair_hidden_dim,
            ffn_embedding_dim=self.ffn_embedding_dim,
            dropout=self.dropout,
            droppath_prob=self.droppath_prob,
            pair_dropout=self.pair_dropout,
            attention_dropout=self.attention_dropout,
            activation_dropout=self.activation_dropout,
            pre_ln=self.pre_ln,
            tri_update=self.tri_update,
        )

    @property
    def dim_out(self):
        """Returns the output dimension of atomic representation."""
        return self.embed_dim

    @property
    def dim_in(self):
        """Returns the atomic input dimension of this descriptor."""
        return self.embed_dim

    @property
    def dim_emb(self):
        """Returns the output dimension of pair representation."""
        return self.pair_embed_dim

    def compute_input_stats(self, merged: List[dict], path: Optional[DPPath] = None):
        """Update mean and stddev for descriptor elements."""
        pass

    def forward(
        self,
        extended_coord,
        nlist,
        atype,
        nlist_type,
        nlist_loc=None,
        atype_tebd=None,
        nlist_tebd=None,
        seq_input=None,
    ):
        """Calculate the atomic and pair representations of this descriptor.

        Args:
        - extended_coord: Copied atom coordinates with shape [nframes, nall, 3].
        - nlist: Neighbor list with shape [nframes, nloc, nnei].
        - atype: Atom type with shape [nframes, nloc].
        - nlist_type: Atom type of neighbors with shape [nframes, nloc, nnei].
        - nlist_loc: Local index of neighbor list with shape [nframes, nloc, nnei].
        - atype_tebd: Atomic type embedding with shape [nframes, nloc, tebd_dim].
        - nlist_tebd: Type embeddings of neighbor with shape [nframes, nloc, nnei, tebd_dim].
        - seq_input: The sequential input from other descriptor with
                    shape [nframes, nloc, tebd_dim] or [nframes * nloc, 1 + nnei, tebd_dim]

        Returns
        -------
        - result: descriptor with shape [nframes, nloc, self.filter_neuron[-1] * self.axis_neuron].
        - ret: environment matrix with shape [nframes, nloc, self.neei, out_size]
        """
        nframes, nloc = nlist.shape[:2]
        nall = extended_coord.shape[1]
        nlist2 = torch.cat(
            [
                torch.arange(0, nloc, device=nlist.device)
                .reshape(1, nloc, 1)
                .expand(nframes, -1, -1),
                nlist,
            ],
            dim=-1,
        )
        nlist_loc2 = torch.cat(
            [
                torch.arange(0, nloc, device=nlist_loc.device)
                .reshape(1, nloc, 1)
                .expand(nframes, -1, -1),
                nlist_loc,
            ],
            dim=-1,
        )
        nlist_type2 = torch.cat([atype.reshape(nframes, nloc, 1), nlist_type], dim=-1)
        nnei2_mask = nlist2 != -1
        padding_mask = nlist2 == -1
        nlist2 = nlist2 * nnei2_mask
        nlist_loc2 = nlist_loc2 * nnei2_mask

        # nframes x nloc x (1 + nnei2) x (1 + nnei2)
        pair_mask = nnei2_mask.unsqueeze(-1) * nnei2_mask.unsqueeze(-2)
        # nframes x nloc x (1 + nnei2) x (1 + nnei2) x head
        attn_mask = torch.zeros(
            [nframes, nloc, 1 + self.nnei, 1 + self.nnei, self.attention_heads],
            device=nlist.device,
            dtype=extended_coord.dtype,
        )
        attn_mask.masked_fill_(padding_mask.unsqueeze(2).unsqueeze(-1), float("-inf"))
        # (nframes x nloc) x head x (1 + nnei2) x (1 + nnei2)
        attn_mask = (
            attn_mask.reshape(
                nframes * nloc, 1 + self.nnei, 1 + self.nnei, self.attention_heads
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # Atomic feature
        # [(nframes x nloc) x (1 + nnei2) x tebd_dim]
        atom_feature = torch.gather(
            atype_tebd,
            dim=1,
            index=nlist_loc2.reshape(nframes, -1)
            .unsqueeze(-1)
            .expand(-1, -1, self.embed_dim),
        ).reshape(nframes * nloc, 1 + self.nnei, self.embed_dim)
        if self.pre_add_seq and seq_input is not None:
            first_dim = seq_input.shape[0]
            if first_dim == nframes * nloc:
                atom_feature += seq_input
            elif first_dim == nframes:
                atom_feature_seq = torch.gather(
                    seq_input,
                    dim=1,
                    index=nlist_loc2.reshape(nframes, -1)
                    .unsqueeze(-1)
                    .expand(-1, -1, self.embed_dim),
                ).reshape(nframes * nloc, 1 + self.nnei, self.embed_dim)
                atom_feature += atom_feature_seq
            else:
                raise RuntimeError
        atom_feature = atom_feature * nnei2_mask.reshape(
            nframes * nloc, 1 + self.nnei, 1
        )

        # Pair feature
        # [(nframes x nloc) x (1 + nnei2)]
        nlist_type2_reshape = nlist_type2.reshape(nframes * nloc, 1 + self.nnei)
        # [(nframes x nloc) x (1 + nnei2) x (1 + nnei2)]
        edge_type = nlist_type2_reshape.unsqueeze(-1) * (
            self.ntypes + 1
        ) + nlist_type2_reshape.unsqueeze(-2)
        # [(nframes x nloc) x (1 + nnei2) x (1 + nnei2) x pair_dim]
        edge_feature = self.edge_type_embedding(edge_type)

        # [(nframes x nloc) x (1 + nnei2) x (1 + nnei2) x 2]
        edge_type_2dim = torch.cat(
            [
                nlist_type2_reshape.view(nframes * nloc, 1 + self.nnei, 1, 1).expand(
                    -1, -1, 1 + self.nnei, -1
                ),
                nlist_type2_reshape.view(nframes * nloc, 1, 1 + self.nnei, 1).expand(
                    -1, 1 + self.nnei, -1, -1
                )
                + self.ntypes,
            ],
            dim=-1,
        )
        # [(nframes x nloc) x (1 + nnei2) x 3]
        coord_selected = torch.gather(
            extended_coord.unsqueeze(1)
            .expand(-1, nloc, -1, -1)
            .reshape(nframes * nloc, nall, 3),
            dim=1,
            index=nlist2.reshape(nframes * nloc, 1 + self.nnei, 1).expand(-1, -1, 3),
        )

        # Update pair features (or and atomic features) with gbf features
        # delta_pos: [(nframes x nloc) x (1 + nnei2) x (1 + nnei2) x 3].
        atomic_feature, pair_feature, delta_pos = self.gaussian_encoder(
            coord_selected, atom_feature, edge_type_2dim, edge_feature
        )
        # [(nframes x nloc) x (1 + nnei2) x (1 + nnei2) x pair_dim]
        attn_bias = pair_feature

        # output: [(nframes x nloc) x (1 + nnei2) x tebd_dim]
        # pair: [(nframes x nloc) x (1 + nnei2) x (1 + nnei2) x pair_dim]
        output, pair = self.backbone(
            atomic_feature,
            pair=attn_bias,
            attn_mask=attn_mask,
            pair_mask=pair_mask,
            atom_mask=nnei2_mask.reshape(nframes * nloc, 1 + self.nnei),
        )

        return output, pair, delta_pos, None
