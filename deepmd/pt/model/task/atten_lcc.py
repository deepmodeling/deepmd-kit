# SPDX-License-Identifier: LGPL-3.0-or-later
import torch
import torch.nn as nn

from deepmd.pt.model.network.network import (
    EnergyHead,
    NodeTaskHead,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
)
from deepmd.pt.utils import (
    env,
)


class FittingNetAttenLcc(Fitting):
    def __init__(
        self, embedding_width, bias_atom_e, pair_embed_dim, attention_heads, **kwargs
    ):
        super().__init__()
        self.embedding_width = embedding_width
        self.engergy_proj = EnergyHead(self.embedding_width, 1)
        self.energe_agg_factor = nn.Embedding(4, 1, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
        nn.init.normal_(self.energe_agg_factor.weight, 0, 0.01)
        bias_atom_e = torch.tensor(bias_atom_e)
        self.register_buffer("bias_atom_e", bias_atom_e)
        self.pair_embed_dim = pair_embed_dim
        self.attention_heads = attention_heads
        self.node_proc = NodeTaskHead(
            self.embedding_width, self.pair_embed_dim, self.attention_heads
        )
        self.node_proc.zero_init()

    def forward(self, output, pair, delta_pos, atype, nframes, nloc):
        # [nframes x nloc x tebd_dim]
        output_nloc = (output[:, 0, :]).reshape(nframes, nloc, self.embedding_width)
        # Optional: GRRG or mean of gbf TODO

        # energy outut
        # [nframes, nloc]
        energy_out = self.engergy_proj(output_nloc).view(nframes, nloc)
        # [nframes, nloc]
        energy_factor = self.energe_agg_factor(torch.zeros_like(atype)).view(
            nframes, nloc
        )
        energy_out = (energy_out * energy_factor) + self.bias_atom_e[atype]
        energy_out = energy_out.sum(dim=-1)

        # vector output
        # predict_force: [(nframes x nloc) x (1 + nnei2) x 3]
        predict_force = self.node_proc(output, pair, delta_pos=delta_pos)
        # predict_force_nloc: [nframes x nloc x 3]
        predict_force_nloc = (predict_force[:, 0, :]).reshape(nframes, nloc, 3)
        return energy_out, predict_force_nloc
