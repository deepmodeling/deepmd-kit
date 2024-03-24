# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)
from deepmd.pt.model.network.network import (
    MaskLMHead,
    NonLinearHead,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
)
from deepmd.pt.utils import (
    env,
)


@fitting_check_output
class DenoiseNet(Fitting):
    def __init__(
        self,
        feature_dim,
        ntypes,
        attn_head=8,
        prefactor=[0.5, 0.5],
        activation_function="gelu",
        **kwargs,
    ):
        """Construct a denoise net.

        Args:
        - ntypes: Element count.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neurons in each hidden layers of the fitting net.
        - bias_atom_e: Average enery per atom for each element.
        - resnet_dt: Using time-step in the ResNet construction.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.ntypes = ntypes
        self.attn_head = attn_head
        self.prefactor = torch.tensor(
            prefactor, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
        )

        self.lm_head = MaskLMHead(
            embed_dim=self.feature_dim,
            output_dim=ntypes,
            activation_fn=activation_function,
            weight=None,
        )

        if not isinstance(self.attn_head, list):
            self.pair2coord_proj = NonLinearHead(
                self.attn_head, 1, activation_fn=activation_function
            )
        else:
            self.pair2coord_proj = []
            self.ndescriptor = len(self.attn_head)
            for ii in range(self.ndescriptor):
                _pair2coord_proj = NonLinearHead(
                    self.attn_head[ii], 1, activation_fn=activation_function
                )
                self.pair2coord_proj.append(_pair2coord_proj)
            self.pair2coord_proj = torch.nn.ModuleList(self.pair2coord_proj)

    def output_def(self):
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "updated_coord",
                    [3],
                    reduciable=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
                OutputVariableDef(
                    "logits",
                    [-1],
                    reduciable=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    def forward(
        self,
        pair_weights,
        diff,
        nlist_mask,
        features,
        sw,
        masked_tokens: Optional[torch.Tensor] = None,
    ):
        """Calculate the updated coord.
        Args:
        - coord: Input noisy coord with shape [nframes, nloc, 3].
        - pair_weights: Input pair weights with shape [nframes, nloc, nnei, head].
        - diff: Input pair relative coord list with shape [nframes, nloc, nnei, 3].
        - nlist_mask: Input nlist mask with shape [nframes, nloc, nnei].

        Returns
        -------
        - denoised_coord: Denoised updated coord with shape [nframes, nloc, 3].
        """
        # [nframes, nloc, nnei, 1]
        logits = self.lm_head(features, masked_tokens=masked_tokens)
        if not isinstance(self.attn_head, list):
            attn_probs = self.pair2coord_proj(pair_weights)
            out_coord = (attn_probs * diff).sum(dim=-2) / (
                sw.sum(dim=-1).unsqueeze(-1) + 1e-6
            )
        else:
            assert len(self.prefactor) == self.ndescriptor
            all_coord_update = []
            assert len(pair_weights) == len(diff) == len(nlist_mask) == self.ndescriptor
            for ii in range(self.ndescriptor):
                _attn_probs = self.pair2coord_proj[ii](pair_weights[ii])
                _coord_update = (_attn_probs * diff[ii]).sum(dim=-2) / (
                    nlist_mask[ii].sum(dim=-1).unsqueeze(-1) + 1e-6
                )
                all_coord_update.append(_coord_update)
            out_coord = self.prefactor[0] * all_coord_update[0]
            for ii in range(self.ndescriptor - 1):
                out_coord += self.prefactor[ii + 1] * all_coord_update[ii + 1]
        return {
            "updated_coord": out_coord,
            "logits": logits,
        }
