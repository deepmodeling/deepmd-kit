# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.pt.model.backbone import (
    BackBone,
)
from deepmd.pt.model.network.network import (
    Evoformer2bEncoder,
)


class Evoformer2bBackBone(BackBone):
    def __init__(
        self,
        nnei,
        layer_num=6,
        attn_head=8,
        atomic_dim=1024,
        pair_dim=100,
        feature_dim=1024,
        ffn_dim=2048,
        post_ln=False,
        final_layer_norm=True,
        final_head_layer_norm=False,
        emb_layer_norm=False,
        atomic_residual=False,
        evo_residual=False,
        residual_factor=1.0,
        activation_function="gelu",
        **kwargs,
    ):
        """Construct an evoformer backBone."""
        super().__init__()
        self.nnei = nnei
        self.layer_num = layer_num
        self.attn_head = attn_head
        self.atomic_dim = atomic_dim
        self.pair_dim = pair_dim
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // attn_head
        assert (
            feature_dim % attn_head == 0
        ), f"feature_dim {feature_dim} must be divided by attn_head {attn_head}!"
        self.ffn_dim = ffn_dim
        self.post_ln = post_ln
        self.final_layer_norm = final_layer_norm
        self.final_head_layer_norm = final_head_layer_norm
        self.emb_layer_norm = emb_layer_norm
        self.activation_function = activation_function
        self.atomic_residual = atomic_residual
        self.evo_residual = evo_residual
        self.residual_factor = float(residual_factor)
        self.encoder = Evoformer2bEncoder(
            nnei=self.nnei,
            layer_num=self.layer_num,
            attn_head=self.attn_head,
            atomic_dim=self.atomic_dim,
            pair_dim=self.pair_dim,
            feature_dim=self.feature_dim,
            ffn_dim=self.ffn_dim,
            post_ln=self.post_ln,
            final_layer_norm=self.final_layer_norm,
            final_head_layer_norm=self.final_head_layer_norm,
            emb_layer_norm=self.emb_layer_norm,
            atomic_residual=self.atomic_residual,
            evo_residual=self.evo_residual,
            residual_factor=self.residual_factor,
            activation_function=self.activation_function,
        )

    def forward(self, atomic_rep, pair_rep, nlist, nlist_type, nlist_mask):
        """Encoder the atomic and pair representations.

        Args:
        - atomic_rep: Atomic representation with shape [nframes, nloc, atomic_dim].
        - pair_rep: Pair representation with shape [nframes, nloc, nnei, pair_dim].
        - nlist: Neighbor list with shape [nframes, nloc, nnei].
        - nlist_type: Neighbor types with shape [nframes, nloc, nnei].
        - nlist_mask: Neighbor mask with shape [nframes, nloc, nnei], `False` if blank.

        Returns
        -------
        - atomic_rep: Atomic representation after encoder with shape [nframes, nloc, feature_dim].
        - transformed_atomic_rep: Transformed atomic representation after encoder with shape [nframes, nloc, atomic_dim].
        - pair_rep: Pair representation after encoder with shape [nframes, nloc, nnei, attn_head].
        - delta_pair_rep: Delta pair representation after encoder with shape [nframes, nloc, nnei, attn_head].
        - norm_x: Normalization loss of atomic_rep.
        - norm_delta_pair_rep: Normalization loss of delta_pair_rep.
        """
        (
            atomic_rep,
            transformed_atomic_rep,
            pair_rep,
            delta_pair_rep,
            norm_x,
            norm_delta_pair_rep,
        ) = self.encoder(atomic_rep, pair_rep, nlist, nlist_type, nlist_mask)
        return (
            atomic_rep,
            transformed_atomic_rep,
            pair_rep,
            delta_pair_rep,
            norm_x,
            norm_delta_pair_rep,
        )
