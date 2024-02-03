# SPDX-License-Identifier: LGPL-3.0-or-later
import torch
import torch.nn.functional as F

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.utils import (
    env,
)


class DenoiseLoss(TaskLoss):
    def __init__(
        self,
        ntypes,
        masked_token_loss=1.0,
        masked_coord_loss=1.0,
        norm_loss=0.01,
        use_l1=True,
        beta=1.00,
        mask_loss_coord=True,
        mask_loss_token=True,
        **kwargs,
    ):
        """Construct a layer to compute loss on coord, and type reconstruction."""
        super().__init__()
        self.ntypes = ntypes
        self.masked_token_loss = masked_token_loss
        self.masked_coord_loss = masked_coord_loss
        self.norm_loss = norm_loss
        self.has_coord = self.masked_coord_loss > 0.0
        self.has_token = self.masked_token_loss > 0.0
        self.has_norm = self.norm_loss > 0.0
        self.use_l1 = use_l1
        self.beta = beta
        self.frac_beta = 1.00 / self.beta
        self.mask_loss_coord = mask_loss_coord
        self.mask_loss_token = mask_loss_token

    def forward(self, model_pred, label, natoms, learning_rate, mae=False):
        """Return loss on coord and type denoise.

        Returns
        -------
        - loss: Loss to minimize.
        """
        updated_coord = model_pred["updated_coord"]
        logits = model_pred["logits"]
        clean_coord = label["clean_coord"]
        clean_type = label["clean_type"]
        coord_mask = label["coord_mask"]
        type_mask = label["type_mask"]

        loss = torch.tensor(0.0, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
        more_loss = {}
        if self.has_coord:
            if self.mask_loss_coord:
                masked_updated_coord = updated_coord[coord_mask]
                masked_clean_coord = clean_coord[coord_mask]
                if masked_updated_coord.size(0) > 0:
                    coord_loss = F.smooth_l1_loss(
                        masked_updated_coord.view(-1, 3),
                        masked_clean_coord.view(-1, 3),
                        reduction="mean",
                        beta=self.beta,
                    )
                else:
                    coord_loss = torch.tensor(
                        0.0, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
                    )
            else:
                coord_loss = F.smooth_l1_loss(
                    updated_coord.view(-1, 3),
                    clean_coord.view(-1, 3),
                    reduction="mean",
                    beta=self.beta,
                )
            loss += self.masked_coord_loss * coord_loss
            more_loss["coord_l1_error"] = coord_loss.detach()
        if self.has_token:
            if self.mask_loss_token:
                masked_logits = logits[type_mask]
                masked_target = clean_type[type_mask]
                if masked_logits.size(0) > 0:
                    token_loss = F.nll_loss(
                        F.log_softmax(masked_logits, dim=-1),
                        masked_target,
                        reduction="mean",
                    )
                else:
                    token_loss = torch.tensor(
                        0.0, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
                    )
            else:
                token_loss = F.nll_loss(
                    F.log_softmax(logits.view(-1, self.ntypes - 1), dim=-1),
                    clean_type.view(-1),
                    reduction="mean",
                )
            loss += self.masked_token_loss * token_loss
            more_loss["token_error"] = token_loss.detach()
        if self.has_norm:
            norm_x = model_pred["norm_x"]
            norm_delta_pair_rep = model_pred["norm_delta_pair_rep"]
            loss += self.norm_loss * (norm_x + norm_delta_pair_rep)
            more_loss["norm_loss"] = norm_x.detach() + norm_delta_pair_rep.detach()

        return loss, more_loss
