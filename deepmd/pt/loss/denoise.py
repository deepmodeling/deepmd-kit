# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import torch
import torch.nn.functional as F

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    GLOBAL_PT_FLOAT_PRECISION,
)
from deepmd.pt.utils.region import (
    phys2inter,
)
from deepmd.utils.data import (
    DataRequirementItem,
)


def get_cell_perturb_matrix(cell_pert_fraction: float):
    # TODO: user fix some component
    if cell_pert_fraction < 0:
        raise RuntimeError("cell_pert_fraction can not be negative")
    e0 = torch.rand(6)
    e = e0 * 2 * cell_pert_fraction - cell_pert_fraction
    cell_pert_matrix = torch.tensor(
        [
            [1 + e[0], 0, 0],
            [e[5], 1 + e[1], 0],
            [e[4], e[3], 1 + e[2]],
        ],
        dtype=env.GLOBAL_PT_FLOAT_PRECISION,
        device=env.DEVICE,
    )
    return cell_pert_matrix, e


class DenoiseLoss(TaskLoss):
    def __init__(
        self,
        mask_token: bool = False,
        mask_coord: bool = True,
        mask_cell: bool = False,
        token_loss: float = 1.0,
        coord_loss: float = 1.0,
        cell_loss: float = 1.0,
        noise_type: str = "gaussian",
        coord_noise: float = 0.2,
        cell_pert_fraction: float = 0.0,
        noise_mode: str = "prob",
        mask_num: int = 1,
        mask_prob: float = 0.2,
        loss_func: str = "rmse",
        **kwargs,
    ) -> None:
        r"""Construct a layer to compute loss on token, coord and cell.

        Parameters
        ----------
        mask_token : bool
            Whether to mask token.
        mask_coord : bool
            Whether to mask coordinate.
        mask_cell : bool
            Whether to mask cell.
        token_loss : float
            The preference factor for token denoise.
        coord_loss : float
            The preference factor for coordinate denoise.
        cell_loss : float
            The preference factor for cell denoise.
        noise_type : str
            The type of noise to add to the coordinate. It can be 'uniform' or 'gaussian'.
        coord_noise : float
            The magnitude of noise to add to the coordinate.
        cell_pert_fraction : float
            A value determines how much will cell deform.
        noise_mode : str
            "'prob' means the noise is added with a probability.'fix_num' means the noise is added with a fixed number."
        mask_num : int
            The number of atoms to mask coordinates. It is only used when noise_mode is 'fix_num'.
        mask_prob : float
            The probability of masking coordinates. It is only used when noise_mode is 'prob'.
        loss_func : str
            The loss function to minimize, it can be 'mae' or 'rmse'.
        **kwargs
            Other keyword arguments.
        """
        super().__init__()
        self.mask_token = mask_token
        self.mask_coord = mask_coord
        self.mask_cell = mask_cell
        self.token_loss = token_loss
        self.coord_loss = coord_loss
        self.cell_loss = cell_loss
        self.noise_type = noise_type
        self.coord_noise = coord_noise
        self.cell_pert_fraction = cell_pert_fraction
        self.noise_mode = noise_mode
        self.mask_num = mask_num
        self.mask_prob = mask_prob
        self.loss_func = loss_func

    def forward(self, input_dict, model, label, natoms, learning_rate, mae=False):
        """Return loss on token,coord and cell.

        Parameters
        ----------
        input_dict : dict[str, torch.Tensor]
            Model inputs.
        model : torch.nn.Module
            Model to be used to output the predictions.
        label : dict[str, torch.Tensor]
            Labels.
        natoms : int
            The local atom number.

        Returns
        -------
        model_pred: dict[str, torch.Tensor]
            Model predictions.
        loss: torch.Tensor
            Loss for model to minimize.
        more_loss: dict[str, torch.Tensor]
            Other losses for display.
        """
        nloc = input_dict["atype"].shape[1]
        nbz = input_dict["atype"].shape[0]
        input_dict["box"] = input_dict["box"].cuda()

        # TODO: Change lattice to lower triangular matrix

        label["clean_coord"] = input_dict["coord"].clone().detach()
        label["clean_box"] = input_dict["box"].clone().detach()
        origin_frac_coord = phys2inter(
            label["clean_coord"], label["clean_box"].reshape(nbz, 3, 3)
        )
        label["clean_frac_coord"] = origin_frac_coord.clone().detach()
        if self.mask_cell:
            strain_components_all = torch.zeros(
                (nbz, 3), dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
            )
            for ii in range(nbz):
                cell_perturb_matrix, strain_components = get_cell_perturb_matrix_HEA(
                    self.cell_noise
                )
                # left-multiplied by `cell_perturb_matrix`` to get the noise box
                input_dict["box"][ii] = torch.matmul(
                    cell_perturb_matrix, input_dict["box"][ii].reshape(3, 3)
                ).reshape(-1)
                input_dict["coord"][ii] = torch.matmul(
                    origin_frac_coord[ii].reshape(nloc, 3),
                    input_dict["box"][ii].reshape(3, 3),
                )
                strain_components_all[ii] = strain_components.reshape(-1)
            label["strain_components"] = strain_components_all.clone().detach()

        if self.mask_coord:
            # add noise to coordinates and update label['updated_coord']
            mask_num = 0
            if self.noise_mode == "fix_num":
                mask_num = self.mask_num
                if nloc < mask_num:
                    mask_num = nloc
            elif self.noise_mode == "prob":
                mask_num = int(self.mask_prob * nloc)
                if mask_num == 0:
                    mask_num = 1
            else:
                NotImplementedError(f"Unknown noise mode {self.noise_mode}!")

            coord_mask_all = torch.zeros(
                input_dict["atype"].shape, dtype=torch.bool, device=env.DEVICE
            )
            for ii in range(nbz):
                noise_on_coord = 0.0
                coord_mask_res = np.random.choice(
                    range(nloc), mask_num, replace=False
                ).tolist()
                coord_mask = np.isin(range(nloc), coord_mask_res)  # nloc
                if self.noise_type == "uniform":
                    noise_on_coord = np.random.uniform(
                        low=-self.noise, high=self.noise, size=(mask_num, 3)
                    )
                elif self.noise_type == "gaussian":
                    noise_on_coord = np.random.normal(
                        loc=0.0, scale=self.noise, size=(mask_num, 3)
                    )
                else:
                    raise NotImplementedError(f"Unknown noise type {self.noise_type}!")

                noise_on_coord = torch.tensor(
                    noise_on_coord,
                    dtype=env.GLOBAL_PT_FLOAT_PRECISION,
                    device=env.DEVICE,
                )  # mask_num 3
                input_dict["coord"][ii][coord_mask, :] += (
                    noise_on_coord  # nbz mask_num 3 //
                )
                coord_mask_all[ii] = torch.tensor(
                    coord_mask, dtype=torch.bool, device=env.DEVICE
                )
            label["coord_mask"] = coord_mask_all
            frac_coord = phys2inter(
                input_dict["coord"], input_dict["box"].reshape(nbz, 3, 3)
            )
            # label["updated_coord"] = (label["clean_frac_coord"] - frac_coord).clone().detach()
            label["updated_coord"] = (
                (
                    (label["clean_frac_coord"] - frac_coord)
                    @ label["clean_box"].reshape(nbz, 3, 3)
                )
                .clone()
                .detach()
            )

        if self.mask_token:
            # TODO: mask_token
            pass

        if (not self.mask_coord) and (not self.mask_cell) and (not self.mask_token):
            raise RuntimeError(
                "At least one of mask_coord, mask_cell and mask_token should be True!"
            )

        model_pred = model(**input_dict)

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}

        diff_coord = (label["updated_coord"] - model_pred["updated_coord"]).reshape(-1)
        diff_cell = (
            label["strain_components"] - model_pred["strain_components"]
        ).reshape(-1)
        if self.loss_func == "rmse":
            l2_coord_loss = torch.mean(torch.square(diff_coord))
            l2_cell_loss = torch.mean(torch.square(diff_cell))
            rmse_f = l2_coord_loss.sqrt()
            rmse_v = l2_cell_loss.sqrt()
            more_loss["rmse_coord"] = rmse_f.detach()
            more_loss["rmse_cell"] = rmse_v.detach()
            loss += self.coord_loss * l2_coord_loss.to(
                GLOBAL_PT_FLOAT_PRECISION
            ) + self.cell_loss * l2_cell_loss.to(GLOBAL_PT_FLOAT_PRECISION)
        elif self.loss_func == "mae":
            l1_coord_loss = F.l1_loss(
                label["updated_coord"], model_pred["updated_coord"], reduction="none"
            )
            l1_cell_loss = F.l1_loss(
                label["strain_components"],
                model_pred["strain_components"],
                reduction="none",
            )
            more_loss["mae_coord"] = l1_coord_loss.mean().detach()
            more_loss["mae_cell"] = l1_cell_loss.mean().detach()
            l1_coord_loss = l1_coord_loss.sum(-1).mean(-1).sum()
            l1_cell_loss = l1_cell_loss.sum()
            loss += self.coord_loss * l1_coord_loss.to(
                GLOBAL_PT_FLOAT_PRECISION
            ) + self.cell_loss * l1_cell_loss.to(GLOBAL_PT_FLOAT_PRECISION)
        else:
            raise RuntimeError(f"Unknown loss function {self.loss_func}!")
        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        return []

    def serialize(self) -> dict:
        pass

    @classmethod
    def deserialize(cls, data: dict) -> "TaskLoss":
        pass
