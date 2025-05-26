#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Optional,
)

import numpy as np
import optax
from tqdm import trange

from deepmd.dpmodel.loss.ener import (
    EnergyLoss,
)
from deepmd.dpmodel.model.transform_output import (
    communicate_extended_output,
)
from deepmd.dpmodel.utils.learning_rate import (
    LearningRateExp,
)
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.jax.env import (
    jax,
    jnp,
    nnx,
)
from deepmd.jax.model.model import (
    get_model,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

log = logging.getLogger(__name__)


class DPTrainer:
    def __init__(self, jdata) -> None:
        self.model = get_model(jdata["model"])
        self.training_param = jdata["training"]
        self.num_steps = self.training_param["numb_steps"]

        def get_lr_and_coef(lr_param):
            lr_type = lr_param.get("type", "exp")
            if lr_type == "exp":
                lr = LearningRateExp(
                    lr_param["start_lr"],
                    lr_param["stop_lr"],
                    lr_param["decay_steps"],
                    self.num_steps,
                )
            else:
                raise RuntimeError("unknown learning_rate type " + lr_type)
            return lr

        learning_rate_param = jdata["learning_rate"]
        self.lr = get_lr_and_coef(learning_rate_param)
        loss_param = jdata.get("loss", {})
        loss_param["starter_learning_rate"] = learning_rate_param["start_lr"]
        self.loss = EnergyLoss.get_loss(loss_param)

        # training
        tr_data = jdata["training"]
        self.disp_file = tr_data.get("disp_file", "lcurve.out")
        self.disp_freq = tr_data.get("disp_freq", 1000)
        self.save_freq = tr_data.get("save_freq", 1000)
        self.save_ckpt = tr_data.get("save_ckpt", "model.ckpt")
        self.max_ckpt_keep = tr_data.get("max_ckpt_keep", 5)
        self.display_in_training = tr_data.get("disp_training", True)
        self.timing_in_training = tr_data.get("time_training", True)
        self.profiling = tr_data.get("profiling", False)
        self.profiling_file = tr_data.get("profiling_file", "timeline.json")
        self.enable_profiler = tr_data.get("enable_profiler", False)
        self.tensorboard = tr_data.get("tensorboard", False)
        self.tensorboard_log_dir = tr_data.get("tensorboard_log_dir", "log")
        self.tensorboard_freq = tr_data.get("tensorboard_freq", 1)
        self.mixed_prec = tr_data.get("mixed_precision", None)
        self.change_bias_after_training = tr_data.get(
            "change_bias_after_training", False
        )
        self.numb_fparam = self.model.get_dim_fparam()

        if tr_data.get("validation_data", None) is not None:
            self.valid_numb_batch = tr_data["validation_data"].get("numb_btch", 1)
        else:
            self.valid_numb_batch = 1

        # if init the graph with the frozen model
        self.frz_model = None
        self.ckpt_meta = None
        self.model_type = None

    @property
    def data_requirements(self) -> list[DataRequirementItem]:
        return self.loss.label_requirement

    def train(self, train_data, valid_data=None) -> None:
        optimizer = nnx.Optimizer(self.model, optax.adam(1e-3))  # reference sharing

        def loss_fn(
            model,
            lr,
            label_dict,
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fp,
            ap,
        ):
            model_dict_lower = self.model.call_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fp,
                ap,
            )
            model_dict = communicate_extended_output(
                model_dict_lower,
                model.model_output_def(),
                mapping,
                do_atomic_virial=False,
            )
            loss, more_loss = self.loss(
                learning_rate=lr,
                natoms=label_dict["coord"].shape[1],
                model_dict=model_dict,
                label_dict=label_dict,
            )
            return loss

        @nnx.jit
        def train_step(
            optimizer,
            lr,
            label_dict,
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fp,
            ap,
        ):
            grads = nnx.grad(loss_fn)(
                optimizer.model,
                lr,
                label_dict,
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fp,
                ap,
            )
            optimizer.update(grads)

        for step in trange(self.num_steps):
            batch_data = train_data.get_batch()
            # numpy to jax
            jax_data = {
                kk: jnp.asarray(vv) if not kk.startswith("find_") else bool(vv.item())
                for kk, vv in batch_data.items()
            }
            extended_coord, extended_atype, nlist, mapping, fp, ap = prepare_input(
                rcut=self.model.get_rcut(),
                sel=self.model.get_sel(),
                coord=jax_data["coord"],
                atype=jax_data["type"],
                box=jax_data["box"] if jax_data["find_box"] else None,
                fparam=jax_data.get("fparam", None),
                aparam=jax_data.get("aparam", None),
            )
            loss = train_step(
                optimizer,
                self.lr.value(step),
                jax_data,
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fp,
                ap,
            )
            # print(step, jnp.sqrt(loss))


def prepare_input(
    *,  # enforce keyword-only arguments
    rcut: float,
    sel: list[int],
    coord: np.ndarray,
    atype: np.ndarray,
    box: Optional[np.ndarray] = None,
    fparam: Optional[np.ndarray] = None,
    aparam: Optional[np.ndarray] = None,
):
    nframes, nloc = atype.shape[:2]
    cc, bb, fp, ap = coord, box, fparam, aparam
    del coord, box, fparam, aparam
    if bb is not None:
        coord_normalized = normalize_coord(
            cc.reshape(nframes, nloc, 3),
            bb.reshape(nframes, 3, 3),
        )
    else:
        coord_normalized = cc.copy()
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
        coord_normalized, atype, bb, rcut
    )
    nlist = build_neighbor_list(
        extended_coord,
        extended_atype,
        nloc,
        rcut,
        sel,
        # types will be distinguished in the lower interface,
        # so it doesn't need to be distinguished here
        distinguish_types=False,
    )
    extended_coord = extended_coord.reshape(nframes, -1, 3)
    return extended_coord, extended_atype, nlist, mapping, fp, ap
