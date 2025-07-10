#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import shutil
import time
from pathlib import (
    Path,
)
from typing import (
    Optional,
)

import numpy as np
import optax
import orbax.checkpoint as ocp

from deepmd.common import (
    symlink_prefix_files,
)
from deepmd.dpmodel.loss.ener import (
    EnergyHessianLoss,
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
    jnp,
    nnx,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.model import (
    get_model,
)
from deepmd.jax.utils.serialization import (
    serialize_from_file,
)
from deepmd.loggers.training import (
    format_training_message,
    format_training_message_per_task,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.model_stat import (
    make_stat_input,
)

log = logging.getLogger(__name__)


class DPTrainer:
    def __init__(
        self,
        jdata,
        init_model: Optional[str] = None,
        restart: Optional[str] = None,
    ) -> None:
        self.init_model = init_model
        self.restart = restart
        self.model_def_script = jdata["model"]
        self.start_step = 0
        if self.init_model is not None:
            model_dict = serialize_from_file(self.init_model)
            self.model = BaseModel.deserialize(model_dict["model"])
        elif self.restart is not None:
            model_dict = serialize_from_file(self.restart)
            self.model = BaseModel.deserialize(model_dict["model"])
            self.start_step = model_dict["@variables"].get("current_step", 0)
        else:
            # from scratch
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

        loss_type = loss_param.get("type", "ener")
        if loss_type == "ener" and loss_param.get("start_pref_h", 0.0) > 0.0:
            self.loss = EnergyHessianLoss.get_loss(loss_param)
            self.model.enable_hessian()
        elif loss_type == "ener":
            self.loss = EnergyLoss.get_loss(loss_param)
        else:
            raise RuntimeError("unknown loss type " + loss_type)

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
        model = self.model
        tx = optax.adam(
            learning_rate=lambda step: self.lr.value(self.start_step + step, xp=jnp),
        )
        optimizer = nnx.Optimizer(model, tx)

        # data stat
        if self.init_model is None and self.restart is None:
            data_stat_nbatch = 10  # TODO
            all_stat = make_stat_input(train_data, data_stat_nbatch, merge_sys=False)
            all_stat["atype"] = all_stat.pop("type")

            # swap dict key and list idx
            all_stat_sys = [
                {
                    kk: jnp.asarray(np.concatenate(vv[ii], axis=0))
                    for kk, vv in all_stat.items()
                    if not kk.startswith("find_")
                }
                for ii in range(train_data.get_nsystems())
            ]
            for ii, single_data in enumerate(all_stat_sys):
                if not train_data.data_systems[ii].pbc:
                    single_data["box"] = None
            model.atomic_model.descriptor.compute_input_stats(all_stat_sys)
            model.atomic_model.fitting.compute_output_stats(
                all_stat, mixed_type=train_data.mixed_type
            )

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
            model_dict_lower = model.call_lower(
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
        def loss_fn_more_loss(
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
            model_dict_lower = model.call_lower(
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
            return more_loss

        @nnx.jit
        def train_step(
            model,
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
                model,
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

        start_time = time.time()
        disp_file_fp = open(self.disp_file, "w")
        for step in range(self.start_step, self.num_steps):
            batch_data = train_data.get_batch()
            # numpy to jax
            jax_data = {
                kk: jnp.asarray(vv) if not kk.startswith("find_") else bool(vv.item())
                for kk, vv in batch_data.items()
            }
            extended_coord, extended_atype, nlist, mapping, fp, ap = prepare_input(
                rcut=model.get_rcut(),
                sel=model.get_sel(),
                coord=jax_data["coord"],
                atype=jax_data["type"],
                box=jax_data["box"] if jax_data["default_mesh"].size > 1 else None,
                fparam=jax_data.get("fparam", None),
                aparam=jax_data.get("aparam", None),
            )
            train_step(
                model,
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
            if self.display_in_training and (
                step == 0 or (step + 1) % self.disp_freq == 0
            ):
                wall_time = time.time() - start_time
                log.info(
                    format_training_message(
                        batch=step + 1,
                        wall_time=wall_time,
                    )
                )
                more_loss = loss_fn_more_loss(
                    optimizer.model,
                    self.lr.value(step),
                    jax_data,
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping,
                    fp,
                    ap,
                )
                if valid_data is not None:
                    valid_batch_data = valid_data.get_batch()
                    jax_valid_data = {
                        kk: jnp.asarray(vv) for kk, vv in valid_batch_data.items()
                    }
                    extended_coord, extended_atype, nlist, mapping, fp, ap = (
                        prepare_input(
                            rcut=model.get_rcut(),
                            sel=model.get_sel(),
                            coord=jax_valid_data["coord"],
                            atype=jax_valid_data["type"],
                            box=jax_valid_data["box"]
                            if jax_valid_data["find_box"]
                            else None,
                            fparam=jax_valid_data.get("fparam", None),
                            aparam=jax_valid_data.get("aparam", None),
                        )
                    )
                    valid_more_loss = loss_fn_more_loss(
                        optimizer.model,
                        self.lr.value(step),
                        jax_valid_data,
                        extended_coord,
                        extended_atype,
                        nlist,
                        mapping,
                        fp,
                        ap,
                    )
                else:
                    valid_more_loss = None
                self.print_on_training(
                    disp_file_fp,
                    train_results=more_loss,
                    valid_results=valid_more_loss,
                    cur_batch=step + 1,
                    cur_lr=self.lr.value(step),
                )
                start_time = time.time()
            if (step + 1) % self.save_freq == 0:
                # save model
                _, state = nnx.split(model)
                ckpt_path = Path(f"{self.save_ckpt}-{step + 1}.jax")
                if ckpt_path.is_dir():
                    # remove old checkpoint if it exists
                    shutil.rmtree(ckpt_path)
                model_def_script_cpy = self.model_def_script.copy()
                model_def_script_cpy["current_step"] = step + 1
                with ocp.Checkpointer(
                    ocp.CompositeCheckpointHandler("state", "model_def_script")
                ) as checkpointer:
                    checkpointer.save(
                        ckpt_path.absolute(),
                        ocp.args.Composite(
                            state=ocp.args.StandardSave(state.to_pure_dict()),
                            model_def_script=ocp.args.JsonSave(model_def_script_cpy),
                        ),
                    )
                log.info(f"Trained model has been saved to: {ckpt_path!s}")
                symlink_prefix_files(f"{self.save_ckpt}-{step + 1}", self.save_ckpt)
                with open("checkpoint", "w") as fp:
                    fp.write(f"{self.save_ckpt}.jax")

        disp_file_fp.close()

    @staticmethod
    def print_on_training(
        fp,
        train_results,
        valid_results,
        cur_batch,
        cur_lr,
    ) -> None:
        print_str = ""
        print_str += f"{cur_batch:7d}"
        if valid_results is not None:
            prop_fmt = "   %11.2e %11.2e"
            for k in valid_results.keys():
                # assert k in train_results.keys()
                print_str += prop_fmt % (valid_results[k], train_results[k])
        else:
            prop_fmt = "   %11.2e"
            for k in train_results.keys():
                print_str += prop_fmt % (train_results[k])
        print_str += f"   {cur_lr:8.1e}\n"
        log.info(
            format_training_message_per_task(
                batch=cur_batch,
                task_name="trn",
                rmse=train_results,
                learning_rate=cur_lr,
            )
        )
        if valid_results is not None:
            log.info(
                format_training_message_per_task(
                    batch=cur_batch,
                    task_name="val",
                    rmse=valid_results,
                    learning_rate=None,
                )
            )
        fp.write(print_str)
        fp.flush()


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
