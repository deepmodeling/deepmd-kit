# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training loop for the pt_expt backend.

Uses ``DeepmdDataSystem`` (numpy-based batch provider) instead of the
pt backend's ``DpLoaderSet`` + ``DataLoader``.  NumPy batches are
converted to torch tensors at the boundary.
"""

import datetime
import functools
import logging
import time
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np
import torch

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.batch import (
    normalize_batch,
    split_batch,
)
from deepmd.dpmodel.utils.learning_rate import (
    LearningRateExp,
)
from deepmd.loggers.training import (
    format_training_message,
    format_training_message_per_task,
)
from deepmd.pt_expt.loss import (
    DOSLoss,
    EnergyLoss,
    EnergySpinLoss,
    PropertyLoss,
    TensorLoss,
)
from deepmd.pt_expt.model import (
    get_model,
)
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt_expt.utils.env import (
    DEVICE,
    GLOBAL_PT_FLOAT_PRECISION,
)
from deepmd.pt_expt.utils.stat import (
    make_stat_input,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.path import (
    DPPath,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: loss factory (reused from pt)
# ---------------------------------------------------------------------------


def get_loss(
    loss_params: dict[str, Any],
    start_lr: float,
    _ntypes: int,
    _model: Any,
) -> EnergyLoss:
    loss_type = loss_params.get("type", "ener")
    if loss_type == "ener":
        loss_params["starter_learning_rate"] = start_lr
        return EnergyLoss(**loss_params)
    elif loss_type == "dos":
        loss_params["starter_learning_rate"] = start_lr
        loss_params["numb_dos"] = _model.model_output_def()["dos"].output_size
        return DOSLoss(**loss_params)
    elif loss_type == "ener_spin":
        loss_params["starter_learning_rate"] = start_lr
        return EnergySpinLoss(**loss_params)
    elif loss_type == "tensor":
        model_output_type = _model.model_output_type()
        if "mask" in model_output_type:
            model_output_type.pop(model_output_type.index("mask"))
        tensor_name = model_output_type[0]
        loss_params["tensor_size"] = _model.model_output_def()[tensor_name].output_size
        loss_params["label_name"] = tensor_name
        if tensor_name == "polarizability":
            tensor_name = "polar"
        loss_params["tensor_name"] = tensor_name
        return TensorLoss(**loss_params)
    elif loss_type == "property":
        task_dim = _model.get_task_dim()
        var_name = _model.get_var_name()
        intensive = _model.get_intensive()
        loss_params["task_dim"] = task_dim
        loss_params["var_name"] = var_name
        loss_params["intensive"] = intensive
        return PropertyLoss(**loss_params)
    else:
        raise ValueError(f"Unsupported loss type for pt_expt: {loss_type}")


def get_additional_data_requirement(_model: Any) -> list[DataRequirementItem]:
    additional_data_requirement: list[DataRequirementItem] = []
    if _model.get_dim_fparam() > 0:
        additional_data_requirement.append(
            DataRequirementItem(
                "fparam",
                _model.get_dim_fparam(),
                atomic=False,
                must=False,
                default=0.0,
            )
        )
    if _model.get_dim_aparam() > 0:
        additional_data_requirement.append(
            DataRequirementItem(
                "aparam", _model.get_dim_aparam(), atomic=True, must=True
            )
        )
    return additional_data_requirement


# ---------------------------------------------------------------------------
# torch.compile helpers
# ---------------------------------------------------------------------------


def _trace_and_compile(
    model: torch.nn.Module,
    ext_coord: torch.Tensor,
    ext_atype: torch.Tensor,
    nlist: torch.Tensor,
    mapping: torch.Tensor,
    fparam: torch.Tensor | None,
    aparam: torch.Tensor | None,
    compile_opts: dict[str, Any],
) -> torch.nn.Module:
    """Trace ``forward_lower`` with ``make_fx`` and compile with ``torch.compile``.

    Uses symbolic tracing (``tracing_mode="symbolic"``) so the resulting
    FX graph captures shape-polymorphic operations.  The graph is then
    compiled with ``torch.compile(dynamic=True)`` and the inductor
    backend, which automatically pads tensor shapes for efficient kernel
    execution (``shape_padding=True``).

    Parameters
    ----------
    model : torch.nn.Module
        The (uncompiled) model.
    ext_coord, ext_atype, nlist, mapping, fparam, aparam
        Sample tensors used to drive the symbolic trace.
    compile_opts : dict
        Options forwarded to ``torch.compile``.  Keys ``dynamic`` and
        ``backend`` are set internally and ignored if provided.

    Returns
    -------
    torch.nn.Module
        The compiled ``forward_lower`` callable.
    """
    from torch.fx.experimental.proxy_tensor import (
        make_fx,
    )

    was_training = model.training
    # Trace in train mode so that create_graph=True is captured inside
    # task_deriv_one.  Without this, the autograd.grad that computes
    # forces is traced with create_graph=False (eval mode), producing
    # force tensors that are detached from model parameters — force loss
    # backprop cannot reach the weights and force RMSE never decreases.
    model.train()

    def fn(
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None,
        fparam: torch.Tensor | None,
        aparam: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        extended_coord = extended_coord.detach().requires_grad_(True)
        return model.forward_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
        )

    # Symbolic tracing captures shape-polymorphic ops, pairing with
    # dynamic=True in torch.compile to handle varying nall without
    # manual padding or recompilation.
    traced_lower = make_fx(
        fn,
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
    )(ext_coord, ext_atype, nlist, mapping, fparam, aparam)

    if not was_training:
        model.eval()

    # Override backend and dynamic — the inductor backend with
    # dynamic=True handles varying shapes automatically.
    compile_opts.pop("dynamic", None)
    compile_opts.pop("backend", None)
    if "options" not in compile_opts:
        compile_opts["options"] = {}
    compile_opts["options"].setdefault("shape_padding", True)

    compiled_lower = torch.compile(
        traced_lower,
        backend="inductor",
        dynamic=True,
        **compile_opts,
    )
    return compiled_lower


class _CompiledModel(torch.nn.Module):
    """Coord extension (eager) -> compiled forward_lower.

    Coord extension and neighbor list construction involve data-dependent
    control flow and are kept in eager mode.  The compiled ``forward_lower``
    handles varying ``nall`` via ``dynamic=True`` — no manual padding or
    recompilation needed.
    """

    def __init__(
        self,
        original_model: torch.nn.Module,
        compiled_forward_lower: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.original_model = original_model
        self.compiled_forward_lower = compiled_forward_lower

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        from deepmd.dpmodel.utils.nlist import (
            build_neighbor_list,
            extend_coord_with_ghosts,
        )
        from deepmd.dpmodel.utils.region import (
            normalize_coord,
        )

        nframes, nloc = atype.shape[:2]
        rcut = self.original_model.get_rcut()
        sel = self.original_model.get_sel()

        # coord extension + nlist (data-dependent, run in eager)
        coord_3d = coord.detach().reshape(nframes, nloc, 3)
        box_flat = box.detach().reshape(nframes, 9) if box is not None else None

        if box_flat is not None:
            coord_norm = normalize_coord(coord_3d, box_flat.reshape(nframes, 3, 3))
        else:
            coord_norm = coord_3d

        ext_coord, ext_atype, mapping = extend_coord_with_ghosts(
            coord_norm, atype, box_flat, rcut
        )
        nlist = build_neighbor_list(
            ext_coord,
            ext_atype,
            nloc,
            rcut,
            sel,
            distinguish_types=False,
        )
        ext_coord = ext_coord.reshape(nframes, -1, 3)
        ext_coord = ext_coord.detach().requires_grad_(True)

        result = self.compiled_forward_lower(
            ext_coord, ext_atype, nlist, mapping, fparam, aparam
        )

        # Translate forward_lower keys -> forward keys.
        # ``extended_force`` lives on all extended atoms (nf, nall, 3).
        # Ghost-atom forces must be scatter-summed back to local atoms
        # via ``mapping`` — the same operation ``communicate_extended_output``
        # performs in the uncompiled path.
        actual_nall = ext_coord.shape[1]
        out: dict[str, torch.Tensor] = {}
        out["atom_energy"] = result["atom_energy"]
        out["energy"] = result["energy"]
        if "extended_force" in result:
            ext_force = result["extended_force"]  # (nf, nall, 3)
            # scatter-sum extended forces onto local atoms
            idx = mapping.unsqueeze(-1).expand_as(ext_force)  # (nf, nall, 3)
            force = torch.zeros(
                nframes, nloc, 3, dtype=ext_force.dtype, device=ext_force.device
            )
            force.scatter_add_(1, idx, ext_force)
            out["force"] = force
        if "virial" in result:
            out["virial"] = result["virial"]
        if "extended_virial" in result:
            out["extended_virial"] = result["extended_virial"]
        if "atom_virial" in result:
            out["atom_virial"] = result["atom_virial"]
        if "mask" in result:
            out["mask"] = result["mask"]
        return out


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Training driver for the pt_expt backend.

    Uses ``DeepmdDataSystem`` for data loading (numpy batches converted
    to torch tensors at the boundary).  Single-task, single-GPU only.

    Parameters
    ----------
    config : dict
        Full training configuration.
    training_data : DeepmdDataSystem
        Training data.
    stat_file_path : DPPath or None
        Path for saving / loading statistics.
    validation_data : DeepmdDataSystem or None
        Validation data.
    init_model : str or None
        Path to a checkpoint to initialise weights from.
    restart_model : str or None
        Path to a checkpoint to *restart* training from (restores step + optimiser).
    """

    def __init__(
        self,
        config: dict[str, Any],
        training_data: DeepmdDataSystem,
        stat_file_path: DPPath | None = None,
        validation_data: DeepmdDataSystem | None = None,
        init_model: str | None = None,
        restart_model: str | None = None,
        finetune_model: str | None = None,
        finetune_links: dict | None = None,
    ) -> None:
        if finetune_model is not None and (
            init_model is not None or restart_model is not None
        ):
            raise ValueError(
                "finetune_model cannot be combined with init_model or restart_model."
            )
        resume_model = init_model or restart_model or finetune_model
        resuming = resume_model is not None
        self.restart_training = restart_model is not None

        model_params = config["model"]
        training_params = config["training"]

        # Iteration config
        self.num_steps = training_params["numb_steps"]
        self.disp_file = training_params.get("disp_file", "lcurve.out")
        self.disp_freq = training_params.get("disp_freq", 1000)
        self.save_ckpt = training_params.get("save_ckpt", "model.ckpt")
        self.save_freq = training_params.get("save_freq", 1000)
        self.display_in_training = training_params.get("disp_training", True)
        self.timing_in_training = training_params.get("time_training", True)
        self.lcurve_should_print_header = True

        # Model ---------------------------------------------------------------
        self.model = get_model(deepcopy(model_params)).to(DEVICE)

        # Loss ----------------------------------------------------------------
        self.loss = get_loss(
            config.get("loss", {}),
            config["learning_rate"]["start_lr"],
            len(model_params["type_map"]),
            self.model,
        )

        # Data requirements ---------------------------------------------------
        data_requirement = self.loss.label_requirement
        data_requirement += get_additional_data_requirement(self.model)
        training_data.add_data_requirements(data_requirement)
        if validation_data is not None:
            validation_data.add_data_requirements(data_requirement)

        self.training_data = training_data
        self.validation_data = validation_data
        self.valid_numb_batch = training_params.get("validation_data", {}).get(
            "numb_btch", 1
        )

        # Statistics ----------------------------------------------------------
        data_stat_nbatch = model_params.get("data_stat_nbatch", 10)

        @functools.lru_cache
        def get_sample() -> list[dict[str, np.ndarray]]:
            return make_stat_input(training_data, data_stat_nbatch)

        finetune_has_new_type = (
            finetune_model is not None
            and finetune_links is not None
            and finetune_links["Default"].get_has_new_type()
        )
        if not resuming or finetune_has_new_type:
            self.model.compute_or_load_stat(
                sampled_func=get_sample,
                stat_file_path=stat_file_path,
            )

        # Learning rate -------------------------------------------------------
        lr_params = config["learning_rate"].copy()
        lr_params["num_steps"] = self.num_steps
        self.lr_schedule = LearningRateExp(**lr_params)

        # Gradient clipping
        self.gradient_max_norm = training_params.get("gradient_max_norm", 0.0)

        # Model wrapper -------------------------------------------------------
        self.wrapper = ModelWrapper(self.model, self.loss, model_params=model_params)
        self.start_step = 0

        # Optimiser -----------------------------------------------------------
        opt_type = training_params.get("opt_type", "Adam")
        initial_lr = float(self.lr_schedule.value(self.start_step))

        if opt_type == "Adam":
            self.optimizer = torch.optim.Adam(self.wrapper.parameters(), lr=initial_lr)
        elif opt_type == "AdamW":
            weight_decay = training_params.get("weight_decay", 0.001)
            self.optimizer = torch.optim.AdamW(
                self.wrapper.parameters(),
                lr=initial_lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: self.lr_schedule.value(step) / initial_lr,
            last_epoch=self.start_step - 1,
        )

        # Resume --------------------------------------------------------------
        if resuming:
            log.info(f"Resuming from {resume_model}.")
            is_pte = resume_model.endswith((".pte", ".pt2"))

            if is_pte:
                # .pte frozen model: no optimizer state, no step counter
                optimizer_state_dict = None
                self.start_step = 0
            else:
                state_dict = torch.load(
                    resume_model, map_location=DEVICE, weights_only=True
                )
                if "model" in state_dict:
                    optimizer_state_dict = (
                        state_dict["optimizer"]
                        if self.restart_training and finetune_model is None
                        else None
                    )
                    state_dict = state_dict["model"]
                else:
                    optimizer_state_dict = None
                self.start_step = (
                    state_dict["_extra_state"]["train_infos"]["step"]
                    if self.restart_training
                    else 0
                )

            if finetune_model is not None and finetune_links is not None:
                # --- Finetune: selective weight loading -----------------------
                finetune_rule = finetune_links["Default"]

                # Build pretrained model and load weights
                if is_pte:
                    from deepmd.pt_expt.model import (
                        BaseModel,
                    )
                    from deepmd.pt_expt.utils.serialization import (
                        serialize_from_file,
                    )

                    data = serialize_from_file(finetune_model)
                    pretrained_model = BaseModel.deserialize(data["model"]).to(DEVICE)
                else:
                    pretrained_model = get_model(
                        deepcopy(state_dict["_extra_state"]["model_params"])
                    ).to(DEVICE)
                pretrained_wrapper = ModelWrapper(pretrained_model)
                if not is_pte:
                    pretrained_wrapper.load_state_dict(state_dict)

                # Change type map if needed
                if (
                    finetune_rule.get_finetune_tmap()
                    != pretrained_wrapper.model.get_type_map()
                ):
                    model_with_new_type_stat = (
                        self.wrapper.model if finetune_rule.get_has_new_type() else None
                    )
                    pretrained_wrapper.model.change_type_map(
                        finetune_rule.get_finetune_tmap(),
                        model_with_new_type_stat=model_with_new_type_stat,
                    )

                # Selectively copy weights: descriptor always from pretrained,
                # fitting from pretrained unless random_fitting is True
                pretrained_state = pretrained_wrapper.state_dict()
                target_state = self.wrapper.state_dict()
                new_state = {}
                for key in target_state:
                    if key == "_extra_state":
                        new_state[key] = target_state[key]
                    elif (
                        finetune_rule.get_random_fitting() and ".descriptor." not in key
                    ):
                        new_state[key] = target_state[key]  # keep random init
                    elif key in pretrained_state:
                        new_state[key] = pretrained_state[key]  # from pretrained
                    else:
                        new_state[key] = target_state[key]  # fallback
                self.wrapper.load_state_dict(new_state)

                # Adjust output bias
                bias_mode = (
                    "change-by-statistic"
                    if not finetune_rule.get_random_fitting()
                    else "set-by-statistic"
                )
                self.model = model_change_out_bias(
                    self.model, get_sample, _bias_adjust_mode=bias_mode
                )
            else:
                # --- Normal resume (init_model / restart) --------------------
                self.wrapper.load_state_dict(state_dict)

            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
                # rebuild scheduler from the resumed step.
                # last_epoch handles the step offset; the lambda must NOT
                # add self.start_step again (that would double-count).
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer,
                    lambda step: self.lr_schedule.value(step) / initial_lr,
                    last_epoch=self.start_step - 1,
                )

        # torch.compile -------------------------------------------------------
        # The model's forward uses torch.autograd.grad (for forces) with
        # create_graph=True so the loss backward can differentiate through
        # forces.  torch.compile does not support this "double backward".
        #
        # Solution: use make_fx to trace the model forward, which decomposes
        # torch.autograd.grad into primitive ops.  The resulting traced
        # module is then compiled by torch.compile — no double backward.
        self.enable_compile = training_params.get("enable_compile", False)
        if self.enable_compile:
            compile_opts = training_params.get("compile_options", {})
            log.info("Compiling model with torch.compile (%s)", compile_opts)
            self._compile_model(compile_opts)

    # ------------------------------------------------------------------
    # torch.compile helpers
    # ------------------------------------------------------------------

    def _compile_model(self, compile_opts: dict[str, Any]) -> None:
        """Replace ``self.model`` with a compiled version.

        The model's ``forward`` uses ``torch.autograd.grad`` (for forces)
        with ``create_graph=True``, which creates a "double backward" that
        ``torch.compile`` cannot handle.

        Solution: use ``make_fx`` with ``tracing_mode="symbolic"`` to trace
        ``forward_lower``, decomposing ``torch.autograd.grad`` into
        primitive ops with symbolic shapes.  The traced graph is compiled
        with ``torch.compile(dynamic=True, backend="inductor")`` so
        varying ``nall`` across batches is handled automatically — no
        manual padding or recompilation needed.

        Coord extension + nlist build (data-dependent control flow) are
        kept outside the compiled region.
        """
        from deepmd.dpmodel.utils.nlist import (
            build_neighbor_list,
            extend_coord_with_ghosts,
        )
        from deepmd.dpmodel.utils.region import (
            normalize_coord,
        )

        model = self.model

        # --- Get one sample batch to drive the symbolic trace ---
        inp, _ = self.get_data(is_train=True)
        coord = inp["coord"].detach()
        atype = inp["atype"].detach()
        box = inp.get("box")
        if box is not None:
            box = box.detach()

        nframes, nloc = atype.shape[:2]
        coord_3d = coord.reshape(nframes, nloc, 3)
        box_flat = box.reshape(nframes, 9) if box is not None else None

        if box_flat is not None:
            coord_norm = normalize_coord(coord_3d, box_flat.reshape(nframes, 3, 3))
        else:
            coord_norm = coord_3d

        ext_coord, ext_atype, mapping = extend_coord_with_ghosts(
            coord_norm, atype, box_flat, model.get_rcut()
        )
        nlist_t = build_neighbor_list(
            ext_coord,
            ext_atype,
            nloc,
            model.get_rcut(),
            model.get_sel(),
            distinguish_types=False,
        )
        ext_coord = ext_coord.reshape(nframes, -1, 3)

        fparam = inp.get("fparam")
        aparam = inp.get("aparam")

        compiled_lower = _trace_and_compile(
            model,
            ext_coord,
            ext_atype,
            nlist_t,
            mapping,
            fparam,
            aparam,
            compile_opts,
        )

        self.wrapper.model = _CompiledModel(model, compiled_lower)
        log.info(
            "Model compiled (tracing_mode=symbolic, dynamic=True, backend=inductor).",
        )

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def get_data(
        self,
        is_train: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fetch a batch and split into input / label dicts.

        Returns
        -------
        input_dict, label_dict
        """
        data_sys = self.training_data if is_train else self.validation_data
        if data_sys is None:
            return {}, {}

        batch = normalize_batch(data_sys.get_batch())
        input_dict, label_dict = split_batch(batch)

        # Convert numpy values to torch tensors.
        for dd in (input_dict, label_dict):
            for key, val in dd.items():
                if val is None:
                    continue
                if isinstance(val, np.ndarray):
                    if np.issubdtype(val.dtype, np.integer):
                        dd[key] = torch.from_numpy(val).to(DEVICE)
                    else:
                        dd[key] = torch.from_numpy(val).to(
                            dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
                        )
                elif isinstance(val, (float, np.bool_)):
                    dd[key] = torch.tensor(
                        float(val), dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
                    )
        # requires_grad on coord for force computation via autograd
        if "coord" in input_dict and input_dict["coord"] is not None:
            input_dict["coord"] = input_dict["coord"].requires_grad_(True)

        return input_dict, label_dict

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int) -> None:
        self.wrapper.train_infos["step"] = step
        state = {
            "model": self.wrapper.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        ckpt_path = f"{self.save_ckpt}-{step}.pt"
        torch.save(state, ckpt_path)
        # symlink latest
        latest = Path(f"{self.save_ckpt}.pt")
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(ckpt_path)
        log.info(f"Saved checkpoint to {ckpt_path}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    @torch.compiler.disable
    def _optimizer_step(self) -> None:
        """Run optimizer and scheduler step outside torch._dynamo.

        Dynamo intercepts tensor creation inside Adam._init_group,
        which can trigger CUDA init on CPU-only builds.
        """
        self.optimizer.step()
        self.scheduler.step()

    def run(self) -> None:
        fout = open(
            self.disp_file,
            mode="w" if not self.restart_training else "a",
            buffering=1,
        )
        log.info("Start to train %d steps.", self.num_steps)

        self.wrapper.train()
        wall_start = time.time()
        last_log_time = wall_start

        for step_id in range(self.start_step, self.num_steps):
            cur_lr = float(self.lr_schedule.value(step_id))

            if self.timing_in_training:
                t_start = time.time()

            # --- forward / backward ---
            self.optimizer.zero_grad(set_to_none=True)
            input_dict, label_dict = self.get_data(is_train=True)

            cur_lr_sched = self.scheduler.get_last_lr()[0]
            model_pred, loss, more_loss = self.wrapper(
                **input_dict, cur_lr=cur_lr_sched, label=label_dict
            )
            loss.backward()

            if self.gradient_max_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.wrapper.parameters(), self.gradient_max_norm
                )

            self._optimizer_step()

            if self.timing_in_training:
                t_end = time.time()

            # --- display ---
            display_step_id = step_id + 1
            if self.display_in_training and (
                display_step_id % self.disp_freq == 0 or display_step_id == 1
            ):
                self.wrapper.eval()

                train_results = {k: v for k, v in more_loss.items() if "l2_" not in k}

                # validation
                valid_results: dict[str, Any] = {}
                if self.validation_data is not None:
                    sum_natoms = 0
                    for _ii in range(self.valid_numb_batch):
                        val_input, val_label = self.get_data(is_train=False)
                        if not val_input:
                            break
                        _, _vloss, _vmore = self.wrapper(
                            **val_input, cur_lr=cur_lr_sched, label=val_label
                        )
                        natoms = int(val_input["atype"].shape[-1])
                        sum_natoms += natoms
                        for k, v in _vmore.items():
                            if "l2_" not in k:
                                valid_results[k] = (
                                    valid_results.get(k, 0.0) + v * natoms
                                )
                    if sum_natoms > 0:
                        valid_results = {
                            k: v / sum_natoms for k, v in valid_results.items()
                        }

                # wall-clock time
                current_time = time.time()
                wall_elapsed = current_time - wall_start
                interval_wall_time = current_time - last_log_time
                last_log_time = current_time
                if self.timing_in_training:
                    step_time = t_end - t_start
                    steps_completed_since_restart = max(
                        1,
                        display_step_id - self.start_step,
                    )
                    eta = int(
                        (self.num_steps - display_step_id)
                        / steps_completed_since_restart
                        * wall_elapsed
                    )
                    log.info(
                        format_training_message(
                            batch=display_step_id,
                            wall_time=interval_wall_time,
                            eta=eta,
                            current_time=datetime.datetime.fromtimestamp(
                                current_time,
                                tz=datetime.timezone.utc,
                            ).astimezone(),
                        )
                    )
                    log.info("step=%d  step_time=%.4fs", display_step_id, step_time)
                else:
                    log.info(
                        format_training_message(
                            batch=display_step_id,
                            wall_time=interval_wall_time,
                        )
                    )

                # log
                log.info(
                    format_training_message_per_task(
                        batch=display_step_id,
                        task_name="trn",
                        rmse=train_results,
                        learning_rate=cur_lr,
                    )
                )
                if valid_results:
                    log.info(
                        format_training_message_per_task(
                            batch=display_step_id,
                            task_name="val",
                            rmse=valid_results,
                            learning_rate=None,
                        )
                    )

                # lcurve file
                if self.lcurve_should_print_header:
                    self.print_header(fout, train_results, valid_results)
                    self.lcurve_should_print_header = False
                self.print_on_training(
                    fout, display_step_id, cur_lr, train_results, valid_results
                )

                self.wrapper.train()

            # --- checkpoint ---
            if display_step_id % self.save_freq == 0:
                self.save_checkpoint(display_step_id)

        # final save
        self.save_checkpoint(self.num_steps)
        wall_total = time.time() - wall_start
        fout.close()
        log.info("Training finished. Total wall time: %.2fs", wall_total)

    # ------------------------------------------------------------------
    # Printing helpers
    # ------------------------------------------------------------------

    def print_header(
        self,
        fout: Any,
        train_results: dict[str, Any],
        valid_results: dict[str, Any],
    ) -> None:
        train_keys = sorted(train_results.keys())
        header = "# {:5s}".format("step")
        if valid_results:
            for k in train_keys:
                header += f"   {k + '_val':>11s} {k + '_trn':>11s}"
        else:
            for k in train_keys:
                header += f"   {k + '_trn':>11s}"
        header += "   {:8s}\n".format("lr")
        fout.write(header)
        fout.flush()

    def print_on_training(
        self,
        fout: Any,
        step_id: int,
        cur_lr: float,
        train_results: dict,
        valid_results: dict,
    ) -> None:
        train_keys = sorted(train_results.keys())
        line = f"{step_id:7d}"
        if valid_results:
            for k in train_keys:
                line += f"   {valid_results.get(k, float('nan')):11.2e} {train_results[k]:11.2e}"
        else:
            for k in train_keys:
                line += f"   {train_results[k]:11.2e}"
        line += f"   {cur_lr:8.1e}\n"
        fout.write(line)
        fout.flush()


def model_change_out_bias(
    _model: Any,
    _sample_func: Any,
    _bias_adjust_mode: str = "change-by-statistic",
) -> Any:
    """Change the output bias of a model based on sampled data.

    Parameters
    ----------
    _model
        The model whose bias should be adjusted.
    _sample_func
        Callable that returns sampled data for bias computation.
    _bias_adjust_mode
        ``"change-by-statistic"`` or ``"set-by-statistic"``.

    Returns
    -------
    The model with updated bias.
    """
    old_bias = deepcopy(_model.get_out_bias())
    _model.change_out_bias(
        _sample_func,
        bias_adjust_mode=_bias_adjust_mode,
    )
    new_bias = deepcopy(_model.get_out_bias())

    from deepmd.dpmodel.model.dp_model import (
        DPModelCommon,
    )

    if isinstance(_model, DPModelCommon) and _bias_adjust_mode == "set-by-statistic":
        _model.get_fitting_net().compute_input_stats(_sample_func)

    model_type_map = _model.get_type_map()
    log.info(
        f"Change output bias of {model_type_map!s} "
        f"from {to_numpy_array(old_bias).reshape(-1)[: len(model_type_map)]!s} "
        f"to {to_numpy_array(new_bias).reshape(-1)[: len(model_type_map)]!s}."
    )
    return _model
