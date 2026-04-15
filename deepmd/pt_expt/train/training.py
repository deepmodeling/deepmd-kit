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
import torch.distributed as dist

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


def _remove_detach_nodes(gm: torch.fx.GraphModule) -> None:
    """Remove ``aten.detach.default`` nodes from an FX graph in-place.

    ``make_fx`` inserts these nodes when recording saved tensors from the
    autograd backward pass (``autograd.grad`` with ``create_graph=True``).
    The detach breaks the gradient connection between saved activations and
    model parameters, causing incorrect second-order derivatives — e.g.
    bias gradients become zero for force-loss training.

    Removing these nodes restores the gradient path so that higher-order
    derivatives flow correctly through the decomposed backward ops.
    """
    graph = gm.graph
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target == torch.ops.aten.detach.default:
            input_node = node.args[0]
            node.replace_all_uses_with(input_node)
            graph.erase_node(node)
    graph.lint()
    gm.recompile()


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

    Parameters
    ----------
    model : torch.nn.Module
        The (uncompiled) model.
    ext_coord, ext_atype, nlist, mapping, fparam, aparam
        Sample tensors (already padded to the desired max_nall).
    compile_opts : dict
        Options forwarded to ``torch.compile`` (excluding ``dynamic``).

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
        extended_coord = extended_coord.requires_grad_(True)
        return model.forward_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
        )

    # Use default tracing_mode="real" (concrete shapes) for best
    # runtime performance.  If data-dependent intermediate shapes
    # change at runtime, the caller catches the error and retraces.
    traced_lower = make_fx(fn)(ext_coord, ext_atype, nlist, mapping, fparam, aparam)

    # make_fx inserts aten.detach.default for saved tensors used in the
    # decomposed autograd.grad backward ops.  These detach nodes break
    # second-order gradient flow (d(force)/d(params) for force training).
    # Removing them restores correct higher-order derivatives.
    _remove_detach_nodes(traced_lower)

    if not was_training:
        model.eval()

    if "backend" not in compile_opts:
        compile_opts["backend"] = "aot_eager"
    compiled_lower = torch.compile(traced_lower, dynamic=False, **compile_opts)
    return compiled_lower


class _CompiledModel(torch.nn.Module):
    """Coord extension (eager) -> pad nall -> compiled forward_lower.

    If a batch's ``nall`` exceeds the current ``max_nall``, the model is
    automatically re-traced and recompiled with a larger pad size.
    """

    def __init__(
        self,
        original_model: torch.nn.Module,
        compiled_forward_lower: torch.nn.Module,
        max_nall: int,
        compile_opts: dict[str, Any],
    ) -> None:
        super().__init__()
        self.original_model = original_model
        self.compiled_forward_lower = compiled_forward_lower
        self._max_nall = max_nall
        self._compile_opts = compile_opts

    def _recompile(
        self,
        ext_coord: torch.Tensor,
        ext_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor,
        fparam: torch.Tensor | None,
        aparam: torch.Tensor | None,
        new_max_nall: int,
    ) -> None:
        """Re-trace and recompile for the given inputs.

        If *new_max_nall* differs from the current ``_max_nall``, the
        inputs are padded (or already padded by the caller).
        """
        # Pad if the caller provides unpadded tensors (nall growth case)
        actual_nall = ext_coord.shape[1]
        pad_n = new_max_nall - actual_nall
        if pad_n > 0:
            ext_coord = torch.nn.functional.pad(ext_coord, (0, 0, 0, pad_n))
            ext_atype = torch.nn.functional.pad(ext_atype, (0, pad_n))
            mapping = torch.nn.functional.pad(mapping, (0, pad_n))

        ext_coord = ext_coord.detach()

        self.compiled_forward_lower = _trace_and_compile(
            self.original_model,
            ext_coord,
            ext_atype,
            nlist,
            mapping,
            fparam,
            aparam,
            self._compile_opts,
        )
        self._max_nall = new_max_nall
        log.info(
            "Recompiled model with max_nall=%d.",
            new_max_nall,
        )

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

        # Grow max_nall if needed (retrace + recompile)
        actual_nall = ext_coord.shape[1]
        if actual_nall > self._max_nall:
            new_max_nall = ((int(actual_nall * 1.2) + 7) // 8) * 8
            log.info(
                "nall=%d exceeds max_nall=%d; recompiling with max_nall=%d.",
                actual_nall,
                self._max_nall,
                new_max_nall,
            )
            self._recompile(
                ext_coord, ext_atype, nlist, mapping, fparam, aparam, new_max_nall
            )

        # Pad to max_nall so compiled graph sees a fixed shape
        pad_n = self._max_nall - actual_nall
        if pad_n > 0:
            ext_coord = torch.nn.functional.pad(ext_coord, (0, 0, 0, pad_n))
            ext_atype = torch.nn.functional.pad(ext_atype, (0, pad_n))
            mapping = torch.nn.functional.pad(mapping, (0, pad_n))
        ext_coord = ext_coord.detach().requires_grad_(True)

        result = self.compiled_forward_lower(
            ext_coord, ext_atype, nlist, mapping, fparam, aparam
        )

        # Translate forward_lower keys -> forward keys.
        # ``extended_force`` lives on all extended atoms (nf, nall, 3).
        # Ghost-atom forces must be scatter-summed back to local atoms
        # via ``mapping`` — the same operation ``communicate_extended_output``
        # performs in the uncompiled path.
        out: dict[str, torch.Tensor] = {}
        out["atom_energy"] = result["atom_energy"]
        out["energy"] = result["energy"]
        if "extended_force" in result:
            ext_force = result["extended_force"]  # (nf, nall_padded, 3)
            # mapping may be padded; only use actual_nall entries
            map_actual = mapping[:, :actual_nall]  # (nf, actual_nall)
            ext_force_actual = ext_force[:, :actual_nall, :]  # (nf, actual_nall, 3)
            # scatter-sum extended forces onto local atoms
            idx = map_actual.unsqueeze(-1).expand_as(
                ext_force_actual
            )  # (nf, actual_nall, 3)
            force = torch.zeros(
                nframes, nloc, 3, dtype=ext_force.dtype, device=ext_force.device
            )
            force.scatter_add_(1, idx, ext_force_actual)
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
    to torch tensors at the boundary).  Supports single-task and multi-task
    training.  Single-GPU only.

    Parameters
    ----------
    config : dict
        Full training configuration.
    training_data : DeepmdDataSystem or dict
        Training data.  Dict of ``{model_key: DeepmdDataSystem}`` for multi-task.
    stat_file_path : DPPath or dict or None
        Path for saving / loading statistics.
    validation_data : DeepmdDataSystem or dict or None
        Validation data.
    init_model : str or None
        Path to a checkpoint to initialise weights from.
    restart_model : str or None
        Path to a checkpoint to *restart* training from (restores step + optimiser).
    shared_links : dict or None
        Parameter sharing rules for multi-task training.
    """

    def __init__(
        self,
        config: dict[str, Any],
        training_data: DeepmdDataSystem | dict,
        stat_file_path: DPPath | dict | None = None,
        validation_data: DeepmdDataSystem | dict | None = None,
        init_model: str | None = None,
        restart_model: str | None = None,
        finetune_model: str | None = None,
        finetune_links: dict | None = None,
        shared_links: dict | None = None,
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

        # Multi-task detection
        self.multi_task = "model_dict" in model_params
        self.model_keys = (
            list(model_params["model_dict"]) if self.multi_task else ["Default"]
        )
        self.num_model = len(self.model_keys)

        # Distributed training detection
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1

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
        if not self.multi_task:
            self.model = get_model(deepcopy(model_params)).to(DEVICE)
        else:
            self.model = {}
            do_case_embd, case_embd_index = _get_case_embd_config(model_params)
            for model_key in self.model_keys:
                self.model[model_key] = get_model(
                    deepcopy(model_params["model_dict"][model_key])
                ).to(DEVICE)
                if do_case_embd and not resuming:
                    self.model[model_key].set_case_embd(case_embd_index[model_key])

        # Loss ----------------------------------------------------------------
        if not self.multi_task:
            self.loss = get_loss(
                config.get("loss", {}),
                config["learning_rate"]["start_lr"],
                len(model_params["type_map"]),
                self.model,
            )
        else:
            self.loss = {}
            for model_key in self.model_keys:
                loss_param = config["loss_dict"][model_key]
                lr_param = config["learning_rate"]["start_lr"]
                ntypes = len(model_params["model_dict"][model_key]["type_map"])
                self.loss[model_key] = get_loss(
                    loss_param, lr_param, ntypes, self.model[model_key]
                )

        # Data requirements ---------------------------------------------------
        if not self.multi_task:
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
        else:
            self.training_data = {}
            self.validation_data = {}
            self.valid_numb_batch = {}
            for model_key in self.model_keys:
                data_requirement = self.loss[model_key].label_requirement
                data_requirement += get_additional_data_requirement(
                    self.model[model_key]
                )
                training_data[model_key].add_data_requirements(data_requirement)
                if validation_data[model_key] is not None:
                    validation_data[model_key].add_data_requirements(data_requirement)
                self.training_data[model_key] = training_data[model_key]
                self.validation_data[model_key] = validation_data[model_key]
                self.valid_numb_batch[model_key] = (
                    training_params["data_dict"][model_key]
                    .get("validation_data", {})
                    .get("numb_btch", 1)
                )

        # Statistics ----------------------------------------------------------
        if not self.multi_task:
            data_stat_nbatch = model_params.get("data_stat_nbatch", 10)

            @functools.lru_cache
            def get_sample() -> list[dict[str, np.ndarray]]:
                return make_stat_input(training_data, data_stat_nbatch)

            finetune_has_new_type = (
                finetune_model is not None
                and finetune_links is not None
                and finetune_links["Default"].get_has_new_type()
            )
            if (not resuming or finetune_has_new_type) and self.rank == 0:
                self.model.compute_or_load_stat(
                    sampled_func=get_sample,
                    stat_file_path=stat_file_path,
                )
            if self.is_distributed:
                self._broadcast_model_stat(self.model)
        else:
            self._finetune_update_stat = False
            self._sample_funcs: dict[str, Any] = {}
            for model_key in self.model_keys:
                _nbatch = model_params["model_dict"][model_key].get(
                    "data_stat_nbatch", 10
                )
                _data = training_data[model_key]
                _stat_path = stat_file_path[model_key] if stat_file_path else None

                def _make_sample(
                    _d: DeepmdDataSystem = _data, _n: int = _nbatch
                ) -> list[dict[str, np.ndarray]]:
                    return make_stat_input(_d, _n)

                self._sample_funcs[model_key] = _make_sample

                _finetune_has_new_type = (
                    finetune_model is not None
                    and finetune_links is not None
                    and model_key in finetune_links
                    and finetune_links[model_key].get_has_new_type()
                )
                if _finetune_has_new_type:
                    self._finetune_update_stat = True
                if (not resuming or _finetune_has_new_type) and self.rank == 0:
                    self.model[model_key].compute_or_load_stat(
                        sampled_func=_make_sample,
                        stat_file_path=_stat_path,
                    )
            if self.is_distributed:
                for model_key in self.model_keys:
                    self._broadcast_model_stat(self.model[model_key])

        # Model probability (multi-task) --------------------------------------
        if self.multi_task:
            from deepmd.dpmodel.utils.training_utils import (
                resolve_model_prob,
            )

            self.model_prob = resolve_model_prob(
                self.model_keys,
                training_params.get("model_prob"),
                training_data,
            )
        else:
            self.model_prob = None

        # Learning rate -------------------------------------------------------
        lr_params = config["learning_rate"].copy()
        lr_params["num_steps"] = self.num_steps
        self.lr_schedule = LearningRateExp(**lr_params)

        # Gradient clipping
        self.gradient_max_norm = training_params.get("gradient_max_norm", 0.0)

        # Model wrapper -------------------------------------------------------
        self.wrapper = ModelWrapper(self.model, self.loss, model_params=model_params)
        self.start_step = 0

        # Shared params (multi-task) ------------------------------------------
        if shared_links is not None:
            _data_stat_protect = np.array(
                [
                    model_params["model_dict"][ii].get("data_stat_protect", 1e-2)
                    for ii in model_params["model_dict"]
                ]
            )
            assert np.allclose(_data_stat_protect, _data_stat_protect[0]), (
                "Model key 'data_stat_protect' must be the same in each branch when multitask!"
            )
            self.wrapper.share_params(
                shared_links,
                resume=(resuming and not self._finetune_update_stat) or self.rank != 0,
                model_key_prob_map=dict(zip(self.model_keys, self.model_prob)),
                data_stat_protect=_data_stat_protect[0],
            )

        # DDP wrapping --------------------------------------------------------
        if self.is_distributed:
            # Multi-task uses only one fitting_net per step, so unused
            # parameters exist in the graph. Single-task doesn't need this.
            _find_unused = self.multi_task
            if DEVICE.type == "cuda":
                from deepmd.pt_expt.utils.env import (
                    LOCAL_RANK,
                )

                torch.cuda.set_device(LOCAL_RANK)
                self.wrapper = torch.nn.parallel.DistributedDataParallel(
                    self.wrapper,
                    device_ids=[LOCAL_RANK],
                    find_unused_parameters=_find_unused,
                    output_device=LOCAL_RANK,
                )
            else:
                # CPU (gloo backend) — no device_ids
                self.wrapper = torch.nn.parallel.DistributedDataParallel(
                    self.wrapper,
                    find_unused_parameters=_find_unused,
                )

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

                # Build pretrained model(s) and load weights
                if is_pte:
                    from deepmd.pt_expt.model import (
                        BaseModel,
                    )
                    from deepmd.pt_expt.utils.serialization import (
                        serialize_from_file,
                    )

                    data = serialize_from_file(finetune_model)
                    pretrained_model_params = data["model_def_script"]
                    pretrained_model = BaseModel.deserialize(data["model"]).to(DEVICE)
                else:
                    pretrained_model_params = state_dict["_extra_state"]["model_params"]

                # Build pretrained model (single-task or multi-task)
                if "model_dict" not in pretrained_model_params:
                    # Single-task pretrained → wrap as {"Default": model}
                    if is_pte:
                        pretrained_models = pretrained_model
                    else:
                        pretrained_models = get_model(
                            deepcopy(pretrained_model_params)
                        ).to(DEVICE)
                else:
                    pretrained_models = {}
                    for pk in pretrained_model_params["model_dict"]:
                        pretrained_models[pk] = get_model(
                            deepcopy(pretrained_model_params["model_dict"][pk])
                        ).to(DEVICE)
                pretrained_wrapper = ModelWrapper(pretrained_models)
                if not is_pte:
                    pretrained_wrapper.load_state_dict(state_dict)

                # Per-branch type map change
                for model_key in self.model_keys:
                    finetune_rule = finetune_links[model_key]
                    _model_key_from = finetune_rule.get_model_branch()
                    if (
                        finetune_rule.get_finetune_tmap()
                        != pretrained_wrapper.model[_model_key_from].get_type_map()
                    ):
                        model_with_new_type_stat = (
                            self._unwrapped.model[model_key]
                            if finetune_rule.get_has_new_type()
                            else None
                        )
                        pretrained_wrapper.model[_model_key_from].change_type_map(
                            finetune_rule.get_finetune_tmap(),
                            model_with_new_type_stat=model_with_new_type_stat,
                        )

                # Selective weight copy (per-branch key remapping)
                pretrained_state = pretrained_wrapper.state_dict()
                target_state = self._unwrapped.state_dict()
                new_state = {}
                for key in target_state:
                    if key == "_extra_state":
                        new_state[key] = target_state[key]
                        continue
                    # Find which model_key this key belongs to
                    matched = False
                    for model_key in self.model_keys:
                        if f".{model_key}." not in key:
                            continue
                        matched = True
                        finetune_rule = finetune_links[model_key]
                        _key_from = finetune_rule.get_model_branch()
                        pretrained_key = key.replace(f".{model_key}.", f".{_key_from}.")
                        use_random = (
                            finetune_rule.get_random_fitting()
                            and ".descriptor." not in key
                        )
                        if use_random:
                            new_state[key] = target_state[key]
                        elif pretrained_key in pretrained_state:
                            new_state[key] = pretrained_state[pretrained_key]
                        else:
                            new_state[key] = target_state[key]
                        break
                    if not matched:
                        new_state[key] = target_state[key]
                self._unwrapped.load_state_dict(new_state)

                # Per-branch bias adjustment (rank 0 only, then broadcast)
                if not self.multi_task:
                    finetune_rule = finetune_links["Default"]
                    bias_mode = (
                        "change-by-statistic"
                        if not finetune_rule.get_random_fitting()
                        else "set-by-statistic"
                    )
                    if self.rank == 0:
                        self.model = model_change_out_bias(
                            self.model, get_sample, _bias_adjust_mode=bias_mode
                        )
                    if self.is_distributed:
                        self._broadcast_model_stat(self.model)
                else:
                    for model_key in self.model_keys:
                        finetune_rule = finetune_links[model_key]
                        if finetune_rule.get_resuming():
                            log.info(f"Model branch {model_key} will resume training.")
                            continue
                        log.info(f"Model branch {model_key} will be fine-tuned.")
                        bias_mode = (
                            "change-by-statistic"
                            if not finetune_rule.get_random_fitting()
                            else "set-by-statistic"
                        )
                        if self.rank == 0:
                            self.model[model_key] = model_change_out_bias(
                                self.model[model_key],
                                self._sample_funcs[model_key],
                                _bias_adjust_mode=bias_mode,
                            )
                        if self.is_distributed:
                            self._broadcast_model_stat(self.model[model_key])
            else:
                # --- Normal resume (init_model / restart) --------------------
                self._unwrapped.load_state_dict(state_dict)

            if shared_links is not None:
                # Re-apply sharing after loading checkpoint
                self._unwrapped.share_params(
                    shared_links,
                    resume=True,
                    model_key_prob_map=dict(zip(self.model_keys, self.model_prob)),
                )

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

        The model's ``forward`` uses ``torch.autograd.grad`` (for force
        computation) with ``create_graph=True``, which creates a "double
        backward" that ``torch.compile`` cannot handle.

        Solution: use ``make_fx`` to trace ``forward_lower``, decomposing
        ``torch.autograd.grad`` into primitive ops.  The coord extension +
        nlist build (data-dependent control flow) are kept outside the
        compiled region.

        To avoid the overhead of symbolic tracing and dynamic shapes, the
        extended-atom dimension (nall) is padded to a fixed maximum
        estimated from the training data.  This allows concrete-shape
        tracing and ``dynamic=False``.  If a batch exceeds the current
        max_nall at runtime, the model is automatically re-traced and
        recompiled with a larger pad size.
        """
        from deepmd.dpmodel.utils.nlist import (
            build_neighbor_list,
            extend_coord_with_ghosts,
        )
        from deepmd.dpmodel.utils.region import (
            normalize_coord,
        )

        for task_key in self.model_keys:
            model = self.wrapper.model[task_key]

            # --- Estimate max_nall by sampling multiple batches ---
            n_sample = 20
            max_nall = 0
            best_sample: (
                tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict] | None
            ) = None

            for _ii in range(n_sample):
                inp, _ = self.get_data(is_train=True, task_key=task_key)
                coord = inp["coord"].detach()
                atype = inp["atype"].detach()
                box = inp.get("box")
                if box is not None:
                    box = box.detach()

                nframes, nloc = atype.shape[:2]
                coord_np = coord.cpu().numpy().reshape(nframes, nloc, 3)
                atype_np = atype.cpu().numpy()
                box_np = (
                    box.cpu().numpy().reshape(nframes, 9) if box is not None else None
                )

                if box_np is not None:
                    coord_norm = normalize_coord(
                        coord_np, box_np.reshape(nframes, 3, 3)
                    )
                else:
                    coord_norm = coord_np

                ext_coord_np, ext_atype_np, mapping_np = extend_coord_with_ghosts(
                    coord_norm, atype_np, box_np, model.get_rcut()
                )
                nlist_np = build_neighbor_list(
                    ext_coord_np,
                    ext_atype_np,
                    nloc,
                    model.get_rcut(),
                    model.get_sel(),
                    distinguish_types=False,
                )
                ext_coord_np = ext_coord_np.reshape(nframes, -1, 3)
                nall = ext_coord_np.shape[1]
                if nall > max_nall:
                    max_nall = nall
                    best_sample = (
                        ext_coord_np,
                        ext_atype_np,
                        mapping_np,
                        nlist_np,
                        nloc,
                        inp,
                    )

            # Add 20 % margin and round up to a multiple of 8.
            max_nall = ((int(max_nall * 1.2) + 7) // 8) * 8
            log.info(
                "Estimated max_nall=%d for compiled model "
                "(task=%s, sampled %d batches).",
                max_nall,
                task_key,
                n_sample,
            )

            # --- Pad the largest sample to max_nall and trace ---
            assert best_sample is not None
            ext_coord_np, ext_atype_np, mapping_np, nlist_np, nloc, sample_input = (
                best_sample
            )
            nframes = ext_coord_np.shape[0]
            actual_nall = ext_coord_np.shape[1]
            pad_n = max_nall - actual_nall

            if pad_n > 0:
                ext_coord_np = np.pad(ext_coord_np, ((0, 0), (0, pad_n), (0, 0)))
                ext_atype_np = np.pad(ext_atype_np, ((0, 0), (0, pad_n)))
                mapping_np = np.pad(mapping_np, ((0, 0), (0, pad_n)))

            ext_coord = torch.tensor(
                ext_coord_np, dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
            )
            ext_atype = torch.tensor(ext_atype_np, dtype=torch.int64, device=DEVICE)
            nlist_t = torch.tensor(nlist_np, dtype=torch.int64, device=DEVICE)
            mapping_t = torch.tensor(mapping_np, dtype=torch.int64, device=DEVICE)
            fparam = sample_input.get("fparam")
            aparam = sample_input.get("aparam")

            task_compile_opts = dict(compile_opts)
            task_compile_opts.pop("dynamic", None)  # always False for padded approach

            compiled_lower = _trace_and_compile(
                model,
                ext_coord,
                ext_atype,
                nlist_t,
                mapping_t,
                fparam,
                aparam,
                task_compile_opts,
            )

            self.wrapper.model[task_key] = _CompiledModel(
                model, compiled_lower, max_nall, task_compile_opts
            )
            log.info(
                "Model compiled with padded nall=%d (task=%s, dynamic=False).",
                max_nall,
                task_key,
            )

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def get_data(
        self,
        is_train: bool = True,
        task_key: str = "Default",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fetch a batch and split into input / label dicts.

        Parameters
        ----------
        is_train : bool
            Whether to fetch from training or validation data.
        task_key : str
            Task key for multi-task training.

        Returns
        -------
        input_dict, label_dict
        """
        if not self.multi_task:
            data_sys = self.training_data if is_train else self.validation_data
        else:
            data_sys = (
                self.training_data[task_key]
                if is_train
                else self.validation_data[task_key]
            )
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
    # DDP helpers
    # ------------------------------------------------------------------

    @property
    def _unwrapped(self) -> "ModelWrapper":
        """Return the raw ModelWrapper, unwrapping DDP if active."""
        if hasattr(self.wrapper, "module"):
            return self.wrapper.module
        return self.wrapper

    @staticmethod
    def _broadcast_model_stat(model: torch.nn.Module) -> None:
        """Broadcast model parameters and buffers from rank 0 to all ranks."""
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        for b in model.buffers():
            dist.broadcast(b, src=0)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int) -> None:
        self._unwrapped.train_infos["step"] = step
        state = {
            "model": self._unwrapped.state_dict(),
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
        from deepmd.utils import random as dp_random

        fout = (
            open(
                self.disp_file,
                mode="w" if not self.restart_training else "a",
                buffering=1,
            )
            if self.rank == 0
            else None
        )
        log.info("Start to train %d steps.", self.num_steps)

        self.wrapper.train()
        wall_start = time.time()
        last_log_time = wall_start

        for step_id in range(self.start_step, self.num_steps):
            cur_lr = float(self.lr_schedule.value(step_id))

            # --- task selection (multi-task) ---
            task_key = "Default"
            if self.multi_task:
                model_index = dp_random.choice(
                    np.arange(self.num_model, dtype=np.int_),
                    p=self.model_prob,
                )
                task_key = self.model_keys[model_index]

            if self.timing_in_training:
                t_start = time.time()

            # --- forward / backward ---
            self.optimizer.zero_grad(set_to_none=True)
            input_dict, label_dict = self.get_data(is_train=True, task_key=task_key)

            cur_lr_sched = self.scheduler.get_last_lr()[0]
            model_pred, loss, more_loss = self.wrapper(
                **input_dict,
                cur_lr=cur_lr_sched,
                label=label_dict,
                task_key=task_key if self.multi_task else None,
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

                if self.rank == 0:
                    if not self.multi_task:
                        train_results = {
                            k: v for k, v in more_loss.items() if "l2_" not in k
                        }

                        # validation
                        valid_results: dict[str, Any] = {}
                        if self.validation_data is not None:
                            sum_natoms = 0
                            for _ii in range(self.valid_numb_batch):
                                val_input, val_label = self.get_data(is_train=False)
                                if not val_input:
                                    break
                                _, _vloss, _vmore = self._unwrapped(
                                    **val_input,
                                    cur_lr=cur_lr_sched,
                                    label=val_label,
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
                    else:
                        # Multi-task: compute loss for ALL tasks
                        train_results = {_key: {} for _key in self.model_keys}
                        valid_results = {_key: {} for _key in self.model_keys}

                        # current task already has loss
                        train_results[task_key] = {
                            k: v for k, v in more_loss.items() if "l2_" not in k
                        }

                        # compute loss for other tasks
                        for _key in self.model_keys:
                            if _key != task_key:
                                self.optimizer.zero_grad()
                                _inp, _lab = self.get_data(is_train=True, task_key=_key)
                                _, _loss, _more = self._unwrapped(
                                    **_inp,
                                    cur_lr=cur_lr_sched,
                                    label=_lab,
                                    task_key=_key,
                                )
                                train_results[_key] = {
                                    k: v for k, v in _more.items() if "l2_" not in k
                                }

                            # validation for each task
                            _vdata = self.validation_data[_key]
                            if _vdata is not None:
                                _sum_natoms = 0
                                _vres: dict[str, Any] = {}
                                for _ii in range(self.valid_numb_batch[_key]):
                                    _vi, _vl = self.get_data(
                                        is_train=False, task_key=_key
                                    )
                                    if not _vi:
                                        break
                                    _, _vloss, _vmore = self._unwrapped(
                                        **_vi,
                                        cur_lr=cur_lr_sched,
                                        label=_vl,
                                        task_key=_key,
                                    )
                                    natoms = int(_vi["atype"].shape[-1])
                                    _sum_natoms += natoms
                                    for k, v in _vmore.items():
                                        if "l2_" not in k:
                                            _vres[k] = _vres.get(k, 0.0) + v * natoms
                                if _sum_natoms > 0:
                                    _vres = {
                                        k: v / _sum_natoms for k, v in _vres.items()
                                    }
                                valid_results[_key] = _vres
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
                    if not self.multi_task:
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
                    else:
                        for _key in self.model_keys:
                            log.info(
                                format_training_message_per_task(
                                    batch=display_step_id,
                                    task_name=_key + "_trn",
                                    rmse=train_results[_key],
                                    learning_rate=cur_lr,
                                )
                            )
                            if valid_results[_key]:
                                log.info(
                                    format_training_message_per_task(
                                        batch=display_step_id,
                                        task_name=_key + "_val",
                                        rmse=valid_results[_key],
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
            if display_step_id % self.save_freq == 0 and self.rank == 0:
                self.save_checkpoint(display_step_id)

        # final save
        if self.rank == 0:
            self.save_checkpoint(self.num_steps)
        wall_total = time.time() - wall_start
        if fout is not None:
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
        header = "# {:5s}".format("step")
        if not self.multi_task:
            train_keys = sorted(train_results.keys())
            if valid_results:
                for k in train_keys:
                    header += f"   {k + '_val':>11s} {k + '_trn':>11s}"
            else:
                for k in train_keys:
                    header += f"   {k + '_trn':>11s}"
        else:
            for model_key in self.model_keys:
                if valid_results[model_key]:
                    for k in sorted(train_results[model_key].keys()):
                        header += f"   {k + '_val_' + model_key:>11s} {k + '_trn_' + model_key:>11s}"
                else:
                    for k in sorted(train_results[model_key].keys()):
                        header += f"   {k + '_trn_' + model_key:>11s}"
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
        line = f"{step_id:7d}"
        if not self.multi_task:
            train_keys = sorted(train_results.keys())
            if valid_results:
                for k in train_keys:
                    line += f"   {valid_results.get(k, float('nan')):11.2e} {train_results[k]:11.2e}"
            else:
                for k in train_keys:
                    line += f"   {train_results[k]:11.2e}"
        else:
            for model_key in self.model_keys:
                if valid_results[model_key]:
                    for k in sorted(valid_results[model_key].keys()):
                        line += f"   {valid_results[model_key][k]:11.2e} {train_results[model_key][k]:11.2e}"
                else:
                    for k in sorted(train_results[model_key].keys()):
                        line += f"   {train_results[model_key][k]:11.2e}"
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


def _get_case_embd_config(
    model_params: dict[str, Any],
) -> tuple[bool, dict[str, int]]:
    """Check whether case embedding is enabled and build the index map.

    Parameters
    ----------
    model_params : dict
        Model parameters containing ``model_dict``.

    Returns
    -------
    do_case_embd : bool
        Whether case embedding is enabled.
    case_embd_index : dict
        Mapping from model key to case index (sorted alphabetically).
    """
    assert "model_dict" in model_params, (
        "Only support setting case embedding for multi-task model!"
    )
    model_keys = list(model_params["model_dict"])
    sorted_model_keys = sorted(model_keys)
    numb_case_embd_list = [
        model_params["model_dict"][mk].get("fitting_net", {}).get("dim_case_embd", 0)
        for mk in sorted_model_keys
    ]
    if not all(item == numb_case_embd_list[0] for item in numb_case_embd_list):
        raise ValueError(
            "All models must have the same dimension of case embedding, "
            f"while the settings are: {numb_case_embd_list}"
        )
    if numb_case_embd_list[0] == 0:
        return False, {}
    case_embd_index = {mk: idx for idx, mk in enumerate(sorted_model_keys)}
    return True, case_embd_index
