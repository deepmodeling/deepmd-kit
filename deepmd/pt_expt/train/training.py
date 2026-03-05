# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training loop for the pt_expt backend.

Uses ``DeepmdDataSystem`` (numpy-based batch provider) instead of the
pt backend's ``DpLoaderSet`` + ``DataLoader``.  NumPy batches are
converted to torch tensors at the boundary.
"""

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

from deepmd.dpmodel.utils.batch import (
    normalize_batch,
    split_batch,
)
from deepmd.dpmodel.utils.learning_rate import (
    LearningRateExp,
)
from deepmd.loggers.training import (
    format_training_message_per_task,
)
from deepmd.pt_expt.loss import (
    EnergyLoss,
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

    Parameters
    ----------
    model : torch.nn.Module
        The (uncompiled) model.  Temporarily set to eval mode for tracing.
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
        extended_coord = extended_coord.detach().requires_grad_(True)
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

    if not was_training:
        model.eval()

    # The inductor backend does not propagate gradients through the
    # make_fx-decomposed autograd.grad ops (second-order gradients for
    # force training).  Use "aot_eager" which correctly preserves the
    # gradient chain while still benefiting from make_fx decomposition.
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
    ) -> None:
        resume_model = init_model or restart_model
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

        if not resuming:
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
            state_dict = torch.load(
                resume_model, map_location=DEVICE, weights_only=True
            )
            if "model" in state_dict:
                optimizer_state_dict = (
                    state_dict["optimizer"] if self.restart_training else None
                )
                state_dict = state_dict["model"]
            else:
                optimizer_state_dict = None

            self.start_step = (
                state_dict["_extra_state"]["train_infos"]["step"]
                if self.restart_training
                else 0
            )
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

        model = self.model

        # --- Estimate max_nall by sampling multiple batches ---
        n_sample = 20
        max_nall = 0
        best_sample: (
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict] | None
        ) = None

        for _ii in range(n_sample):
            inp, _ = self.get_data(is_train=True)
            coord = inp["coord"].detach()
            atype = inp["atype"].detach()
            box = inp.get("box")
            if box is not None:
                box = box.detach()

            nframes, nloc = atype.shape[:2]
            coord_np = coord.cpu().numpy().reshape(nframes, nloc, 3)
            atype_np = atype.cpu().numpy()
            box_np = box.cpu().numpy().reshape(nframes, 9) if box is not None else None

            if box_np is not None:
                coord_norm = normalize_coord(coord_np, box_np.reshape(nframes, 3, 3))
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
            "Estimated max_nall=%d for compiled model (sampled %d batches).",
            max_nall,
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

        compile_opts.pop("dynamic", None)  # always False for padded approach

        compiled_lower = _trace_and_compile(
            model,
            ext_coord,
            ext_atype,
            nlist_t,
            mapping_t,
            fparam,
            aparam,
            compile_opts,
        )

        self.wrapper.model = _CompiledModel(
            model, compiled_lower, max_nall, compile_opts
        )
        log.info(
            "Model compiled with padded nall=%d (tracing_mode=real, dynamic=False).",
            max_nall,
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
                wall_elapsed = time.time() - wall_start
                if self.timing_in_training:
                    step_time = t_end - t_start
                    log.info(
                        "step=%d  wall=%.2fs  step_time=%.4fs",
                        display_step_id,
                        wall_elapsed,
                        step_time,
                    )
                else:
                    log.info("step=%d  wall=%.2fs", display_step_id, wall_elapsed)

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
