# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training entrypoint for the pt_expt backend."""

import argparse
import logging
import os
from dataclasses import (
    replace,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import h5py

from deepmd.dpmodel.train import (
    AbstractTrainEntrypoint,
    TrainEntrypointOptions,
    TrainingTaskConfig,
    iter_training_task_configs,
    make_task_maps,
    print_data_summaries,
)
from deepmd.dpmodel.utils.lmdb_data import (
    is_lmdb,
)
from deepmd.pt_expt.train import (
    training,
)
from deepmd.pt_expt.utils.lmdb_dataset import (
    LmdbDataSystem,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
    get_data,
    process_systems,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.summary import SummaryPrinter as BaseSummaryPrinter

log = logging.getLogger(__name__)

_PT_EXPT_MODEL_SUFFIXES = (".pt", ".pte", ".pt2")


def _ensure_pt_expt_model_suffix(model_path: str | None) -> str | None:
    """Append the default checkpoint suffix when a model path is a prefix."""
    if model_path is not None and not model_path.endswith(_PT_EXPT_MODEL_SUFFIXES):
        return f"{model_path}.pt"
    return model_path


def _update_changed_model_tensors(
    target_state_dict: dict[str, Any],
    source_state_dict: dict[str, Any],
) -> None:
    """Copy changed tensors into an existing state dict without breaking aliases."""
    import torch

    for key, source_value in source_state_dict.items():
        if key == "_extra_state":
            continue
        if key not in target_state_dict:
            target_state_dict[key] = (
                source_value.detach().clone()
                if torch.is_tensor(source_value)
                else source_value
            )
            continue
        target_value = target_state_dict[key]
        if torch.is_tensor(target_value) and torch.is_tensor(source_value):
            if (
                target_value.shape == source_value.shape
                and target_value.dtype == source_value.dtype
            ):
                if not torch.equal(target_value, source_value):
                    target_value.copy_(source_value)
            else:
                target_state_dict[key] = source_value.detach().clone()
        elif target_value != source_value:
            target_state_dict[key] = source_value


def _detect_lmdb_path(systems_raw: Any) -> str | None:
    """Return the LMDB path when ``systems_raw`` is a scalar LMDB string.

    Returns ``None`` for non-LMDB inputs. Raises ``ValueError`` if
    ``systems_raw`` is a list containing any LMDB path, so both
    ``_get_neighbor_stat_data`` and ``_build_data_system`` fail with the
    same clear message instead of the opaque error from
    :func:`process_systems` / :class:`DeepmdData`.
    """
    if isinstance(systems_raw, str) and is_lmdb(systems_raw):
        return systems_raw
    if isinstance(systems_raw, list) and any(
        isinstance(s, str) and is_lmdb(s) for s in systems_raw
    ):
        raise ValueError(
            "LMDB datasets must be passed as a scalar 'systems' string "
            "(e.g. 'systems': '/path/to/data.lmdb'); list-form systems "
            "with LMDB paths are not supported."
        )
    return None


def _get_neighbor_stat_data(
    dataset_params: dict[str, Any],
    type_map: list[str] | None,
) -> Any:
    """Return a data proxy suitable for ``BaseModel.update_sel`` (neighbor stat).

    Routes a scalar LMDB ``systems`` path through dpmodel's
    ``make_neighbor_stat_data``; falls back to the legacy ``get_data`` for
    npy/HDF5 directories.
    """
    lmdb_path = _detect_lmdb_path(dataset_params.get("systems"))
    if lmdb_path is not None:
        from deepmd.dpmodel.utils.lmdb_data import (
            make_neighbor_stat_data,
        )

        return make_neighbor_stat_data(lmdb_path, type_map)
    return get_data(dataset_params, 0, type_map, None)


def _build_data_system(
    dataset_params: dict[str, Any],
    type_map: list[str],
    seed: int | None = None,
) -> DeepmdDataSystem | LmdbDataSystem:
    """Build a data system from dataset config, routing LMDB paths to LmdbDataSystem.

    A scalar ``systems`` value pointing at an LMDB directory triggers the
    LMDB adapter; otherwise we fall through to the legacy
    :class:`DeepmdDataSystem` path with system expansion.
    """
    systems_raw = dataset_params["systems"]
    lmdb_path = _detect_lmdb_path(systems_raw)
    if lmdb_path is not None:
        return LmdbDataSystem(
            lmdb_path=lmdb_path,
            type_map=type_map,
            batch_size=dataset_params["batch_size"],
            auto_prob_style=dataset_params.get("auto_prob"),
            seed=seed,
        )
    systems = process_systems(
        systems_raw,
        patterns=dataset_params.get("rglob_patterns", None),
    )
    return DeepmdDataSystem(
        systems=systems,
        batch_size=dataset_params["batch_size"],
        test_size=1,
        type_map=type_map,
        trn_all_set=True,
        sys_probs=dataset_params.get("sys_probs", None),
        auto_prob_style=dataset_params.get("auto_prob", "prob_sys_size"),
    )


def _ensure_stat_file_path(stat_file_path: str | None) -> DPPath | None:
    """Create a stat-file target and return a DPPath wrapper."""
    if stat_file_path is None:
        return None
    path = Path(stat_file_path)
    if not path.exists():
        if stat_file_path.endswith((".h5", ".hdf5")):
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w"):
                pass
        else:
            path.mkdir(parents=True, exist_ok=True)
    return DPPath(stat_file_path, "a")


def get_trainer(
    config: dict[str, Any],
    init_model: str | None = None,
    restart_model: str | None = None,
    finetune_model: str | None = None,
    finetune_links: dict | None = None,
    shared_links: dict | None = None,
) -> training.Trainer:
    """Build a :class:`training.Trainer` from a normalised config."""
    training_params = config["training"]
    multi_task = "model_dict" in config["model"]

    data_seed = training_params.get("seed", None)

    def factory(
        task_config: TrainingTaskConfig,
    ) -> tuple[DeepmdDataSystem | LmdbDataSystem, Any | None, DPPath | None]:
        type_map = list(task_config.model_params["type_map"])
        train_data = _build_data_system(
            dict(task_config.training_data_params), type_map, seed=data_seed
        )
        validation_data = None
        if task_config.validation_data_params is not None:
            validation_data = _build_data_system(
                dict(task_config.validation_data_params), type_map, seed=data_seed
            )
        return (
            train_data,
            validation_data,
            _ensure_stat_file_path(task_config.stat_file),
        )

    train_data_map, validation_data_map, stat_file_path_map = make_task_maps(
        config, factory
    )
    print_data_summaries(train_data_map, validation_data_map)
    if multi_task:
        train_data = train_data_map
        validation_data = validation_data_map
        stat_file_path = stat_file_path_map
    else:
        task_key = next(iter(train_data_map))
        train_data = train_data_map[task_key]
        validation_data = validation_data_map[task_key]
        stat_file_path = stat_file_path_map[task_key]

    trainer = training.Trainer(
        config,
        train_data,
        stat_file_path=stat_file_path,
        validation_data=validation_data,
        init_model=init_model,
        restart_model=restart_model,
        finetune_model=finetune_model,
        finetune_links=finetune_links,
        shared_links=shared_links,
    )
    return trainer


class SummaryPrinter(BaseSummaryPrinter):
    """Summary printer for pt_expt."""

    def is_built_with_cuda(self) -> bool:
        """Check if PyTorch was built with CUDA."""
        import torch

        return torch.version.cuda is not None

    def is_built_with_rocm(self) -> bool:
        """Check if PyTorch was built with ROCm."""
        import torch

        return torch.version.hip is not None

    def get_compute_device(self) -> str:
        """Get the selected compute device."""
        from deepmd.pt_expt.utils.env import (
            DEVICE,
        )

        return str(DEVICE)

    def get_ngpus(self) -> int:
        """Get the number of visible CUDA devices."""
        import torch

        return torch.cuda.device_count()

    def get_backend_info(self) -> dict:
        """Get backend information."""
        import torch

        return {
            "Backend": "PyTorch Experimental",
            "PT Ver": f"v{torch.__version__}-g{torch.version.git_version[:11]}",
        }

    def get_device_name(self) -> str | None:
        """Return the current CUDA device name when available."""
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(torch.cuda.current_device())
        return None


class PTExptTrainEntrypoint(AbstractTrainEntrypoint):
    """pt_expt implementation of the common training entrypoint pipeline."""

    def __init__(self) -> None:
        self.finetune_links: dict[str, Any] | None = None
        self.shared_links: dict[str, Any] | None = None
        self._owns_process_group = False

    def prepare_options(
        self,
        options: TrainEntrypointOptions,
    ) -> TrainEntrypointOptions:
        """Normalize checkpoint prefixes accepted by the train CLI."""
        return replace(
            options,
            init_model=_ensure_pt_expt_model_suffix(options.init_model),
            restart=_ensure_pt_expt_model_suffix(options.restart),
        )

    def preprocess_config(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
    ) -> dict[str, Any]:
        """Apply pt_expt multi-task, finetune, and pretrained-model preprocessing."""
        import torch

        from deepmd.pt_expt.utils.env import (
            DEVICE,
        )

        self.finetune_links = None
        self.shared_links = None

        if self.is_multi_task(config):
            from deepmd.pt_expt.utils.multi_task import (
                preprocess_shared_params,
            )

            config["model"], self.shared_links = preprocess_shared_params(
                config["model"]
            )
            if "RANDOM" in config["model"]["model_dict"]:
                raise ValueError("Model name can not be 'RANDOM' in multi-task mode!")

        if options.finetune is not None:
            from deepmd.pt_expt.utils.finetune import (
                get_finetune_rules,
            )

            config["model"], self.finetune_links = get_finetune_rules(
                options.finetune,
                config["model"],
                model_branch=options.model_branch,
                change_model_params=options.use_pretrain_script,
            )

        if options.init_model is not None and options.use_pretrain_script:
            init_state_dict = torch.load(
                options.init_model, map_location=DEVICE, weights_only=True
            )
            if "model" in init_state_dict:
                init_state_dict = init_state_dict["model"]
            config["model"] = init_state_dict["_extra_state"]["model_params"]

        return config

    def update_neighbor_stat(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
        *,
        multi_task: bool,
    ) -> tuple[dict[str, Any], None]:
        """Update pt_expt descriptor selections from neighbor statistics."""
        log.info(
            "Calculate neighbor statistics... "
            "(add --skip-neighbor-stat to skip this step)"
        )
        from deepmd.pt_expt.model import (
            BaseModel,
        )

        for task_config in iter_training_task_configs(config):
            type_map = task_config.model_params.get("type_map")
            train_data = _get_neighbor_stat_data(
                dict(task_config.training_data_params), type_map
            )
            updated_model_params, _ = BaseModel.update_sel(
                train_data, type_map, dict(task_config.model_params)
            )
            if multi_task:
                config["model"]["model_dict"][task_config.key] = updated_model_params
            else:
                config["model"] = updated_model_params
        return config, None

    def print_summary(self) -> None:
        """Print pt_expt backend summary."""
        SummaryPrinter()()

    def setup_run(
        self,
        options: TrainEntrypointOptions,
        config: dict[str, Any],
    ) -> None:
        """Initialize pt_expt distributed training when launched by torchrun/srun."""
        self._owns_process_group = False
        if os.environ.get("LOCAL_RANK") is not None:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return
            dist.init_process_group(backend="cuda:nccl,cpu:gloo")
            self._owns_process_group = True

    def teardown_run(
        self,
        options: TrainEntrypointOptions,
        config: dict[str, Any],
    ) -> None:
        """Destroy the pt_expt distributed process group if this entrypoint made one."""
        if not self._owns_process_group:
            return
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        self._owns_process_group = False

    def run_training(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
        neighbor_stat: Any,
    ) -> None:
        """Build and run the pt_expt trainer."""
        trainer = get_trainer(
            config,
            options.init_model,
            options.restart,
            finetune_model=options.finetune,
            finetune_links=self.finetune_links,
            shared_links=self.shared_links,
        )
        trainer.run()


def train(
    input_file: str,
    init_model: str | None = None,
    restart: str | None = None,
    finetune: str | None = None,
    model_branch: str = "",
    use_pretrain_script: bool = False,
    skip_neighbor_stat: bool = False,
    output: str = "out.json",
) -> None:
    """Run training with the pt_expt backend.

    Parameters
    ----------
    input_file : str
        Path to the JSON configuration file.
    init_model : str or None
        Path to a checkpoint to initialise weights from.
    restart : str or None
        Path to a checkpoint to restart training from.
    finetune : str or None
        Path to a pretrained checkpoint to fine-tune from.
    model_branch : str
        Branch to select from a multi-task pretrained model.
    use_pretrain_script : bool
        If True, copy descriptor/fitting params from the pretrained model.
    skip_neighbor_stat : bool
        Skip neighbour statistics calculation.
    output : str
        Where to dump the normalised config.
    """
    PTExptTrainEntrypoint().run(
        TrainEntrypointOptions(
            input_file=input_file,
            output=output,
            init_model=init_model,
            restart=restart,
            finetune=finetune,
            model_branch=model_branch,
            use_pretrain_script=use_pretrain_script,
            skip_neighbor_stat=skip_neighbor_stat,
        )
    )


def freeze(
    model: str,
    output: str = "frozen_model.pte",
    head: str | None = None,
    lower_kind: str = "nlist",
) -> None:
    """Freeze a pt_expt checkpoint into a .pte exported model.

    Parameters
    ----------
    model : str
        Path to the checkpoint file (.pt).
    output : str
        Path for the output .pte file.
    head : str or None
        Head to freeze in multi-task mode.
    lower_kind : str
        Lower-level export form: ``"nlist"`` (default, dense neighbor-list lower)
        or ``"graph"`` (NeighborGraph edge-list lower). ``"graph"`` is only valid
        for graph-eligible models (``mixed_types`` and ``uses_graph_lower``:
        dpa1/se_atten with concat type embedding and no ``exclude_types``,
        attention layers included) and selects the C++ graph inference path;
        the per-atom virial is enabled for it (near-free in the graph path:
        one extra scatter off the shared single backward). NOTE: for
        ``smooth_type_embedding=True`` the carry-all graph attention
        intentionally drops the dense layout's sel-padding terms from the
        softmax denominator, so graph-form results are sel-independent and
        differ from the legacy dense lower by up to ~1e-4 (see
        ``DescrptDPA1.call_graph``).
    """
    import torch

    from deepmd.pt_expt.model.get_model import (
        get_model,
    )
    from deepmd.pt_expt.train.wrapper import (
        ModelWrapper,
    )
    from deepmd.pt_expt.utils.env import (
        DEVICE,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    state_dict = torch.load(model, map_location=DEVICE, weights_only=True)
    if "model" in state_dict:
        state_dict = state_dict["model"]

    extra_state = state_dict.get("_extra_state")
    if not isinstance(extra_state, dict) or "model_params" not in extra_state:
        raise ValueError(
            f"Unsupported checkpoint format at '{model}': missing "
            "'_extra_state.model_params' in model state dict."
        )
    model_params = extra_state["model_params"]

    multi_task = "model_dict" in model_params
    if multi_task:
        if head is None:
            raise ValueError(
                "Multi-task model requires --head to specify which model to freeze. "
                f"Available heads: {list(model_params['model_dict'].keys())}"
            )
        if head not in model_params["model_dict"]:
            raise ValueError(
                f"Head '{head}' not found. "
                f"Available: {list(model_params['model_dict'].keys())}"
            )
        # Build full multi-task wrapper, load weights, extract single head
        model_dict = {}
        for key in model_params["model_dict"]:
            from copy import (
                deepcopy,
            )

            model_dict[key] = get_model(deepcopy(model_params["model_dict"][key]))
        wrapper = ModelWrapper(model_dict)
        wrapper.load_state_dict(state_dict)

        m = wrapper.model[head]
        single_model_params = model_params["model_dict"][head]
    else:
        m = get_model(model_params)
        wrapper = ModelWrapper(m)
        wrapper.load_state_dict(state_dict)
        single_model_params = model_params

    m.eval()

    # The graph lower is opt-in and only valid for graph-eligible models
    # (dpa1 with concat tebd and no type exclusion; attention layers included
    # -- the carry-all pair enumeration exports via unbacked SymInts). Fail
    # fast with a clear message rather than emitting a broken .pt2. Enable the
    # per-atom virial for the graph form -- it is near-free there (one extra
    # scatter off the single shared backward).
    do_atomic_virial = False
    if lower_kind == "graph":
        from deepmd.pt_expt.train.training import (
            _model_uses_graph_lower,
        )

        if not _model_uses_graph_lower(m):
            raise ValueError(
                "lower_kind='graph' requires a graph-eligible model "
                "(mixed_types and a descriptor exposing uses_graph_lower()==True, "
                "currently dpa1 with tebd_input_mode='concat' and no "
                "exclude_types). Use lower_kind='nlist' for this model."
            )
        do_atomic_virial = True

    model_dict_serialized = m.serialize()
    deserialize_to_file(
        output,
        {"model": model_dict_serialized, "model_def_script": single_model_params},
        do_atomic_virial=do_atomic_virial,
        lower_kind=lower_kind,
    )
    log.info("Saved frozen model to %s (lower_kind=%s)", output, lower_kind)


def change_bias(
    input_file: str,
    mode: str = "change",
    bias_value: list | None = None,
    datafile: str | None = None,
    system: str = ".",
    numb_batch: int = 0,
    model_branch: str | None = None,
    output: str | None = None,
) -> None:
    """Change the output bias of a pt_expt model.

    Parameters
    ----------
    input_file : str
        Path to the model file (.pt checkpoint or .pte frozen model).
    mode : str
        ``"change"`` or ``"set"``.
    bias_value : list or None
        User-defined bias values (one per type).
    datafile : str or None
        File listing data system paths.
    system : str
        Data system path (used when *datafile* is None).
    numb_batch : int
        Number of batches for statistics (0 = all).
    model_branch : str or None
        Branch name for multi-task models.
    output : str or None
        Output file path.
    """
    import torch

    from deepmd.common import (
        expand_sys_str,
    )
    from deepmd.dpmodel.common import (
        to_numpy_array,
    )
    from deepmd.pt_expt.model.get_model import (
        get_model,
    )
    from deepmd.pt_expt.train.training import (
        get_additional_data_requirement,
        get_loss,
        model_change_out_bias,
    )
    from deepmd.pt_expt.train.wrapper import (
        ModelWrapper,
    )
    from deepmd.pt_expt.utils.env import (
        DEVICE,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
        serialize_from_file,
    )
    from deepmd.pt_expt.utils.stat import (
        make_stat_input,
    )

    if input_file.endswith(".pt"):
        old_state_dict = torch.load(input_file, map_location=DEVICE, weights_only=True)
        if "model" in old_state_dict:
            model_state_dict = old_state_dict["model"]
        else:
            model_state_dict = old_state_dict
        extra_state = model_state_dict.get("_extra_state")
        if not isinstance(extra_state, dict) or "model_params" not in extra_state:
            raise ValueError(
                f"Unsupported checkpoint format at '{input_file}': missing "
                "'_extra_state.model_params' in model state dict."
            )
        model_params = extra_state["model_params"]
    elif input_file.endswith((".pte", ".pt2")):
        pte_data = serialize_from_file(input_file)
        from deepmd.pt_expt.model.model import (
            BaseModel,
        )

        model_to_change = BaseModel.deserialize(pte_data["model"])
        model_params = pte_data.get("model_def_script")
    else:
        raise RuntimeError(
            "The model provided must be a checkpoint file with a .pt extension "
            "or a frozen model with a .pte/.pt2 extension"
        )

    if mode == "change":
        bias_adjust_mode = "change-by-statistic"
    elif mode == "set":
        bias_adjust_mode = "set-by-statistic"
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Expected 'change' or 'set'.")

    if input_file.endswith(".pt"):
        multi_task = "model_dict" in model_params
        if multi_task:
            raise NotImplementedError(
                "Multi-task change-bias is not yet supported for the pt_expt backend."
            )
        type_map = model_params["type_map"]
        model = get_model(model_params)
        wrapper = ModelWrapper(model)
        wrapper.load_state_dict(model_state_dict)
        model_to_change = model

    if input_file.endswith((".pte", ".pt2")):
        type_map = model_to_change.get_type_map()

    if bias_value is not None:
        if "energy" not in model_to_change.model_output_type():
            raise ValueError("User-defined bias is only available for energy models!")
        if len(bias_value) != len(type_map):
            raise ValueError(
                f"The number of elements in the bias ({len(bias_value)}) must match "
                f"the number of types in type_map ({len(type_map)}): {type_map}."
            )
        old_bias = model_to_change.get_out_bias()
        bias_to_set = torch.tensor(
            bias_value, dtype=old_bias.dtype, device=old_bias.device
        ).view(old_bias.shape)
        model_to_change.set_out_bias(bias_to_set)
        log.info(
            f"Change output bias of {type_map!s} "
            f"from {to_numpy_array(old_bias).reshape(-1)!s} "
            f"to {to_numpy_array(bias_to_set).reshape(-1)!s}."
        )
    else:
        if datafile is not None:
            with open(datafile) as datalist:
                all_sys = datalist.read().splitlines()
        else:
            all_sys = expand_sys_str(system)
        data_systems = process_systems(all_sys)
        data = DeepmdDataSystem(
            systems=data_systems,
            batch_size=1,
            test_size=1,
            rcut=model_to_change.get_rcut(),
            type_map=type_map,
        )
        mock_loss = get_loss({"inference": True}, 1.0, len(type_map), model_to_change)
        data.add_data_requirements(mock_loss.label_requirement)
        data.add_data_requirements(get_additional_data_requirement(model_to_change))
        if numb_batch != 0:
            nbatches = numb_batch
        else:
            # Cap at the minimum across systems so no system wraps and
            # overweights short systems (matching PT behavior).
            nbatches = min(data.get_nbatches())
        sampled_data = make_stat_input(data, nbatches)
        model_to_change = model_change_out_bias(
            model_to_change, sampled_data, _bias_adjust_mode=bias_adjust_mode
        )

    if input_file.endswith(".pt"):
        output_path = (
            output if output is not None else input_file.replace(".pt", "_updated.pt")
        )
        wrapper = ModelWrapper(model_to_change)
        _update_changed_model_tensors(model_state_dict, wrapper.state_dict())
        torch.save(old_state_dict, output_path)
    elif input_file.endswith((".pte", ".pt2")):
        output_path = (
            output
            if output is not None
            else input_file.replace(".pte", "_updated.pte").replace(
                ".pt2", "_updated.pt2"
            )
        )
        model_dict = model_to_change.serialize()
        deserialize_to_file(
            output_path, {"model": model_dict, "model_def_script": model_params}
        )
    log.info(f"Saved model to {output_path}")


def main(args: list[str] | argparse.Namespace | None = None) -> None:
    """Entry point for the pt_expt backend CLI.

    Parameters
    ----------
    args : list[str] | argparse.Namespace | None
        Command-line arguments or pre-parsed namespace.
    """
    from deepmd.loggers.loggers import (
        set_log_handles,
    )
    from deepmd.main import (
        parse_args,
    )

    if not isinstance(args, argparse.Namespace):
        FLAGS = parse_args(args=args)
    else:
        FLAGS = args

    set_log_handles(
        FLAGS.log_level,
        Path(FLAGS.log_path) if FLAGS.log_path else None,
        mpi_log=None,
    )
    log.info("DeePMD-kit backend: pt_expt (PyTorch Exportable)")

    if FLAGS.command == "train":
        train(
            input_file=FLAGS.INPUT,
            init_model=FLAGS.init_model,
            restart=FLAGS.restart,
            finetune=FLAGS.finetune,
            model_branch=FLAGS.model_branch,
            use_pretrain_script=FLAGS.use_pretrain_script,
            skip_neighbor_stat=FLAGS.skip_neighbor_stat,
            output=FLAGS.output,
        )
    elif FLAGS.command == "freeze":
        if Path(FLAGS.checkpoint_folder).is_dir():
            checkpoint_path = Path(FLAGS.checkpoint_folder)
            # pt_expt training saves a symlink "model.ckpt.pt" → latest ckpt
            default_ckpt = checkpoint_path / "model.ckpt.pt"
            if default_ckpt.exists():
                FLAGS.model = str(default_ckpt)
            else:
                raise FileNotFoundError(
                    f"Cannot find checkpoint in '{checkpoint_path}'. "
                    "Expected 'model.ckpt.pt' (created by pt_expt training)."
                )
        else:
            model_path = Path(FLAGS.checkpoint_folder)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint path '{model_path}' does not exist."
                )
            FLAGS.model = str(model_path)
        _lower_kind = getattr(FLAGS, "lower_kind", "nlist")
        if not FLAGS.output.endswith((".pte", ".pt2")):
            # Default suffix: .pt2 for the graph export (an AOTI .pt2 archive is
            # what the C++ graph path consumes), .pte otherwise. Explicit user
            # .pte / .pt2 suffixes are preserved for both.
            _default_suffix = ".pt2" if _lower_kind == "graph" else ".pte"
            FLAGS.output = str(Path(FLAGS.output).with_suffix(_default_suffix))
        freeze(
            model=FLAGS.model,
            output=FLAGS.output,
            head=FLAGS.head,
            lower_kind=_lower_kind,
        )
    elif FLAGS.command == "change-bias":
        change_bias(
            input_file=FLAGS.INPUT,
            mode=FLAGS.mode,
            bias_value=FLAGS.bias_value,
            datafile=FLAGS.datafile,
            system=FLAGS.system,
            numb_batch=FLAGS.numb_batch,
            model_branch=FLAGS.model_branch,
            output=FLAGS.output,
        )
    elif FLAGS.command == "compress":
        from deepmd.pt_expt.entrypoints.compress import (
            enable_compression,
        )

        if not FLAGS.input.endswith((".pte", ".pt2")):
            FLAGS.input = str(Path(FLAGS.input).with_suffix(".pte"))
        if not FLAGS.output.endswith((".pte", ".pt2")):
            FLAGS.output = str(Path(FLAGS.output).with_suffix(".pte"))
        enable_compression(
            input_file=FLAGS.input,
            output=FLAGS.output,
            stride=FLAGS.step,
            extrapolate=FLAGS.extrapolate,
            check_frequency=FLAGS.frequency,
            training_script=FLAGS.training_script,
        )
    else:
        raise RuntimeError(
            f"Unsupported command '{FLAGS.command}' for the pt_expt backend."
        )
