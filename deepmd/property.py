# SPDX-License-Identifier: LGPL-3.0-or-later
"""High-level property prediction helpers built on existing DeePMD-kit flows.

This module intentionally does not introduce a new command surface. Instead, it
builds standard property-training inputs and delegates to the existing PyTorch
training / freezing / inference entrypoints.

The high-level API is designed to feel closer to unimol-tools:
- ``PropertyTrainer(...).fit()`` for training/fine-tuning
- ``PropertyPredictor(load_model=...).predict(...)`` for inference

Here ``model_name`` is interpreted as the pretrained model name / alias.
"""

from __future__ import (
    annotations,
)

import copy
import json
import logging
import tempfile
from pathlib import (
    Path,
)
from typing import (
    Any,
)

from deepmd.infer.deep_property import (
    DeepProperty,
)
from deepmd.pretrained.download import (
    resolve_model_path,
)
from deepmd.pretrained.registry import (
    available_model_names,
)

log = logging.getLogger(__name__)

DEFAULT_PROPERTY_TEMPLATE: dict[str, Any] = {
    "model": {
        "type_map": [],
        "descriptor": {
            "type": "dpa1",
            "sel": 120,
            "rcut_smth": 0.5,
            "rcut": 6.0,
            "neuron": [25, 50, 100],
            "tebd_dim": 8,
            "axis_neuron": 16,
            "type_one_side": True,
            "attn": 128,
            "attn_layer": 0,
            "attn_dotr": True,
            "attn_mask": False,
            "activation_function": "tanh",
            "scaling_factor": 1.0,
            "normalize": True,
            "temperature": 1.0,
        },
        "fitting_net": {
            "type": "property",
            "intensive": True,
            "task_dim": 1,
            "property_name": "property",
            "neuron": [240, 240, 240],
            "resnet_dt": True,
            "seed": 1,
        },
    },
    "learning_rate": {
        "type": "exp",
        "decay_steps": 5000,
        "start_lr": 2.0e-4,
        "stop_lr": 3.51e-8,
    },
    "loss": {
        "type": "property",
        "metric": ["mae"],
        "loss_func": "smooth_mae",
        "beta": 1.0,
    },
    "training": {
        "training_data": {
            "systems": [],
            "batch_size": 1,
        },
        "validation_data": {
            "systems": [],
            "batch_size": 1,
        },
        "numb_steps": 1000000,
        "gradient_max_norm": 5.0,
        "seed": 10,
        "disp_file": "lcurve.out",
        "disp_freq": 100,
        "save_freq": 2000,
    },
}


def resolve_model_name(
    model_name: str | None,
    *,
    cache_dir: str | Path | None = None,
) -> str | None:
    """Resolve a pretrained model name / alias to a local file path.

    If the provided string matches a built-in pretrained alias, download/resolve
    it to a local file path. Otherwise it is returned unchanged, so callers may
    also pass a local checkpoint path here.
    """
    if model_name is None:
        return None
    if model_name in available_model_names():
        return str(
            resolve_model_path(
                model_name,
                cache_dir=Path(cache_dir) if cache_dir is not None else None,
                logger=log,
            )
        )
    return model_name


def resolve_finetune_model(
    finetune_model: str | None,
    *,
    cache_dir: str | Path | None = None,
) -> str | None:
    """Backward-compatible alias of :func:`resolve_model_name`."""
    return resolve_model_name(finetune_model, cache_dir=cache_dir)


class PropertyTrainer:
    """Draft high-level property trainer.

    The public constructor is intentionally shaped closer to unimol-tools.
    In this draft implementation, actual training currently runs through the
    existing DeePMD-Kit PyTorch property workflow after low-level systems have
    been prepared.

    Use :meth:`from_systems` to provide already-prepared DeePMD property
    datasets. High-level ``fit(data=...)`` support for SMILES/CSV inputs will be
    added in follow-up work.
    """

    def __init__(
        self,
        *,
        task: str = "regression",
        data_type: str = "molecule",
        epochs: int = 10,
        learning_rate: float = 2.0e-4,
        batch_size: int = 16,
        metrics: list[str] | None = None,
        save_path: str = "out.json",
        smiles_col: str = "SMILES",
        target_cols: list[str] | str | None = None,
        target_col_prefix: str = "TARGET",
        remove_hs: bool = False,
        model_name: str | None = None,
        load_model_dir: str | None = None,
        cache_dir: str | Path | None = None,
        seed: int = 10,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.data_type = data_type
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.metrics = metrics
        self.save_path = save_path
        self.smiles_col = smiles_col
        self.target_cols = target_cols
        self.target_col_prefix = target_col_prefix
        self.remove_hs = remove_hs
        self.model_name = model_name
        self.load_model_dir = load_model_dir
        self.cache_dir = cache_dir
        self.seed = seed
        self.extra_kwargs = kwargs

        self._system_options: dict[str, Any] | None = None

    @classmethod
    def from_systems(
        cls,
        *,
        type_map: list[str],
        train_systems: list[str],
        valid_systems: list[str] | None = None,
        property_name: str = "property",
        task_dim: int = 1,
        intensive: bool = True,
        sel: int = 120,
        rcut: float = 6.0,
        rcut_smth: float = 0.5,
        validation_batch_size: int | None = None,
        numb_steps: int = 1000000,
        stop_lr: float = 3.51e-8,
        decay_steps: int = 5000,
        disp_file: str = "lcurve.out",
        disp_freq: int = 100,
        save_freq: int = 2000,
        model_branch: str = "",
        skip_neighbor_stat: bool = False,
        use_pretrain_script: bool | None = None,
        force_load: bool = False,
        **kwargs: Any,
    ) -> PropertyTrainer:
        trainer = cls(**kwargs)
        trainer._system_options = {
            "type_map": type_map,
            "train_systems": train_systems,
            "valid_systems": valid_systems,
            "property_name": property_name,
            "task_dim": task_dim,
            "intensive": intensive,
            "sel": sel,
            "rcut": rcut,
            "rcut_smth": rcut_smth,
            "validation_batch_size": validation_batch_size,
            "numb_steps": numb_steps,
            "stop_lr": stop_lr,
            "decay_steps": decay_steps,
            "disp_file": disp_file,
            "disp_freq": disp_freq,
            "save_freq": save_freq,
            "model_branch": model_branch,
            "skip_neighbor_stat": skip_neighbor_stat,
            "use_pretrain_script": use_pretrain_script,
            "force_load": force_load,
        }
        return trainer

    def build_input(self) -> dict[str, Any]:
        """Build a standard property-training input dict from prepared systems."""
        if self._system_options is None:
            raise RuntimeError(
                "No prepared systems have been attached. "
                "Use PropertyTrainer.from_systems(...) in this draft implementation."
            )

        sysopt = self._system_options
        type_map = sysopt["type_map"]
        train_systems = sysopt["train_systems"]
        valid_systems = sysopt["valid_systems"]
        task_dim = sysopt["task_dim"]

        if not type_map:
            raise ValueError("type_map must not be empty")
        if not train_systems:
            raise ValueError("train_systems must not be empty")
        if task_dim < 1:
            raise ValueError("task_dim must be a positive integer")
        if self.task != "regression":
            raise NotImplementedError(
                "PropertyTrainer currently supports only task='regression'"
            )
        if self.data_type != "molecule":
            raise NotImplementedError(
                "PropertyTrainer currently supports only data_type='molecule'"
            )

        config = copy.deepcopy(DEFAULT_PROPERTY_TEMPLATE)
        config["model"]["type_map"] = type_map
        config["model"]["descriptor"]["sel"] = sysopt["sel"]
        config["model"]["descriptor"]["rcut"] = sysopt["rcut"]
        config["model"]["descriptor"]["rcut_smth"] = sysopt["rcut_smth"]
        config["model"]["fitting_net"]["property_name"] = sysopt["property_name"]
        config["model"]["fitting_net"]["task_dim"] = task_dim
        config["model"]["fitting_net"]["intensive"] = sysopt["intensive"]
        config["learning_rate"]["decay_steps"] = sysopt["decay_steps"]
        config["learning_rate"]["start_lr"] = self.learning_rate
        config["learning_rate"]["stop_lr"] = sysopt["stop_lr"]
        if self.metrics is not None:
            config["loss"]["metric"] = self.metrics
        config["training"]["training_data"]["systems"] = train_systems
        config["training"]["training_data"]["batch_size"] = self.batch_size
        config["training"]["validation_data"]["systems"] = (
            valid_systems if valid_systems is not None else train_systems
        )
        config["training"]["validation_data"]["batch_size"] = (
            sysopt["validation_batch_size"]
            if sysopt["validation_batch_size"] is not None
            else self.batch_size
        )
        config["training"]["numb_steps"] = sysopt["numb_steps"]
        config["training"]["seed"] = self.seed
        config["training"]["disp_file"] = sysopt["disp_file"]
        config["training"]["disp_freq"] = sysopt["disp_freq"]
        config["training"]["save_freq"] = sysopt["save_freq"]
        return config

    def write_input(self, path: str | Path) -> Path:
        """Write the generated input JSON to ``path`` and return it."""
        path = Path(path)
        path.write_text(json.dumps(self.build_input(), indent=4) + "\n")
        return path

    def fit(self, data: Any | None = None) -> Path:
        """Train the property model.

        Parameters
        ----------
        data
            Placeholder for future high-level molecule/CSV/SMILES inputs.
            In this draft implementation, passing ``data`` is not yet supported;
            use :meth:`from_systems` to construct the trainer from prepared
            DeePMD property systems.

        Returns
        -------
        Path
            The output JSON path written by the underlying training flow.
        """
        if data is not None:
            raise NotImplementedError(
                "High-level fit(data=...) support is not implemented in this draft. "
                "Use PropertyTrainer.from_systems(...) first."
            )

        from deepmd.pt.entrypoints.main import train as pt_train

        config = self.build_input()
        finetune_model = resolve_model_name(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        if finetune_model is None and self.load_model_dir is not None:
            finetune_model = resolve_model_name(
                self.load_model_dir,
                cache_dir=self.cache_dir,
            )
        use_pretrain_script = self._system_options["use_pretrain_script"]
        if use_pretrain_script is None:
            use_pretrain_script = finetune_model is not None

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="deepmd_property_",
            delete=False,
        ) as fp:
            json.dump(config, fp, indent=4)
            fp.write("\n")
            input_file = fp.name

        try:
            pt_train(
                input_file=input_file,
                init_model=None,
                restart=None,
                finetune=finetune_model,
                init_frz_model=None,
                model_branch=self._system_options["model_branch"],
                skip_neighbor_stat=self._system_options["skip_neighbor_stat"],
                use_pretrain_script=use_pretrain_script,
                force_load=self._system_options["force_load"],
                output=self.save_path,
            )
        finally:
            Path(input_file).unlink(missing_ok=True)

        return Path(self.save_path)


class PropertyPredictor:
    """A thin wrapper around :class:`deepmd.infer.deep_property.DeepProperty`."""

    def __init__(self, load_model: str) -> None:
        self.model = load_model
        self._predictor = DeepProperty(load_model)

    def predict(
        self,
        coords: Any,
        cells: Any,
        atom_types: Any,
        *,
        atomic: bool = False,
        fparam: Any | None = None,
        aparam: Any | None = None,
        mixed_type: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Predict properties for the provided structures."""
        return self._predictor.eval(
            coords,
            cells,
            atom_types,
            atomic=atomic,
            fparam=fparam,
            aparam=aparam,
            mixed_type=mixed_type,
            **kwargs,
        )


__all__ = [
    "DEFAULT_PROPERTY_TEMPLATE",
    "PropertyPredictor",
    "PropertyTrainer",
    "resolve_finetune_model",
    "resolve_model_name",
]
