# SPDX-License-Identifier: LGPL-3.0-or-later
"""High-level property training interface."""

from __future__ import (
    annotations,
)

import json
from pathlib import (
    Path,
)
from typing import (
    Any,
)

from deepmd_property_tools.config import (
    ConfigHandler,
)
from deepmd_property_tools.data import (
    DataHub,
)
from deepmd_property_tools.tasks import (
    Trainer,
)
from deepmd_property_tools.weights import (
    WeightHub,
)


class PropertyTrain:
    def __init__(
        self,
        task: str = "regression",
        property_name: str = "Property",
        property_col: str = "Property",
        save_path: str | Path = "./exp_property",
        epochs: int | None = None,
        batch_size: int | str | None = None,
        metrics: str | list[str] | None = None,
        data_type: str = "molecule",
        model_name: str = "dpa3",
        model_size: str = "5m",
        numb_steps: int | None = None,
        finetune: str | Path | None = None,
        nproc_per_node: int = 1,
        train_ratio: float = 0.9,
        mol_template: str = "id{row}.mol",
        smiles_col: str = "SMILES",
        overlap_tol: float = 1e-6,
        seed: int = 42,
        overwrite: bool = True,
        freeze: bool = False,
        use_pretrain_script: bool = False,
        skip_neighbor_stat: bool = False,
        force_load: bool = False,
        model_branch: str = "",
        input_updates: dict[str, Any] | None = None,
        **params: Any,
    ) -> None:
        if params:
            names = ", ".join(sorted(params))
            raise TypeError(f"Unexpected PropertyTrain argument(s): {names}")
        if task != "regression":
            raise ValueError(
                "DeePMD property tools currently support task='regression'"
            )
        if data_type != "molecule":
            raise ValueError(
                "DeePMD property tools currently support data_type='molecule'"
            )
        if model_name != "dpa3":
            raise ValueError(
                "DeePMD property tools currently support model_name='dpa3'"
            )
        self.task = task
        self.data_type = data_type
        self.model_name = model_name
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.metrics = metrics
        self.property_name = property_name
        self.property_col = property_col
        self.save_path = Path(save_path)
        self.numb_steps = (
            numb_steps if numb_steps is not None else self._epochs_to_steps(epochs)
        )
        self.finetune = (
            None
            if finetune is None
            else WeightHub(root=self.save_path.parent).get(finetune)
        )
        self.nproc_per_node = nproc_per_node
        self.train_ratio = train_ratio
        self.mol_template = mol_template
        self.smiles_col = smiles_col
        self.overlap_tol = overlap_tol
        self.seed = seed
        self.overwrite = overwrite
        self.freeze_model = freeze
        self.use_pretrain_script = use_pretrain_script
        self.skip_neighbor_stat = skip_neighbor_stat
        self.force_load = force_load
        self.model_branch = model_branch
        if input_updates is None:
            input_updates = {}
        if batch_size is not None:
            input_updates = ConfigHandler.merge(
                input_updates,
                {"training": {"training_data": {"batch_size": batch_size}}},
            )
        if metrics is not None:
            metric_list = [metrics] if isinstance(metrics, str) else list(metrics)
            input_updates = ConfigHandler.merge(
                input_updates, {"loss": {"metric": metric_list}}
            )
        self.input_updates = input_updates
        self.datahub: DataHub | None = None

    def fit(self, data: dict[str, Any] | str | Path) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.datahub = DataHub(
            data=data,
            is_train=True,
            save_path=self.save_path,
            property_name=self.property_name,
            property_col=self.property_col,
            train_ratio=self.train_ratio,
            mol_template=self.mol_template,
            smiles_col=self.smiles_col,
            overlap_tol=self.overlap_tol,
            seed=self.seed,
            overwrite=self.overwrite,
            numb_steps=self.numb_steps,
            input_updates=self.input_updates,
        )
        self._save_config()
        trainer = Trainer(
            save_path=self.save_path,
            finetune=self.finetune,
            nproc_per_node=self.nproc_per_node,
            freeze=self.freeze_model,
            use_pretrain_script=self.use_pretrain_script,
            skip_neighbor_stat=self.skip_neighbor_stat,
            force_load=self.force_load,
            model_branch=self.model_branch,
        )
        trainer.run(self.datahub.result.input_path)

    def _save_config(self) -> None:
        if self.datahub is None or self.datahub.result is None:
            return
        config = {
            "task": self.task,
            "data_type": self.data_type,
            "model_name": self.model_name,
            "model_size": self.model_size,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "metrics": self.metrics,
            "property_name": self.property_name,
            "property_col": self.property_col,
            "smiles_col": self.smiles_col,
            "type_map": self.datahub.result.type_map,
            "input_path": str(self.datahub.result.input_path),
            "prepared_data": str(self.datahub.result.output_dir),
            "frozen_model": str(self.save_path / "frozen_model.pth"),
            "checkpoint": str(self.save_path / "model.ckpt.pt"),
        }
        (self.save_path / "property_tools_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _epochs_to_steps(epochs: int | None) -> int:
        if epochs is None:
            return 1000000
        return max(1, int(epochs)) * 1000
