# SPDX-License-Identifier: LGPL-3.0-or-later
"""Full validation support for the JAX trainer."""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from deepmd.dpmodel.train.validation import (
    LOG_COLUMN_ORDER,
    FullValidatorBase,
)
from deepmd.jax.env import (
    jnp,
)
from deepmd.jax.utils.auto_batch_size import (
    AutoBatchSize,
)
from deepmd.utils.eval_metrics import (
    FULL_VALIDATION_WEIGHTED_METRIC_KEYS,
    compute_energy_type_metrics,
)
from deepmd.utils.weight_avg import (
    weighted_average,
)

if TYPE_CHECKING:
    from deepmd.jax.model.base_model import (
        BaseModel,
    )


class JAXFullValidator(FullValidatorBase):
    """Run full validation for a single-task JAX energy model."""

    def __init__(
        self,
        *,
        validating_params: dict[str, Any],
        validation_data: Any,
        model: BaseModel,
        state_store: dict[str, Any],
        num_steps: int,
        rank: int,
        restart_training: bool,
        checkpoint_dir: Any = None,
    ) -> None:
        self.validation_data = validation_data
        self.model = model
        self.auto_batch_size = AutoBatchSize(silent=True)
        super().__init__(
            validating_params=validating_params,
            state_store=state_store,
            num_steps=num_steps,
            rank=rank,
            restart_training=restart_training,
            checkpoint_dir=checkpoint_dir,
            best_checkpoint_suffix=".jax",
        )

    def evaluate_all_systems(self) -> dict[str, float]:
        """Evaluate every validation system and aggregate metrics."""
        system_metrics = [
            self._evaluate_system(data_system)
            for data_system in self._iter_validation_data_systems()
        ]
        aggregated = weighted_average([metric for metric in system_metrics if metric])
        return {
            metric_key: float(aggregated[metric_key])
            for _, metric_key in LOG_COLUMN_ORDER
            if metric_key in aggregated
        }

    def _iter_validation_data_systems(self) -> Any:
        """Yield DeepmdData-like validation systems."""
        validation_data = self.validation_data
        if hasattr(validation_data, "data_systems"):
            yield from validation_data.data_systems
            return
        if hasattr(validation_data, "get_test"):
            yield validation_data
            return
        if hasattr(validation_data, "systems"):
            for dataset in validation_data.systems:
                yield getattr(dataset, "data_system", dataset)
            return
        raise TypeError(
            "JAX full validation expects a DeepmdDataSystem, DeepmdData-like "
            f"object, or loader set; got {type(validation_data)!r}."
        )

    def _evaluate_system(self, data_system: Any) -> dict[str, tuple[float, float]]:
        """Evaluate one validation system."""
        test_data = data_system.get_test()
        natoms = int(test_data["type"].shape[1])
        nframes = int(test_data["coord"].shape[0])
        has_pbc = bool(getattr(data_system, "pbc", False))
        include_virial = has_pbc and bool(test_data.get("find_virial", 0.0))
        prediction = self._predict_outputs(
            coord=test_data["coord"].reshape(nframes, -1),
            atom_types=test_data["type"],
            box=test_data["box"] if has_pbc else None,
            fparam=test_data["fparam"]
            if self.model.get_dim_fparam() > 0
            and bool(test_data.get("find_fparam", 0.0))
            else None,
            aparam=test_data["aparam"] if self.model.get_dim_aparam() > 0 else None,
            include_virial=include_virial,
            natoms=natoms,
            nframes=nframes,
        )
        shared_metrics = compute_energy_type_metrics(
            prediction=prediction,
            test_data=test_data,
            natoms=natoms,
            has_pbc=has_pbc,
        )
        return shared_metrics.as_weighted_average_errors(
            FULL_VALIDATION_WEIGHTED_METRIC_KEYS
        )

    def _predict_outputs(
        self,
        *,
        coord: np.ndarray,
        atom_types: np.ndarray,
        box: np.ndarray | None,
        fparam: np.ndarray | None,
        aparam: np.ndarray | None,
        include_virial: bool,
        natoms: int,
        nframes: int,
    ) -> dict[str, np.ndarray]:
        """Predict energy, force, and virial for the full validation batch."""

        def predict_batch(
            coord_batch: np.ndarray,
            atom_types_batch: np.ndarray,
            box_batch: np.ndarray | None,
            fparam_batch: np.ndarray | None,
            aparam_batch: np.ndarray | None,
        ) -> dict[str, np.ndarray]:
            from deepmd.jax.train.trainer import (
                _evaluate_model_dict,
                prepare_input,
            )

            coord_input = jnp.asarray(coord_batch.reshape(-1, natoms * 3))
            type_input = jnp.asarray(atom_types_batch)
            box_input = (
                jnp.asarray(box_batch.reshape(-1, 9)) if box_batch is not None else None
            )
            fparam_input = (
                jnp.asarray(fparam_batch.reshape(-1, self.model.get_dim_fparam()))
                if fparam_batch is not None
                else None
            )
            aparam_input = (
                jnp.asarray(
                    aparam_batch.reshape(-1, natoms, self.model.get_dim_aparam())
                )
                if aparam_batch is not None
                else None
            )
            extended_coord, extended_atype, nlist, mapping, fp, ap = prepare_input(
                rcut=self.model.get_rcut(),
                sel=self.model.get_sel(),
                coord=coord_input,
                atype=type_input,
                box=box_input,
                fparam=fparam_input,
                aparam=aparam_input,
            )
            batch_output = _evaluate_model_dict(
                self.model,
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fp,
                ap,
            )
            prediction = {
                "energy": np.asarray(batch_output["energy"]).reshape(-1, 1),
                "force": np.asarray(batch_output["force"]).reshape(-1, natoms * 3),
            }
            if include_virial:
                if "virial" not in batch_output:
                    raise KeyError(
                        "Full validation requested virial metrics, but model "
                        "output does not contain `virial`."
                    )
                prediction["virial"] = np.asarray(batch_output["virial"]).reshape(-1, 9)
            return prediction

        batch_prediction = self.auto_batch_size.execute_all(
            predict_batch,
            nframes,
            natoms,
            coord,
            atom_types,
            box,
            fparam,
            aparam,
        )
        prediction = {
            "energy": np.asarray(batch_prediction["energy"]),
            "force": np.asarray(batch_prediction["force"]),
        }
        if include_virial:
            prediction["virial"] = np.asarray(batch_prediction["virial"])
        return prediction
