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
    FullValidatorBase,
)
from deepmd.jax.env import (
    jax,
    jnp,
    nnx,
)
from deepmd.jax.utils.auto_batch_size import (
    AutoBatchSize,
)
from deepmd.jax.utils.graph_batch import (
    prepare_graph_batch,
)
from deepmd.jax.utils.precision import (
    MatmulPrecisionPolicy,
)
from deepmd.utils.eval_metrics import (
    ENERGY_FULL_VALIDATION_PROFILE,
    SPIN_FULL_VALIDATION_PROFILE,
)
from deepmd.utils.weight_avg import (
    merge_weighted_errors,
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
        symm_virial: bool = False,
        checkpoint_dir: Any = None,
        full_val_file: Any = None,
        best_checkpoint_prefix: str = "best.ckpt",
        precision_policy: MatmulPrecisionPolicy | None = None,
    ) -> None:
        self.validation_data = validation_data
        self.model = model
        self.auto_batch_size = AutoBatchSize(maximum_batch_size=1024, silent=True)
        self.matmul_precision = precision_policy or MatmulPrecisionPolicy()
        super().__init__(
            validating_params=validating_params,
            state_store=state_store,
            num_steps=num_steps,
            rank=rank,
            restart_training=restart_training,
            symm_virial=symm_virial,
            checkpoint_dir=checkpoint_dir,
            full_val_file=full_val_file,
            best_checkpoint_prefix=best_checkpoint_prefix,
            best_checkpoint_suffix=".jax",
            profile=(
                SPIN_FULL_VALIDATION_PROFILE
                if model.has_spin()
                else ENERGY_FULL_VALIDATION_PROFILE
            ),
        )
        self._evaluate_batch = self._make_batch_evaluator()

    def _make_batch_evaluator(self) -> Any:
        """Compile the lower-interface evaluator for each static batch shape."""
        from deepmd.jax.train.trainer import (
            _evaluate_inputs,
        )

        precision = self.matmul_precision
        evaluation_model = nnx.clone(self.model)
        evaluation_model.eval()
        graph, _ = nnx.split(evaluation_model)

        def evaluate(state: Any, inputs: dict[str, Any]) -> dict[str, Any]:
            model = nnx.merge(graph, state)
            with precision.applied(training=False):
                return _evaluate_inputs(model, inputs)

        compiled = jax.jit(evaluate)

        def evaluate_model(model: BaseModel, inputs: dict[str, Any]) -> dict[str, Any]:
            # Failed asynchronous executions must not replace live parameters
            # with error-bearing output buffers before automatic batch retries.
            return compiled(nnx.state(model), inputs)

        return evaluate_model

    def evaluate_all_systems(self) -> dict[str, float]:
        """Evaluate every validation system and aggregate metrics."""
        self.model.eval()
        system_metrics = [
            self._evaluate_system(data_system)
            for data_system in self._iter_validation_data_systems()
        ]
        aggregated = weighted_average([metric for metric in system_metrics if metric])
        return {
            metric_key: float(aggregated[metric_key])
            for metric_key, _, _ in self.table_column_specs
            if metric_key in aggregated
        }

    def propagate_error(self, error_message: str | None) -> str | None:
        """Broadcast rank-0 full-validation failures to all JAX processes."""
        if jax.process_count() <= 1:
            return error_message
        from jax.experimental import (
            multihost_utils,
        )

        has_error = multihost_utils.broadcast_one_to_all(
            np.asarray(error_message is not None, dtype=np.bool_),
            is_source=self.rank == 0,
        )
        if not bool(np.asarray(has_error).item()):
            return None
        return error_message or "Full validation failed on rank 0; see rank-0 logs."

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
        """Evaluate one system while bounding the memory held by lazy readers."""
        has_pbc = bool(getattr(data_system, "pbc", False))
        batches = (
            data_system.iter_test(chunk_atoms=self.auto_batch_size.current_batch_size)
            if hasattr(data_system, "iter_test")
            else (data_system.get_test(),)
        )
        combined: dict[str, tuple[float, float]] = {}
        for test_data in batches:
            current = self._evaluate_test_data(test_data, has_pbc=has_pbc)
            combined = merge_weighted_errors([combined, current])
        return combined

    def _evaluate_test_data(
        self, test_data: dict[str, Any], *, has_pbc: bool
    ) -> dict[str, tuple[float, float]]:
        """Evaluate aligned labels from one homogeneous validation chunk."""
        natoms = int(test_data["type"].shape[1])
        nframes = int(test_data["coord"].shape[0])
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
            spin=test_data.get("spin"),
            charge_spin=test_data.get("charge_spin"),
            include_virial=include_virial,
            natoms=natoms,
            nframes=nframes,
        )
        return self.profile.compute_system_metrics(
            prediction=prediction,
            test_data=test_data,
            natoms=natoms,
            has_pbc=has_pbc,
            symm_virial=self.symm_virial,
        )

    def _predict_outputs(
        self,
        *,
        coord: np.ndarray,
        atom_types: np.ndarray,
        box: np.ndarray | None,
        fparam: np.ndarray | None,
        aparam: np.ndarray | None,
        spin: np.ndarray | None = None,
        charge_spin: np.ndarray | None = None,
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
            spin_batch: np.ndarray | None,
            charge_spin_batch: np.ndarray | None,
        ) -> dict[str, np.ndarray]:
            from deepmd.jax.train.trainer import (
                convert_numpy_data_to_jax_data,
            )

            if self.model.uses_full_neighbor_list():
                _, inputs = prepare_graph_batch(
                    {
                        "atype": atom_types_batch,
                        "coord": coord_batch,
                        "box": box_batch,
                        "fparam": fparam_batch,
                        "aparam": aparam_batch,
                        "spin": spin_batch,
                        "charge_spin": charge_spin_batch,
                    },
                    rcut=self.model.get_rcut(),
                    ntypes=self.model.get_ntypes(),
                    pair_exclude_types=self.model.atomic_model.pair_exclude_types,
                    bucket_subdivisions=1,
                )
                batch_output = self._evaluate_batch(
                    self.model, convert_numpy_data_to_jax_data(inputs)
                )
                batch_frames = atom_types_batch.shape[0]
                physical = atom_types_batch.reshape(-1) >= 0
                for key in ("force", "force_mag", "mask_mag"):
                    if key not in batch_output:
                        continue
                    compact = np.asarray(batch_output[key])
                    restored = np.zeros(
                        (physical.size, *compact.shape[1:]), dtype=compact.dtype
                    )
                    restored[physical] = compact[: np.count_nonzero(physical)]
                    batch_output[key] = restored.reshape(batch_frames, natoms, -1)
                batch_output["energy"] = batch_output["energy"][:batch_frames]
                batch_output["virial"] = batch_output["virial"][:batch_frames]
            else:
                batch_output = self._predict_dense_batch(
                    coord_batch,
                    atom_types_batch,
                    box_batch,
                    fparam_batch,
                    aparam_batch,
                    spin_batch,
                    charge_spin_batch,
                    natoms,
                )
            prediction = {
                "energy": np.asarray(batch_output["energy"]).reshape(-1, 1),
                "force": np.asarray(batch_output["force"]).reshape(-1, natoms * 3),
            }
            if include_virial:
                prediction["virial"] = np.asarray(batch_output["virial"]).reshape(-1, 9)
            if self.profile.needs_spin:
                prediction["force_mag"] = np.asarray(batch_output["force_mag"])
                prediction["mask_mag"] = np.asarray(batch_output["mask_mag"])
            return prediction

        return self.auto_batch_size.execute_all(
            predict_batch,
            nframes,
            natoms,
            coord,
            atom_types,
            box,
            fparam,
            aparam,
            spin,
            charge_spin,
        )

    def _predict_dense_batch(
        self,
        coord_batch: np.ndarray,
        atom_types_batch: np.ndarray,
        box_batch: np.ndarray | None,
        fparam_batch: np.ndarray | None,
        aparam_batch: np.ndarray | None,
        spin_batch: np.ndarray | None,
        charge_spin_batch: np.ndarray | None,
        natoms: int,
    ) -> dict[str, Any]:
        """Evaluate models using the extended-coordinate neighbor-list contract."""
        from deepmd.jax.train.trainer import (
            prepare_input,
        )

        extended_coord, extended_atype, nlist, mapping, fp, ap = prepare_input(
            rcut=self.model.get_rcut(),
            sel=self.model.get_sel(),
            coord=jnp.asarray(coord_batch.reshape(-1, natoms * 3)),
            atype=jnp.asarray(atom_types_batch),
            box=jnp.asarray(box_batch.reshape(-1, 9))
            if box_batch is not None
            else None,
            fparam=(
                jnp.asarray(fparam_batch.reshape(-1, self.model.get_dim_fparam()))
                if fparam_batch is not None
                else None
            ),
            aparam=(
                jnp.asarray(
                    aparam_batch.reshape(-1, natoms, self.model.get_dim_aparam())
                )
                if aparam_batch is not None
                else None
            ),
            pair_excl=self.model.atomic_model.pair_excl,
        )
        return self._evaluate_batch(
            self.model,
            {
                "extended_coord": extended_coord,
                "extended_atype": extended_atype,
                "nlist": nlist,
                "mapping": mapping,
                "fp": fp,
                "ap": ap,
                "spin": jnp.asarray(spin_batch) if spin_batch is not None else None,
                "charge_spin": jnp.asarray(charge_spin_batch)
                if charge_spin_batch is not None
                else None,
            },
        )
