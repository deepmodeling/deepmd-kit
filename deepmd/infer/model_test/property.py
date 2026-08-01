# SPDX-License-Identifier: LGPL-3.0-or-later
"""Testing of models fitting an arbitrary per-frame property."""

from deepmd.infer.model_test.base import (
    ChunkContext,
    ModelTester,
    _write_per_frame_details,
)
from deepmd.utils.data import (
    DeepmdData,
)
from deepmd.utils.eval_metrics import (
    mae,
    rmse,
)

__all__ = ["PropertyTester"]


class PropertyTester(ModelTester):
    """Test a model of an arbitrary per-frame property."""

    report = (
        ("mae_property", "PROPERTY MAE            : {} units"),
        ("rmse_property", "PROPERTY RMSE           : {} units"),
        ("mae_aproperty", "Atomic PROPERTY MAE     : {} units"),
        ("rmse_aproperty", "Atomic PROPERTY RMSE    : {} units"),
    )
    per_system_only = ("mae_aproperty", "rmse_aproperty")

    def add_data_requirements(self, data: DeepmdData) -> None:
        """Declare the labels a property test consumes."""
        dp = self.dp
        var_name = dp.get_var_name()
        assert isinstance(var_name, str)
        data.add(var_name, dp.task_dim, atomic=False, must=True, high_prec=True)
        if self.atomic:
            data.add(
                f"atom_{var_name}",
                dp.task_dim,
                atomic=True,
                must=True,
                high_prec=True,
            )
        if dp.get_dim_fparam() > 0:
            data.add(
                "fparam", dp.get_dim_fparam(), atomic=False, must=True, high_prec=False
            )
        if dp.get_dim_aparam() > 0:
            data.add(
                "aparam", dp.get_dim_aparam(), atomic=True, must=True, high_prec=False
            )

    def evaluate_chunk(
        self,
        data: DeepmdData,
        test_data: dict,
        context: ChunkContext,
    ) -> dict[str, tuple[float, float]]:
        """Evaluate one chunk of a property test."""
        dp = self.dp
        var_name = dp.get_var_name()
        mixed_type = data.mixed_type
        natoms = len(test_data["type"][0])
        nframes = test_data["box"].shape[0]

        coord = test_data["coord"].reshape([nframes, -1])
        box = test_data["box"] if data.pbc else None
        if mixed_type:
            atype = test_data["type"].reshape([nframes, -1])
        else:
            atype = test_data["type"][0]
        fparam = test_data["fparam"] if dp.get_dim_fparam() > 0 else None
        aparam = test_data["aparam"] if dp.get_dim_aparam() > 0 else None

        ret = dp.eval(
            coord,
            box,
            atype,
            fparam=fparam,
            aparam=aparam,
            atomic=self.atomic,
            mixed_type=mixed_type,
        )
        prediction = ret[0].reshape([nframes, dp.task_dim])

        diff = prediction - test_data[var_name]
        errors: dict[str, tuple[float, float]] = {
            "mae_property": (mae(diff), prediction.size),
            "rmse_property": (rmse(diff), prediction.size),
        }

        atom_prediction = None
        if self.atomic:
            atom_prediction = ret[1].reshape([nframes, natoms * dp.task_dim])
            atom_diff = atom_prediction - test_data[f"atom_{var_name}"]
            errors["mae_aproperty"] = (mae(atom_diff), atom_prediction.size)
            errors["rmse_aproperty"] = (rmse(atom_diff), atom_prediction.size)

        if context.detail_path is not None:
            _write_per_frame_details(
                context,
                suffix="property",
                reference=test_data[var_name],
                prediction=prediction,
            )
            if self.atomic:
                _write_per_frame_details(
                    context,
                    suffix="aproperty",
                    reference=test_data[f"atom_{var_name}"],
                    prediction=atom_prediction,
                )

        return errors
