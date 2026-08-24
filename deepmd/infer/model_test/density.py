# SPDX-License-Identifier: LGPL-3.0-or-later
"""Testing of models predicting charge density on grid points."""

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

__all__ = ["DensityTester"]


class DensityTester(ModelTester):
    """Test a model of charge density on grid points."""

    report = (
        ("mae_density", "DENSITY MAE             : {} units"),
        ("rmse_density", "DENSITY RMSE            : {} units"),
    )

    def add_data_requirements(self, data: DeepmdData) -> None:
        """Declare the labels a density test consumes."""
        dp = self.dp
        # The grid and the density are defined on grid points rather than on
        # atoms, and their extent (ngrid) is not known until the data is
        # loaded. They are declared "atomic" so the loader keeps the
        # frame-major layout without reshaping to natoms; see the grid/density
        # early return in DeepmdData._load_data.
        data.add("grid", 3, atomic=True, must=True, high_prec=True)
        data.add("density", 1, atomic=True, must=True, high_prec=True)
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
        """Evaluate one chunk of a density test."""
        dp = self.dp
        mixed_type = data.mixed_type
        nframes = test_data["box"].shape[0]

        coord = test_data["coord"].reshape([nframes, -1])
        box = test_data["box"] if data.pbc else None
        if mixed_type:
            atype = test_data["type"].reshape([nframes, -1])
        else:
            atype = test_data["type"][0]
        fparam = test_data["fparam"] if dp.get_dim_fparam() > 0 else None
        aparam = test_data["aparam"] if dp.get_dim_aparam() > 0 else None
        grid = test_data["grid"]

        prediction = dp.eval(
            coord,
            box,
            atype,
            grid,
            fparam=fparam,
            aparam=aparam,
            mixed_type=mixed_type,
        ).reshape(nframes, -1)
        label = test_data["density"].reshape(nframes, -1)

        diff = prediction - label
        errors: dict[str, tuple[float, float]] = {
            "mae_density": (mae(diff), diff.size),
            "rmse_density": (rmse(diff), diff.size),
        }

        if context.detail_path is not None:
            _write_per_frame_details(
                context,
                suffix="density",
                reference=label,
                prediction=prediction,
            )

        return errors
