# SPDX-License-Identifier: LGPL-3.0-or-later
"""Testing of density-of-states models."""

import logging

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

log = logging.getLogger(__name__)

__all__ = ["DosTester"]


class DosTester(ModelTester):
    """Test a model of the electronic density of states."""

    report = (
        ("mae_dos", "DOS MAE            : {} Occupation/eV"),
        ("rmse_dos", "DOS RMSE           : {} Occupation/eV"),
        ("mae_dosa", "DOS MAE/Natoms     : {} Occupation/eV"),
        ("rmse_dosa", "DOS RMSE/Natoms    : {} Occupation/eV"),
        ("mae_ados", "Atomic DOS MAE     : {} Occupation/eV"),
        ("rmse_ados", "Atomic DOS RMSE    : {} Occupation/eV"),
    )
    per_system_only = ("mae_ados", "rmse_ados")

    def add_data_requirements(self, data: DeepmdData) -> None:
        """Declare the labels a density-of-states test consumes."""
        dp = self.dp
        data.add("dos", dp.numb_dos, atomic=False, must=True, high_prec=True)
        if self.atomic:
            data.add("atom_dos", dp.numb_dos, atomic=True, must=True, high_prec=True)
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
        """Evaluate one chunk of a density-of-states test."""
        dp = self.dp
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
        dos = ret[0].reshape([nframes, dp.numb_dos])

        diff_dos = dos - test_data["dos"]
        mae_dos = mae(diff_dos)
        rmse_dos = rmse(diff_dos)
        errors: dict[str, tuple[float, float]] = {
            "mae_dos": (mae_dos, dos.size),
            "mae_dosa": (mae_dos / natoms, dos.size),
            "rmse_dos": (rmse_dos, dos.size),
            "rmse_dosa": (rmse_dos / natoms, dos.size),
        }

        ados = None
        if self.atomic:
            ados = ret[1].reshape([nframes, natoms * dp.numb_dos])
            diff_ados = ados - test_data["atom_dos"]
            errors["mae_ados"] = (mae(diff_ados), ados.size)
            errors["rmse_ados"] = (rmse(diff_ados), ados.size)

        if context.detail_path is not None:
            _write_per_frame_details(
                context,
                suffix="dos",
                reference=test_data["dos"],
                prediction=dos,
            )
            if self.atomic:
                _write_per_frame_details(
                    context,
                    suffix="ados",
                    reference=test_data["atom_dos"],
                    prediction=ados,
                )

        return errors


# ---------------------------------------------------------------------------
# Property models
# ---------------------------------------------------------------------------
