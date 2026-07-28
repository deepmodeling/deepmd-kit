# SPDX-License-Identifier: LGPL-3.0-or-later
"""Testing of atomic tensor models, such as dipole and polarizability."""

import logging
from typing import (
    ClassVar,
)

import numpy as np

from deepmd.infer.model_test.base import (
    ChunkContext,
    ModelTester,
    save_txt_file,
)
from deepmd.utils.data import (
    DeepmdData,
)
from deepmd.utils.eval_metrics import (
    rmse,
)

log = logging.getLogger(__name__)

__all__ = ["DipoleTester", "PolarTester", "TensorTester"]


class TensorTester(ModelTester):
    """Test a model of an atomic tensor, summed over atoms unless per-atom.

    A tensor model is evaluated over the atoms its selected types cover, so
    both the reported error and the detail layout follow from the number of
    such atoms.
    """

    #: Label of the per-frame quantity, and of its per-atom counterpart.
    label: ClassVar[str]
    atomic_label: ClassVar[str]
    #: Number of components of the tensor.
    ndof: ClassVar[int]
    #: Per-component names used in the detail header.
    components: ClassVar[tuple[str, ...]]

    def add_data_requirements(self, data: DeepmdData) -> None:
        """Declare the label a tensor test consumes."""
        data.add(
            self.atomic_label if self.atomic else self.label,
            self.ndof,
            atomic=self.atomic,
            must=True,
            high_prec=False,
            type_sel=self.dp.get_sel_type(),
            output_natoms_for_type_sel=True,
        )

    def evaluate_chunk(
        self,
        data: DeepmdData,
        test_data: dict,
        context: ChunkContext,
    ) -> dict[str, tuple[float, float]]:
        """Evaluate one chunk of a tensor test."""
        nframes = test_data["box"].shape[0]
        coord = test_data["coord"].reshape([nframes, -1])
        box = test_data["box"] if data.pbc else None
        atype = test_data["type"][0]
        prediction = self.dp.eval(coord, box, atype).reshape([nframes, -1])

        sel_type = self.dp.get_sel_type()
        sel_natoms = int(sum(sum(atype == ii) for ii in sel_type))

        if self.atomic:
            sel_mask = np.isin(atype, sel_type)
            prediction = prediction.reshape((nframes, -1, self.ndof))[
                :, sel_mask, :
            ].reshape((nframes, -1))
            reference = (
                test_data[self.atomic_label]
                .reshape((nframes, -1, self.ndof))[:, sel_mask, :]
                .reshape((nframes, -1))
            )
        else:
            prediction = np.sum(prediction.reshape((nframes, -1, self.ndof)), axis=1)
            reference = test_data[self.label]

        rmse_tensor = rmse(prediction - reference)
        errors: dict[str, tuple[float, float]] = {
            "rmse": (rmse_tensor, prediction.size)
        }
        if not self.atomic:
            errors["rmse_sqrtn"] = (rmse_tensor / np.sqrt(sel_natoms), prediction.size)
            errors["rmse_n"] = (rmse_tensor / sel_natoms, prediction.size)

        if context.detail_path is not None:
            width = self.ndof * (sel_natoms if self.atomic else 1)
            save_txt_file(
                context.detail_path.with_suffix(".out"),
                np.concatenate(
                    (
                        np.reshape(reference, [-1, width]),
                        np.reshape(prediction, [-1, width]),
                    ),
                    axis=1,
                ),
                header=self._detail_header(sel_natoms),
                append=context.append_detail,
            )

        return errors

    def _detail_header(self, sel_natoms: int) -> str:
        """Return the detail-file header for the layout in use."""
        if not self.atomic:
            return " ".join(
                [f"data_{name}" for name in self.components]
                + [f"pred_{name}" for name in self.components]
            )
        return " ".join(
            [
                f"{prefix}_{name}{number}"
                for prefix in ("data", "pred")
                for number in range(1, sel_natoms + 1)
                for name in self.components
            ]
        )


class DipoleTester(TensorTester):
    """Test a dipole model."""

    label = "dipole"
    atomic_label = "atomic_dipole"
    ndof = 3
    components = ("x", "y", "z")
    report = (
        ("rmse", "Dipole  RMSE       : {}"),
        ("rmse_sqrtn", "Dipole  RMSE/sqrtN : {}"),
        ("rmse_n", "Dipole  RMSE/N     : {}"),
    )
    report_footer = "The unit of error is the same as the unit of provided label."
    per_system_only = ("rmse_sqrtn", "rmse_n")


class PolarTester(TensorTester):
    """Test a polarizability model."""

    label = "polarizability"
    atomic_label = "atomic_polarizability"
    ndof = 9
    components = ("pxx", "pxy", "pxz", "pyx", "pyy", "pyz", "pzx", "pzy", "pzz")
    report = (
        ("rmse", "Polarizability  RMSE       : {}"),
        ("rmse_sqrtn", "Polarizability  RMSE/sqrtN : {}"),
        ("rmse_n", "Polarizability  RMSE/N     : {}"),
    )
    report_footer = "The unit of error is the same as the unit of provided label."
    per_system_only = ("rmse_sqrtn", "rmse_n")
