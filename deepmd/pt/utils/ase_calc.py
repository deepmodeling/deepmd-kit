# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    ClassVar,
)

import dpdata
import numpy as np
from ase import (
    Atoms,
)
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
)

from deepmd.pt.infer.deep_eval import (
    DeepPot,
)


class DPCalculator(Calculator):
    implemented_properties: ClassVar[list] = [
        "energy",
        "free_energy",
        "forces",
        "virial",
        "stress",
    ]

    def __init__(self, model):
        Calculator.__init__(self)
        self.dp = DeepPot(model)
        self.type_map = self.dp.type_map

    def calculate(self, atoms: Atoms, properties, system_changes) -> None:
        Calculator.calculate(self, atoms, properties, system_changes)
        system = dpdata.System(atoms, fmt="ase/structure")
        type_trans = np.array(
            [self.type_map.index(i) for i in system.data["atom_names"]]
        )
        input_coords = system.data["coords"]
        input_cells = system.data["cells"]
        input_types = list(type_trans[system.data["atom_types"]])
        model_predict = self.dp.eval(input_coords, input_cells, input_types)
        self.results = {
            "energy": model_predict[0].item(),
            "free_energy": model_predict[0].item(),
            "forces": model_predict[1].reshape(-1, 3),
            "virial": model_predict[2].reshape(3, 3),
        }

        # convert virial into stress for lattice relaxation
        if "stress" in properties:
            if sum(atoms.get_pbc()) > 0 or (atoms.cell is not None):
                # the usual convention (tensile stress is positive)
                # stress = -virial / volume
                stress = (
                    -0.5
                    * (self.results["virial"].copy() + self.results["virial"].copy().T)
                    / atoms.get_volume()
                )
                # Voigt notation
                self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]
            else:
                raise PropertyNotImplementedError
