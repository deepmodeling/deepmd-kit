# SPDX-License-Identifier: LGPL-3.0-or-later
"""Manage testing models in a standard way.

For each model, a YAML file ending with `-testcase.yaml` must be given. It should contains the following keys:

- `key`: The key of the model.
- `filename`: The path to the model file.
- `ntypes`: The number of atomic types.
- `rcut`: The cutoff radius.
- `type_map`: The mapping between atomic types and atomic names.
- `dim_fparam`: The number of frame parameters.
- `dim_aparam`: The number of atomic parameters.
- `results`: A list of results. Each result should contain the following keys:
    - `atype`: The atomic types.
    - `coord`: The atomic coordinates.
    - `box`: The simulation box.
    - `atomic_energy` or `energy` (optional): The atomic energies or the total energy.
    - `force` (optional): The atomic forces.
    - `atomic_virial` or `virial` (optional): The atomic virials or the total virial.
"""

import tempfile
from functools import (
    lru_cache,
)
from pathlib import (
    Path,
)
from typing import (
    Optional,
)

import numpy as np
import yaml

from deepmd.entrypoints.convert_backend import (
    convert_backend,
)

this_directory = Path(__file__).parent.resolve()
# create a temporary directory under this directory
# to store the temporary model files
# it will be deleted when the program exits
tempdir = tempfile.TemporaryDirectory(dir=this_directory, prefix="deepmd_test_models_")


class Result:
    """Test results.

    Parameters
    ----------
    data : dict
        Dictionary containing the results.

    Attributes
    ----------
    atype : np.ndarray
        The atomic types.
    nloc : int
        The number of atoms.
    coord : np.ndarray
        The atomic coordinates.
    box : np.ndarray
        The simulation box.
    atomic_energy : np.ndarray
        The atomic energies.
    energy : np.ndarray
        The total energy.
    force : np.ndarray
        The atomic forces.
    atomic_virial : np.ndarray
        The atomic virials.
    virial : np.ndarray
        The total virial.
    """

    def __init__(self, data: dict) -> None:
        self.atype = np.array(data["atype"], dtype=np.int64)
        self.nloc = self.atype.size
        self.coord = np.array(data["coord"], dtype=np.float64).reshape(self.nloc, 3)
        if data["box"] is not None:
            self.box = np.array(data["box"], dtype=np.float64).reshape(3, 3)
        else:
            self.box = None
        if "fparam" in data:
            self.fparam = np.array(data["fparam"], dtype=np.float64).ravel()
        else:
            self.fparam = None
        if "aparam" in data:
            self.aparam = np.array(data["aparam"], dtype=np.float64).reshape(
                self.nloc, -1
            )
        else:
            self.aparam = None
        if "atomic_energy" in data:
            self.atomic_energy = np.array(
                data["atomic_energy"], dtype=np.float64
            ).reshape(self.nloc, 1)
            self.energy = np.sum(self.atomic_energy, axis=0)
        elif "energy" in data:
            self.atomic_energy = None
            self.energy = np.array(data["energy"], dtype=np.float64).reshape(1)
        else:
            self.atomic_energy = None
            self.energy = None
        if "force" in data:
            self.force = np.array(data["force"], dtype=np.float64).reshape(self.nloc, 3)
        else:
            self.force = None
        if "atomic_virial" in data:
            self.atomic_virial = np.array(
                data["atomic_virial"], dtype=np.float64
            ).reshape(self.nloc, 9)
            self.virial = np.sum(self.atomic_virial, axis=0)
        elif "virial" in data:
            self.atomic_virial = None
            self.virial = np.array(data["virial"], dtype=np.float64).reshape(9)
        else:
            self.atomic_virial = None
            self.virial = None
        if "descriptor" in data:
            self.descriptor = np.array(data["descriptor"], dtype=np.float64).reshape(
                self.nloc, -1
            )
        else:
            self.descriptor = None


class Case:
    """Test case.

    Parameters
    ----------
    filename : str
        The path to the test case file.
    """

    def __init__(self, filename: str) -> None:
        with open(filename) as file:
            config = yaml.safe_load(file)
        self.key = config["key"]
        self.filename = str(Path(filename).parent / config["filename"])
        self.results = [Result(data) for data in config["results"]]
        self.ntypes = config["ntypes"]
        self.rcut = config["rcut"]
        self.type_map = config["type_map"]
        self.dim_fparam = config["dim_fparam"]
        self.dim_aparam = config["dim_aparam"]
        self.model_def_script = config.get("model_def_script")

    @lru_cache
    def get_model(self, suffix: str, out_file: Optional[str] = None) -> str:
        """Get the model file with the specified suffix.

        Parameters
        ----------
        suffix : str
            The suffix of the model file.
        out_file : str, optional
            The path to the output model file. If not given, a temporary file will be created.

        Returns
        -------
        str
            The path to the model file.
        """
        # generate a temporary model file
        if out_file is None:
            out_file = tempfile.NamedTemporaryFile(
                suffix=suffix, dir=tempdir.name, delete=False, prefix=self.key + "_"
            ).name
        convert_backend(INPUT=self.filename, OUTPUT=out_file)
        return out_file


@lru_cache
def get_cases() -> dict[str, Case]:
    """Get all test cases.

    Returns
    -------
    dict[str, Case]
        A dictionary containing all test cases.

    Examples
    --------
    To get a specific case:

    >>> get_cases()["se_e2_a"]
    """
    cases = {}
    for ff in this_directory.glob("*-testcase.yaml"):
        case = Case(ff)
        cases[case.key] = case
    return cases
