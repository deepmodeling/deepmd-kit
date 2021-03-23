"""ASE calculator interface module."""

from typing import TYPE_CHECKING, Dict, List, Optional, Union
from pathlib import Path

from deepmd import DeepPotential
from ase.calculators.calculator import Calculator, all_changes

if TYPE_CHECKING:
    from ase import Atoms

__all__ = ["DP"]


class DP(Calculator):
    """Implementation of ASE deepmd calculator.

    Implemented propertie are `energy`, `forces` and `stress`

    Parameters
    ----------
    model : Union[str, Path]
        path to the model
    label : str, optional
        calculator label, by default "DP"
    type_dict : Dict[str, int], optional
        mapping of element types and their numbers, best left None and the calculator
        will infer this information from model, by default None

    Examples
    --------
    Compute potential energy

    >>> from ase import Atoms
    >>> from deepmd.calculator import DP
    >>> water = Atoms('H2O',
    >>>             positions=[(0.7601, 1.9270, 1),
    >>>                        (1.9575, 1, 1),
    >>>                        (1., 1., 1.)],
    >>>             cell=[100, 100, 100],
    >>>             calculator=DP(model="frozen_model.pb"))
    >>> print(water.get_potential_energy())
    >>> print(water.get_forces())

    Run BFGS structure optimization

    >>> from ase.optimize import BFGS
    >>> dyn = BFGS(water)
    >>> dyn.run(fmax=1e-6)
    >>> print(water.get_positions())
    """

    name = "DP"
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: Union[str, "Path"],
        label: str = "DP",
        type_dict: Dict[str, int] = None,
        **kwargs
    ) -> None:
        Calculator.__init__(self, label=label, **kwargs)
        self.dp = DeepPotential(str(Path(model).resolve()))
        if type_dict:
            self.type_dict = type_dict
        else:
            self.type_dict = dict(
                zip(self.dp.get_type_map(), range(self.dp.get_ntypes()))
            )

    def calculate(
        self,
        atoms: Optional["Atoms"] = None,
        properties: List[str] = ["energy", "forces", "stress"],
        system_changes: List[str] = all_changes,
    ):
        """Run calculation with deepmd model.

        Parameters
        ----------
        atoms : Optional[Atoms], optional
            atoms object to run the calculation on, by default None
        properties : List[str], optional
            unused, only for function signature compatibility,
            by default ["energy", "forces", "stress"]
        system_changes : List[str], optional
            unused, only for function signature compatibility, by default all_changes
        """
        if atoms is not None:
            self.atoms = atoms.copy()

        coord = self.atoms.get_positions().reshape([1, -1])
        if sum(self.atoms.get_pbc()) > 0:
            cell = self.atoms.get_cell().reshape([1, -1])
        else:
            cell = None
        symbols = self.atoms.get_chemical_symbols()
        atype = [self.type_dict[k] for k in symbols]
        e, f, v = self.dp.eval(coords=coord, cells=cell, atom_types=atype)
        self.results["energy"] = e[0]
        self.results["forces"] = f[0]
        self.results["stress"] = v[0]
