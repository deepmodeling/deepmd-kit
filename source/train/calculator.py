"""An ASE calculator interface.

Example:
```python
from ase import Atoms
from deepmd.calculator import DP

water = Atoms('H2O',
              positions=[(0.7601, 1.9270, 1),
                         (1.9575, 1, 1),
                         (1., 1., 1.)],
              cell=[100, 100, 100],
              calculator=DP(model="frozen_model.pb"))
print(water.get_potential_energy())
print(water.get_forces())
```

Optimization is also available:
```python
from ase.optimize import BFGS
dyn = BFGS(water)
dyn.run(fmax=1e-6)
print(water.get_positions())
```
"""

from ase.calculators.calculator import (
    Calculator, all_changes, PropertyNotImplementedError
)
import deepmd.DeepPot as DeepPot


class DP(Calculator):
    name = "DP"
    implemented_properties = ["energy", "forces", "virial", "stress"]

    def __init__(self, model, label="DP", type_dict=None, **kwargs):
        Calculator.__init__(self, label=label, **kwargs)
        self.dp = DeepPot(model)
        if type_dict:
            self.type_dict=type_dict
        else:
            self.type_dict = dict(zip(self.dp.get_type_map(), range(self.dp.get_ntypes())))

    def calculate(self, atoms=None, properties=["energy", "forces", "virial"], system_changes=all_changes):
        coord = atoms.get_positions().reshape([1, -1])
        if sum(atoms.get_pbc())>0:
           cell = atoms.get_cell().reshape([1, -1])
        else:
           cell = None
        symbols = atoms.get_chemical_symbols()
        atype = [self.type_dict[k] for k in symbols]
        e, f, v = self.dp.eval(coords=coord, cells=cell, atom_types=atype)
        self.results['energy'] = e[0][0]
        self.results['forces'] = f[0]
        self.results['virial'] = v[0].reshape(3,3)

        # convert virial into stress for lattice relaxation
        if "stress" in properties:
            if sum(atoms.get_pbc()) > 0:
                # the usual convention (tensile stress is positive)
                # stress = -virial / volume
                stress = -0.5*(v[0].copy()+v[0].copy().T) / atoms.get_volume()
                # Voigt notation 
                self.results['stress'] = stress.flat[[0,4,8,5,2,1]] 
            else:
                raise PropertyNotImplementedError
