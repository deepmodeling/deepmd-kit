
import numpy as np 
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.atom import Atom as ase_Atom
from ase.atoms import Atoms as ase_Atoms

from deepmd.nvnmd.utils.fio import Fio
from deepmd.nvnmd.utils.config import nvnmd_cfg
from tensorflow.python.ops.gen_math_ops import is_nan

# require ase package

class Atoms:
    def __init__(self) -> None:
        pass 
    
    def load(self, file_name):
        if file_name.endswith('.xsf'):
            return self.load_xsf(file_name)
    
    def load_xsf(self, file_name):
        Fio().exits(file_name)
        atoms = ase_read(file_name)
        return atoms 

    def spe2atn(self, type_map):
        return [ase_Atom(t).number for t in type_map]

    def resort_by_type(self, atoms, type_map):
        type_map_an = self.spe2atn(type_map)
        spe = atoms.get_atomic_numbers()
        natom = len(spe)
        ntype = len(type_map)
        #
        symb = ''
        coords = atoms.get_positions()
        coords2 = []
        for tt in range(ntype):
            n = 0
            for ii in range(natom):
                if spe[ii] == type_map_an[tt]:
                    n += 1
                    coords2.append(coords[ii])
            symb += type_map[tt] + str(n)
        coords2 = np.array(coords2)
        #
        atoms2 = ase_Atoms(
            symbols=symb,
            positions=coords2,
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc()
        )
        return atoms2 
    
    def extend(self, lst, rij, spe, crd, box):
        natom = len(spe)
        nnei = np.size(lst) // natom 
        #
        lst = np.int32(lst)
        lst = np.reshape(lst, [natom, nnei])
        rij = np.reshape(rij, [natom, nnei, 3])
        spe = np.reshape(spe, [natom, 1])
        crd = np.reshape(crd, [natom, 3])
        box = np.reshape(box, [3, 3])
        # build pbc coords
        spes = np.zeros([27, natom, 1])
        crds = np.zeros([27, natom, 3])
        ct = 0
        for xx in [0, 1, -1]:
            for yy in [0, 1, -1]:
                for zz in [0, 1, -1]:
                    v = np.array([xx, yy, zz])
                    spes[ct] = spe
                    crds[ct] = crd + np.matmul(v, box)
                    ct += 1
        # find pbc neighbor
        is_nei = np.zeros(27 * natom, dtype=np.int32)
        for ii in range(natom):
            for jj in range(nnei):
                ij = lst[ii, jj]
                if (ij != -1):
                    drij = crds[:,ij] - crd[ii] - rij[ii, jj]
                    drij = np.sum(np.power(drij, 2), axis=1) 
                    ij2 = np.where(drij < 1e-5)[0]
                    ij2 = ij2 * natom + ij
                    lst[ii, jj] = ij2 
                    is_nei[ij2] = 1
        # find new atoms: center atoms and neighboring atoms
        spe2 = []
        crd2 = []
        idx = []
        idx2 = np.zeros(27*natom, dtype=np.int32) - 1
        is_nei[0:natom] = 1
        spes = np.reshape(spes, [-1])
        crds = np.reshape(crds, [-1, 3])
        ct = 0
        for ii in range(27*natom):
            if (is_nei[ii] == 1):
                spe2.append(spes[ii])
                crd2.append(crds[ii])
                idx.append(ii)
                idx2[ii] = ct
                ct += 1 
        spe2 = np.reshape(spe2, [-1, 1])
        crd2 = np.reshape(crd2, [-1, 3])
        # rebuild local lst
        for ii in range(natom):
            for jj in range(nnei):
                ij = lst[ii, jj]
                lst[ii, jj] = -1 if (ij == -1) else idx2[ij]
        return lst, spe2, crd2 


