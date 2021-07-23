import os,sys
import numpy as np
import unittest

from deepmd.common import select_idx_map


class TestSelIdx (unittest.TestCase) :
    def test_add (self) :
        atom_type = np.array([0,1,2,2,1,0], dtype = int)
        type_sel = np.array([1,0], dtype = int)
        idx_map = select_idx_map(atom_type, type_sel)
        new_atom_type = atom_type[idx_map]
        self.assertEqual(list(idx_map), [0, 5, 1, 4])
        self.assertEqual(list(new_atom_type), [0, 0, 1, 1])

if __name__ == '__main__':
    unittest.main()
