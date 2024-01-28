import unittest
import torch
import numpy as np
from unittest.mock import patch
from deepmd.pt.model.model.pair_tab import PairTabModel

class TestPairTab(unittest.TestCase):

    @patch('numpy.loadtxt')
    def setUp(self, mock_loadtxt) -> None:

        file_path = 'dummy_path'
        mock_loadtxt.return_value = np.array([
                                    [0.005, 1.   , 2.   , 3.   ],
                                    [0.01 , 0.8  , 1.6  , 2.4  ],
                                    [0.015, 0.5  , 1.   , 1.5  ],
                                    [0.02 , 0.25 , 0.4  , 0.75 ]])

        self.model = PairTabModel(
            tab_file = file_path,
            rcut =  0.1,
            sel = 2
        )

        self.extended_coord = torch.tensor([
            [[0.01,0.01,0.01],
            [0.01,0.02,0.01],
            [0.01,0.01,0.02],
            [0.02,0.01,0.01]],

            [[0.01,0.01,0.01],
            [0.01,0.02,0.01],
            [0.01,0.01,0.02],
            [0.05,0.01,0.01]],
        ])

        # nframes=2, nall=4
        self.extended_atype = torch.tensor([
            [0,1,0,1],
            [0,0,1,1]
        ])

        # nframes=2, nloc=2, nnei=2
        self.nlist = torch.tensor([
            [[1,2],[0,2]], 
            [[1,2],[0,3]]
        ])

    def test_without_mask(self):
        
        result = self.model.forward_atomic(self.extended_coord, self.extended_atype,self.nlist)
        expected_result = torch.tensor([[2.4000, 2.7085],
                                        [2.4000, 0.8000]])
        
        torch.testing.assert_allclose(result,expected_result)

    def test_with_mask(self):

        self.nlist = torch.tensor([
            [[1,-1],[0,2]], 
            [[1,2],[0,3]]
        ])
        
        result = self.model.forward_atomic(self.extended_coord, self.extended_atype,self.nlist)
        expected_result = torch.tensor([[1.6000, 2.7085],
                                        [2.4000, 0.8000]])
        
        torch.testing.assert_allclose(result,expected_result)

    def test_jit(self):
        model = torch.jit.script(self.model)

    if __name__ == '__main__':
        unittest.main()