#!/usr/bin/env python3

import numpy as np
from typing import Tuple, List
from deepmd.infer.deep_eval import DeepTensor

class DeepPolar (DeepTensor) :
    def __init__(self, 
                 model_file : str, 
                 default_tf_graph : bool = False
    ) -> None:
        """
        Constructor

        Parameters
        ----------
        model_file : str
                The name of the frozen model file.
        default_tf_graph : bool
                If uses the default tf graph, otherwise build a new tf graph for evaluation
        """
        DeepTensor.__init__(self, model_file, 'polar', 9, default_tf_graph = default_tf_graph)

    
class DeepGlobalPolar (DeepTensor) :
    def __init__(self, 
                 model_file : str, 
                 default_tf_graph : bool = False
    ) -> None:
        """
        Constructor

        Parameters
        ----------
        model_file : str
                The name of the frozen model file.
        default_tf_graph : bool
                If uses the default tf graph, otherwise build a new tf graph for evaluation
        """
        DeepTensor.__init__(self, model_file, 'global_polar', 9, default_tf_graph = default_tf_graph)

    def eval(self,
             coords : np.array,
             cells : np.array,
             atom_types : List[int],
    ) -> np.array:
        """
        Evaluate the model

        Parameters
        ----------
        coords
                The coordinates of atoms. 
                The array should be of size nframes x natoms x 3
        cells
                The cell of the region. 
                If None then non-PBC is assumed, otherwise using PBC. 
                The array should be of size nframes x 9
        atom_types
                The atom types
                The list should contain natoms ints

        Returns
        -------
        polar
                The system polarizability
        """
        return DeepTensor.eval(self, coords, cells, atom_types, atomic = False)
