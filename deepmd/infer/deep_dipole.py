#!/usr/bin/env python3

from typing import Tuple, List
from deepmd.infer.deep_eval import DeepTensor

class DeepDipole (DeepTensor) :
    def __init__(self, 
                 model_file : str, 
                 load_prefix : str = 'load', 
                 default_tf_graph : bool = False
    ) -> None:
        """
        Constructor

        Parameters
        ----------
        model_file : str
                The name of the frozen model file.
        load_prefix: str
                The prefix in the load computational graph
        default_tf_graph : bool
                If uses the default tf graph, otherwise build a new tf graph for evaluation
        """
        DeepTensor.__init__(self, model_file, 'dipole', 3, load_prefix = load_prefix, default_tf_graph = default_tf_graph)

