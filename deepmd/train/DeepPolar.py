#!/usr/bin/env python3

from deepmd.DeepEval import DeepTensor

class DeepPolar (DeepTensor) :
    def __init__(self, 
                 model_file, 
                 default_tf_graph = False) :
        DeepTensor.__init__(self, model_file, 'polar', 9, default_tf_graph = default_tf_graph)

    
class DeepGlobalPolar (DeepTensor) :
    def __init__(self, 
                 model_file, 
                 default_tf_graph = False) :
        DeepTensor.__init__(self, model_file, 'global_polar', 9, default_tf_graph = default_tf_graph)

    def eval(self,
             coords, 
             cells, 
             atom_types) :
        return DeepTensor.eval(self, coords, cells, atom_types, atomic = False)
