#!/usr/bin/env python3

from deepmd.DeepEval import DeepTensor

class DeepWFC (DeepTensor) :
    def __init__(self, 
                 model_file, 
                 default_tf_graph = False) :
        DeepTensor.__init__(self, model_file, 'wfc', 12, default_tf_graph = default_tf_graph)

