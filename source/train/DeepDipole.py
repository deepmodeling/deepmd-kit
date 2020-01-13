#!/usr/bin/env python3

from deepmd.DeepEval import DeepTensor

class DeepDipole (DeepTensor) :
    def __init__(self, 
                 model_file, 
                 load_prefix = 'load') :
        DeepTensor.__init__(self, model_file, 'dipole', 3, load_prefix = load_prefix)

