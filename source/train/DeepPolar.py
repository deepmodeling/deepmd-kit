#!/usr/bin/env python3

import os,sys
import numpy as np
from deepmd.DeepEval import DeepTensor

class DeepPolar (DeepTensor) :
    def __init__(self, 
                 model_file) :
        DeepTensor.__init__(self, model_file, 'polar', 9)

    
class DeepGlobalPolar (DeepTensor) :
    def __init__(self, 
                 model_file) :
        DeepTensor.__init__(self, model_file, 'global_polar', 9)

    def eval(self,
             coords, 
             cells, 
             atom_types) :
        return DeepTensor.eval(self, coords, cells, atom_types, atomic = False)
