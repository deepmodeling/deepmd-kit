#!/usr/bin/env python3

import os,sys
import numpy as np
import tensorflow as tf
from deepmd.DeepEval import DeepTensor

class DeepWFC (DeepTensor) :
    def __init__(self, 
                 model_file) :
        DeepTensor.__init__(self, model_file, 'wfc', 12)

