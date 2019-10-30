import os,sys,warnings
import numpy as np
from deepmd.env import tf
from deepmd.common import ClassArg

class LearningRateExp (object) :
    def __init__ (self, 
                  jdata) :
        args = ClassArg()\
               .add('decay_steps',      int,    must = True)\
               .add('decay_rate',       float,  must = True)\
               .add('start_lr',         float,  must = True)
        class_data = args.parse(jdata)
        self.decay_steps_ = class_data['decay_steps']
        self.decay_rate_ = class_data['decay_rate']
        self.start_lr_ = class_data['start_lr']

    def build(self, global_step) :
        return tf.train.exponential_decay(self.start_lr_, 
                                          global_step,
                                          self.decay_steps_,
                                          self.decay_rate_, 
                                          staircase=True)
    def start_lr(self) :
        return self.start_lr_

    def value (self, 
              batch) :
        return self.start_lr_ * np.power (self.decay_rate_, (batch // self.decay_steps_))

