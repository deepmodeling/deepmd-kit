import os,sys,warnings
import numpy as np
import tensorflow as tf
from deepmd.common import j_must_have, j_must_have_d, j_have

class LearningRateExp (object) :
    def __init__ (self, 
                  jdata) :
        self.decay_steps_ = j_must_have(jdata, 'decay_steps')
        self.decay_rate_ = j_must_have(jdata, 'decay_rate')
        self.start_lr_ = j_must_have(jdata, 'start_lr')        

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

