import numpy as np
from deepmd.env import tf
from deepmd.common import ClassArg

class LearningRateExp (object) :
    def __init__ (self, 
                  jdata) :
        args = ClassArg()\
               .add('decay_steps',      int,    must = False)\
               .add('decay_rate',       float,  must = False)\
               .add('start_lr',         float,  must = True)\
               .add('stop_lr',          float,  must = False)
        self.cd = args.parse(jdata)
        self.start_lr_ = self.cd['start_lr']

    def build(self, global_step, stop_batch = None) :
        if stop_batch is None:            
            self.decay_steps_ = self.cd['decay_steps'] if self.cd['decay_steps'] is not None else 5000
            self.decay_rate_  = self.cd['decay_rate']  if self.cd['decay_rate']  is not None else 0.95
        else:
            self.stop_lr_  = self.cd['stop_lr'] if self.cd['stop_lr'] is not None else 5e-8
            default_ds = 100 if stop_batch // 10 > 100 else stop_batch // 100 + 1
            self.decay_steps_ = self.cd['decay_steps'] if self.cd['decay_steps'] is not None else default_ds
            if self.decay_steps_ >= stop_batch:
                self.decay_steps_ = default_ds
            self.decay_rate_ = np.exp(np.log(self.stop_lr_ / self.start_lr_) / (stop_batch / self.decay_steps_))
            
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

