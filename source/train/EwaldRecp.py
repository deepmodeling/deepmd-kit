import platform
import os
import numpy as np
from deepmd.env import tf
from deepmd.common import ClassArg
from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.RunOptions import global_ener_float_precision
from deepmd.RunOptions import global_cvt_2_tf_float
from deepmd.RunOptions import global_cvt_2_ener_float

if platform.system() == "Windows":
    ext = "dll"
elif platform.system() == "Darwin":
    ext = "dylib"
else:
    ext = "so"

module_path = os.path.dirname(os.path.realpath(__file__)) + "/"
assert (os.path.isfile (module_path  + "libop_abi.{}".format(ext) )), "op module does not exist"
op_module = tf.load_op_library(module_path + "libop_abi.{}".format(ext))

class EwaldRecp () :
    def __init__(self, 
                 hh,
                 beta):
        self.hh = hh
        self.beta = beta
        self.sess = tf.Session()

    def eval(self, 
             coord, 
             charge, 
             box) :
        coord = np.array(coord)
        charge = np.array(charge)
        box = np.array(box)
        nframes = charge.shape[0]
        natoms = charge.shape[1]
        coord = np.reshape(coord, [nframes, 3*natoms])
        box = np.reshape(box, [nframes, 9])
        # place holders
        t_coord      = tf.placeholder(global_tf_float_precision, [None, natoms * 3], name='t_coord')
        t_charge     = tf.placeholder(global_tf_float_precision, [None, natoms], name='t_charge')
        t_box        = tf.placeholder(global_tf_float_precision, [None, 9], name='t_box')
        t_nloc    = tf.placeholder(tf.int32, [1], name = "t_nloc")
        
        t_energy, t_force, t_virial \
            = op_module.ewald_recp(t_coord, t_charge, t_nloc, t_box, 
                                   ewald_h = self.hh,
                                   ewald_beta = self.beta)

        [energy, force, virial] \
            = self.sess.run([t_energy, t_force, t_virial], 
                            feed_dict = {
                                t_coord:  coord,
                                t_charge: charge,
                                t_box:    box,
                                t_nloc:   [natoms],
                            })

        return energy, force, virial
             
             
