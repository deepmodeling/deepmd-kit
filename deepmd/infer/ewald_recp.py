import numpy as np
from typing import Tuple, List

from deepmd.env import tf
from deepmd.common import ClassArg
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import GLOBAL_ENER_FLOAT_PRECISION
from deepmd.env import global_cvt_2_tf_float
from deepmd.env import global_cvt_2_ener_float
from deepmd.env import op_module
from deepmd.env import default_tf_session_config

class EwaldRecp () :
    """
    Evaluate the reciprocal part of the Ewald sum
    """
    def __init__(self, 
                 hh,
                 beta):
        """
        Constructor 

        Parameters
        ----------
        hh
                Grid spacing of the reciprocal part of Ewald sum. Unit: A
        beta
                Splitting parameter of the Ewald sum. Unit: A^{-1}
        """
        self.hh = hh
        self.beta = beta
        with tf.Graph().as_default() as graph:
            # place holders
            self.t_nloc       = tf.placeholder(tf.int32, [1], name = "t_nloc")
            self.t_coord      = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='t_coord')
            self.t_charge     = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='t_charge')
            self.t_box        = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='t_box')
            # output            
            self.t_energy, self.t_force, self.t_virial \
                = op_module.ewald_recp(self.t_coord, self.t_charge, self.t_nloc, self.t_box, 
                                       ewald_h = self.hh,
                                       ewald_beta = self.beta)
        self.sess = tf.Session(graph=graph, config=default_tf_session_config)

    def eval(self, 
             coord : np.array, 
             charge : np.array, 
             box : np.array
    ) -> Tuple[np.array, np.array, np.array] :
        """
        Evaluate 
        
        Parameters
        ----------
        coord
                The coordinates of atoms
        charge
                The atomic charge
        box
                The simulation region. PBC is assumed

        Returns
        -------
        e
                The energy 
        f
                The force 
        v
                The virial 
        """
        coord = np.array(coord)
        charge = np.array(charge)
        box = np.array(box)
        nframes = charge.shape[0]
        natoms = charge.shape[1]
        coord = np.reshape(coord, [nframes * 3 * natoms])
        charge = np.reshape(charge, [nframes * natoms])
        box = np.reshape(box, [nframes * 9])

        [energy, force, virial] \
            = self.sess.run([self.t_energy, self.t_force, self.t_virial], 
                            feed_dict = {
                                self.t_coord:  coord,
                                self.t_charge: charge,
                                self.t_box:    box,
                                self.t_nloc:   [natoms],
                            })

        return energy, force, virial
             
             
