import platform
import os,sys,warnings
import numpy as np
from deepmd.env import tf
from deepmd.common import ClassArg
from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.RunOptions import global_ener_float_precision
from deepmd.RunOptions import global_cvt_2_tf_float
from deepmd.RunOptions import global_cvt_2_ener_float

from deepmd.DescrptSeA import DescrptSeA
from deepmd.DescrptSeR import DescrptSeR

if platform.system() == "Windows":
    ext = "dll"
elif platform.system() == "Darwin":
    ext = "dylib"
else:
    ext = "so"
module_path = os.path.dirname(os.path.realpath(__file__)) + "/"
assert (os.path.isfile (module_path  + "libop_abi.{}".format(ext) )), "op module does not exist"
op_module = tf.load_op_library(module_path + "libop_abi.{}".format(ext))

class DescrptSeAR ():
    def __init__ (self, jdata):
        args = ClassArg()\
               .add('a',      dict,   must = True) \
               .add('r',      dict,   must = True) 
        class_data = args.parse(jdata)
        self.param_a = class_data['a']
        self.param_r = class_data['r']
        self.descrpt_a = DescrptSeA(self.param_a)
        self.descrpt_r = DescrptSeR(self.param_r)        
        assert(self.descrpt_a.get_ntypes() == self.descrpt_r.get_ntypes())

    def get_rcut (self) :
        return np.max([self.descrpt_a.get_rcut(), self.descrpt_r.get_rcut()])

    def get_ntypes (self) :
        return self.descrpt_r.get_ntypes()

    def get_dim_out (self) :
        return (self.descrpt_a.get_dim_out() + self.descrpt_r.get_dim_out())

    def get_nlist_a (self) :
        return self.descrpt_a.nlist, self.descrpt_a.rij, self.descrpt_a.sel_a, self.descrpt_a.sel_r

    def get_nlist_r (self) :
        return self.descrpt_r.nlist, self.descrpt_r.rij, self.descrpt_r.sel_a, self.descrpt_r.sel_r

    def compute_dstats (self,
                        data_coord, 
                        data_box, 
                        data_atype, 
                        natoms_vec,
                        mesh) :    
        davg_a, dstd_a = self.descrpt_a.compute_dstats(data_coord, data_box, data_atype, natoms_vec, mesh)
        davg_r, dstd_r = self.descrpt_r.compute_dstats(data_coord, data_box, data_atype, natoms_vec, mesh)
        return [davg_a, davg_r], [dstd_a, dstd_r]


    def build (self, 
               coord_, 
               atype_,
               natoms,
               box, 
               mesh,
               davg,
               dstd,
               suffix = '', 
               reuse = None):
        # dout
        self.dout_a = self.descrpt_a.build(coord_, atype_, natoms, box, mesh, davg[0], dstd[0], suffix=suffix+'_a', reuse=reuse)
        self.dout_r = self.descrpt_r.build(coord_, atype_, natoms, box, mesh, davg[1], dstd[1], suffix=suffix,      reuse=reuse)
        self.dout_a = tf.reshape(self.dout_a, [-1, self.descrpt_a.get_dim_out()])
        self.dout_r = tf.reshape(self.dout_r, [-1, self.descrpt_r.get_dim_out()])
        self.dout = tf.concat([self.dout_a, self.dout_r], axis = 1)
        self.dout = tf.reshape(self.dout, [-1, natoms[0] * self.get_dim_out()])
        return self.dout


    def prod_force_virial(self, atom_ener, natoms) :
        f_a, v_a, av_a = self.descrpt_a.prod_force_virial(atom_ener, natoms)
        f_r, v_r, av_r = self.descrpt_r.prod_force_virial(atom_ener, natoms)
        force = f_a + f_r
        virial = v_a + v_r
        atom_virial = av_a + av_r
        return force, virial, atom_virial
        



