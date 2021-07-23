import numpy as np
from deepmd.env import tf
from deepmd.common import ClassArg

from .se_a import DescrptSeA
from .se_r import DescrptSeR
from deepmd.env import op_module

class DescrptSeAR ():
    def __init__ (self, jdata):
        args = ClassArg()\
               .add('a',      dict,   must = True) \
               .add('r',      dict,   must = True) 
        class_data = args.parse(jdata)
        self.param_a = class_data['a']
        self.param_r = class_data['r']
        self.descrpt_a = DescrptSeA(**self.param_a)
        self.descrpt_r = DescrptSeR(**self.param_r)
        assert(self.descrpt_a.get_ntypes() == self.descrpt_r.get_ntypes())
        self.davg = None
        self.dstd = None

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

    def compute_input_stats (self,
                        data_coord, 
                        data_box, 
                        data_atype, 
                        natoms_vec,
                             mesh, 
                             input_dict) :    
        self.descrpt_a.compute_input_stats(data_coord, data_box, data_atype, natoms_vec, mesh, input_dict)
        self.descrpt_r.compute_input_stats(data_coord, data_box, data_atype, natoms_vec, mesh, input_dict)
        self.davg = [self.descrpt_a.davg, self.descrpt_r.davg]
        self.dstd = [self.descrpt_a.dstd, self.descrpt_r.dstd]


    def build (self, 
               coord_, 
               atype_,
               natoms,
               box, 
               mesh,
               input_dict,
               suffix = '', 
               reuse = None):
        davg = self.davg
        dstd = self.dstd
        if davg is None:
            davg = [np.zeros([self.descrpt_a.ntypes, self.descrpt_a.ndescrpt]), 
                    np.zeros([self.descrpt_r.ntypes, self.descrpt_r.ndescrpt])]
        if dstd is None:
            dstd = [np.ones ([self.descrpt_a.ntypes, self.descrpt_a.ndescrpt]), 
                    np.ones ([self.descrpt_r.ntypes, self.descrpt_r.ndescrpt])]
        # dout
        self.dout_a = self.descrpt_a.build(coord_, atype_, natoms, box, mesh, input_dict, suffix=suffix+'_a', reuse=reuse)
        self.dout_r = self.descrpt_r.build(coord_, atype_, natoms, box, mesh, input_dict, suffix=suffix     , reuse=reuse)
        self.dout_a = tf.reshape(self.dout_a, [-1, self.descrpt_a.get_dim_out()])
        self.dout_r = tf.reshape(self.dout_r, [-1, self.descrpt_r.get_dim_out()])
        self.dout = tf.concat([self.dout_a, self.dout_r], axis = 1)
        self.dout = tf.reshape(self.dout, [-1, natoms[0] * self.get_dim_out()])
        
        tf.summary.histogram('embedding_net_output', self.dout)
        return self.dout


    def prod_force_virial(self, atom_ener, natoms) :
        f_a, v_a, av_a = self.descrpt_a.prod_force_virial(atom_ener, natoms)
        f_r, v_r, av_r = self.descrpt_r.prod_force_virial(atom_ener, natoms)
        force = f_a + f_r
        virial = v_a + v_r
        atom_virial = av_a + av_r
        tf.summary.histogram('force', force)
        tf.summary.histogram('virial', virial)
        tf.summary.histogram('atom_virial', atom_virial)
        return force, virial, atom_virial
        



