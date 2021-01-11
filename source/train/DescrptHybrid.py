import numpy as np
from deepmd.env import tf
from deepmd.common import ClassArg
from deepmd.env import op_module
from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.DescrptLocFrame import DescrptLocFrame
from deepmd.descrpt_se_a import DescrptSeA
from deepmd.descrpt_se_a_t import DescrptSeAT
from deepmd.descrpt_se_a_ebd import DescrptSeAEbd
from deepmd.descrpt_se_a_ef import DescrptSeAEf
from deepmd.descrpt_se_r import DescrptSeR

class DescrptHybrid ():
    def __init__ (self, jdata):
        args = ClassArg()\
               .add('list', list, must = True)
        class_data = args.parse(jdata)
        dict_list = class_data['list']
        self.numb_descrpt = len(dict_list)
        self.descrpt_list = []
        self.descrpt_type = []
        for ii in dict_list:
            this_type = ii.get('type')
            if this_type == 'loc_frame':
                this_descrpt = DescrptLocFrame(ii)
            elif this_type == 'se_a' :
                this_descrpt = DescrptSeA(**ii)
            elif this_type == 'se_at' :
                this_descrpt = DescrptSeAT(ii)
            elif this_type == 'se_a_ebd' :
                this_descrpt = DescrptSeAEbd(ii)
            elif this_type == 'se_a_ef' :
                this_descrpt = DescrptSeAEf(ii)
            elif this_type == 'se_r' :
                this_descrpt = DescrptSeR(ii)
            else :
                raise RuntimeError('unknow model type ' + this_type)
            self.descrpt_list.append(this_descrpt)
            self.descrpt_type.append(this_type)
        for ii in range(1, self.numb_descrpt):
            assert(self.descrpt_list[ii].get_ntypes() == 
                   self.descrpt_list[ 0].get_ntypes()), \
                   f'number of atom types in {ii}th descrptor does not match others'


    def get_rcut (self) :
        all_rcut = [ii.get_rcut() for ii in self.descrpt_list]
        return np.max(all_rcut)


    def get_ntypes (self) :
        return self.descrpt_list[0].get_ntypes()


    def get_dim_out (self) :
        all_dim_out = [ii.get_dim_out() for ii in self.descrpt_list]
        return sum(all_dim_out)


    def get_nlist_i(self, ii):
        return self.descrpt_list[ii].nlist, self.descrpt_list[ii].rij, self.descrpt_list[ii].sel_a, self.descrpt_list[ii].sel_r
    

    def compute_input_stats (self,
                             data_coord, 
                             data_box, 
                             data_atype, 
                             natoms_vec,
                             mesh, 
                             input_dict) :
        for ii in self.descrpt_list:
            ii.compute_input_stats(data_coord, data_box, data_atype, natoms_vec, mesh, input_dict)
    

    def build (self, 
               coord_, 
               atype_,
               natoms,
               box, 
               mesh,
               input_dict,
               suffix = '', 
               reuse = None):
        with tf.variable_scope('descrpt_attr' + suffix, reuse = reuse) :
            t_rcut = tf.constant(self.get_rcut(), 
                                 name = 'rcut', 
                                 dtype = global_tf_float_precision)
            t_ntypes = tf.constant(self.get_ntypes(), 
                                   name = 'ntypes', 
                                   dtype = tf.int32)
        all_dout = []
        for tt,ii in zip(self.descrpt_type,self.descrpt_list):
            dout = ii.build(coord_, atype_, natoms, box, mesh, input_dict, suffix=suffix+f'_{tt}', reuse=reuse)
            dout = tf.reshape(dout, [-1, ii.get_dim_out()])
            all_dout.append(dout)
        dout = tf.concat(all_dout, axis = 1)
        dout = tf.reshape(dout, [-1, natoms[0] * self.get_dim_out()])
        return dout
        

    def prod_force_virial(self, atom_ener, natoms) :
        for idx,ii in enumerate(self.descrpt_list):
            ff, vv, av = ii.prod_force_virial(atom_ener, natoms)
            if idx == 0:
                force = ff
                virial = vv
                atom_virial = av
            else:
                force += ff
                virial += vv
                atom_virial += av
        return force, virial, atom_virial
