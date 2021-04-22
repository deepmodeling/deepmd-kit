import numpy as np
from typing import Tuple, List

from deepmd.env import tf
from deepmd.common import ClassArg, get_activation_func, get_precision, add_data_requirement
from deepmd.utils.network import one_layer
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import op_module
from deepmd.env import default_tf_session_config
from deepmd.utils.network import embedding_net,share_embedding_network_oneside,share_embedding_network_twoside
from .se_a import DescrptSeA

class DescrptSeAEbd (DescrptSeA):
    def __init__ (self, 
                  rcut: float,
                  rcut_smth: float,
                  sel: List[str],
                  neuron: List[int] = [24,48,96],
                  axis_neuron: int = 8,
                  resnet_dt: bool = False,
                  trainable: bool = True,
                  seed: int = 1,
                  type_filter:list[int] = [],
                  type_one_side: bool = True,
                  numb_aparam : int = 0,
                  set_davg_zero: bool = False,
                  activation_function: str = 'tanh',
                  precision: str = 'default',
                  exclude_types: List[int] = [],
    ) -> None:
        """
        Constructor

        Parameters
        ----------
        rcut
                The cut-off radius
        rcut_smth
                From where the environment matrix should be smoothed
        sel : list[str]
                sel[i] specifies the maxmum number of type i atoms in the cut-off radius
        neuron : list[int]
                Number of neurons in each hidden layers of the embedding net
        axis_neuron
                Number of the axis neuron (number of columns of the sub-matrix of the embedding matrix)
        resnet_dt
                Time-step `dt` in the resnet construction:
                y = x + dt * \phi (Wx + b)
        trainable
                If the weights of embedding net are trainable.
        seed
                Random seed for initializing the network parameters.
        type_one_side
                Try to build N_types embedding nets. Otherwise, building N_types^2 embedding nets
        type_nchanl
                Number of channels for type representation
        type_nlayer
                Number of hidden layers for the type embedding net (skip connected).
        numb_aparam
                Number of atomic parameters. If >0 it will be embedded with atom types.
        set_davg_zero
                Set the shift of embedding net input to zero.
        activation_function
                The activation function in the embedding net. Supported options are {0}
        precision
                The precision of the embedding net parameters. Supported options are {1}
        """
        # args = ClassArg()\
        #        .add('type_nchanl',      int,    default = 4) \
        #        .add('type_nlayer',      int,    default = 2) \
        #        .add('type_one_side',    bool,   default = True) \
        #        .add('numb_aparam',      int,    default = 0)
        # class_data = args.parse(jdata)
        DescrptSeA.__init__(self, 
                            rcut,
                            rcut_smth,
                            sel,
                            neuron = neuron,
                            axis_neuron = axis_neuron,
                            resnet_dt = resnet_dt,
                            trainable = trainable,
                            seed = seed,
                            type_one_side = type_one_side,
                            set_davg_zero = set_davg_zero,
                            activation_function = activation_function,
                            precision = precision
        )
        self.type_filter = type_filter
        self.type_one_side = type_one_side
        self.numb_aparam = numb_aparam
        if self.numb_aparam > 0:
            add_data_requirement('aparam', 3, atomic=True, must=True, high_prec=False)



    def build (self, 
               coord_ : tf.Tensor, 
               atype_ : tf.Tensor,
               natoms : tf.Tensor,
               box_ : tf.Tensor, 
               mesh : tf.Tensor,
               input_dict : dict, 
               reuse : bool = None,
               suffix : str = ''
        ) -> tf.Tensor:
        """
        Build the computational graph for the descriptor

        Parameters
        ----------
        coord_
                The coordinate of atoms
        atype_
                The type of atoms
        natoms
                The number of atoms. This tensor has the length of Ntypes + 2
                natoms[0]: number of local atoms
                natoms[1]: total number of atoms held by this processor
                natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        mesh
                For historical reasons, only the length of the Tensor matters.
                if size of mesh == 6, pbc is assumed. 
                if size of mesh == 0, no-pbc is assumed. 
        input_dict
                Dictionary for additional inputs
        reuse
                The weights in the networks should be reused when get the variable.
        suffix
                Name suffix to identify this descriptor

        Returns
        -------
        descriptor
                The output descriptor
        """
        nei_type = np.array([])
        for ii in range(self.ntypes):
            nei_type = np.append(nei_type, ii * np.ones(self.sel_a[ii])) # like a mask 
        self.nei_type = tf.get_variable('t_nei_type', 
                                        [self.nnei],
                                        dtype = GLOBAL_TF_FLOAT_PRECISION,
                                        trainable = False,
                                        initializer = tf.constant_initializer(nei_type))
        #self.type_embedding = tf.get_variable('t_embed',shape=[self.ntypes,self.type_filter[-1]],trainable=True,initializer=tf.random_normal_initializer(),dtype=GLOBAL_TF_FLOAT_PRECISION)
        davg = self.davg
        dstd = self.dstd
        #self.type_filter[-1] = self.filter_neuron[0]
        with tf.variable_scope('descrpt_attr' + suffix, reuse = reuse) :
            if davg is None:
                davg = np.zeros([self.ntypes, self.ndescrpt]) 
            if dstd is None:
                dstd = np.ones ([self.ntypes, self.ndescrpt])
            t_rcut = tf.constant(np.max([self.rcut_r, self.rcut_a]), 
                                 name = 'rcut', 
                                 dtype = GLOBAL_TF_FLOAT_PRECISION)
            t_ntypes = tf.constant(self.ntypes, 
                                   name = 'ntypes', 
                                   dtype = tf.int32)
            t_ndescrpt = tf.constant(self.ndescrpt, 
                                     name = 'ndescrpt', 
                                     dtype = tf.int32)            
            t_sel = tf.constant(self.sel_a, 
                                name = 'sel', 
                                dtype = tf.int32)            
            self.t_avg = tf.get_variable('t_avg', 
                                         davg.shape, 
                                         dtype = GLOBAL_TF_FLOAT_PRECISION,
                                         trainable = False,
                                         initializer = tf.constant_initializer(davg))
            self.t_std = tf.get_variable('t_std', 
                                         dstd.shape, 
                                         dtype = GLOBAL_TF_FLOAT_PRECISION,
                                         trainable = False,
                                         initializer = tf.constant_initializer(dstd))

        coord = tf.reshape (coord_, [-1, natoms[1] * 3])
        box   = tf.reshape (box_, [-1, 9])
        atype = tf.reshape (atype_, [-1, natoms[1]])

        self.descrpt, self.descrpt_deriv, self.rij, self.nlist \
            = op_module.prod_env_mat_a (coord,
                                       atype,
                                       natoms,
                                       box,
                                       mesh,
                                       self.t_avg,
                                       self.t_std,
                                       rcut_a = self.rcut_a,
                                       rcut_r = self.rcut_r,
                                       rcut_r_smth = self.rcut_r_smth,
                                       sel_a = self.sel_a,
                                       sel_r = self.sel_r)
        # only used when tensorboard was set as true
        tf.summary.histogram('descrpt', self.descrpt)
        tf.summary.histogram('rij', self.rij)
        tf.summary.histogram('nlist', self.nlist)

        self.descrpt_reshape = tf.reshape(self.descrpt, [-1, self.ndescrpt])
        self.descrpt_reshape = tf.identity(self.descrpt_reshape, name = 'o_rmat')
        self.descrpt_deriv = tf.identity(self.descrpt_deriv, name = 'o_rmat_deriv')
        self.rij = tf.identity(self.rij, name = 'o_rij')
        self.nlist = tf.identity(self.nlist, name = 'o_nlist')

        self.dout, self.qmat,type_embedding = self._pass_filter(self.descrpt_reshape, 
                                                 atype,
                                                 natoms, 
                                                 input_dict,
                                                 suffix = suffix, 
                                                 reuse = reuse, 
                                                 trainable = self.trainable)

        # only used when tensorboard was set as true
        tf.summary.histogram('embedding_net_output', self.dout)
        return type_embedding,self.dout



    def _type_embed(self, 
                    atype,
                    ndim = 1,
                    reuse = None, 
                    suffix = '',
                    trainable = True):
        ebd_type = tf.cast(atype, self.filter_precision)
        #ebd_type = ebd_type / float(self.ntypes)
        ebd_type = tf.reshape(ebd_type, [-1, self.ntypes])
        name = 'type_embed_net_' 
        with tf.variable_scope(name, reuse=reuse):
          ebd_type = embedding_net(ebd_type,
                                 [self.ntypes]+self.type_filter,
                                 activation_fn = self.filter_activation_fn,
                                 precision = self.filter_precision,
                                 resnet_dt = self.filter_resnet_dt,
                                 seed = self.seed,
                                 trainable = trainable)

        ebd_type = tf.reshape(ebd_type, [-1, self.type_filter[-1]]) # nnei * type_filter[-1]
        return ebd_type                       


    def _embedding_net(self, 
                       inputs,
                       natoms,
                       filter_neuron,
                       activation_fn=tf.nn.tanh, 
                       stddev=1.0,
                       bavg=0.0,
                       name='linear', 
                       reuse=None,
                       seed=None, 
                       trainable = True):
        '''
        inputs:  nf x na x (nei x 4)
        outputs: nf x na x nei x output_size
        '''
        # natom x (nei x 4)
        inputs = tf.reshape(inputs, [-1, self.ndescrpt])
        shape = inputs.get_shape().as_list()
        outputs_size = [1] + filter_neuron
        with tf.variable_scope(name, reuse=reuse):
            xyz_scatter_total = []
            # with natom x (nei x 4)  
            inputs_i = inputs
            shape_i = inputs_i.get_shape().as_list()
            # with (natom x nei) x 4  
            inputs_reshape = tf.reshape(inputs_i, [-1, 4]) # Ri
            # with (natom x nei) x 1
            xyz_scatter = tf.reshape(tf.slice(inputs_reshape, [0,0],[-1,1]),[-1,1]) # the col of sij
            # with (natom x nei) x out_size
            xyz_scatter = embedding_net(xyz_scatter, 
                                        [1]+self.filter_neuron, 
                                        self.filter_precision, 
                                        activation_fn = activation_fn, 
                                        resnet_dt = self.filter_resnet_dt,
                                        stddev = stddev,
                                        bavg = bavg,
                                        seed = seed,
                                        trainable = trainable)
            # natom x nei x out_size
            xyz_scatter = tf.reshape(xyz_scatter, (-1, shape_i[1]//4, outputs_size[-1]))
            xyz_scatter_total.append(xyz_scatter)
        # natom x nei x outputs_size
        xyz_scatter = tf.concat(xyz_scatter_total, axis=1)
        # nf x natom x nei x outputs_size
        xyz_scatter = tf.reshape(xyz_scatter, [tf.shape(inputs)[0], natoms[0], self.nnei, outputs_size[-1]])
        return xyz_scatter

    def _share_embedding_net_twoside(self, 
                                   inputs,
                                   natoms,
                                   atype,
                                   nframes,
                                   filter_neuron,
                                   activation_fn=tf.nn.tanh, 
                                   stddev=1.0,
                                   bavg=0.0,
                                   name='linear', 
                                   reuse=None,
                                   seed=None, 
                                   trainable = True):
        '''
        inputs:  nf x na x (nei x 4)
        outputs: nf x na x nei x output_size
        '''
        # natom x (nei x 4)
        #init_shape = inputs.get_shape().as_list()
        #nf = init_shape[0]

        inputs = tf.reshape(inputs, [-1, self.ndescrpt])
        shape = inputs.get_shape().as_list()
        outputs_size = [1] + filter_neuron
        with tf.variable_scope(name, reuse=reuse):
            xyz_scatter_total = []
            # with natom x (nei x 4)  
            inputs_i = inputs
            shape_i = inputs_i.get_shape().as_list()
            # with (natom x nei) x 4  
            inputs_reshape = tf.reshape(inputs_i, [-1, 4]) # Ri
            # with (natom x nei) x 1
            xyz_scatter = tf.reshape(tf.slice(inputs_reshape, [0,0],[-1,1]),[-1,1]) # the col of sij
            
            # with [ntypes, nchanl]
                  
            #self.nei_type = tf.reshape(self.nei_type,[-1,1])
          
            nei_embed = self._type_embed( tf.one_hot(tf.cast(self.nei_type,dtype=tf.int32),int(self.ntypes)),
                                        reuse = reuse,
                                        trainable = True,
                                        suffix = '') #nnei*nchnl
            atm_embed = self._type_embed( tf.one_hot(tf.cast(atype,dtype=tf.int32),int(self.ntypes)),
                                        reuse = True,
                                        trainable = True,
                                        suffix = '') #(nf*natom)*nchnl        
  
            
            print('*'*20+'nei_embed')
            print(nei_embed.get_shape().as_list())
            print('*'*20+'atm_embed')
            print(atm_embed.get_shape().as_list())
            
            _atom_type_ = []
            for ii in range(self.ntypes):
              _atom_type_.append(ii)
            _atom_type_ = tf.convert_to_tensor(_atom_type_,dtype = GLOBAL_TF_FLOAT_PRECISION)
            tmp_type_embedding = self._type_embed( tf.one_hot(tf.cast(_atom_type_,dtype=tf.int32),int(self.ntypes)),
                                        reuse = True,
                                        trainable = True,
                                        suffix = '')  
            nei_embed = tf.tile(nei_embed,(nframes*natoms[0],1))
            nei_embed = tf.reshape(nei_embed,[-1,self.type_filter[-1]])
            
            atm_embed = tf.tile(atm_embed,(1,self.nnei))
            atm_embed = tf.reshape(atm_embed,[-1,self.type_filter[-1]])
            filter_tmp = [1+2*self.type_filter[-1]]+self.filter_neuron
            
            xyz_scatter = share_embedding_network_twoside(xyz_scatter, 
                                        atype,
                                        self.nei_type,
                                        nei_embed,
                                        atm_embed,
                                        filter_tmp, 
                                        self.filter_precision, 
                                        activation_fn = activation_fn, 
                                        resnet_dt = self.filter_resnet_dt,
                                        stddev = stddev,
                                        bavg = bavg,
                                        seed = seed,
                                        trainable = trainable)                                 
            # with (natom x nei) x out_size
            # natom x nei x out_size
            
            xyz_scatter = tf.reshape(xyz_scatter, (-1, shape_i[1]//4, outputs_size[-1]) )
            xyz_scatter_total.append(xyz_scatter)
        # natom x nei x outputs_size
        xyz_scatter = tf.concat(xyz_scatter_total, axis=1)
        # nf x natom x nei x outputs_size
        xyz_scatter = tf.reshape(xyz_scatter, [tf.shape(inputs)[0], natoms[0], self.nnei, outputs_size[-1]])
        print('*'*20)
        print(xyz_scatter.get_shape().as_list())
        return xyz_scatter,tmp_type_embedding
        
    def _share_embedding_net_oneside(self, 
                       inputs,
                       natoms,
                       atype,
                       nframes,
                       filter_neuron,
                       activation_fn=tf.nn.tanh, 
                       stddev=1.0,
                       bavg=0.0,
                       name='linear', 
                       reuse=None,
                       seed=None, 
                       trainable = True):
        '''
        inputs:  nf x na x (nei x 4)
        outputs: nf x na x nei x output_size
        '''
        # natom x (nei x 4)
        #init_shape = inputs.get_shape().as_list()
        #nf = init_shape[0]

        inputs = tf.reshape(inputs, [-1, self.ndescrpt])
        shape = inputs.get_shape().as_list()
        outputs_size = [1] + filter_neuron
        with tf.variable_scope(name, reuse=reuse):
            xyz_scatter_total = []
            # with natom x (nei x 4)  
            inputs_i = inputs
            shape_i = inputs_i.get_shape().as_list()
            # with (natom x nei) x 4  
            inputs_reshape = tf.reshape(inputs_i, [-1, 4]) # Ri
            # with (natom x nei) x 1
            xyz_scatter = tf.reshape(tf.slice(inputs_reshape, [0,0],[-1,1]),[-1,1]) # the col of sij
            
            _atom_type_ = []
            for ii in range(self.ntypes):
              _atom_type_.append(ii)
            _atom_type_ = tf.convert_to_tensor(_atom_type_,dtype = GLOBAL_TF_FLOAT_PRECISION)
            # with [ntypes, nchanl]
            
            
            #self.nei_type = tf.reshape(self.nei_type,[-1,1])
            
            nei_embed = self._type_embed( tf.one_hot(tf.cast(self.nei_type,dtype=tf.int32),int(self.ntypes)), 
                                        reuse = reuse,
                                        trainable = True,
                                        suffix = '') #nnei*nchnl
            '''
            atm_embed = self._type_embed( atype, 
                                        reuse = True,
                                        trainable = True,
                                        suffix = '') #(nf*natom)*nchnl
            '''     
            print('*'*20+'nei_embed')
            print(nei_embed.get_shape().as_list())
            #print('*'*20+'atm_embed')
            #print(atm_embed.get_shape().as_list())
            _atom_type_ = []
            for ii in range(self.ntypes):
              _atom_type_.append(ii)
            _atom_type_ = tf.convert_to_tensor(_atom_type_,dtype = GLOBAL_TF_FLOAT_PRECISION)
            tmp_type_embedding = self._type_embed( tf.one_hot(tf.cast(_atom_type_,dtype=tf.int32),int(self.ntypes)),
                                        reuse = True,
                                        trainable = True,
                                        suffix = '') 
            
            nei_embed = tf.tile(nei_embed,(nframes*natoms[0],1))
            nei_embed = tf.reshape(nei_embed,[-1,self.type_filter[-1]])
            
            #atm_embed = tf.tile(atm_embed,(1,self.nnei))
            #atm_embed = tf.reshape(atm_embed,[-1,self.type_filter[-1]])
            filter_tmp = [1+self.type_filter[-1]]+self.filter_neuron
            
            xyz_scatter = share_embedding_network_oneside(xyz_scatter, 
                                        atype,
                                        self.nei_type,
                                        nei_embed,
                                        filter_tmp, 
                                        self.filter_precision, 
                                        activation_fn = activation_fn, 
                                        resnet_dt = self.filter_resnet_dt,
                                        stddev = stddev,
                                        bavg = bavg,
                                        seed = seed,
                                        trainable = trainable)                                 
            # with (natom x nei) x out_size
            # natom x nei x out_size
            
            xyz_scatter = tf.reshape(xyz_scatter, (-1, shape_i[1]//4, outputs_size[-1]) )
            xyz_scatter_total.append(xyz_scatter)
        # natom x nei x outputs_size
        xyz_scatter = tf.concat(xyz_scatter_total, axis=1)
        # nf x natom x nei x outputs_size
        xyz_scatter = tf.reshape(xyz_scatter, [tf.shape(inputs)[0], natoms[0], self.nnei, outputs_size[-1]])
        print('*'*20)
        print(xyz_scatter.get_shape().as_list())
        return xyz_scatter,tmp_type_embedding
    





    def _pass_filter(self, 
                     inputs,
                     atype,
                     natoms,
                     input_dict,
                     reuse = None,
                     suffix = '', 
                     trainable = True) :
        # nf x na x ndescrpt
        # nf x na x (nnei x 4)
        
        inputs = tf.reshape(inputs, [-1, natoms[0], self.ndescrpt]) # total input
        layer, qmat,type_embedding = self._share_filter(tf.cast(inputs, self.filter_precision), 
                                       atype,
                                       natoms,
                                       input_dict,
                                       name='filter_type_all'+suffix, 
                                       reuse=reuse, 
                                       seed = self.seed, 
                                       trainable = trainable, 
                                       activation_fn = self.filter_activation_fn)
        output      = tf.reshape(layer, [tf.shape(inputs)[0], natoms[0] * self.get_dim_out()])
        output_qmat = tf.reshape(qmat,  [tf.shape(inputs)[0], natoms[0] * self.get_dim_rot_mat_1() * 3])
        return output, output_qmat,type_embedding



    
    def _share_filter(self, 
                    inputs, 
                    atype,
                    natoms,
                    input_dict,
                    activation_fn=tf.nn.tanh, 
                    stddev=1.0,
                    bavg=0.0,
                    name='linear', 
                    reuse=None,
                    seed=None, 
                    trainable = True):
        outputs_size = self.filter_neuron[-1]
        outputs_size_2 = self.n_axis_neuron
        # nf x natom x (nei x 4)
        nframes = tf.shape(inputs)[0]

        shape = tf.reshape(inputs, [-1, self.ndescrpt]).get_shape().as_list() #[natom, ndescrpt]
        
        # nf x natom x nei x outputs_size        
        if self.type_one_side:
          mat_g,type_embedding = self._share_embedding_net_oneside(inputs,
                                    natoms,
                                    atype,
                                    nframes,
                                    self.filter_neuron, 
                                    activation_fn = activation_fn, 
                                    stddev = stddev,
                                    bavg = bavg,
                                    name = name, 
                                    reuse = reuse,
                                    seed = seed,
                                    trainable = trainable)
        else:
          mat_g,type_embedding = self._share_embedding_net_twoside(inputs,
                                    natoms,
                                    atype,
                                    nframes,
                                    self.filter_neuron, 
                                    activation_fn = activation_fn, 
                                    stddev = stddev,
                                    bavg = bavg,
                                    name = name, 
                                    reuse = reuse,
                                    seed = seed,
                                    trainable = trainable)
        # (nf x natom) x nei x outputs_size        
        mat_g = tf.reshape(mat_g, [nframes*natoms[0], self.nnei, outputs_size])
        # natom x nei x 4
        inputs_reshape = tf.reshape(inputs, [-1, shape[1]//4, 4])

        # natom x 4 x outputs_size
        xyz_scatter_1 = tf.matmul(inputs_reshape, mat_g, transpose_a = True)
        print('*'*20)
        print(xyz_scatter_1.get_shape().as_list())
        xyz_scatter_1 = xyz_scatter_1 * (4.0 / shape[1])
        # natom x 4 x outputs_size_2     
        xyz_scatter_2 = tf.slice(xyz_scatter_1, [0,0,0],[-1,-1,outputs_size_2])
        # # natom x 3 x outputs_size_2
        # qmat = tf.slice(xyz_scatter_2, [0,1,0], [-1, 3, -1])
        # natom x 3 x outputs_size_1
        qmat = tf.slice(xyz_scatter_1, [0,1,0], [-1, 3, -1])
        # natom x outputs_size_2 x 3
        qmat = tf.transpose(qmat, perm = [0, 2, 1])
        # natom x outputs_size x outputs_size_2
        result = tf.matmul(xyz_scatter_1, xyz_scatter_2, transpose_a = True)
        # natom x (outputs_size x outputs_size_2)
        result = tf.reshape(result, [-1, outputs_size_2 * outputs_size])

        return result, qmat,type_embedding

