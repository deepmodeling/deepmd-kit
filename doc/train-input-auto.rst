.. _`model`: 

model: 
    | type: ``dict``
    | argument path: ``model``

    .. _`model/type_map`: 

    type_map: 
        | type: ``list``, optional
        | argument path: ``model/type_map``

        A list of strings. Give the name to each type of atoms. It is noted that the number of atom type of training system must be less than 128 in a GPU environment.

    .. _`model/data_stat_nbatch`: 

    data_stat_nbatch: 
        | type: ``int``, optional, default: ``10``
        | argument path: ``model/data_stat_nbatch``

        The model determines the normalization from the statistics of the data. This key specifies the number of `frames` in each `system` used for statistics.

    .. _`model/data_stat_protect`: 

    data_stat_protect: 
        | type: ``float``, optional, default: ``0.01``
        | argument path: ``model/data_stat_protect``

        Protect parameter for atomic energy regression.

    .. _`model/use_srtab`: 

    use_srtab: 
        | type: ``str``, optional
        | argument path: ``model/use_srtab``

        The table for the short-range pairwise interaction added on top of DP. The table is a text data file with (N_t + 1) * N_t / 2 + 1 columes. The first colume is the distance between atoms. The second to the last columes are energies for pairs of certain types. For example we have two atom types, 0 and 1. The columes from 2nd to 4th are for 0-0, 0-1 and 1-1 correspondingly.

    .. _`model/smin_alpha`: 

    smin_alpha: 
        | type: ``float``, optional
        | argument path: ``model/smin_alpha``

        The short-range tabulated interaction will be swithed according to the distance of the nearest neighbor. This distance is calculated by softmin. This parameter is the decaying parameter in the softmin. It is only required when `use_srtab` is provided.

    .. _`model/sw_rmin`: 

    sw_rmin: 
        | type: ``float``, optional
        | argument path: ``model/sw_rmin``

        The lower boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided.

    .. _`model/sw_rmax`: 

    sw_rmax: 
        | type: ``float``, optional
        | argument path: ``model/sw_rmax``

        The upper boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided.

    .. _`model/type_embedding`: 

    type_embedding: 
        | type: ``dict``, optional
        | argument path: ``model/type_embedding``

        The type embedding.

        .. _`model/type_embedding/neuron`: 

        neuron: 
            | type: ``list``, optional, default: ``[2, 4, 8]``
            | argument path: ``model/type_embedding/neuron``

            Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built.

        .. _`model/type_embedding/activation_function`: 

        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/type_embedding/activation_function``

            The activation function in the embedding net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        .. _`model/type_embedding/resnet_dt`: 

        resnet_dt: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/type_embedding/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        .. _`model/type_embedding/precision`: 

        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/type_embedding/precision``

            The precision of the embedding net parameters, supported options are "default", "float16", "float32", "float64".

        .. _`model/type_embedding/trainable`: 

        trainable: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/type_embedding/trainable``

            If the parameters in the embedding net are trainable

        .. _`model/type_embedding/seed`: 

        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/type_embedding/seed``

            Random seed for parameter initialization

    .. _`model/descriptor`: 

    descriptor: 
        | type: ``dict``
        | argument path: ``model/descriptor``

        The descriptor of atomic environment.


        Depending on the value of *type*, different sub args are accepted. 

        .. _`model/descriptor/type`: 

        type:
            | type: ``str`` (flag key)
            | argument path: ``model/descriptor/type`` 
            | possible choices: |code:model/descriptor[loc_frame]|_, |code:model/descriptor[se_e2_a]|_, |code:model/descriptor[se_e2_r]|_, |code:model/descriptor[se_e3]|_, |code:model/descriptor[se_a_tpe]|_, |code:model/descriptor[hybrid]|_

            The type of the descritpor. See explanation below. 

            - `loc_frame`: Defines a local frame at each atom, and the compute the descriptor as local coordinates under this frame.

            - `se_e2_a`: Used by the smooth edition of Deep Potential. The full relative coordinates are used to construct the descriptor.

            - `se_e2_r`: Used by the smooth edition of Deep Potential. Only the distance between atoms is used to construct the descriptor.

            - `se_e3`: Used by the smooth edition of Deep Potential. The full relative coordinates are used to construct the descriptor. Three-body embedding will be used by this descriptor.

            - `se_a_tpe`: Used by the smooth edition of Deep Potential. The full relative coordinates are used to construct the descriptor. Type embedding will be used by this descriptor.

            - `hybrid`: Concatenate of a list of descriptors as a new descriptor.

            .. |code:model/descriptor[loc_frame]| replace:: ``loc_frame``
            .. _`code:model/descriptor[loc_frame]`: `model/descriptor[loc_frame]`_
            .. |code:model/descriptor[se_e2_a]| replace:: ``se_e2_a``
            .. _`code:model/descriptor[se_e2_a]`: `model/descriptor[se_e2_a]`_
            .. |code:model/descriptor[se_e2_r]| replace:: ``se_e2_r``
            .. _`code:model/descriptor[se_e2_r]`: `model/descriptor[se_e2_r]`_
            .. |code:model/descriptor[se_e3]| replace:: ``se_e3``
            .. _`code:model/descriptor[se_e3]`: `model/descriptor[se_e3]`_
            .. |code:model/descriptor[se_a_tpe]| replace:: ``se_a_tpe``
            .. _`code:model/descriptor[se_a_tpe]`: `model/descriptor[se_a_tpe]`_
            .. |code:model/descriptor[hybrid]| replace:: ``hybrid``
            .. _`code:model/descriptor[hybrid]`: `model/descriptor[hybrid]`_

        .. |flag:model/descriptor/type| replace:: *type*
        .. _`flag:model/descriptor/type`: `model/descriptor/type`_


        .. _`model/descriptor[loc_frame]`: 

        When |flag:model/descriptor/type|_ is set to ``loc_frame``: 

        .. _`model/descriptor[loc_frame]/sel_a`: 

        sel_a: 
            | type: ``list``
            | argument path: ``model/descriptor[loc_frame]/sel_a``

            A list of integers. The length of the list should be the same as the number of atom types in the system. `sel_a[i]` gives the selected number of type-i neighbors. The full relative coordinates of the neighbors are used by the descriptor.

        .. _`model/descriptor[loc_frame]/sel_r`: 

        sel_r: 
            | type: ``list``
            | argument path: ``model/descriptor[loc_frame]/sel_r``

            A list of integers. The length of the list should be the same as the number of atom types in the system. `sel_r[i]` gives the selected number of type-i neighbors. Only relative distance of the neighbors are used by the descriptor. sel_a[i] + sel_r[i] is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius.

        .. _`model/descriptor[loc_frame]/rcut`: 

        rcut: 
            | type: ``float``, optional, default: ``6.0``
            | argument path: ``model/descriptor[loc_frame]/rcut``

            The cut-off radius. The default value is 6.0

        .. _`model/descriptor[loc_frame]/axis_rule`: 

        axis_rule: 
            | type: ``list``
            | argument path: ``model/descriptor[loc_frame]/axis_rule``

            A list of integers. The length should be 6 times of the number of types. 

            - axis_rule[i*6+0]: class of the atom defining the first axis of type-i atom. 0 for neighbors with full coordinates and 1 for neighbors only with relative distance.

            - axis_rule[i*6+1]: type of the atom defining the first axis of type-i atom.

            - axis_rule[i*6+2]: index of the axis atom defining the first axis. Note that the neighbors with the same class and type are sorted according to their relative distance.

            - axis_rule[i*6+3]: class of the atom defining the first axis of type-i atom. 0 for neighbors with full coordinates and 1 for neighbors only with relative distance.

            - axis_rule[i*6+4]: type of the atom defining the second axis of type-i atom.

            - axis_rule[i*6+5]: class of the atom defining the second axis of type-i atom. 0 for neighbors with full coordinates and 1 for neighbors only with relative distance.


        .. _`model/descriptor[se_e2_a]`: 

        When |flag:model/descriptor/type|_ is set to ``se_e2_a`` (or its alias ``se_a``): 

        .. _`model/descriptor[se_e2_a]/sel`: 

        sel: 
            | type: ``list``
            | argument path: ``model/descriptor[se_e2_a]/sel``

            A list of integers. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius. It is noted that the total sel value must be less than 4096 in a GPU environment.

        .. _`model/descriptor[se_e2_a]/rcut`: 

        rcut: 
            | type: ``float``, optional, default: ``6.0``
            | argument path: ``model/descriptor[se_e2_a]/rcut``

            The cut-off radius.

        .. _`model/descriptor[se_e2_a]/rcut_smth`: 

        rcut_smth: 
            | type: ``float``, optional, default: ``0.5``
            | argument path: ``model/descriptor[se_e2_a]/rcut_smth``

            Where to start smoothing. For example the 1/r term is smoothed from `rcut` to `rcut_smth`

        .. _`model/descriptor[se_e2_a]/neuron`: 

        neuron: 
            | type: ``list``, optional, default: ``[10, 20, 40]``
            | argument path: ``model/descriptor[se_e2_a]/neuron``

            Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built.

        .. _`model/descriptor[se_e2_a]/axis_neuron`: 

        axis_neuron: 
            | type: ``int``, optional, default: ``4``
            | argument path: ``model/descriptor[se_e2_a]/axis_neuron``

            Size of the submatrix of G (embedding matrix).

        .. _`model/descriptor[se_e2_a]/activation_function`: 

        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/descriptor[se_e2_a]/activation_function``

            The activation function in the embedding net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        .. _`model/descriptor[se_e2_a]/resnet_dt`: 

        resnet_dt: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_e2_a]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        .. _`model/descriptor[se_e2_a]/type_one_side`: 

        type_one_side: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_e2_a]/type_one_side``

            Try to build N_types embedding nets. Otherwise, building N_types^2 embedding nets

        .. _`model/descriptor[se_e2_a]/precision`: 

        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/descriptor[se_e2_a]/precision``

            The precision of the embedding net parameters, supported options are "default", "float16", "float32", "float64".

        .. _`model/descriptor[se_e2_a]/trainable`: 

        trainable: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/descriptor[se_e2_a]/trainable``

            If the parameters in the embedding net is trainable

        .. _`model/descriptor[se_e2_a]/seed`: 

        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/descriptor[se_e2_a]/seed``

            Random seed for parameter initialization

        .. _`model/descriptor[se_e2_a]/exclude_types`: 

        exclude_types: 
            | type: ``list``, optional, default: ``[]``
            | argument path: ``model/descriptor[se_e2_a]/exclude_types``

            The Excluded types

        .. _`model/descriptor[se_e2_a]/set_davg_zero`: 

        set_davg_zero: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_e2_a]/set_davg_zero``

            Set the normalization average to zero. This option should be set when `atom_ener` in the energy fitting is used


        .. _`model/descriptor[se_e2_r]`: 

        When |flag:model/descriptor/type|_ is set to ``se_e2_r`` (or its alias ``se_r``): 

        .. _`model/descriptor[se_e2_r]/sel`: 

        sel: 
            | type: ``list``
            | argument path: ``model/descriptor[se_e2_r]/sel``

            A list of integers. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius. It is noted that the total sel value must be less than 4096 in a GPU environment.

        .. _`model/descriptor[se_e2_r]/rcut`: 

        rcut: 
            | type: ``float``, optional, default: ``6.0``
            | argument path: ``model/descriptor[se_e2_r]/rcut``

            The cut-off radius.

        .. _`model/descriptor[se_e2_r]/rcut_smth`: 

        rcut_smth: 
            | type: ``float``, optional, default: ``0.5``
            | argument path: ``model/descriptor[se_e2_r]/rcut_smth``

            Where to start smoothing. For example the 1/r term is smoothed from `rcut` to `rcut_smth`

        .. _`model/descriptor[se_e2_r]/neuron`: 

        neuron: 
            | type: ``list``, optional, default: ``[10, 20, 40]``
            | argument path: ``model/descriptor[se_e2_r]/neuron``

            Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built.

        .. _`model/descriptor[se_e2_r]/activation_function`: 

        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/descriptor[se_e2_r]/activation_function``

            The activation function in the embedding net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        .. _`model/descriptor[se_e2_r]/resnet_dt`: 

        resnet_dt: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_e2_r]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        .. _`model/descriptor[se_e2_r]/type_one_side`: 

        type_one_side: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_e2_r]/type_one_side``

            Try to build N_types embedding nets. Otherwise, building N_types^2 embedding nets

        .. _`model/descriptor[se_e2_r]/precision`: 

        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/descriptor[se_e2_r]/precision``

            The precision of the embedding net parameters, supported options are "default", "float16", "float32", "float64".

        .. _`model/descriptor[se_e2_r]/trainable`: 

        trainable: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/descriptor[se_e2_r]/trainable``

            If the parameters in the embedding net are trainable

        .. _`model/descriptor[se_e2_r]/seed`: 

        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/descriptor[se_e2_r]/seed``

            Random seed for parameter initialization

        .. _`model/descriptor[se_e2_r]/exclude_types`: 

        exclude_types: 
            | type: ``list``, optional, default: ``[]``
            | argument path: ``model/descriptor[se_e2_r]/exclude_types``

            The Excluded types

        .. _`model/descriptor[se_e2_r]/set_davg_zero`: 

        set_davg_zero: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_e2_r]/set_davg_zero``

            Set the normalization average to zero. This option should be set when `atom_ener` in the energy fitting is used


        .. _`model/descriptor[se_e3]`: 

        When |flag:model/descriptor/type|_ is set to ``se_e3`` (or its aliases ``se_at``, ``se_a_3be``, ``se_t``): 

        .. _`model/descriptor[se_e3]/sel`: 

        sel: 
            | type: ``list``
            | argument path: ``model/descriptor[se_e3]/sel``

            A list of integers. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius. It is noted that the total sel value must be less than 4096 in a GPU environment.

        .. _`model/descriptor[se_e3]/rcut`: 

        rcut: 
            | type: ``float``, optional, default: ``6.0``
            | argument path: ``model/descriptor[se_e3]/rcut``

            The cut-off radius.

        .. _`model/descriptor[se_e3]/rcut_smth`: 

        rcut_smth: 
            | type: ``float``, optional, default: ``0.5``
            | argument path: ``model/descriptor[se_e3]/rcut_smth``

            Where to start smoothing. For example the 1/r term is smoothed from `rcut` to `rcut_smth`

        .. _`model/descriptor[se_e3]/neuron`: 

        neuron: 
            | type: ``list``, optional, default: ``[10, 20, 40]``
            | argument path: ``model/descriptor[se_e3]/neuron``

            Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built.

        .. _`model/descriptor[se_e3]/activation_function`: 

        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/descriptor[se_e3]/activation_function``

            The activation function in the embedding net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        .. _`model/descriptor[se_e3]/resnet_dt`: 

        resnet_dt: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_e3]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        .. _`model/descriptor[se_e3]/precision`: 

        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/descriptor[se_e3]/precision``

            The precision of the embedding net parameters, supported options are "default", "float16", "float32", "float64".

        .. _`model/descriptor[se_e3]/trainable`: 

        trainable: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/descriptor[se_e3]/trainable``

            If the parameters in the embedding net are trainable

        .. _`model/descriptor[se_e3]/seed`: 

        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/descriptor[se_e3]/seed``

            Random seed for parameter initialization

        .. _`model/descriptor[se_e3]/set_davg_zero`: 

        set_davg_zero: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_e3]/set_davg_zero``

            Set the normalization average to zero. This option should be set when `atom_ener` in the energy fitting is used


        .. _`model/descriptor[se_a_tpe]`: 

        When |flag:model/descriptor/type|_ is set to ``se_a_tpe`` (or its alias ``se_a_ebd``): 

        .. _`model/descriptor[se_a_tpe]/sel`: 

        sel: 
            | type: ``list``
            | argument path: ``model/descriptor[se_a_tpe]/sel``

            A list of integers. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius. It is noted that the total sel value must be less than 4096 in a GPU environment.

        .. _`model/descriptor[se_a_tpe]/rcut`: 

        rcut: 
            | type: ``float``, optional, default: ``6.0``
            | argument path: ``model/descriptor[se_a_tpe]/rcut``

            The cut-off radius.

        .. _`model/descriptor[se_a_tpe]/rcut_smth`: 

        rcut_smth: 
            | type: ``float``, optional, default: ``0.5``
            | argument path: ``model/descriptor[se_a_tpe]/rcut_smth``

            Where to start smoothing. For example the 1/r term is smoothed from `rcut` to `rcut_smth`

        .. _`model/descriptor[se_a_tpe]/neuron`: 

        neuron: 
            | type: ``list``, optional, default: ``[10, 20, 40]``
            | argument path: ``model/descriptor[se_a_tpe]/neuron``

            Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built.

        .. _`model/descriptor[se_a_tpe]/axis_neuron`: 

        axis_neuron: 
            | type: ``int``, optional, default: ``4``
            | argument path: ``model/descriptor[se_a_tpe]/axis_neuron``

            Size of the submatrix of G (embedding matrix).

        .. _`model/descriptor[se_a_tpe]/activation_function`: 

        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/descriptor[se_a_tpe]/activation_function``

            The activation function in the embedding net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        .. _`model/descriptor[se_a_tpe]/resnet_dt`: 

        resnet_dt: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_a_tpe]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        .. _`model/descriptor[se_a_tpe]/type_one_side`: 

        type_one_side: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_a_tpe]/type_one_side``

            Try to build N_types embedding nets. Otherwise, building N_types^2 embedding nets

        .. _`model/descriptor[se_a_tpe]/precision`: 

        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/descriptor[se_a_tpe]/precision``

            The precision of the embedding net parameters, supported options are "default", "float16", "float32", "float64".

        .. _`model/descriptor[se_a_tpe]/trainable`: 

        trainable: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/descriptor[se_a_tpe]/trainable``

            If the parameters in the embedding net is trainable

        .. _`model/descriptor[se_a_tpe]/seed`: 

        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/descriptor[se_a_tpe]/seed``

            Random seed for parameter initialization

        .. _`model/descriptor[se_a_tpe]/exclude_types`: 

        exclude_types: 
            | type: ``list``, optional, default: ``[]``
            | argument path: ``model/descriptor[se_a_tpe]/exclude_types``

            The Excluded types

        .. _`model/descriptor[se_a_tpe]/set_davg_zero`: 

        set_davg_zero: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_a_tpe]/set_davg_zero``

            Set the normalization average to zero. This option should be set when `atom_ener` in the energy fitting is used

        .. _`model/descriptor[se_a_tpe]/type_nchanl`: 

        type_nchanl: 
            | type: ``int``, optional, default: ``4``
            | argument path: ``model/descriptor[se_a_tpe]/type_nchanl``

            number of channels for type embedding

        .. _`model/descriptor[se_a_tpe]/type_nlayer`: 

        type_nlayer: 
            | type: ``int``, optional, default: ``2``
            | argument path: ``model/descriptor[se_a_tpe]/type_nlayer``

            number of hidden layers of type embedding net

        .. _`model/descriptor[se_a_tpe]/numb_aparam`: 

        numb_aparam: 
            | type: ``int``, optional, default: ``0``
            | argument path: ``model/descriptor[se_a_tpe]/numb_aparam``

            dimension of atomic parameter. if set to a value > 0, the atomic parameters are embedded.


        .. _`model/descriptor[hybrid]`: 

        When |flag:model/descriptor/type|_ is set to ``hybrid``: 

        .. _`model/descriptor[hybrid]/list`: 

        list: 
            | type: ``list``
            | argument path: ``model/descriptor[hybrid]/list``

            A list of descriptor definitions

    .. _`model/fitting_net`: 

    fitting_net: 
        | type: ``dict``
        | argument path: ``model/fitting_net``

        The fitting of physical properties.


        Depending on the value of *type*, different sub args are accepted. 

        .. _`model/fitting_net/type`: 

        type:
            | type: ``str`` (flag key), default: ``ener``
            | argument path: ``model/fitting_net/type`` 
            | possible choices: |code:model/fitting_net[ener]|_, |code:model/fitting_net[dipole]|_, |code:model/fitting_net[polar]|_

            The type of the fitting. See explanation below. 

            - `ener`: Fit an energy model (potential energy surface).

            - `dipole`: Fit an atomic dipole model. Global dipole labels or atomic dipole labels for all the selected atoms (see `sel_type`) should be provided by `dipole.npy` in each data system. The file either has number of frames lines and 3 times of number of selected atoms columns, or has number of frames lines and 3 columns. See `loss` parameter.

            - `polar`: Fit an atomic polarizability model. Global polarizazbility labels or atomic polarizability labels for all the selected atoms (see `sel_type`) should be provided by `polarizability.npy` in each data system. The file eith has number of frames lines and 9 times of number of selected atoms columns, or has number of frames lines and 9 columns. See `loss` parameter.



            .. |code:model/fitting_net[ener]| replace:: ``ener``
            .. _`code:model/fitting_net[ener]`: `model/fitting_net[ener]`_
            .. |code:model/fitting_net[dipole]| replace:: ``dipole``
            .. _`code:model/fitting_net[dipole]`: `model/fitting_net[dipole]`_
            .. |code:model/fitting_net[polar]| replace:: ``polar``
            .. _`code:model/fitting_net[polar]`: `model/fitting_net[polar]`_

        .. |flag:model/fitting_net/type| replace:: *type*
        .. _`flag:model/fitting_net/type`: `model/fitting_net/type`_


        .. _`model/fitting_net[ener]`: 

        When |flag:model/fitting_net/type|_ is set to ``ener``: 

        .. _`model/fitting_net[ener]/numb_fparam`: 

        numb_fparam: 
            | type: ``int``, optional, default: ``0``
            | argument path: ``model/fitting_net[ener]/numb_fparam``

            The dimension of the frame parameter. If set to >0, file `fparam.npy` should be included to provided the input fparams.

        .. _`model/fitting_net[ener]/numb_aparam`: 

        numb_aparam: 
            | type: ``int``, optional, default: ``0``
            | argument path: ``model/fitting_net[ener]/numb_aparam``

            The dimension of the atomic parameter. If set to >0, file `aparam.npy` should be included to provided the input aparams.

        .. _`model/fitting_net[ener]/neuron`: 

        neuron: 
            | type: ``list``, optional, default: ``[120, 120, 120]``
            | argument path: ``model/fitting_net[ener]/neuron``

            The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built.

        .. _`model/fitting_net[ener]/activation_function`: 

        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/fitting_net[ener]/activation_function``

            The activation function in the fitting net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        .. _`model/fitting_net[ener]/precision`: 

        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/fitting_net[ener]/precision``

            The precision of the fitting net parameters, supported options are "default", "float16", "float32", "float64".

        .. _`model/fitting_net[ener]/resnet_dt`: 

        resnet_dt: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/fitting_net[ener]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        .. _`model/fitting_net[ener]/trainable`: 

        trainable: 
            | type: ``bool`` | ``list``, optional, default: ``True``
            | argument path: ``model/fitting_net[ener]/trainable``

            Whether the parameters in the fitting net are trainable. This option can be

            - bool: True if all parameters of the fitting net are trainable, False otherwise.

            - list of bool: Specifies if each layer is trainable. Since the fitting net is composed by hidden layers followed by a output layer, the length of tihs list should be equal to len(`neuron`)+1.

        .. _`model/fitting_net[ener]/rcond`: 

        rcond: 
            | type: ``float``, optional, default: ``0.001``
            | argument path: ``model/fitting_net[ener]/rcond``

            The condition number used to determine the inital energy shift for each type of atoms.

        .. _`model/fitting_net[ener]/seed`: 

        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/fitting_net[ener]/seed``

            Random seed for parameter initialization of the fitting net

        .. _`model/fitting_net[ener]/atom_ener`: 

        atom_ener: 
            | type: ``list``, optional, default: ``[]``
            | argument path: ``model/fitting_net[ener]/atom_ener``

            Specify the atomic energy in vacuum for each type


        .. _`model/fitting_net[dipole]`: 

        When |flag:model/fitting_net/type|_ is set to ``dipole``: 

        .. _`model/fitting_net[dipole]/neuron`: 

        neuron: 
            | type: ``list``, optional, default: ``[120, 120, 120]``
            | argument path: ``model/fitting_net[dipole]/neuron``

            The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built.

        .. _`model/fitting_net[dipole]/activation_function`: 

        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/fitting_net[dipole]/activation_function``

            The activation function in the fitting net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        .. _`model/fitting_net[dipole]/resnet_dt`: 

        resnet_dt: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/fitting_net[dipole]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        .. _`model/fitting_net[dipole]/precision`: 

        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/fitting_net[dipole]/precision``

            The precision of the fitting net parameters, supported options are "default", "float16", "float32", "float64".

        .. _`model/fitting_net[dipole]/sel_type`: 

        sel_type: 
            | type: ``int`` | ``NoneType`` | ``list``, optional
            | argument path: ``model/fitting_net[dipole]/sel_type``

            The atom types for which the atomic dipole will be provided. If not set, all types will be selected.

        .. _`model/fitting_net[dipole]/seed`: 

        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/fitting_net[dipole]/seed``

            Random seed for parameter initialization of the fitting net


        .. _`model/fitting_net[polar]`: 

        When |flag:model/fitting_net/type|_ is set to ``polar``: 

        .. _`model/fitting_net[polar]/neuron`: 

        neuron: 
            | type: ``list``, optional, default: ``[120, 120, 120]``
            | argument path: ``model/fitting_net[polar]/neuron``

            The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built.

        .. _`model/fitting_net[polar]/activation_function`: 

        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/fitting_net[polar]/activation_function``

            The activation function in the fitting net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        .. _`model/fitting_net[polar]/resnet_dt`: 

        resnet_dt: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/fitting_net[polar]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        .. _`model/fitting_net[polar]/precision`: 

        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/fitting_net[polar]/precision``

            The precision of the fitting net parameters, supported options are "default", "float16", "float32", "float64".

        .. _`model/fitting_net[polar]/fit_diag`: 

        fit_diag: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/fitting_net[polar]/fit_diag``

            Fit the diagonal part of the rotational invariant polarizability matrix, which will be converted to normal polarizability matrix by contracting with the rotation matrix.

        .. _`model/fitting_net[polar]/scale`: 

        scale: 
            | type: ``float`` | ``list``, optional, default: ``1.0``
            | argument path: ``model/fitting_net[polar]/scale``

            The output of the fitting net (polarizability matrix) will be scaled by ``scale``

        .. _`model/fitting_net[polar]/shift_diag`: 

        shift_diag: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/fitting_net[polar]/shift_diag``

            Whether to shift the diagonal of polar, which is beneficial to training. Default is true.

        .. _`model/fitting_net[polar]/sel_type`: 

        sel_type: 
            | type: ``int`` | ``NoneType`` | ``list``, optional
            | argument path: ``model/fitting_net[polar]/sel_type``

            The atom types for which the atomic polarizability will be provided. If not set, all types will be selected.

        .. _`model/fitting_net[polar]/seed`: 

        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/fitting_net[polar]/seed``

            Random seed for parameter initialization of the fitting net

    .. _`model/modifier`: 

    modifier: 
        | type: ``dict``, optional
        | argument path: ``model/modifier``

        The modifier of model output.


        Depending on the value of *type*, different sub args are accepted. 

        .. _`model/modifier/type`: 

        type:
            | type: ``str`` (flag key)
            | argument path: ``model/modifier/type`` 
            | possible choices: |code:model/modifier[dipole_charge]|_

            The type of modifier. See explanation below.

            -`dipole_charge`: Use WFCC to model the electronic structure of the system. Correct the long-range interaction

            .. |code:model/modifier[dipole_charge]| replace:: ``dipole_charge``
            .. _`code:model/modifier[dipole_charge]`: `model/modifier[dipole_charge]`_

        .. |flag:model/modifier/type| replace:: *type*
        .. _`flag:model/modifier/type`: `model/modifier/type`_


        .. _`model/modifier[dipole_charge]`: 

        When |flag:model/modifier/type|_ is set to ``dipole_charge``: 

        .. _`model/modifier[dipole_charge]/model_name`: 

        model_name: 
            | type: ``str``
            | argument path: ``model/modifier[dipole_charge]/model_name``

            The name of the frozen dipole model file.

        .. _`model/modifier[dipole_charge]/model_charge_map`: 

        model_charge_map: 
            | type: ``list``
            | argument path: ``model/modifier[dipole_charge]/model_charge_map``

            The charge of the WFCC. The list length should be the same as the `sel_type <model/fitting_net[dipole]/sel_type_>`_. 

        .. _`model/modifier[dipole_charge]/sys_charge_map`: 

        sys_charge_map: 
            | type: ``list``
            | argument path: ``model/modifier[dipole_charge]/sys_charge_map``

            The charge of real atoms. The list length should be the same as the `type_map <model/type_map_>`_

        .. _`model/modifier[dipole_charge]/ewald_beta`: 

        ewald_beta: 
            | type: ``float``, optional, default: ``0.4``
            | argument path: ``model/modifier[dipole_charge]/ewald_beta``

            The splitting parameter of Ewald sum. Unit is A^-1

        .. _`model/modifier[dipole_charge]/ewald_h`: 

        ewald_h: 
            | type: ``float``, optional, default: ``1.0``
            | argument path: ``model/modifier[dipole_charge]/ewald_h``

            The grid spacing of the FFT grid. Unit is A

    .. _`model/compress`: 

    compress: 
        | type: ``dict``, optional
        | argument path: ``model/compress``

        Model compression configurations


        Depending on the value of *type*, different sub args are accepted. 

        .. _`model/compress/type`: 

        type:
            | type: ``str`` (flag key), default: ``se_e2_a``
            | argument path: ``model/compress/type`` 
            | possible choices: |code:model/compress[se_e2_a]|_

            The type of model compression, which should be consistent with the descriptor type.

            .. |code:model/compress[se_e2_a]| replace:: ``se_e2_a``
            .. _`code:model/compress[se_e2_a]`: `model/compress[se_e2_a]`_

        .. |flag:model/compress/type| replace:: *type*
        .. _`flag:model/compress/type`: `model/compress/type`_


        .. _`model/compress[se_e2_a]`: 

        When |flag:model/compress/type|_ is set to ``se_e2_a`` (or its alias ``se_a``): 

        .. _`model/compress[se_e2_a]/compress`: 

        compress: 
            | type: ``bool``
            | argument path: ``model/compress[se_e2_a]/compress``

            The name of the frozen model file.

        .. _`model/compress[se_e2_a]/model_file`: 

        model_file: 
            | type: ``str``
            | argument path: ``model/compress[se_e2_a]/model_file``

            The input model file, which will be compressed by the DeePMD-kit.

        .. _`model/compress[se_e2_a]/table_config`: 

        table_config: 
            | type: ``list``
            | argument path: ``model/compress[se_e2_a]/table_config``

            The arguments of model compression, including extrapolate(scale of model extrapolation), stride(uniform stride of tabulation's first and second table), and frequency(frequency of tabulation overflow check).


.. _`loss`: 

loss: 
    | type: ``dict``, optional
    | argument path: ``loss``

    The definition of loss function. The loss type should be set to `tensor`, `ener` or left unset.
    \.


    Depending on the value of *type*, different sub args are accepted. 

    .. _`loss/type`: 

    type:
        | type: ``str`` (flag key), default: ``ener``
        | argument path: ``loss/type`` 
        | possible choices: |code:loss[ener]|_, |code:loss[tensor]|_

        The type of the loss. When the fitting type is `ener`, the loss type should be set to `ener` or left unset. When the fitting type is `dipole` or `polar`, the loss type should be set to `tensor`. 
        \.

        .. |code:loss[ener]| replace:: ``ener``
        .. _`code:loss[ener]`: `loss[ener]`_
        .. |code:loss[tensor]| replace:: ``tensor``
        .. _`code:loss[tensor]`: `loss[tensor]`_

    .. |flag:loss/type| replace:: *type*
    .. _`flag:loss/type`: `loss/type`_


    .. _`loss[ener]`: 

    When |flag:loss/type|_ is set to ``ener``: 

    .. _`loss[ener]/start_pref_e`: 

    start_pref_e: 
        | type: ``float`` | ``int``, optional, default: ``0.02``
        | argument path: ``loss[ener]/start_pref_e``

        The prefactor of energy loss at the start of the training. Should be larger than or equal to 0. If set to none-zero value, the energy label should be provided by file energy.npy in each data system. If both start_pref_energy and limit_pref_energy are set to 0, then the energy will be ignored.

    .. _`loss[ener]/limit_pref_e`: 

    limit_pref_e: 
        | type: ``float`` | ``int``, optional, default: ``1.0``
        | argument path: ``loss[ener]/limit_pref_e``

        The prefactor of energy loss at the limit of the training, Should be larger than or equal to 0. i.e. the training step goes to infinity.

    .. _`loss[ener]/start_pref_f`: 

    start_pref_f: 
        | type: ``float`` | ``int``, optional, default: ``1000``
        | argument path: ``loss[ener]/start_pref_f``

        The prefactor of force loss at the start of the training. Should be larger than or equal to 0. If set to none-zero value, the force label should be provided by file force.npy in each data system. If both start_pref_force and limit_pref_force are set to 0, then the force will be ignored.

    .. _`loss[ener]/limit_pref_f`: 

    limit_pref_f: 
        | type: ``float`` | ``int``, optional, default: ``1.0``
        | argument path: ``loss[ener]/limit_pref_f``

        The prefactor of force loss at the limit of the training, Should be larger than or equal to 0. i.e. the training step goes to infinity.

    .. _`loss[ener]/start_pref_v`: 

    start_pref_v: 
        | type: ``float`` | ``int``, optional, default: ``0.0``
        | argument path: ``loss[ener]/start_pref_v``

        The prefactor of virial loss at the start of the training. Should be larger than or equal to 0. If set to none-zero value, the virial label should be provided by file virial.npy in each data system. If both start_pref_virial and limit_pref_virial are set to 0, then the virial will be ignored.

    .. _`loss[ener]/limit_pref_v`: 

    limit_pref_v: 
        | type: ``float`` | ``int``, optional, default: ``0.0``
        | argument path: ``loss[ener]/limit_pref_v``

        The prefactor of virial loss at the limit of the training, Should be larger than or equal to 0. i.e. the training step goes to infinity.

    .. _`loss[ener]/start_pref_ae`: 

    start_pref_ae: 
        | type: ``float`` | ``int``, optional, default: ``0.0``
        | argument path: ``loss[ener]/start_pref_ae``

        The prefactor of atom_ener loss at the start of the training. Should be larger than or equal to 0. If set to none-zero value, the atom_ener label should be provided by file atom_ener.npy in each data system. If both start_pref_atom_ener and limit_pref_atom_ener are set to 0, then the atom_ener will be ignored.

    .. _`loss[ener]/limit_pref_ae`: 

    limit_pref_ae: 
        | type: ``float`` | ``int``, optional, default: ``0.0``
        | argument path: ``loss[ener]/limit_pref_ae``

        The prefactor of atom_ener loss at the limit of the training, Should be larger than or equal to 0. i.e. the training step goes to infinity.

    .. _`loss[ener]/relative_f`: 

    relative_f: 
        | type: ``float`` | ``NoneType``, optional
        | argument path: ``loss[ener]/relative_f``

        If provided, relative force error will be used in the loss. The difference of force will be normalized by the magnitude of the force in the label with a shift given by `relative_f`, i.e. DF_i / ( || F || + relative_f ) with DF denoting the difference between prediction and label and || F || denoting the L2 norm of the label.


    .. _`loss[tensor]`: 

    When |flag:loss/type|_ is set to ``tensor``: 

    .. _`loss[tensor]/pref`: 

    pref: 
        | type: ``float`` | ``int``
        | argument path: ``loss[tensor]/pref``

        The prefactor of the weight of global loss. It should be larger than or equal to 0. If controls the weight of loss corresponding to global label, i.e. 'polarizability.npy` or `dipole.npy`, whose shape should be #frames x [9 or 3]. If it's larger than 0.0, this npy should be included.

    .. _`loss[tensor]/pref_atomic`: 

    pref_atomic: 
        | type: ``float`` | ``int``
        | argument path: ``loss[tensor]/pref_atomic``

        The prefactor of the weight of atomic loss. It should be larger than or equal to 0. If controls the weight of loss corresponding to atomic label, i.e. `atomic_polarizability.npy` or `atomic_dipole.npy`, whose shape should be #frames x ([9 or 3] x #selected atoms). If it's larger than 0.0, this npy should be included. Both `pref` and `pref_atomic` should be provided, and either can be set to 0.0.


.. _`learning_rate`: 

learning_rate: 
    | type: ``dict``
    | argument path: ``learning_rate``

    The definitio of learning rate


    Depending on the value of *type*, different sub args are accepted. 

    .. _`learning_rate/type`: 

    type:
        | type: ``str`` (flag key), default: ``exp``
        | argument path: ``learning_rate/type`` 
        | possible choices: |code:learning_rate[exp]|_

        The type of the learning rate.

        .. |code:learning_rate[exp]| replace:: ``exp``
        .. _`code:learning_rate[exp]`: `learning_rate[exp]`_

    .. |flag:learning_rate/type| replace:: *type*
    .. _`flag:learning_rate/type`: `learning_rate/type`_


    .. _`learning_rate[exp]`: 

    When |flag:learning_rate/type|_ is set to ``exp``: 

    .. _`learning_rate[exp]/start_lr`: 

    start_lr: 
        | type: ``float``, optional, default: ``0.001``
        | argument path: ``learning_rate[exp]/start_lr``

        The learning rate the start of the training.

    .. _`learning_rate[exp]/stop_lr`: 

    stop_lr: 
        | type: ``float``, optional, default: ``1e-08``
        | argument path: ``learning_rate[exp]/stop_lr``

        The desired learning rate at the end of the training.

    .. _`learning_rate[exp]/decay_steps`: 

    decay_steps: 
        | type: ``int``, optional, default: ``5000``
        | argument path: ``learning_rate[exp]/decay_steps``

        The learning rate is decaying every this number of training steps.


.. _`training`: 

training: 
    | type: ``dict``
    | argument path: ``training``

    The training options.

    .. _`training/training_data`: 

    training_data: 
        | type: ``dict``
        | argument path: ``training/training_data``

        Configurations of training data.

        .. _`training/training_data/systems`: 

        systems: 
            | type: ``list`` | ``str``
            | argument path: ``training/training_data/systems``

            The data systems for training. This key can be provided with a list that specifies the systems, or be provided with a string by which the prefix of all systems are given and the list of the systems is automatically generated.

        .. _`training/training_data/set_prefix`: 

        set_prefix: 
            | type: ``str``, optional, default: ``set``
            | argument path: ``training/training_data/set_prefix``

            The prefix of the sets in the `systems <training/training_data/systems_>`_.

        .. _`training/training_data/batch_size`: 

        batch_size: 
            | type: ``int`` | ``list`` | ``str``, optional, default: ``auto``
            | argument path: ``training/training_data/batch_size``

            This key can be 

            - list: the length of which is the same as the `systems <training/training_data/systems_>`_. The batch size of each system is given by the elements of the list.

            - int: all `systems <training/training_data/systems_>`_ use the same batch size.

            - string "auto": automatically determines the batch size so that the batch_size times the number of atoms in the system is no less than 32.

            - string "auto:N": automatically determines the batch size so that the batch_size times the number of atoms in the system is no less than N.

        .. _`training/training_data/auto_prob`: 

        auto_prob: 
            | type: ``str``, optional, default: ``prob_sys_size``, alias: *auto_prob_style*
            | argument path: ``training/training_data/auto_prob``

            Determine the probability of systems automatically. The method is assigned by this key and can be

            - "prob_uniform"  : the probability all the systems are equal, namely 1.0/self.get_nsystems()

            - "prob_sys_size" : the probability of a system is proportional to the number of batches in the system

            - "prob_sys_size;stt_idx:end_idx:weight;stt_idx:end_idx:weight;..." : the list of systems is devided into blocks. A block is specified by `stt_idx:end_idx:weight`, where `stt_idx` is the starting index of the system, `end_idx` is then ending (not including) index of the system, the probabilities of the systems in this block sums up to `weight`, and the relatively probabilities within this block is proportional to the number of batches in the system.

        .. _`training/training_data/sys_probs`: 

        sys_probs: 
            | type: ``NoneType`` | ``list``, optional, default: ``None``, alias: *sys_weights*
            | argument path: ``training/training_data/sys_probs``

            A list of float if specified. Should be of the same length as `systems`, specifying the probability of each system.

    .. _`training/validation_data`: 

    validation_data: 
        | type: ``NoneType`` | ``dict``, optional, default: ``None``
        | argument path: ``training/validation_data``

        Configurations of validation data. Similar to that of training data, except that a `numb_btch` argument may be configured

        .. _`training/validation_data/systems`: 

        systems: 
            | type: ``list`` | ``str``
            | argument path: ``training/validation_data/systems``

            The data systems for validation. This key can be provided with a list that specifies the systems, or be provided with a string by which the prefix of all systems are given and the list of the systems is automatically generated.

        .. _`training/validation_data/set_prefix`: 

        set_prefix: 
            | type: ``str``, optional, default: ``set``
            | argument path: ``training/validation_data/set_prefix``

            The prefix of the sets in the `systems <training/validation_data/systems_>`_.

        .. _`training/validation_data/batch_size`: 

        batch_size: 
            | type: ``int`` | ``list`` | ``str``, optional, default: ``auto``
            | argument path: ``training/validation_data/batch_size``

            This key can be 

            - list: the length of which is the same as the `systems <training/validation_data/systems_>`_. The batch size of each system is given by the elements of the list.

            - int: all `systems <training/validation_data/systems_>`_ use the same batch size.

            - string "auto": automatically determines the batch size so that the batch_size times the number of atoms in the system is no less than 32.

            - string "auto:N": automatically determines the batch size so that the batch_size times the number of atoms in the system is no less than N.

        .. _`training/validation_data/auto_prob`: 

        auto_prob: 
            | type: ``str``, optional, default: ``prob_sys_size``, alias: *auto_prob_style*
            | argument path: ``training/validation_data/auto_prob``

            Determine the probability of systems automatically. The method is assigned by this key and can be

            - "prob_uniform"  : the probability all the systems are equal, namely 1.0/self.get_nsystems()

            - "prob_sys_size" : the probability of a system is proportional to the number of batches in the system

            - "prob_sys_size;stt_idx:end_idx:weight;stt_idx:end_idx:weight;..." : the list of systems is devided into blocks. A block is specified by `stt_idx:end_idx:weight`, where `stt_idx` is the starting index of the system, `end_idx` is then ending (not including) index of the system, the probabilities of the systems in this block sums up to `weight`, and the relatively probabilities within this block is proportional to the number of batches in the system.

        .. _`training/validation_data/sys_probs`: 

        sys_probs: 
            | type: ``NoneType`` | ``list``, optional, default: ``None``, alias: *sys_weights*
            | argument path: ``training/validation_data/sys_probs``

            A list of float if specified. Should be of the same length as `systems`, specifying the probability of each system.

        .. _`training/validation_data/numb_btch`: 

        numb_btch: 
            | type: ``int``, optional, default: ``1``, alias: *numb_batch*
            | argument path: ``training/validation_data/numb_btch``

            An integer that specifies the number of systems to be sampled for each validation period.

    .. _`training/numb_steps`: 

    numb_steps: 
        | type: ``int``, alias: *stop_batch*
        | argument path: ``training/numb_steps``

        Number of training batch. Each training uses one batch of data.

    .. _`training/seed`: 

    seed: 
        | type: ``int`` | ``NoneType``, optional
        | argument path: ``training/seed``

        The random seed for getting frames from the training data set.

    .. _`training/disp_file`: 

    disp_file: 
        | type: ``str``, optional, default: ``lcueve.out``
        | argument path: ``training/disp_file``

        The file for printing learning curve.

    .. _`training/disp_freq`: 

    disp_freq: 
        | type: ``int``, optional, default: ``1000``
        | argument path: ``training/disp_freq``

        The frequency of printing learning curve.

    .. _`training/numb_test`: 

    numb_test: 
        | type: ``int`` | ``list`` | ``str``, optional, default: ``1``
        | argument path: ``training/numb_test``

        Number of frames used for the test during training.

    .. _`training/save_freq`: 

    save_freq: 
        | type: ``int``, optional, default: ``1000``
        | argument path: ``training/save_freq``

        The frequency of saving check point.

    .. _`training/save_ckpt`: 

    save_ckpt: 
        | type: ``str``, optional, default: ``model.ckpt``
        | argument path: ``training/save_ckpt``

        The file name of saving check point.

    .. _`training/disp_training`: 

    disp_training: 
        | type: ``bool``, optional, default: ``True``
        | argument path: ``training/disp_training``

        Displaying verbose information during training.

    .. _`training/time_training`: 

    time_training: 
        | type: ``bool``, optional, default: ``True``
        | argument path: ``training/time_training``

        Timing durining training.

    .. _`training/profiling`: 

    profiling: 
        | type: ``bool``, optional, default: ``False``
        | argument path: ``training/profiling``

        Profiling during training.

    .. _`training/profiling_file`: 

    profiling_file: 
        | type: ``str``, optional, default: ``timeline.json``
        | argument path: ``training/profiling_file``

        Output file for profiling.

    .. _`training/tensorboard`: 

    tensorboard: 
        | type: ``bool``, optional, default: ``False``
        | argument path: ``training/tensorboard``

        Enable tensorboard

    .. _`training/tensorboard_log_dir`: 

    tensorboard_log_dir: 
        | type: ``str``, optional, default: ``log``
        | argument path: ``training/tensorboard_log_dir``

        The log directory of tensorboard outputs

