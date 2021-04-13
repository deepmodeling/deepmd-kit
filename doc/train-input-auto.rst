.. raw:: html

   <a id="model"></a>
model: 
    | type: ``dict``
    | argument path: ``model``

    .. raw:: html

       <a id="model/type_map"></a>
    type_map: 
        | type: ``list``, optional
        | argument path: ``model/type_map``

        A list of strings. Give the name to each type of atoms.

    .. raw:: html

       <a id="model/data_stat_nbatch"></a>
    data_stat_nbatch: 
        | type: ``int``, optional, default: ``10``
        | argument path: ``model/data_stat_nbatch``

        The model determines the normalization from the statistics of the data. This key specifies the number of `frames` in each `system` used for statistics.

    .. raw:: html

       <a id="model/data_stat_protect"></a>
    data_stat_protect: 
        | type: ``float``, optional, default: ``0.01``
        | argument path: ``model/data_stat_protect``

        Protect parameter for atomic energy regression.

    .. raw:: html

       <a id="model/use_srtab"></a>
    use_srtab: 
        | type: ``str``, optional
        | argument path: ``model/use_srtab``

        The table for the short-range pairwise interaction added on top of DP. The table is a text data file with (N_t + 1) * N_t / 2 + 1 columes. The first colume is the distance between atoms. The second to the last columes are energies for pairs of certain types. For example we have two atom types, 0 and 1. The columes from 2nd to 4th are for 0-0, 0-1 and 1-1 correspondingly.

    .. raw:: html

       <a id="model/smin_alpha"></a>
    smin_alpha: 
        | type: ``float``, optional
        | argument path: ``model/smin_alpha``

        The short-range tabulated interaction will be swithed according to the distance of the nearest neighbor. This distance is calculated by softmin. This parameter is the decaying parameter in the softmin. It is only required when `use_srtab` is provided.

    .. raw:: html

       <a id="model/sw_rmin"></a>
    sw_rmin: 
        | type: ``float``, optional
        | argument path: ``model/sw_rmin``

        The lower boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided.

    .. raw:: html

       <a id="model/sw_rmax"></a>
    sw_rmax: 
        | type: ``float``, optional
        | argument path: ``model/sw_rmax``

        The upper boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided.

    .. raw:: html

       <a id="model/descriptor"></a>
    descriptor: 
        | type: ``dict``
        | argument path: ``model/descriptor``

        The descriptor of atomic environment.


        Depending on the value of *type*, different sub args are accepted. 

        .. raw:: html

           <a id="model/descriptor/type"></a>
        type:
            | type: ``str`` (flag key)
            | argument path: ``model/descriptor/type`` 

            The type of the descritpor. Valid types are `loc_frame <#model/descriptor[loc_frame]>`__, `se_a <#model/descriptor[se_a]>`__, `se_r <#model/descriptor[se_r]>`__, `se_a_3be <#model/descriptor[se_a_3be]>`__, `se_a_tpe <#model/descriptor[se_a_tpe]>`__, `hybrid <#model/descriptor[hybrid]>`__. 

            - `loc_frame`: Defines a local frame at each atom, and the compute the descriptor as local coordinates under this frame.

            - `se_a`: Used by the smooth edition of Deep Potential. The full relative coordinates are used to construct the descriptor.

            - `se_r`: Used by the smooth edition of Deep Potential. Only the distance between atoms is used to construct the descriptor.

            - `se_a_3be`: Used by the smooth edition of Deep Potential. The full relative coordinates are used to construct the descriptor. Three-body embedding will be used by this descriptor.

            - `se_a_tpe`: Used by the smooth edition of Deep Potential. The full relative coordinates are used to construct the descriptor. Type embedding will be used by this descriptor.

            - `hybrid`: Concatenate of a list of descriptors as a new descriptor.

            - `se_ar`: A hybrid of `se_a` and `se_r`. Typically `se_a` has a smaller cut-off while the `se_r` has a larger cut-off. Deprecated, use `hybrid` instead.


        .. raw:: html

           <a id="model/descriptor[loc_frame]"></a>
        When *type* is set to ``loc_frame``: 

        .. raw:: html

           <a id="model/descriptor[loc_frame]/sel_a"></a>
        sel_a: 
            | type: ``list``
            | argument path: ``model/descriptor[loc_frame]/sel_a``

            A list of integers. The length of the list should be the same as the number of atom types in the system. `sel_a[i]` gives the selected number of type-i neighbors. The full relative coordinates of the neighbors are used by the descriptor.

        .. raw:: html

           <a id="model/descriptor[loc_frame]/sel_r"></a>
        sel_r: 
            | type: ``list``
            | argument path: ``model/descriptor[loc_frame]/sel_r``

            A list of integers. The length of the list should be the same as the number of atom types in the system. `sel_r[i]` gives the selected number of type-i neighbors. Only relative distance of the neighbors are used by the descriptor. sel_a[i] + sel_r[i] is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius.

        .. raw:: html

           <a id="model/descriptor[loc_frame]/rcut"></a>
        rcut: 
            | type: ``float``, optional, default: ``6.0``
            | argument path: ``model/descriptor[loc_frame]/rcut``

            The cut-off radius. The default value is 6.0

        .. raw:: html

           <a id="model/descriptor[loc_frame]/axis_rule"></a>
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


        .. raw:: html

           <a id="model/descriptor[se_a]"></a>
        When *type* is set to ``se_a``: 

        .. raw:: html

           <a id="model/descriptor[se_a]/sel"></a>
        sel: 
            | type: ``list``
            | argument path: ``model/descriptor[se_a]/sel``

            A list of integers. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius.

        .. raw:: html

           <a id="model/descriptor[se_a]/rcut"></a>
        rcut: 
            | type: ``float``, optional, default: ``6.0``
            | argument path: ``model/descriptor[se_a]/rcut``

            The cut-off radius.

        .. raw:: html

           <a id="model/descriptor[se_a]/rcut_smth"></a>
        rcut_smth: 
            | type: ``float``, optional, default: ``0.5``
            | argument path: ``model/descriptor[se_a]/rcut_smth``

            Where to start smoothing. For example the 1/r term is smoothed from `rcut` to `rcut_smth`

        .. raw:: html

           <a id="model/descriptor[se_a]/neuron"></a>
        neuron: 
            | type: ``list``, optional, default: ``[10, 20, 40]``
            | argument path: ``model/descriptor[se_a]/neuron``

            Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built.

        .. raw:: html

           <a id="model/descriptor[se_a]/axis_neuron"></a>
        axis_neuron: 
            | type: ``int``, optional, default: ``4``
            | argument path: ``model/descriptor[se_a]/axis_neuron``

            Size of the submatrix of G (embedding matrix).

        .. raw:: html

           <a id="model/descriptor[se_a]/activation_function"></a>
        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/descriptor[se_a]/activation_function``

            The activation function in the embedding net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        .. raw:: html

           <a id="model/descriptor[se_a]/resnet_dt"></a>
        resnet_dt: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_a]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        .. raw:: html

           <a id="model/descriptor[se_a]/type_one_side"></a>
        type_one_side: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_a]/type_one_side``

            Try to build N_types embedding nets. Otherwise, building N_types^2 embedding nets

        .. raw:: html

           <a id="model/descriptor[se_a]/precision"></a>
        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/descriptor[se_a]/precision``

            The precision of the embedding net parameters, supported options are "default", "float16", "float32", "float64".

        .. raw:: html

           <a id="model/descriptor[se_a]/trainable"></a>
        trainable: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/descriptor[se_a]/trainable``

            If the parameters in the embedding net is trainable

        .. raw:: html

           <a id="model/descriptor[se_a]/seed"></a>
        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/descriptor[se_a]/seed``

            Random seed for parameter initialization

        .. raw:: html

           <a id="model/descriptor[se_a]/exclude_types"></a>
        exclude_types: 
            | type: ``list``, optional, default: ``[]``
            | argument path: ``model/descriptor[se_a]/exclude_types``

            The Excluded types

        .. raw:: html

           <a id="model/descriptor[se_a]/set_davg_zero"></a>
        set_davg_zero: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_a]/set_davg_zero``

            Set the normalization average to zero. This option should be set when `atom_ener` in the energy fitting is used


        .. raw:: html

           <a id="model/descriptor[se_r]"></a>
        When *type* is set to ``se_r``: 

        .. raw:: html

           <a id="model/descriptor[se_r]/sel"></a>
        sel: 
            | type: ``list``
            | argument path: ``model/descriptor[se_r]/sel``

            A list of integers. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius.

        .. raw:: html

           <a id="model/descriptor[se_r]/rcut"></a>
        rcut: 
            | type: ``float``, optional, default: ``6.0``
            | argument path: ``model/descriptor[se_r]/rcut``

            The cut-off radius.

        .. raw:: html

           <a id="model/descriptor[se_r]/rcut_smth"></a>
        rcut_smth: 
            | type: ``float``, optional, default: ``0.5``
            | argument path: ``model/descriptor[se_r]/rcut_smth``

            Where to start smoothing. For example the 1/r term is smoothed from `rcut` to `rcut_smth`

        .. raw:: html

           <a id="model/descriptor[se_r]/neuron"></a>
        neuron: 
            | type: ``list``, optional, default: ``[10, 20, 40]``
            | argument path: ``model/descriptor[se_r]/neuron``

            Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built.

        .. raw:: html

           <a id="model/descriptor[se_r]/activation_function"></a>
        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/descriptor[se_r]/activation_function``

            The activation function in the embedding net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        .. raw:: html

           <a id="model/descriptor[se_r]/resnet_dt"></a>
        resnet_dt: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_r]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        .. raw:: html

           <a id="model/descriptor[se_r]/type_one_side"></a>
        type_one_side: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_r]/type_one_side``

            Try to build N_types embedding nets. Otherwise, building N_types^2 embedding nets

        .. raw:: html

           <a id="model/descriptor[se_r]/precision"></a>
        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/descriptor[se_r]/precision``

            The precision of the embedding net parameters, supported options are "default", "float16", "float32", "float64".

        .. raw:: html

           <a id="model/descriptor[se_r]/trainable"></a>
        trainable: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/descriptor[se_r]/trainable``

            If the parameters in the embedding net is trainable

        .. raw:: html

           <a id="model/descriptor[se_r]/seed"></a>
        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/descriptor[se_r]/seed``

            Random seed for parameter initialization

        .. raw:: html

           <a id="model/descriptor[se_r]/exclude_types"></a>
        exclude_types: 
            | type: ``list``, optional, default: ``[]``
            | argument path: ``model/descriptor[se_r]/exclude_types``

            The Excluded types

        .. raw:: html

           <a id="model/descriptor[se_r]/set_davg_zero"></a>
        set_davg_zero: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_r]/set_davg_zero``

            Set the normalization average to zero. This option should be set when `atom_ener` in the energy fitting is used


        .. raw:: html

           <a id="model/descriptor[se_a_3be]"></a>
        When *type* is set to ``se_a_3be``: 

        .. raw:: html

           <a id="model/descriptor[se_a_3be]/sel"></a>
        sel: 
            | type: ``list``
            | argument path: ``model/descriptor[se_a_3be]/sel``

            A list of integers. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius.

        .. raw:: html

           <a id="model/descriptor[se_a_3be]/rcut"></a>
        rcut: 
            | type: ``float``, optional, default: ``6.0``
            | argument path: ``model/descriptor[se_a_3be]/rcut``

            The cut-off radius.

        .. raw:: html

           <a id="model/descriptor[se_a_3be]/rcut_smth"></a>
        rcut_smth: 
            | type: ``float``, optional, default: ``0.5``
            | argument path: ``model/descriptor[se_a_3be]/rcut_smth``

            Where to start smoothing. For example the 1/r term is smoothed from `rcut` to `rcut_smth`

        .. raw:: html

           <a id="model/descriptor[se_a_3be]/neuron"></a>
        neuron: 
            | type: ``list``, optional, default: ``[10, 20, 40]``
            | argument path: ``model/descriptor[se_a_3be]/neuron``

            Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built.

        .. raw:: html

           <a id="model/descriptor[se_a_3be]/activation_function"></a>
        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/descriptor[se_a_3be]/activation_function``

            The activation function in the embedding net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        .. raw:: html

           <a id="model/descriptor[se_a_3be]/resnet_dt"></a>
        resnet_dt: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_a_3be]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        .. raw:: html

           <a id="model/descriptor[se_a_3be]/precision"></a>
        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/descriptor[se_a_3be]/precision``

            The precision of the embedding net parameters, supported options are "default", "float16", "float32", "float64".

        .. raw:: html

           <a id="model/descriptor[se_a_3be]/trainable"></a>
        trainable: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/descriptor[se_a_3be]/trainable``

            If the parameters in the embedding net is trainable

        .. raw:: html

           <a id="model/descriptor[se_a_3be]/seed"></a>
        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/descriptor[se_a_3be]/seed``

            Random seed for parameter initialization

        .. raw:: html

           <a id="model/descriptor[se_a_3be]/set_davg_zero"></a>
        set_davg_zero: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_a_3be]/set_davg_zero``

            Set the normalization average to zero. This option should be set when `atom_ener` in the energy fitting is used


        .. raw:: html

           <a id="model/descriptor[se_a_tpe]"></a>
        When *type* is set to ``se_a_tpe``: 

        .. raw:: html

           <a id="model/descriptor[se_a_tpe]/sel"></a>
        sel: 
            | type: ``list``
            | argument path: ``model/descriptor[se_a_tpe]/sel``

            A list of integers. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius.

        .. raw:: html

           <a id="model/descriptor[se_a_tpe]/rcut"></a>
        rcut: 
            | type: ``float``, optional, default: ``6.0``
            | argument path: ``model/descriptor[se_a_tpe]/rcut``

            The cut-off radius.

        .. raw:: html

           <a id="model/descriptor[se_a_tpe]/rcut_smth"></a>
        rcut_smth: 
            | type: ``float``, optional, default: ``0.5``
            | argument path: ``model/descriptor[se_a_tpe]/rcut_smth``

            Where to start smoothing. For example the 1/r term is smoothed from `rcut` to `rcut_smth`

        .. raw:: html

           <a id="model/descriptor[se_a_tpe]/neuron"></a>
        neuron: 
            | type: ``list``, optional, default: ``[10, 20, 40]``
            | argument path: ``model/descriptor[se_a_tpe]/neuron``

            Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built.

        .. raw:: html

           <a id="model/descriptor[se_a_tpe]/axis_neuron"></a>
        axis_neuron: 
            | type: ``int``, optional, default: ``4``
            | argument path: ``model/descriptor[se_a_tpe]/axis_neuron``

            Size of the submatrix of G (embedding matrix).

        .. raw:: html

           <a id="model/descriptor[se_a_tpe]/activation_function"></a>
        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/descriptor[se_a_tpe]/activation_function``

            The activation function in the embedding net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        .. raw:: html

           <a id="model/descriptor[se_a_tpe]/resnet_dt"></a>
        resnet_dt: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_a_tpe]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        .. raw:: html

           <a id="model/descriptor[se_a_tpe]/type_one_side"></a>
        type_one_side: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_a_tpe]/type_one_side``

            Try to build N_types embedding nets. Otherwise, building N_types^2 embedding nets

        .. raw:: html

           <a id="model/descriptor[se_a_tpe]/precision"></a>
        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/descriptor[se_a_tpe]/precision``

            The precision of the embedding net parameters, supported options are "default", "float16", "float32", "float64".

        .. raw:: html

           <a id="model/descriptor[se_a_tpe]/trainable"></a>
        trainable: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/descriptor[se_a_tpe]/trainable``

            If the parameters in the embedding net is trainable

        .. raw:: html

           <a id="model/descriptor[se_a_tpe]/seed"></a>
        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/descriptor[se_a_tpe]/seed``

            Random seed for parameter initialization

        .. raw:: html

           <a id="model/descriptor[se_a_tpe]/exclude_types"></a>
        exclude_types: 
            | type: ``list``, optional, default: ``[]``
            | argument path: ``model/descriptor[se_a_tpe]/exclude_types``

            The Excluded types

        .. raw:: html

           <a id="model/descriptor[se_a_tpe]/set_davg_zero"></a>
        set_davg_zero: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_a_tpe]/set_davg_zero``

            Set the normalization average to zero. This option should be set when `atom_ener` in the energy fitting is used

        .. raw:: html

           <a id="model/descriptor[se_a_tpe]/type_nchanl"></a>
        type_nchanl: 
            | type: ``int``, optional, default: ``4``
            | argument path: ``model/descriptor[se_a_tpe]/type_nchanl``

            number of channels for type embedding

        .. raw:: html

           <a id="model/descriptor[se_a_tpe]/type_nlayer"></a>
        type_nlayer: 
            | type: ``int``, optional, default: ``2``
            | argument path: ``model/descriptor[se_a_tpe]/type_nlayer``

            number of hidden layers of type embedding net

        .. raw:: html

           <a id="model/descriptor[se_a_tpe]/numb_aparam"></a>
        numb_aparam: 
            | type: ``int``, optional, default: ``0``
            | argument path: ``model/descriptor[se_a_tpe]/numb_aparam``

            dimension of atomic parameter. if set to a value > 0, the atomic parameters are embedded.


        .. raw:: html

           <a id="model/descriptor[hybrid]"></a>
        When *type* is set to ``hybrid``: 

        .. raw:: html

           <a id="model/descriptor[hybrid]/list"></a>
        list: 
            | type: ``list``
            | argument path: ``model/descriptor[hybrid]/list``

            A list of descriptor definitions


        .. raw:: html

           <a id="model/descriptor[se_ar]"></a>
        When *type* is set to ``se_ar``: 

        .. raw:: html

           <a id="model/descriptor[se_ar]/a"></a>
        a: 
            | type: ``dict``
            | argument path: ``model/descriptor[se_ar]/a``

            The parameters of descriptor `se_a <#model/descriptor[se_a]>`__

        .. raw:: html

           <a id="model/descriptor[se_ar]/r"></a>
        r: 
            | type: ``dict``
            | argument path: ``model/descriptor[se_ar]/r``

            The parameters of descriptor `se_r <#model/descriptor[se_r]>`__

    .. raw:: html

       <a id="model/fitting_net"></a>
    fitting_net: 
        | type: ``dict``
        | argument path: ``model/fitting_net``

        The fitting of physical properties.


        Depending on the value of *type*, different sub args are accepted. 

        .. raw:: html

           <a id="model/fitting_net/type"></a>
        type:
            | type: ``str`` (flag key), default: ``ener``
            | argument path: ``model/fitting_net/type`` 

            The type of the fitting. Valid types are `ener`, `dipole`, `polar` and `global_polar`. 

            - `ener`: Fit an energy model (potential energy surface).

            - `dipole`: Fit an atomic dipole model. Atomic dipole labels for all the selected atoms (see `sel_type`) should be provided by `dipole.npy` in each data system. The file has number of frames lines and 3 times of number of selected atoms columns.

            - `polar`: Fit an atomic polarizability model. Atomic polarizability labels for all the selected atoms (see `sel_type`) should be provided by `polarizability.npy` in each data system. The file has number of frames lines and 9 times of number of selected atoms columns.

            - `global_polar`: Fit a polarizability model. Polarizability labels should be provided by `polarizability.npy` in each data system. The file has number of frames lines and 9 columns.


        .. raw:: html

           <a id="model/fitting_net[ener]"></a>
        When *type* is set to ``ener``: 

        .. raw:: html

           <a id="model/fitting_net[ener]/numb_fparam"></a>
        numb_fparam: 
            | type: ``int``, optional, default: ``0``
            | argument path: ``model/fitting_net[ener]/numb_fparam``

            The dimension of the frame parameter. If set to >0, file `fparam.npy` should be included to provided the input fparams.

        .. raw:: html

           <a id="model/fitting_net[ener]/numb_aparam"></a>
        numb_aparam: 
            | type: ``int``, optional, default: ``0``
            | argument path: ``model/fitting_net[ener]/numb_aparam``

            The dimension of the atomic parameter. If set to >0, file `aparam.npy` should be included to provided the input aparams.

        .. raw:: html

           <a id="model/fitting_net[ener]/neuron"></a>
        neuron: 
            | type: ``list``, optional, default: ``[120, 120, 120]``
            | argument path: ``model/fitting_net[ener]/neuron``

            The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built.

        .. raw:: html

           <a id="model/fitting_net[ener]/activation_function"></a>
        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/fitting_net[ener]/activation_function``

            The activation function in the fitting net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        .. raw:: html

           <a id="model/fitting_net[ener]/precision"></a>
        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/fitting_net[ener]/precision``

            The precision of the fitting net parameters, supported options are "default", "float16", "float32", "float64".

        .. raw:: html

           <a id="model/fitting_net[ener]/resnet_dt"></a>
        resnet_dt: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/fitting_net[ener]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        .. raw:: html

           <a id="model/fitting_net[ener]/trainable"></a>
        trainable: 
            | type: ``bool`` | ``list``, optional, default: ``True``
            | argument path: ``model/fitting_net[ener]/trainable``

            Whether the parameters in the fitting net are trainable. This option can be

            - bool: True if all parameters of the fitting net are trainable, False otherwise.

            - list of bool: Specifies if each layer is trainable. Since the fitting net is composed by hidden layers followed by a output layer, the length of tihs list should be equal to len(`neuron`)+1.

        .. raw:: html

           <a id="model/fitting_net[ener]/rcond"></a>
        rcond: 
            | type: ``float``, optional, default: ``0.001``
            | argument path: ``model/fitting_net[ener]/rcond``

            The condition number used to determine the inital energy shift for each type of atoms.

        .. raw:: html

           <a id="model/fitting_net[ener]/seed"></a>
        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/fitting_net[ener]/seed``

            Random seed for parameter initialization of the fitting net

        .. raw:: html

           <a id="model/fitting_net[ener]/atom_ener"></a>
        atom_ener: 
            | type: ``list``, optional, default: ``[]``
            | argument path: ``model/fitting_net[ener]/atom_ener``

            Specify the atomic energy in vacuum for each type


        .. raw:: html

           <a id="model/fitting_net[dipole]"></a>
        When *type* is set to ``dipole``: 

        .. raw:: html

           <a id="model/fitting_net[dipole]/neuron"></a>
        neuron: 
            | type: ``list``, optional, default: ``[120, 120, 120]``
            | argument path: ``model/fitting_net[dipole]/neuron``

            The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built.

        .. raw:: html

           <a id="model/fitting_net[dipole]/activation_function"></a>
        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/fitting_net[dipole]/activation_function``

            The activation function in the fitting net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        .. raw:: html

           <a id="model/fitting_net[dipole]/resnet_dt"></a>
        resnet_dt: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/fitting_net[dipole]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        .. raw:: html

           <a id="model/fitting_net[dipole]/precision"></a>
        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/fitting_net[dipole]/precision``

            The precision of the fitting net parameters, supported options are "default", "float16", "float32", "float64".

        .. raw:: html

           <a id="model/fitting_net[dipole]/sel_type"></a>
        sel_type: 
            | type: ``int`` | ``NoneType`` | ``list``, optional
            | argument path: ``model/fitting_net[dipole]/sel_type``

            The atom types for which the atomic dipole will be provided. If not set, all types will be selected.

        .. raw:: html

           <a id="model/fitting_net[dipole]/seed"></a>
        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/fitting_net[dipole]/seed``

            Random seed for parameter initialization of the fitting net


        .. raw:: html

           <a id="model/fitting_net[polar]"></a>
        When *type* is set to ``polar``: 

        .. raw:: html

           <a id="model/fitting_net[polar]/neuron"></a>
        neuron: 
            | type: ``list``, optional, default: ``[120, 120, 120]``
            | argument path: ``model/fitting_net[polar]/neuron``

            The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built.

        .. raw:: html

           <a id="model/fitting_net[polar]/activation_function"></a>
        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/fitting_net[polar]/activation_function``

            The activation function in the fitting net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        .. raw:: html

           <a id="model/fitting_net[polar]/resnet_dt"></a>
        resnet_dt: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/fitting_net[polar]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        .. raw:: html

           <a id="model/fitting_net[polar]/precision"></a>
        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/fitting_net[polar]/precision``

            The precision of the fitting net parameters, supported options are "default", "float16", "float32", "float64".

        .. raw:: html

           <a id="model/fitting_net[polar]/fit_diag"></a>
        fit_diag: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/fitting_net[polar]/fit_diag``

            Fit the diagonal part of the rotational invariant polarizability matrix, which will be converted to normal polarizability matrix by contracting with the rotation matrix.

        .. raw:: html

           <a id="model/fitting_net[polar]/scale"></a>
        scale: 
            | type: ``float`` | ``list``, optional, default: ``1.0``
            | argument path: ``model/fitting_net[polar]/scale``

            The output of the fitting net (polarizability matrix) will be scaled by ``scale``

        .. raw:: html

           <a id="model/fitting_net[polar]/diag_shift"></a>
        diag_shift: 
            | type: ``float`` | ``list``, optional, default: ``0.0``
            | argument path: ``model/fitting_net[polar]/diag_shift``

            The diagonal part of the polarizability matrix  will be shifted by ``diag_shift``. The shift operation is carried out after ``scale``.

        .. raw:: html

           <a id="model/fitting_net[polar]/sel_type"></a>
        sel_type: 
            | type: ``int`` | ``NoneType`` | ``list``, optional
            | argument path: ``model/fitting_net[polar]/sel_type``

            The atom types for which the atomic polarizability will be provided. If not set, all types will be selected.

        .. raw:: html

           <a id="model/fitting_net[polar]/seed"></a>
        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/fitting_net[polar]/seed``

            Random seed for parameter initialization of the fitting net


        .. raw:: html

           <a id="model/fitting_net[global_polar]"></a>
        When *type* is set to ``global_polar``: 

        .. raw:: html

           <a id="model/fitting_net[global_polar]/neuron"></a>
        neuron: 
            | type: ``list``, optional, default: ``[120, 120, 120]``
            | argument path: ``model/fitting_net[global_polar]/neuron``

            The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built.

        .. raw:: html

           <a id="model/fitting_net[global_polar]/activation_function"></a>
        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/fitting_net[global_polar]/activation_function``

            The activation function in the fitting net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        .. raw:: html

           <a id="model/fitting_net[global_polar]/resnet_dt"></a>
        resnet_dt: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/fitting_net[global_polar]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        .. raw:: html

           <a id="model/fitting_net[global_polar]/precision"></a>
        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/fitting_net[global_polar]/precision``

            The precision of the fitting net parameters, supported options are "default", "float16", "float32", "float64".

        .. raw:: html

           <a id="model/fitting_net[global_polar]/fit_diag"></a>
        fit_diag: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/fitting_net[global_polar]/fit_diag``

            Fit the diagonal part of the rotational invariant polarizability matrix, which will be converted to normal polarizability matrix by contracting with the rotation matrix.

        .. raw:: html

           <a id="model/fitting_net[global_polar]/scale"></a>
        scale: 
            | type: ``float`` | ``list``, optional, default: ``1.0``
            | argument path: ``model/fitting_net[global_polar]/scale``

            The output of the fitting net (polarizability matrix) will be scaled by ``scale``

        .. raw:: html

           <a id="model/fitting_net[global_polar]/diag_shift"></a>
        diag_shift: 
            | type: ``float`` | ``list``, optional, default: ``0.0``
            | argument path: ``model/fitting_net[global_polar]/diag_shift``

            The diagonal part of the polarizability matrix  will be shifted by ``diag_shift``. The shift operation is carried out after ``scale``.

        .. raw:: html

           <a id="model/fitting_net[global_polar]/sel_type"></a>
        sel_type: 
            | type: ``int`` | ``NoneType`` | ``list``, optional
            | argument path: ``model/fitting_net[global_polar]/sel_type``

            The atom types for which the atomic polarizability will be provided. If not set, all types will be selected.

        .. raw:: html

           <a id="model/fitting_net[global_polar]/seed"></a>
        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/fitting_net[global_polar]/seed``

            Random seed for parameter initialization of the fitting net


.. raw:: html

   <a id="loss"></a>
loss: 
    | type: ``dict``, optional
    | argument path: ``loss``

    The definition of loss function. The type of the loss depends on the type of the fitting. For fitting type `ener`, the prefactors before energy, force, virial and atomic energy losses may be provided. For fitting type `dipole`, `polar` and `global_polar`, the loss may be an empty `dict` or unset.


    Depending on the value of *type*, different sub args are accepted. 

    .. raw:: html

       <a id="loss/type"></a>
    type:
        | type: ``str`` (flag key), default: ``ener``
        | argument path: ``loss/type`` 

        The type of the loss. For fitting type `ener`, the loss type should be set to `ener` or left unset. For tensorial fitting types `dipole`, `polar` and `global_polar`, the type should be left unset.
        \.


    .. raw:: html

       <a id="loss[ener]"></a>
    When *type* is set to ``ener``: 

    .. raw:: html

       <a id="loss[ener]/start_pref_e"></a>
    start_pref_e: 
        | type: ``float`` | ``int``, optional, default: ``0.02``
        | argument path: ``loss[ener]/start_pref_e``

        The prefactor of energy loss at the start of the training. Should be larger than or equal to 0. If set to none-zero value, the energy label should be provided by file energy.npy in each data system. If both start_pref_energy and limit_pref_energy are set to 0, then the energy will be ignored.

    .. raw:: html

       <a id="loss[ener]/limit_pref_e"></a>
    limit_pref_e: 
        | type: ``float`` | ``int``, optional, default: ``1.0``
        | argument path: ``loss[ener]/limit_pref_e``

        The prefactor of energy loss at the limit of the training, Should be larger than or equal to 0. i.e. the training step goes to infinity.

    .. raw:: html

       <a id="loss[ener]/start_pref_f"></a>
    start_pref_f: 
        | type: ``float`` | ``int``, optional, default: ``1000``
        | argument path: ``loss[ener]/start_pref_f``

        The prefactor of force loss at the start of the training. Should be larger than or equal to 0. If set to none-zero value, the force label should be provided by file force.npy in each data system. If both start_pref_force and limit_pref_force are set to 0, then the force will be ignored.

    .. raw:: html

       <a id="loss[ener]/limit_pref_f"></a>
    limit_pref_f: 
        | type: ``float`` | ``int``, optional, default: ``1.0``
        | argument path: ``loss[ener]/limit_pref_f``

        The prefactor of force loss at the limit of the training, Should be larger than or equal to 0. i.e. the training step goes to infinity.

    .. raw:: html

       <a id="loss[ener]/start_pref_v"></a>
    start_pref_v: 
        | type: ``float`` | ``int``, optional, default: ``0.0``
        | argument path: ``loss[ener]/start_pref_v``

        The prefactor of virial loss at the start of the training. Should be larger than or equal to 0. If set to none-zero value, the virial label should be provided by file virial.npy in each data system. If both start_pref_virial and limit_pref_virial are set to 0, then the virial will be ignored.

    .. raw:: html

       <a id="loss[ener]/limit_pref_v"></a>
    limit_pref_v: 
        | type: ``float`` | ``int``, optional, default: ``0.0``
        | argument path: ``loss[ener]/limit_pref_v``

        The prefactor of virial loss at the limit of the training, Should be larger than or equal to 0. i.e. the training step goes to infinity.

    .. raw:: html

       <a id="loss[ener]/start_pref_ae"></a>
    start_pref_ae: 
        | type: ``float`` | ``int``, optional, default: ``0.0``
        | argument path: ``loss[ener]/start_pref_ae``

        The prefactor of virial loss at the start of the training. Should be larger than or equal to 0. If set to none-zero value, the virial label should be provided by file virial.npy in each data system. If both start_pref_virial and limit_pref_virial are set to 0, then the virial will be ignored.

    .. raw:: html

       <a id="loss[ener]/limit_pref_ae"></a>
    limit_pref_ae: 
        | type: ``float`` | ``int``, optional, default: ``0.0``
        | argument path: ``loss[ener]/limit_pref_ae``

        The prefactor of virial loss at the limit of the training, Should be larger than or equal to 0. i.e. the training step goes to infinity.

    .. raw:: html

       <a id="loss[ener]/relative_f"></a>
    relative_f: 
        | type: ``float`` | ``NoneType``, optional
        | argument path: ``loss[ener]/relative_f``

        If provided, relative force error will be used in the loss. The difference of force will be normalized by the magnitude of the force in the label with a shift given by `relative_f`, i.e. DF_i / ( || F || + relative_f ) with DF denoting the difference between prediction and label and || F || denoting the L2 norm of the label.


.. raw:: html

   <a id="learning_rate"></a>
learning_rate: 
    | type: ``dict``
    | argument path: ``learning_rate``

    The definitio of learning rate


    Depending on the value of *type*, different sub args are accepted. 

    .. raw:: html

       <a id="learning_rate/type"></a>
    type:
        | type: ``str`` (flag key), default: ``exp``
        | argument path: ``learning_rate/type`` 

        The type of the learning rate. Current type `exp`, the exponentially decaying learning rate is supported.


    .. raw:: html

       <a id="learning_rate[exp]"></a>
    When *type* is set to ``exp``: 

    .. raw:: html

       <a id="learning_rate[exp]/start_lr"></a>
    start_lr: 
        | type: ``float``, optional, default: ``0.001``
        | argument path: ``learning_rate[exp]/start_lr``

        The learning rate the start of the training.

    .. raw:: html

       <a id="learning_rate[exp]/stop_lr"></a>
    stop_lr: 
        | type: ``float``, optional, default: ``1e-08``
        | argument path: ``learning_rate[exp]/stop_lr``

        The desired learning rate at the end of the training.

    .. raw:: html

       <a id="learning_rate[exp]/decay_steps"></a>
    decay_steps: 
        | type: ``int``, optional, default: ``5000``
        | argument path: ``learning_rate[exp]/decay_steps``

        The learning rate is decaying every this number of training steps.


.. raw:: html

   <a id="training"></a>
training: 
    | type: ``dict``
    | argument path: ``training``

    The training options

    .. raw:: html

       <a id="training/systems"></a>
    systems: 
        | type: ``list`` | ``str``
        | argument path: ``training/systems``

        The data systems. This key can be provided with a listthat specifies the systems, or be provided with a string by which the prefix of all systems are given and the list of the systems is automatically generated.

    .. raw:: html

       <a id="training/set_prefix"></a>
    set_prefix: 
        | type: ``str``, optional, default: ``set``
        | argument path: ``training/set_prefix``

        The prefix of the sets in the `systems <#training/systems>`__.

    .. raw:: html

       <a id="training/auto_prob"></a>
    auto_prob: 
        | type: ``str``, optional, default: ``prob_sys_size``
        | argument path: ``training/auto_prob``

        Determine the probability of systems automatically. The method is assigned by this key and can be

        - "prob_uniform"  : the probability all the systems are equal, namely 1.0/self.get_nsystems()

        - "prob_sys_size" : the probability of a system is proportional to the number of batches in the system

        - "prob_sys_size;stt_idx:end_idx:weight;stt_idx:end_idx:weight;..." : the list of systems is devided into blocks. A block is specified by `stt_idx:end_idx:weight`, where `stt_idx` is the starting index of the system, `end_idx` is then ending (not including) index of the system, the probabilities of the systems in this block sums up to `weight`, and the relatively probabilities within this block is proportional to the number of batches in the system.

    .. raw:: html

       <a id="training/sys_probs"></a>
    sys_probs: 
        | type: ``NoneType`` | ``list``, optional, default: ``None``
        | argument path: ``training/sys_probs``

        A list of float, should be of the same length as `train_systems`, specifying the probability of each system.

    .. raw:: html

       <a id="training/batch_size"></a>
    batch_size: 
        | type: ``int`` | ``list`` | ``str``, optional, default: ``auto``
        | argument path: ``training/batch_size``

        This key can be 

        - list: the length of which is the same as the `systems <#training/systems>`__. The batch size of each system is given by the elements of the list.

        - int: all `systems <#training/systems>`__ use the same batch size.

        - string "auto": automatically determines the batch size so that the batch_size times the number of atoms in the system is no less than 32.

        - string "auto:N": automatically determines the batch size so that the batch_size times the number of atoms in the system is no less than N.

    .. raw:: html

       <a id="training/numb_steps"></a>
    numb_steps: 
        | type: ``int``
        | argument path: ``training/numb_steps``

        Number of training batch. Each training uses one batch of data.

    .. raw:: html

       <a id="training/seed"></a>
    seed: 
        | type: ``int`` | ``NoneType``, optional
        | argument path: ``training/seed``

        The random seed for getting frames from the training data set.

    .. raw:: html

       <a id="training/disp_file"></a>
    disp_file: 
        | type: ``str``, optional, default: ``lcueve.out``
        | argument path: ``training/disp_file``

        The file for printing learning curve.

    .. raw:: html

       <a id="training/disp_freq"></a>
    disp_freq: 
        | type: ``int``, optional, default: ``1000``
        | argument path: ``training/disp_freq``

        The frequency of printing learning curve.

    .. raw:: html

       <a id="training/numb_test"></a>
    numb_test: 
        | type: ``int`` | ``list`` | ``str``, optional, default: ``1``
        | argument path: ``training/numb_test``

        Number of frames used for the test during training.

    .. raw:: html

       <a id="training/save_freq"></a>
    save_freq: 
        | type: ``int``, optional, default: ``1000``
        | argument path: ``training/save_freq``

        The frequency of saving check point.

    .. raw:: html

       <a id="training/save_ckpt"></a>
    save_ckpt: 
        | type: ``str``, optional, default: ``model.ckpt``
        | argument path: ``training/save_ckpt``

        The file name of saving check point.

    .. raw:: html

       <a id="training/disp_training"></a>
    disp_training: 
        | type: ``bool``, optional, default: ``True``
        | argument path: ``training/disp_training``

        Displaying verbose information during training.

    .. raw:: html

       <a id="training/time_training"></a>
    time_training: 
        | type: ``bool``, optional, default: ``True``
        | argument path: ``training/time_training``

        Timing durining training.

    .. raw:: html

       <a id="training/profiling"></a>
    profiling: 
        | type: ``bool``, optional, default: ``False``
        | argument path: ``training/profiling``

        Profiling during training.

    .. raw:: html

       <a id="training/profiling_file"></a>
    profiling_file: 
        | type: ``str``, optional, default: ``timeline.json``
        | argument path: ``training/profiling_file``

        Output file for profiling.

    .. raw:: html

       <a id="training/tensorboard"></a>
    tensorboard: 
        | type: ``bool``, optional, default: ``False``
        | argument path: ``training/tensorboard``

        Enable tensorboard

    .. raw:: html

       <a id="training/tensorboard_log_dir"></a>
    tensorboard_log_dir: 
        | type: ``str``, optional, default: ``log``
        | argument path: ``training/tensorboard_log_dir``

        The log directory of tensorboard outputs

