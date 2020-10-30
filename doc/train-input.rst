model: 
    | type: ``dict``
    | argument path: ``model``

    type_map: 
        | type: ``list``, optional
        | argument path: ``model/type_map``

        A list of strings. Give the name to each type of atoms.

    data_stat_nbatch: 
        | type: ``int``, optional, default: ``10``
        | argument path: ``model/data_stat_nbatch``

        The model determines the normalization from the statistics of the data. This key specifies the number of `frames` in each `system` used for statistics.

    descriptor: 
        | type: ``dict``
        | argument path: ``model/descriptor``

        The descriptor of atomic environment.


        Depending on the value of *type*, different sub args are accepted. 

        type:
            | type: ``str`` (flag key)
            | argument path: ``model/descriptor/type`` 


        When *type* is set to ``loc_frame``: 

        sel_a: 
            | type: ``list``
            | argument path: ``model/descriptor[loc_frame]/sel_a``

            A list of integers. The length of the list should be the same as the number of atom types in the system. `sel_a[i]` gives the selected number of type-i neighbors. The full relative coordinates of the neighbors are used by the descriptor.

        sel_r: 
            | type: ``list``
            | argument path: ``model/descriptor[loc_frame]/sel_r``

            A list of integers. The length of the list should be the same as the number of atom types in the system. `sel_r[i]` gives the selected number of type-i neighbors. Only relative distance of the neighbors are used by the descriptor. sel_a[i] + sel_r[i] is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius.

        rcut: 
            | type: ``float``, optional, default: ``6.0``
            | argument path: ``model/descriptor[loc_frame]/rcut``

            The cut-off radius. The default value is 6.0

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


        When *type* is set to ``se_a``: 

        sel: 
            | type: ``list``
            | argument path: ``model/descriptor[se_a]/sel``

            A list of integers. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius.

        rcut: 
            | type: ``float``, optional, default: ``6.0``
            | argument path: ``model/descriptor[se_a]/rcut``

            The cut-off radius.

        rcut_smth: 
            | type: ``float``, optional, default: ``0.5``
            | argument path: ``model/descriptor[se_a]/rcut_smth``

            Where to start smoothing. For example the 1/r term is smoothed from `rcut` to `rcut_smth`

        neuron: 
            | type: ``list``, optional, default: ``[10, 20, 40]``
            | argument path: ``model/descriptor[se_a]/neuron``

            Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built.

        axis_neuron: 
            | type: ``int``, optional, default: ``4``
            | argument path: ``model/descriptor[se_a]/axis_neuron``

            Size of the submatrix of G (embedding matrix).

        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/descriptor[se_a]/activation_function``

            The activation function in the embedding net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        resnet_dt: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_a]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        type_one_side: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_a]/type_one_side``

            Try to build N_types embedding nets. Otherwise, building N_types^2 embedding nets

        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/descriptor[se_a]/precision``

            The precision of the embedding net parameters, supported options are "float64", "float32", "float16".

        trainable: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/descriptor[se_a]/trainable``

            If the parameters in the embedding net is trainable

        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/descriptor[se_a]/seed``

            Random seed for parameter initialization

        exclude_types: 
            | type: ``list``, optional, default: ``[]``
            | argument path: ``model/descriptor[se_a]/exclude_types``

            The Excluded types

        set_davg_zero: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_a]/set_davg_zero``

            Set the normalization average to zero. This option should be set when `atom_ener` in the energy fitting is used


        When *type* is set to ``se_r``: 

        sel: 
            | type: ``list``
            | argument path: ``model/descriptor[se_r]/sel``

            A list of integers. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius.

        rcut: 
            | type: ``float``, optional, default: ``6.0``
            | argument path: ``model/descriptor[se_r]/rcut``

            The cut-off radius.

        rcut_smth: 
            | type: ``float``, optional, default: ``0.5``
            | argument path: ``model/descriptor[se_r]/rcut_smth``

            Where to start smoothing. For example the 1/r term is smoothed from `rcut` to `rcut_smth`

        neuron: 
            | type: ``list``, optional, default: ``[10, 20, 40]``
            | argument path: ``model/descriptor[se_r]/neuron``

            Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built.

        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/descriptor[se_r]/activation_function``

            The activation function in the embedding net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        resnet_dt: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_r]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        type_one_side: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_r]/type_one_side``

            Try to build N_types embedding nets. Otherwise, building N_types^2 embedding nets

        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/descriptor[se_r]/precision``

            The precision of the embedding net parameters, supported options are "float64", "float32", "float16".

        trainable: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/descriptor[se_r]/trainable``

            If the parameters in the embedding net is trainable

        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/descriptor[se_r]/seed``

            Random seed for parameter initialization

        exclude_types: 
            | type: ``list``, optional, default: ``[]``
            | argument path: ``model/descriptor[se_r]/exclude_types``

            The Excluded types

        set_davg_zero: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model/descriptor[se_r]/set_davg_zero``

            Set the normalization average to zero. This option should be set when `atom_ener` in the energy fitting is used


        When *type* is set to ``se_ar``: 

        a: 
            | type: ``dict``
            | argument path: ``model/descriptor[se_ar]/a``

            The parameters of descriptor `se_a`

        r: 
            | type: ``dict``
            | argument path: ``model/descriptor[se_ar]/r``

            The parameters of descriptor `se_r`

    fitting_net: 
        | type: ``dict``
        | argument path: ``model/fitting_net``

        The fitting of physical properties.


        Depending on the value of *type*, different sub args are accepted. 

        type:
            | type: ``str`` (flag key), default: ``ener``
            | argument path: ``model/fitting_net/type`` 


        When *type* is set to ``ener``: 

        numb_fparam: 
            | type: ``int``, optional, default: ``0``
            | argument path: ``model/fitting_net[ener]/numb_fparam``

            The dimension of the frame parameter. If set to >0, file `fparam.npy` should be included to provided the input fparams.

        numb_aparam: 
            | type: ``int``, optional, default: ``0``
            | argument path: ``model/fitting_net[ener]/numb_aparam``

            The dimension of the atomic parameter. If set to >0, file `aparam.npy` should be included to provided the input aparams.

        neuron: 
            | type: ``list``, optional, default: ``[120, 120, 120]``
            | argument path: ``model/fitting_net[ener]/neuron``

            The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built.

        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/fitting_net[ener]/activation_function``

            The activation function in the fitting net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/fitting_net[ener]/precision``

            The precision of the fitting net parameters, supported options are "float64", "float32", "float16".

        resnet_dt: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/fitting_net[ener]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        trainable: 
            | type: ``list`` | ``bool``, optional, default: ``True``
            | argument path: ``model/fitting_net[ener]/trainable``

            Whether the parameters in the fitting net are trainable. This option can be

            - bool: True if all parameters of the fitting net are trainable, False otherwise.

            - list of bool: Specifies if each layer is trainable. Since the fitting net is composed by hidden layers followed by a output layer, the length of tihs list should be equal to len(`neuron`)+1.

        rcond: 
            | type: ``float``, optional, default: ``0.001``
            | argument path: ``model/fitting_net[ener]/rcond``

            The condition number used to determine the inital energy shift for each type of atoms.

        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/fitting_net[ener]/seed``

            Random seed for parameter initialization of the fitting net

        atom_ener: 
            | type: ``list``, optional, default: ``[]``
            | argument path: ``model/fitting_net[ener]/atom_ener``

            Specify the atomic energy in vacuum for each type


        When *type* is set to ``dipole``: 

        neuron: 
            | type: ``list``, optional, default: ``[120, 120, 120]``
            | argument path: ``model/fitting_net[dipole]/neuron``

            The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built.

        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/fitting_net[dipole]/activation_function``

            The activation function in the fitting net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        resnet_dt: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/fitting_net[dipole]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/fitting_net[dipole]/precision``

            The precision of the fitting net parameters, supported options are "float64", "float32", "float16".

        sel_type: 
            | type: ``list`` | ``int`` | ``NoneType``, optional
            | argument path: ``model/fitting_net[dipole]/sel_type``

            The atom types for which the atomic dipole will be provided. If not set, all types will be selected.

        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/fitting_net[dipole]/seed``

            Random seed for parameter initialization of the fitting net


        When *type* is set to ``polar``: 

        neuron: 
            | type: ``list``, optional, default: ``[120, 120, 120]``
            | argument path: ``model/fitting_net[polar]/neuron``

            The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built.

        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/fitting_net[polar]/activation_function``

            The activation function in the fitting net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        resnet_dt: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/fitting_net[polar]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/fitting_net[polar]/precision``

            The precision of the fitting net parameters, supported options are "float64", "float32", "float16".

        fit_diag: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/fitting_net[polar]/fit_diag``

            Fit the diagonal part of the rotational invariant polarizability matrix, which will be converted to normal polarizability matrix by contracting with the rotation matrix.

        scale: 
            | type: ``list`` | ``float``, optional, default: ``1.0``
            | argument path: ``model/fitting_net[polar]/scale``

            The output of the fitting net (polarizability matrix) will be scaled by ``scale``

        diag_shift: 
            | type: ``list`` | ``float``, optional, default: ``0.0``
            | argument path: ``model/fitting_net[polar]/diag_shift``

            The diagonal part of the polarizability matrix  will be shifted by ``diag_shift``. The shift operation is carried out after ``scale``.

        sel_type: 
            | type: ``list`` | ``int`` | ``NoneType``, optional
            | argument path: ``model/fitting_net[polar]/sel_type``

            The atom types for which the atomic polarizability will be provided. If not set, all types will be selected.

        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/fitting_net[polar]/seed``

            Random seed for parameter initialization of the fitting net


        When *type* is set to ``global_polar``: 

        neuron: 
            | type: ``list``, optional, default: ``[120, 120, 120]``
            | argument path: ``model/fitting_net[global_polar]/neuron``

            The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built.

        activation_function: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model/fitting_net[global_polar]/activation_function``

            The activation function in the fitting net. Supported activation functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        resnet_dt: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/fitting_net[global_polar]/resnet_dt``

            Whether to use a "Timestep" in the skip connection

        precision: 
            | type: ``str``, optional, default: ``float64``
            | argument path: ``model/fitting_net[global_polar]/precision``

            The precision of the fitting net parameters, supported options are "float64", "float32", "float16".

        fit_diag: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model/fitting_net[global_polar]/fit_diag``

            Fit the diagonal part of the rotational invariant polarizability matrix, which will be converted to normal polarizability matrix by contracting with the rotation matrix.

        scale: 
            | type: ``list`` | ``float``, optional, default: ``1.0``
            | argument path: ``model/fitting_net[global_polar]/scale``

            The output of the fitting net (polarizability matrix) will be scaled by ``scale``

        diag_shift: 
            | type: ``list`` | ``float``, optional, default: ``0.0``
            | argument path: ``model/fitting_net[global_polar]/diag_shift``

            The diagonal part of the polarizability matrix  will be shifted by ``diag_shift``. The shift operation is carried out after ``scale``.

        sel_type: 
            | type: ``list`` | ``int`` | ``NoneType``, optional
            | argument path: ``model/fitting_net[global_polar]/sel_type``

            The atom types for which the atomic polarizability will be provided. If not set, all types will be selected.

        seed: 
            | type: ``int`` | ``NoneType``, optional
            | argument path: ``model/fitting_net[global_polar]/seed``

            Random seed for parameter initialization of the fitting net


loss: 
    | type: ``dict``
    | argument path: ``loss``

    The definition of loss function. The type of the loss depends on the type of the fitting. For fitting type `ener`, the prefactors before energy, force, virial and atomic energy losses may be provided. For fitting type `dipole`, `polar` and `global_polar`, the loss may be an empty `dict` or unset.


    Depending on the value of *type*, different sub args are accepted. 

    type:
        | type: ``str`` (flag key), default: ``ener``
        | argument path: ``loss/type`` 


    When *type* is set to ``ener``: 

    start_pref_e: 
        | type: ``int`` | ``float``, optional, default: ``0.02``
        | argument path: ``loss[ener]/start_pref_e``

        The prefactor of energy loss at the start of the training. Should be larger than or equal to 0. If set to none-zero value, the energy label should be provided by file energy.npy in each data system. If both start_pref_energy and limit_pref_energy are set to 0, then the energy will be ignored.

    limit_pref_e: 
        | type: ``int`` | ``float``, optional, default: ``1.0``
        | argument path: ``loss[ener]/limit_pref_e``

        The prefactor of energy loss at the limit of the training, Should be larger than or equal to 0. i.e. the training step goes to infinity.

    start_pref_f: 
        | type: ``int`` | ``float``, optional, default: ``1000``
        | argument path: ``loss[ener]/start_pref_f``

        The prefactor of force loss at the start of the training. Should be larger than or equal to 0. If set to none-zero value, the force label should be provided by file force.npy in each data system. If both start_pref_force and limit_pref_force are set to 0, then the force will be ignored.

    limit_pref_f: 
        | type: ``int`` | ``float``, optional, default: ``1.0``
        | argument path: ``loss[ener]/limit_pref_f``

        The prefactor of force loss at the limit of the training, Should be larger than or equal to 0. i.e. the training step goes to infinity.

    start_pref_v: 
        | type: ``int`` | ``float``, optional, default: ``0.0``
        | argument path: ``loss[ener]/start_pref_v``

        The prefactor of virial loss at the start of the training. Should be larger than or equal to 0. If set to none-zero value, the virial label should be provided by file virial.npy in each data system. If both start_pref_virial and limit_pref_virial are set to 0, then the virial will be ignored.

    limit_pref_v: 
        | type: ``int`` | ``float``, optional, default: ``0.0``
        | argument path: ``loss[ener]/limit_pref_v``

        The prefactor of virial loss at the limit of the training, Should be larger than or equal to 0. i.e. the training step goes to infinity.

    start_pref_ae: 
        | type: ``int`` | ``float``, optional, default: ``0.0``
        | argument path: ``loss[ener]/start_pref_ae``

        The prefactor of virial loss at the start of the training. Should be larger than or equal to 0. If set to none-zero value, the virial label should be provided by file virial.npy in each data system. If both start_pref_virial and limit_pref_virial are set to 0, then the virial will be ignored.

    limit_pref_ae: 
        | type: ``int`` | ``float``, optional, default: ``0.0``
        | argument path: ``loss[ener]/limit_pref_ae``

        The prefactor of virial loss at the limit of the training, Should be larger than or equal to 0. i.e. the training step goes to infinity.

    relative_f: 
        | type: ``float`` | ``NoneType``, optional
        | argument path: ``loss[ener]/relative_f``

        If provided, relative force error will be used in the loss. The difference of force will be normalized by the magnitude of the force in the label with a shift given by `relative_f`, i.e. DF_i / ( || F || + relative_f ) with DF denoting the difference between prediction and label and || F || denoting the L2 norm of the label.


learning_rate: 
    | type: ``dict``
    | argument path: ``learning_rate``

    The learning rate options

    start_lr: 
        | type: ``float``, optional, default: ``0.001``
        | argument path: ``learning_rate/start_lr``

        The learning rate the start of the training.

    stop_lr: 
        | type: ``float``, optional, default: ``1e-08``
        | argument path: ``learning_rate/stop_lr``

        The desired learning rate at the end of the training.

    decay_steps: 
        | type: ``int``, optional, default: ``5000``
        | argument path: ``learning_rate/decay_steps``

        The learning rate is decaying every this number of training steps.


training: 
    | type: ``dict``
    | argument path: ``training``

    The training options

    systems: 
        | type: ``list`` | ``str``
        | argument path: ``training/systems``

        The data systems. This key can be provided with a listthat specifies the systems, or be provided with a string by which the prefix of all systems are given and the list of the systems is automatically generated.

    set_prefix: 
        | type: ``str``, optional, default: ``set``
        | argument path: ``training/set_prefix``

        The prefix of the sets in the systems.

    stop_batch: 
        | type: ``int``
        | argument path: ``training/stop_batch``

        Number of training batch. Each training uses one batch of data.

    batch_size: 
        | type: ``list`` | ``str`` | ``int``, optional, default: ``auto``
        | argument path: ``training/batch_size``

        This key can be 

        - list: the length of which is the same as the `systems`. The batch size of each system is given by the elements of the list.

        - int: all `systems` uses the same batch size.

        - string "auto": automatically determines the batch size os that the batch_size times the number of atoms in the system is no less than 32.

        - string "auto:N": automatically determines the batch size os that the batch_size times the number of atoms in the system is no less than N.

    seed: 
        | type: ``int`` | ``NoneType``, optional
        | argument path: ``training/seed``

        The random seed for training.

    disp_file: 
        | type: ``str``, optional, default: ``lcueve.out``
        | argument path: ``training/disp_file``

        The file for printing learning curve.

    disp_freq: 
        | type: ``int``, optional, default: ``1000``
        | argument path: ``training/disp_freq``

        The frequency of printing learning curve.

    numb_test: 
        | type: ``int``, optional, default: ``1``
        | argument path: ``training/numb_test``

        Number of frames used for the test during training.

    save_freq: 
        | type: ``int``, optional, default: ``1000``
        | argument path: ``training/save_freq``

        The frequency of saving check point.

    save_ckpt: 
        | type: ``str``, optional, default: ``model.ckpt``
        | argument path: ``training/save_ckpt``

        The file name of saving check point.

    disp_training: 
        | type: ``bool``, optional, default: ``True``
        | argument path: ``training/disp_training``

        Displaying verbose information during training.

    time_training: 
        | type: ``bool``, optional, default: ``True``
        | argument path: ``training/time_training``

        Timing durining training.

    profiling: 
        | type: ``bool``, optional, default: ``False``
        | argument path: ``training/profiling``

        Profiling during training.

    profiling_file: 
        | type: ``str``, optional, default: ``timeline.json``
        | argument path: ``training/profiling_file``

        Output file for profiling.

