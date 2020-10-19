model: ``dict``
    Argument path: model

    type_map: ``list``, optional
        Argument path: model/type_map
        A list of strings. Give the name to each type of atoms.

    data_stat_nbatch: ``int``, optional
        Argument path: model/data_stat_nbatch
        The model determines the normalization from the statistics of the
        data. This key specifies the number of `frames` in each `system` used
        for statistics.

    descriptor: ``dict``
        Argument path: model/descriptor
        The descriptor of atomic environment.

        Depending on the value of *type*, different sub args are accepted. 

        type: ``str``
            Argument path: model/descriptor/type
            The type of the descritpor. Valid types are `loc_frame`, `se_a`,
            `se_r` and `se_ar`. 
            - `loc_frame`: Defines a local frame at each
            atom, and the compute the descriptor as local coordinates under this
            frame.
            - `se_a`: Used by the smooth edition of Deep Potential. The
            full relative coordinates are used to construct the descriptor.
            -
            `se_r`: Used by the smooth edition of Deep Potential. Only the
            distance between atoms is used to construct the descriptor.
            - `se_ar`:
            A hybrid of `se_a` and `se_r`. Typically `se_a` has a smaller cut-off
            while the `se_r` has a larger cut-off.

        When *type* is set to ``loc_frame``: 

        sel_a: ``list``
            Argument path: model/descriptor/loc_frame/sel_a
            A list of integers. The length of the list should be the same as the
            number of atom types in the system. `sel_a[i]` gives the selected
            number of type-i neighbors. The full relative coordinates of the
            neighbors are used by the descriptor.

        sel_r: ``list``
            Argument path: model/descriptor/loc_frame/sel_r
            A list of integers. The length of the list should be the same as the
            number of atom types in the system. `sel_r[i]` gives the selected
            number of type-i neighbors. Only relative distance of the neighbors
            are used by the descriptor. sel_a[i] + sel_r[i] is recommended to be
            larger than the maximally possible number of type-i neighbors in the
            cut-off radius.

        rcut: ``float``, optional
            Argument path: model/descriptor/loc_frame/rcut
            The cut-off radius. The default value is 6.0

        axis_rule: ``list``
            Argument path: model/descriptor/loc_frame/axis_rule
            A list of integers. The length should be 6 times of the number of
            types. 

            - axis_rule[i*6+0]: class of the atom defining the first axis
            of type-i atom. 0 for neighbors with full coordinates and 1 for
            neighbors only with relative distance.

            - axis_rule[i*6+1]: type of
            the atom defining the first axis of type-i atom.

            - axis_rule[i*6+2]:
            index of the axis atom defining the first axis. Note that the
            neighbors with the same class and type are sorted according to their
            relative distance.
            - axis_rule[i*6+3]: class of the atom defining the
            first axis of type-i atom. 0 for neighbors with full coordinates and 1
            for neighbors only with relative distance.
            - axis_rule[i*6+4]: type of
            the atom defining the second axis of type-i atom.
            - axis_rule[i*6+5]:
            class of the atom defining the second axis of type-i atom. 0 for
            neighbors with full coordinates and 1 for neighbors only with relative
            distance.

        When *type* is set to ``se_a``: 

        sel: ``list``
            Argument path: model/descriptor/se_a/sel
            A list of integers. The length of the list should be the same as the
            number of atom types in the system. `sel[i]` gives the selected number
            of type-i neighbors. `sel[i]` is recommended to be larger than the
            maximally possible number of type-i neighbors in the cut-off radius.

        rcut: ``float``, optional
            Argument path: model/descriptor/se_a/rcut
            The cut-off radius.

        rcut_smth: ``float``, optional
            Argument path: model/descriptor/se_a/rcut_smth
            Where to start smoothing. For example the 1/r term is smoothed from
            `rcut` to `rcut_smth`

        neuron: ``list``, optional
            Argument path: model/descriptor/se_a/neuron
            Number of neurons in each hidden layers of the embedding net. When two
            layers are of the same size or one layer is twice as large as the
            previous layer, a skip connection is built.

        axis_neuron: ``int``, optional
            Argument path: model/descriptor/se_a/axis_neuron
            Size of the submatrix of G (embedding matrix).

        activation_function: ``str``, optional
            Argument path: model/descriptor/se_a/activation_function
            The activation function in the embedding net. Supported activation
            functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        resnet_dt: ``bool``, optional
            Argument path: model/descriptor/se_a/resnet_dt
            Whether to use a "Timestep" in the skip connection

        type_one_side: ``bool``, optional
            Argument path: model/descriptor/se_a/type_one_side
            Try to build N_types embedding nets. Otherwise, building N_types^2
            embedding nets

        precision: ``str``, optional
            Argument path: model/descriptor/se_a/precision
            The precision of the embedding net parameters, supported options are
            "float64", "float32", "float16".

        trainable: ``bool``, optional
            Argument path: model/descriptor/se_a/trainable
            If the parameters in the embedding net is trainable

        seed: ``int``|``NoneType``, optional
            Argument path: model/descriptor/se_a/seed
            Random seed for parameter initialization

        exclude_types: ``list``, optional
            Argument path: model/descriptor/se_a/exclude_types
            The Excluded types

        set_davg_zero: ``bool``, optional
            Argument path: model/descriptor/se_a/set_davg_zero
            Set the normalization average to zero. This option should be set when
            `atom_ener` in the energy fitting is used

        When *type* is set to ``se_r``: 

        sel: ``list``
            Argument path: model/descriptor/se_r/sel
            A list of integers. The length of the list should be the same as the
            number of atom types in the system. `sel[i]` gives the selected number
            of type-i neighbors. `sel[i]` is recommended to be larger than the
            maximally possible number of type-i neighbors in the cut-off radius.

        rcut: ``float``, optional
            Argument path: model/descriptor/se_r/rcut
            The cut-off radius.

        rcut_smth: ``float``, optional
            Argument path: model/descriptor/se_r/rcut_smth
            Where to start smoothing. For example the 1/r term is smoothed from
            `rcut` to `rcut_smth`

        neuron: ``list``, optional
            Argument path: model/descriptor/se_r/neuron
            Number of neurons in each hidden layers of the embedding net. When two
            layers are of the same size or one layer is twice as large as the
            previous layer, a skip connection is built.

        activation_function: ``str``, optional
            Argument path: model/descriptor/se_r/activation_function
            The activation function in the embedding net. Supported activation
            functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        resnet_dt: ``bool``, optional
            Argument path: model/descriptor/se_r/resnet_dt
            Whether to use a "Timestep" in the skip connection

        type_one_side: ``bool``, optional
            Argument path: model/descriptor/se_r/type_one_side
            Try to build N_types embedding nets. Otherwise, building N_types^2
            embedding nets

        precision: ``str``, optional
            Argument path: model/descriptor/se_r/precision
            The precision of the embedding net parameters, supported options are
            "float64", "float32", "float16".

        trainable: ``bool``, optional
            Argument path: model/descriptor/se_r/trainable
            If the parameters in the embedding net is trainable

        seed: ``int``|``NoneType``, optional
            Argument path: model/descriptor/se_r/seed
            Random seed for parameter initialization

        exclude_types: ``list``, optional
            Argument path: model/descriptor/se_r/exclude_types
            The Excluded types

        set_davg_zero: ``bool``, optional
            Argument path: model/descriptor/se_r/set_davg_zero
            Set the normalization average to zero. This option should be set when
            `atom_ener` in the energy fitting is used

        When *type* is set to ``se_ar``: 

        a: ``dict``
            Argument path: model/descriptor/se_ar/a
            The parameters of descriptor `se_a`

        r: ``dict``
            Argument path: model/descriptor/se_ar/r
            The parameters of descriptor `se_r`

    fitting_net: ``dict``
        Argument path: model/fitting_net
        The fitting of physical properties.

        Depending on the value of *type*, different sub args are accepted. 

        type: ``str``, default: ``ener``
            Argument path: model/fitting_net/type
            The type of the fitting. Valid types are `ener`, `dipole`, `polar` and
            `global_polar`. 
            - `ener`: Fit an energy model (potential energy
            surface).
            - `dipole`: Fit an atomic dipole model. Atomic dipole labels
            for all the selected atoms (see `sel_type`) should be provided by
            `dipole.npy` in each data system. The file has number of frames lines
            and 3 times of number of selected atoms columns.
            - `polar`: Fit an
            atomic polarizability model. Atomic polarizability labels for all the
            selected atoms (see `sel_type`) should be provided by
            `polarizability.npy` in each data system. The file has number of
            frames lines and 9 times of number of selected atoms columns.
            -
            `global_polar`: Fit a polarizability model. Polarizability labels
            should be provided by `polarizability.npy` in each data system. The
            file has number of frames lines and 9 columns.

        When *type* is set to ``ener``: 

        numb_fparam: ``int``, optional
            Argument path: model/fitting_net/ener/numb_fparam
            The dimension of the frame parameter. If set to >0, file `fparam.npy`
            should be included to provided the input fparams.

        numb_aparam: ``int``, optional
            Argument path: model/fitting_net/ener/numb_aparam
            The dimension of the atomic parameter. If set to >0, file `aparam.npy`
            should be included to provided the input aparams.

        neuron: ``list``, optional
            Argument path: model/fitting_net/ener/neuron
            The number of neurons in each hidden layers of the fitting net. When
            two hidden layers are of the same size, a skip connection is built.

        activation_function: ``str``, optional
            Argument path: model/fitting_net/ener/activation_function
            The activation function in the fitting net. Supported activation
            functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        precision: ``str``, optional
            Argument path: model/fitting_net/ener/precision
            The precision of the fitting net parameters, supported options are
            "float64", "float32", "float16".

        resnet_dt: ``bool``, optional
            Argument path: model/fitting_net/ener/resnet_dt
            Whether to use a "Timestep" in the skip connection

        trainable: ``bool``|``list``, optional
            Argument path: model/fitting_net/ener/trainable
            Whether the parameters in the fitting net are trainable. This option
            can be
            - bool: True if all parameters of the fitting net are
            trainable, False otherwise.
            - list of bool: Specifies if each layer is
            trainable. Since the fitting net is composed by hidden layers followed
            by a output layer, the length of tihs list should be equal to
            len(`neuron`)+1.

        rcond: ``float``, optional
            Argument path: model/fitting_net/ener/rcond
            The condition number used to determine the inital energy shift for
            each type of atoms.

        seed: ``int``|``NoneType``, optional
            Argument path: model/fitting_net/ener/seed
            Random seed for parameter initialization of the fitting net

        atom_ener: ``list``, optional
            Argument path: model/fitting_net/ener/atom_ener
            Specify the atomic energy in vacuum for each type

        When *type* is set to ``dipole``: 

        neuron: ``list``, optional
            Argument path: model/fitting_net/dipole/neuron
            The number of neurons in each hidden layers of the fitting net. When
            two hidden layers are of the same size, a skip connection is built.

        activation_function: ``str``, optional
            Argument path: model/fitting_net/dipole/activation_function
            The activation function in the fitting net. Supported activation
            functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        resnet_dt: ``bool``, optional
            Argument path: model/fitting_net/dipole/resnet_dt
            Whether to use a "Timestep" in the skip connection

        precision: ``str``, optional
            Argument path: model/fitting_net/dipole/precision
            The precision of the fitting net parameters, supported options are
            "float64", "float32", "float16".

        sel_type: ``int``|``NoneType``|``list``, optional
            Argument path: model/fitting_net/dipole/sel_type
            The atom types for which the atomic dipole will be provided. If not
            set, all types will be selected.

        seed: ``int``|``NoneType``, optional
            Argument path: model/fitting_net/dipole/seed
            Random seed for parameter initialization of the fitting net

        When *type* is set to ``polar``: 

        neuron: ``list``, optional
            Argument path: model/fitting_net/polar/neuron
            The number of neurons in each hidden layers of the fitting net. When
            two hidden layers are of the same size, a skip connection is built.

        activation_function: ``str``, optional
            Argument path: model/fitting_net/polar/activation_function
            The activation function in the fitting net. Supported activation
            functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        resnet_dt: ``bool``, optional
            Argument path: model/fitting_net/polar/resnet_dt
            Whether to use a "Timestep" in the skip connection

        precision: ``str``, optional
            Argument path: model/fitting_net/polar/precision
            The precision of the fitting net parameters, supported options are
            "float64", "float32", "float16".

        fit_diag: ``bool``, optional
            Argument path: model/fitting_net/polar/fit_diag
            The diagonal part of the polarizability matrix  will be shifted by
            `fit_diag`. The shift operation is carried out after `scale`.

        scale: ``float``|``list``, optional
            Argument path: model/fitting_net/polar/scale
            The output of the fitting net (polarizability matrix) will be scaled
            by `scale`

        diag_shift: ``float``|``list``, optional
            Argument path: model/fitting_net/polar/diag_shift
            The diagonal part of the polarizability matrix  will be shifted by
            `fit_diag`. The shift operation is carried out after `scale`.

        sel_type: ``int``|``NoneType``|``list``, optional
            Argument path: model/fitting_net/polar/sel_type
            The atom types for which the atomic polarizability will be provided.
            If not set, all types will be selected.

        seed: ``int``|``NoneType``, optional
            Argument path: model/fitting_net/polar/seed
            Random seed for parameter initialization of the fitting net

        When *type* is set to ``global_polar``: 

        neuron: ``list``, optional
            Argument path: model/fitting_net/global_polar/neuron
            The number of neurons in each hidden layers of the fitting net. When
            two hidden layers are of the same size, a skip connection is built.

        activation_function: ``str``, optional
            Argument path: model/fitting_net/global_polar/activation_function
            The activation function in the fitting net. Supported activation
            functions are "relu", "relu6", "softplus", "sigmoid", "tanh", "gelu".

        resnet_dt: ``bool``, optional
            Argument path: model/fitting_net/global_polar/resnet_dt
            Whether to use a "Timestep" in the skip connection

        precision: ``str``, optional
            Argument path: model/fitting_net/global_polar/precision
            The precision of the fitting net parameters, supported options are
            "float64", "float32", "float16".

        fit_diag: ``bool``, optional
            Argument path: model/fitting_net/global_polar/fit_diag
            The diagonal part of the polarizability matrix  will be shifted by
            `fit_diag`. The shift operation is carried out after `scale`.

        scale: ``float``|``list``, optional
            Argument path: model/fitting_net/global_polar/scale
            The output of the fitting net (polarizability matrix) will be scaled
            by `scale`

        diag_shift: ``float``|``list``, optional
            Argument path: model/fitting_net/global_polar/diag_shift
            The diagonal part of the polarizability matrix  will be shifted by
            `fit_diag`. The shift operation is carried out after `scale`.

        sel_type: ``int``|``NoneType``|``list``, optional
            Argument path: model/fitting_net/global_polar/sel_type
            The atom types for which the atomic polarizability will be provided.
            If not set, all types will be selected.

        seed: ``int``|``NoneType``, optional
            Argument path: model/fitting_net/global_polar/seed
            Random seed for parameter initialization of the fitting net

loss: ``dict``
    Argument path: loss
    The definition of loss function. The type of the loss depends on the
    type of the fitting. For fitting type `ener`, the prefactors before
    energy, force, virial and atomic energy losses may be provided. For
    fitting type `dipole`, `polar` and `global_polar`, the loss may be an
    empty `dict` or unset.

    Depending on the value of *type*, different sub args are accepted. 

    type: ``str``, default: ``ener``
        Argument path: loss/type
        The type of the loss. For fitting type `ener`, the loss type should be
        set to `ener` or left unset. For tensorial fitting types `dipole`,
        `polar` and `global_polar`, the type should be left unset.
        \.

    When *type* is set to ``ener``: 

    start_pref_e: ``float``|``int``, optional
        Argument path: loss/ener/start_pref_e
        The prefactor of energy loss at the start of the training. Should be
        larger than or equal to 0. If set to none-zero value, the energy label
        should be provided by file energy.npy in each data system. If both
        start_pref_energy and limit_pref_energy are set to 0, then the energy
        will be ignored.

    limit_pref_e: ``float``|``int``, optional
        Argument path: loss/ener/limit_pref_e
        The prefactor of energy loss at the limit of the training, Should be
        larger than or equal to 0. i.e. the training step goes to infinity.

    start_pref_f: ``float``|``int``, optional
        Argument path: loss/ener/start_pref_f
        The prefactor of force loss at the start of the training. Should be
        larger than or equal to 0. If set to none-zero value, the force label
        should be provided by file force.npy in each data system. If both
        start_pref_force and limit_pref_force are set to 0, then the force
        will be ignored.

    limit_pref_f: ``float``|``int``, optional
        Argument path: loss/ener/limit_pref_f
        The prefactor of force loss at the limit of the training, Should be
        larger than or equal to 0. i.e. the training step goes to infinity.

    start_pref_v: ``float``|``int``, optional
        Argument path: loss/ener/start_pref_v
        The prefactor of virial loss at the start of the training. Should be
        larger than or equal to 0. If set to none-zero value, the virial label
        should be provided by file virial.npy in each data system. If both
        start_pref_virial and limit_pref_virial are set to 0, then the virial
        will be ignored.

    limit_pref_v: ``float``|``int``, optional
        Argument path: loss/ener/limit_pref_v
        The prefactor of virial loss at the limit of the training, Should be
        larger than or equal to 0. i.e. the training step goes to infinity.

    start_pref_ae: ``float``|``int``, optional
        Argument path: loss/ener/start_pref_ae
        The prefactor of virial loss at the start of the training. Should be
        larger than or equal to 0. If set to none-zero value, the virial label
        should be provided by file virial.npy in each data system. If both
        start_pref_virial and limit_pref_virial are set to 0, then the virial
        will be ignored.

    limit_pref_ae: ``float``|``int``, optional
        Argument path: loss/ener/limit_pref_ae
        The prefactor of virial loss at the limit of the training, Should be
        larger than or equal to 0. i.e. the training step goes to infinity.

    relative_f: ``float``|``NoneType``, optional
        Argument path: loss/ener/relative_f
        If provided, relative force error will be used in the loss. The
        difference of force will be normalized by the magnitude of the force
        in the label with a shift given by `relative_f`, i.e. DF_i / ( || F ||
        + relative_f ) with DF denoting the difference between prediction and
        label and || F || denoting the L2 norm of the label.

learning_rate: ``dict``
    Argument path: learning_rate
    The learning rate options

    start_lr: ``float``, optional
        Argument path: learning_rate/start_lr
        The learning rate the start of the training.

    stop_lr: ``float``, optional
        Argument path: learning_rate/stop_lr
        The desired learning rate at the end of the training.

    decay_steps: ``int``, optional
        Argument path: learning_rate/decay_steps
        The learning rate is decaying every this number of training steps.

training: ``dict``
    Argument path: training
    The training options

    systems: ``list``|``str``
        Argument path: training/systems
        The data systems. This key can be provided with a listthat specifies
        the systems, or be provided with a string by which the prefix of all
        systems are given and the list of the systems is automatically
        generated.

    set_prefix: ``str``, optional
        Argument path: training/set_prefix
        The prefix of the sets in the systems.

    stop_batch: ``int``
        Argument path: training/stop_batch
        Number of training batch. Each training uses one batch of data.

    batch_size: ``int``|``list``|``str``, optional
        Argument path: training/batch_size
        This key can be 
        - list: the length of which is the same as the
        `systems`. The batch size of each system is given by the elements of
        the list.
        - int: all `systems` uses the same batch size.
        - string
        "auto": automatically determines the batch size os that the batch_size
        times the number of atoms in the system is no less than 32.
        - string
        "auto:N": automatically determines the batch size os that the
        batch_size times the number of atoms in the system is no less than N.

    seed: ``int``|``NoneType``, optional
        Argument path: training/seed
        The random seed for training.

    disp_file: ``str``, optional
        Argument path: training/disp_file
        The file for printing learning curve.

    disp_freq: ``int``, optional
        Argument path: training/disp_freq
        The frequency of printing learning curve.

    numb_test: ``int``, optional
        Argument path: training/numb_test
        Number of frames used for the test during training.

    save_freq: ``int``, optional
        Argument path: training/save_freq
        The frequency of saving check point.

    save_ckpt: ``str``, optional
        Argument path: training/save_ckpt
        The file name of saving check point.

    disp_training: ``bool``, optional
        Argument path: training/disp_training
        Displaying verbose information during training.

    time_training: ``bool``, optional
        Argument path: training/time_training
        Timing durining training.

    profiling: ``bool``, optional
        Argument path: training/profiling
        Profiling during training.

    profiling_file: ``str``, optional
        Argument path: training/profiling_file
        Output file for profiling.
