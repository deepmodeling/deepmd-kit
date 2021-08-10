from dargs import dargs, Argument, Variant
from .argcheck import list_to_doc, make_link
from .argcheck import type_embedding_args, descrpt_se_a_tpe_args, modifier_dipole_charge, modifier_variant_type_args
from .argcheck import model_compression, model_compression_type_args
from .argcheck import start_pref, limit_pref
from .argcheck import training_data_args, validation_data_args, make_index, gen_doc
# from deepmd.common import ACTIVATION_FN_DICT, PRECISION_DICT
ACTIVATION_FN_DICT = {}
PRECISION_DICT = {}



#  --- Descriptor configurations: --- #
def descrpt_local_frame_args():
    doc_sel_a = 'A list of integers. The length of the list should be the same as the number of atom types in the system. `sel_a[i]` gives the selected number of type-i neighbors. The full relative coordinates of the neighbors are used by the descriptor.'
    doc_sel_r = 'A list of integers. The length of the list should be the same as the number of atom types in the system. `sel_r[i]` gives the selected number of type-i neighbors. Only relative distance of the neighbors are used by the descriptor. sel_a[i] + sel_r[i] is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius.'
    doc_rcut = 'The cut-off radius. The default value is 6.0'
    doc_axis_rule = 'A list of integers. The length should be 6 times of the number of types. \n\n\
- axis_rule[i*6+0]: class of the atom defining the first axis of type-i atom. 0 for neighbors with full coordinates and 1 for neighbors only with relative distance.\n\n\
- axis_rule[i*6+1]: type of the atom defining the first axis of type-i atom.\n\n\
- axis_rule[i*6+2]: index of the axis atom defining the first axis. Note that the neighbors with the same class and type are sorted according to their relative distance.\n\n\
- axis_rule[i*6+3]: class of the atom defining the first axis of type-i atom. 0 for neighbors with full coordinates and 1 for neighbors only with relative distance.\n\n\
- axis_rule[i*6+4]: type of the atom defining the second axis of type-i atom.\n\n\
- axis_rule[i*6+5]: class of the atom defining the second axis of type-i atom. 0 for neighbors with full coordinates and 1 for neighbors only with relative distance.'

    return [
        Argument("name", str, optional=True, default='', doc="Descriptor name."),
        Argument("sel_a", list, optional=False, doc=doc_sel_a),
        Argument("sel_r", list, optional=False, doc=doc_sel_r),
        Argument("rcut", float, optional=True, default=6.0, doc=doc_rcut),
        Argument("axis_rule", list, optional=False, doc=doc_axis_rule)
    ]


def descrpt_se_a_args():
    doc_sel = 'A list of integers. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius. It is noted that the total sel value must be less than 4096 in a GPU environment.'
    doc_rcut = 'The cut-off radius.'
    doc_rcut_smth = 'Where to start smoothing. For example the 1/r term is smoothed from `rcut` to `rcut_smth`'
    doc_neuron = 'Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built.'
    doc_axis_neuron = 'Size of the submatrix of G (embedding matrix).'
    doc_activation_function = f'The activation function in the embedding net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())}'
    doc_resnet_dt = 'Whether to use a "Timestep" in the skip connection'
    doc_type_one_side = 'Try to build N_types embedding nets. Otherwise, building N_types^2 embedding nets'
    doc_precision = f'The precision of the embedding net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())}'
    doc_trainable = 'If the parameters in the embedding net is trainable'
    doc_seed = 'Random seed for parameter initialization'
    doc_exclude_types = 'The Excluded types'
    doc_set_davg_zero = 'Set the normalization average to zero. This option should be set when `atom_ener` in the energy fitting is used'

    return [
        Argument("name", str, optional=True, default='', doc="Descriptor name."),
        Argument("sel", list, optional=False, doc=doc_sel),
        Argument("rcut", float, optional=True, default=6.0, doc=doc_rcut),
        Argument("rcut_smth", float, optional=True, default=0.5, doc=doc_rcut_smth),
        Argument("neuron", list, optional=True, default=[10, 20, 40], doc=doc_neuron),
        Argument("axis_neuron", int, optional=True, default=4, doc=doc_axis_neuron),
        Argument("activation_function", str, optional=True, default='tanh', doc=doc_activation_function),
        Argument("resnet_dt", bool, optional=True, default=False, doc=doc_resnet_dt),
        Argument("type_one_side", bool, optional=True, default=False, doc=doc_type_one_side),
        Argument("precision", str, optional=True, default="float64", doc=doc_precision),
        Argument("trainable", bool, optional=True, default=True, doc=doc_trainable),
        Argument("seed", [int, None], optional=True, doc=doc_seed),
        Argument("exclude_types", list, optional=True, default=[], doc=doc_exclude_types),
        Argument("set_davg_zero", bool, optional=True, default=False, doc=doc_set_davg_zero)
    ]


def descrpt_se_t_args():
    doc_sel = 'A list of integers. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius. It is noted that the total sel value must be less than 4096 in a GPU environment.'
    doc_rcut = 'The cut-off radius.'
    doc_rcut_smth = 'Where to start smoothing. For example the 1/r term is smoothed from `rcut` to `rcut_smth`'
    doc_neuron = 'Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built.'
    doc_activation_function = f'The activation function in the embedding net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())}'
    doc_resnet_dt = 'Whether to use a "Timestep" in the skip connection'
    doc_precision = f'The precision of the embedding net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())}'
    doc_trainable = 'If the parameters in the embedding net are trainable'
    doc_seed = 'Random seed for parameter initialization'
    doc_set_davg_zero = 'Set the normalization average to zero. This option should be set when `atom_ener` in the energy fitting is used'

    return [
        Argument("name", str, optional=True, default='', doc="Descriptor name."),
        Argument("sel", list, optional=False, doc=doc_sel),
        Argument("rcut", float, optional=True, default=6.0, doc=doc_rcut),
        Argument("rcut_smth", float, optional=True, default=0.5, doc=doc_rcut_smth),
        Argument("neuron", list, optional=True, default=[10, 20, 40], doc=doc_neuron),
        Argument("activation_function", str, optional=True, default='tanh', doc=doc_activation_function),
        Argument("resnet_dt", bool, optional=True, default=False, doc=doc_resnet_dt),
        Argument("precision", str, optional=True, default="float64", doc=doc_precision),
        Argument("trainable", bool, optional=True, default=True, doc=doc_trainable),
        Argument("seed", [int, None], optional=True, doc=doc_seed),
        Argument("set_davg_zero", bool, optional=True, default=False, doc=doc_set_davg_zero)
    ]




def descrpt_se_r_args():
    doc_sel = 'A list of integers. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius. It is noted that the total sel value must be less than 4096 in a GPU environment.'
    doc_rcut = 'The cut-off radius.'
    doc_rcut_smth = 'Where to start smoothing. For example the 1/r term is smoothed from `rcut` to `rcut_smth`'
    doc_neuron = 'Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built.'
    doc_activation_function = f'The activation function in the embedding net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())}'
    doc_resnet_dt = 'Whether to use a "Timestep" in the skip connection'
    doc_type_one_side = 'Try to build N_types embedding nets. Otherwise, building N_types^2 embedding nets'
    doc_precision = f'The precision of the embedding net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())}'
    doc_trainable = 'If the parameters in the embedding net are trainable'
    doc_seed = 'Random seed for parameter initialization'
    doc_exclude_types = 'The Excluded types'
    doc_set_davg_zero = 'Set the normalization average to zero. This option should be set when `atom_ener` in the energy fitting is used'

    return [
        Argument("name", str, optional=True, default='', doc="Descriptor name."),
        Argument("sel", list, optional=False, doc=doc_sel),
        Argument("rcut", float, optional=True, default=6.0, doc=doc_rcut),
        Argument("rcut_smth", float, optional=True, default=0.5, doc=doc_rcut_smth),
        Argument("neuron", list, optional=True, default=[10, 20, 40], doc=doc_neuron),
        Argument("activation_function", str, optional=True, default='tanh', doc=doc_activation_function),
        Argument("resnet_dt", bool, optional=True, default=False, doc=doc_resnet_dt),
        Argument("type_one_side", bool, optional=True, default=False, doc=doc_type_one_side),
        Argument("precision", str, optional=True, default="float64", doc=doc_precision),
        Argument("trainable", bool, optional=True, default=True, doc=doc_trainable),
        Argument("seed", [int, None], optional=True, doc=doc_seed),
        Argument("exclude_types", list, optional=True, default=[], doc=doc_exclude_types),
        Argument("set_davg_zero", bool, optional=True, default=False, doc=doc_set_davg_zero)
    ]


def descrpt_se_ar_args():
    link = make_link('se_a', 'model/descriptor[se_a]')
    doc_a = f'The parameters of descriptor {link}'
    link = make_link('se_r', 'model/descriptor[se_r]')
    doc_r = f'The parameters of descriptor {link}'

    return [
        Argument("name", str, optional=True, default='', doc="Descriptor name."),
        Argument("a", dict, optional=False, doc=doc_a),
        Argument("r", dict, optional=False, doc=doc_r),
    ]


def descrpt_hybrid_args():
    doc_list = f'A list of descriptor definitions'

    return [
        Argument("name", str, optional=True, default='', doc="Descriptor name."),
        Argument("list", list, optional=False, doc=doc_list)
    ]


def descrpt_se_conv1d_args():
    doc_conv_windows = 'window sizes of each convolutional layer.'
    doc_conv_neurons = 'number of neurons in each convolutional layer.'
    doc_conv_residual = 'whether use residual convolution.'
    doc_conv_activation_fn = 'the activation function of convolutional layers.'

    return descrpt_se_a_args() + [
        Argument("conv_windows", list, optional=True, default=[], doc=doc_conv_windows),
        Argument("conv_neurons", list, optional=True, default=[], doc=doc_conv_neurons),
        Argument("conv_residual", bool, optional=True, default=False, doc=doc_conv_residual),
        Argument("conv_activation_fn", str, optional=True, default='tanh', doc=doc_conv_activation_fn)
    ]


def descrpt_se_conv_geo_args():
    doc_conv_geo_windows = 'window sizes of each convolutional layer for geometric features.'
    doc_conv_geo_neurons = 'number of neurons in each convolutional layer for geometric features.'
    doc_conv_geo_residual = 'whether use residual convolution for geometric features.'
    doc_conv_geo_activation_fn = 'the activation function of convolutional layers for geometric features.'

    return descrpt_se_conv1d_args() + [
        Argument("conv_geo_windows", list, optional=True, default=[], doc=doc_conv_geo_windows),
        Argument("conv_geo_neurons", list, optional=True, default=[], doc=doc_conv_geo_neurons),
        Argument("conv_geo_residual", bool, optional=True, default=False, doc=doc_conv_geo_residual),
        Argument("conv_geo_activation_fn", str, optional=True, default='tanh', doc=doc_conv_geo_activation_fn)
    ]


def descrpt_variant_type_args():
    link_lf = make_link('loc_frame', 'model/descriptor[loc_frame]')
    link_se_e2_a = make_link('se_e2_a', 'model/descriptor[se_e2_a]')
    link_se_e2_r = make_link('se_e2_r', 'model/descriptor[se_e2_r]')
    link_se_e3 = make_link('se_e3', 'model/descriptor[se_e3]')
    link_se_a_tpe = make_link('se_a_tpe', 'model/descriptor[se_a_tpe]')
    link_hybrid = make_link('hybrid', 'model/descriptor[hybrid]')
    doc_descrpt_type = f'The type of the descritpor. See explanation below. \n\n\
- `loc_frame`: Defines a local frame at each atom, and the compute the descriptor as local coordinates under this frame.\n\n\
- `se_e2_a`: Used by the smooth edition of Deep Potential. The full relative coordinates are used to construct the descriptor.\n\n\
- `se_e2_r`: Used by the smooth edition of Deep Potential. Only the distance between atoms is used to construct the descriptor.\n\n\
- `se_e3`: Used by the smooth edition of Deep Potential. The full relative coordinates are used to construct the descriptor. Three-body embedding will be used by this descriptor.\n\n\
- `se_a_tpe`: Used by the smooth edition of Deep Potential. The full relative coordinates are used to construct the descriptor. Type embedding will be used by this descriptor.\n\n\
- `hybrid`: Concatenate of a list of descriptors as a new descriptor.'

    return Variant("type", [
        Argument("loc_frame", dict, descrpt_local_frame_args()),
        Argument("se_e2_a", dict, descrpt_se_a_args(), alias=['se_a']),
        Argument("se_e2_r", dict, descrpt_se_r_args(), alias=['se_r']),
        Argument("se_e3", dict, descrpt_se_t_args(), alias=['se_at', 'se_a_3be', 'se_t']),
        Argument("se_a_tpe", dict, descrpt_se_a_tpe_args(), alias=['se_a_ebd']),
        Argument("se_conv1d", dict, descrpt_se_conv1d_args(), alias=['se_conv']),
        Argument("se_conv_geo", dict, descrpt_se_conv_geo_args(), alias=['se_seq_geo']),
        Argument("hybrid", dict, descrpt_hybrid_args()),
    ], doc=doc_descrpt_type)


#  --- Fitting net configurations: --- #
def fitting_ener():
    doc_numb_fparam = 'The dimension of the frame parameter. If set to >0, file `fparam.npy` should be included to provided the input fparams.'
    doc_numb_aparam = 'The dimension of the atomic parameter. If set to >0, file `aparam.npy` should be included to provided the input aparams.'
    doc_neuron = 'The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built.'
    doc_activation_function = f'The activation function in the fitting net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())}'
    doc_precision = f'The precision of the fitting net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())}'
    doc_resnet_dt = 'Whether to use a "Timestep" in the skip connection'
    doc_trainable = 'Whether the parameters in the fitting net are trainable. This option can be\n\n\
- bool: True if all parameters of the fitting net are trainable, False otherwise.\n\n\
- list of bool: Specifies if each layer is trainable. Since the fitting net is composed by hidden layers followed by a output layer, the length of tihs list should be equal to len(`neuron`)+1.'
    doc_rcond = 'The condition number used to determine the inital energy shift for each type of atoms.'
    doc_seed = 'Random seed for parameter initialization of the fitting net'
    doc_atom_ener = 'Specify the atomic energy in vacuum for each type'

    return [
        Argument("name", str, optional=True, default='', doc="Fitting net name."),
        Argument("numb_fparam", int, optional=True, default=0, doc=doc_numb_fparam),
        Argument("numb_aparam", int, optional=True, default=0, doc=doc_numb_aparam),
        Argument("neuron", list, optional=True, default=[120, 120, 120], doc=doc_neuron),
        Argument("activation_function", str, optional=True, default='tanh', doc=doc_activation_function),
        Argument("precision", str, optional=True, default='float64', doc=doc_precision),
        Argument("resnet_dt", bool, optional=True, default=True, doc=doc_resnet_dt),
        Argument("trainable", [list, bool], optional=True, default=True, doc=doc_trainable),
        Argument("rcond", float, optional=True, default=1e-3, doc=doc_rcond),
        Argument("seed", [int, None], optional=True, doc=doc_seed),
        Argument("atom_ener", list, optional=True, default=[], doc=doc_atom_ener)
    ]


def fitting_polar():
    doc_neuron = 'The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built.'
    doc_activation_function = f'The activation function in the fitting net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())}'
    doc_resnet_dt = 'Whether to use a "Timestep" in the skip connection'
    doc_precision = f'The precision of the fitting net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())}'
    doc_scale = 'The output of the fitting net (polarizability matrix) will be scaled by ``scale``'
    # doc_diag_shift = 'The diagonal part of the polarizability matrix  will be shifted by ``diag_shift``. The shift operation is carried out after ``scale``.'
    doc_fit_diag = 'Fit the diagonal part of the rotational invariant polarizability matrix, which will be converted to normal polarizability matrix by contracting with the rotation matrix.'
    doc_sel_type = 'The atom types for which the atomic polarizability will be provided. If not set, all types will be selected.'
    doc_seed = 'Random seed for parameter initialization of the fitting net'

    # YWolfeee: user can decide whether to use shift diag
    doc_shift_diag = 'Whether to shift the diagonal of polar, which is beneficial to training. Default is true.'

    return [
        Argument("name", str, optional=True, default='', doc="Fitting net name."),
        Argument("neuron", list, optional=True, default=[120, 120, 120], doc=doc_neuron),
        Argument("activation_function", str, optional=True, default='tanh', doc=doc_activation_function),
        Argument("resnet_dt", bool, optional=True, default=True, doc=doc_resnet_dt),
        Argument("precision", str, optional=True, default='float64', doc=doc_precision),
        Argument("fit_diag", bool, optional=True, default=True, doc=doc_fit_diag),
        Argument("scale", [list, float], optional=True, default=1.0, doc=doc_scale),
        # Argument("diag_shift", [list,float], optional = True, default = 0.0, doc = doc_diag_shift),
        Argument("shift_diag", bool, optional=True, default=True, doc=doc_shift_diag),
        Argument("sel_type", [list, int, None], optional=True, doc=doc_sel_type),
        Argument("seed", [int, None], optional=True, doc=doc_seed)
    ]


# def fitting_global_polar():
#    return fitting_polar()


def fitting_dipole():
    doc_neuron = 'The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built.'
    doc_activation_function = f'The activation function in the fitting net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())}'
    doc_resnet_dt = 'Whether to use a "Timestep" in the skip connection'
    doc_precision = f'The precision of the fitting net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())}'
    doc_sel_type = 'The atom types for which the atomic dipole will be provided. If not set, all types will be selected.'
    doc_seed = 'Random seed for parameter initialization of the fitting net'
    return [
        Argument("name", str, optional=True, default='', doc="Fitting net name."),
        Argument("neuron", list, optional=True, default=[120, 120, 120], doc=doc_neuron),
        Argument("activation_function", str, optional=True, default='tanh', doc=doc_activation_function),
        Argument("resnet_dt", bool, optional=True, default=True, doc=doc_resnet_dt),
        Argument("precision", str, optional=True, default='float64', doc=doc_precision),
        Argument("sel_type", [list, int, None], optional=True, doc=doc_sel_type),
        Argument("seed", [int, None], optional=True, doc=doc_seed)
    ]

def fitting_variant_type_args():
    doc_descrpt_type = 'The type of the fitting. See explanation below. \n\n\
- `ener`: Fit an energy model (potential energy surface).\n\n\
- `dipole`: Fit an atomic dipole model. Global dipole labels or atomic dipole labels for all the selected atoms (see `sel_type`) should be provided by `dipole.npy` in each data system. The file either has number of frames lines and 3 times of number of selected atoms columns, or has number of frames lines and 3 columns. See `loss` parameter.\n\n\
- `polar`: Fit an atomic polarizability model. Global polarizazbility labels or atomic polarizability labels for all the selected atoms (see `sel_type`) should be provided by `polarizability.npy` in each data system. The file eith has number of frames lines and 9 times of number of selected atoms columns, or has number of frames lines and 9 columns. See `loss` parameter.\n\n'

    return Variant("type", [Argument("ener", dict, fitting_ener()),
                            Argument("dipole", dict, fitting_dipole()),
                            Argument("polar", dict, fitting_polar()),
                            ],
                   optional=True,
                   default_tag='ener',
                   doc=doc_descrpt_type)

def model_args():
    doc_type_map = 'A list of strings. Give the name to each type of atoms. It is noted that the number of atom type of training system must be less than 128 in a GPU environment.'
    doc_data_stat_nbatch = 'The model determines the normalization from the statistics of the data. This key specifies the number of `frames` in each `system` used for statistics.'
    doc_data_stat_protect = 'Protect parameter for atomic energy regression.'
    doc_type_embedding = "The type embedding."
    doc_descrpt = 'A list of DeepMD descriptors identified by key arg `name`.'
    doc_fitting = 'A list of DeepMD fitting networks identified by key arg `name`.'
    doc_modifier = 'The modifier of model output.'
    doc_use_srtab = 'The table for the short-range pairwise interaction added on top of DP. The table is a text data file with (N_t + 1) * N_t / 2 + 1 columes. The first colume is the distance between atoms. The second to the last columes are energies for pairs of certain types. For example we have two atom types, 0 and 1. The columes from 2nd to 4th are for 0-0, 0-1 and 1-1 correspondingly.'
    doc_smin_alpha = 'The short-range tabulated interaction will be swithed according to the distance of the nearest neighbor. This distance is calculated by softmin. This parameter is the decaying parameter in the softmin. It is only required when `use_srtab` is provided.'
    doc_sw_rmin = 'The lower boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided.'
    doc_sw_rmax = 'The upper boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided.'
    doc_compress_config = 'Model compression configurations'

    ca = Argument("model", dict,
                  [Argument("type_map", list, optional=True, doc=doc_type_map),
                   Argument("data_stat_nbatch", int, optional=True, default=10, doc=doc_data_stat_nbatch),
                   Argument("data_stat_protect", float, optional=True, default=1e-2, doc=doc_data_stat_protect),
                   Argument("use_srtab", str, optional=True, doc=doc_use_srtab),
                   Argument("smin_alpha", float, optional=True, doc=doc_smin_alpha),
                   Argument("sw_rmin", float, optional=True, doc=doc_sw_rmin),
                   Argument("sw_rmax", float, optional=True, doc=doc_sw_rmax),
                   Argument("type_embedding", dict, type_embedding_args(), [], optional=True, doc=doc_type_embedding),
                   Argument("descriptor", list, doc=doc_descrpt),
                   Argument("fitting_net", list, doc=doc_fitting),
                   Argument("modifier", dict, [], [modifier_variant_type_args()], optional=True, doc=doc_modifier),
                   Argument("compress", dict, [], [model_compression_type_args()], optional=True,
                            doc=doc_compress_config)
                   ])
    # print(ca.gen_doc())
    return ca


#  --- Learning rate configurations: --- #
def learning_rate_exp():
    doc_start_lr = 'The learning rate the start of the training.'
    doc_stop_lr = 'The desired learning rate at the end of the training.'
    doc_decay_steps = 'The learning rate is decaying every this number of training steps.'

    args = [
        Argument("name", str, optional=True, default='', doc="Learning rate name."),
        Argument("start_lr", float, optional=True, default=1e-3, doc=doc_start_lr),
        Argument("stop_lr", float, optional=True, default=1e-8, doc=doc_stop_lr),
        Argument("decay_steps", int, optional=True, default=5000, doc=doc_decay_steps)
    ]
    return args

def learning_rate_variant_type_args():
    doc_lr = 'The type of the learning rate.'

    return Variant("type",
                   [Argument("exp", dict, learning_rate_exp())],
                   optional=True,
                   default_tag='exp',
                   doc=doc_lr)

def learning_rate_args():
    doc_lr = "A list of learning rates identified by key arg `name`."
    return Argument("learning_rate", list, doc=doc_lr)


#  --- Loss configurations: --- #
def loss_ener():
    doc_start_pref_e = start_pref('energy')
    doc_limit_pref_e = limit_pref('energy')
    doc_start_pref_f = start_pref('force')
    doc_limit_pref_f = limit_pref('force')
    doc_start_pref_v = start_pref('virial')
    doc_limit_pref_v = limit_pref('virial')
    doc_start_pref_ae = start_pref('atom_ener')
    doc_limit_pref_ae = limit_pref('atom_ener')
    doc_relative_f = 'If provided, relative force error will be used in the loss. The difference of force will be normalized by the magnitude of the force in the label with a shift given by `relative_f`, i.e. DF_i / ( || F || + relative_f ) with DF denoting the difference between prediction and label and || F || denoting the L2 norm of the label.'
    return [
        Argument("name", str, optional=True, default='', doc="loss name."),
        Argument("start_pref_e", [float, int], optional=True, default=0.02, doc=doc_start_pref_e),
        Argument("limit_pref_e", [float, int], optional=True, default=1.00, doc=doc_limit_pref_e),
        Argument("start_pref_f", [float, int], optional=True, default=1000, doc=doc_start_pref_f),
        Argument("limit_pref_f", [float, int], optional=True, default=1.00, doc=doc_limit_pref_f),
        Argument("start_pref_v", [float, int], optional=True, default=0.00, doc=doc_start_pref_v),
        Argument("limit_pref_v", [float, int], optional=True, default=0.00, doc=doc_limit_pref_v),
        Argument("start_pref_ae", [float, int], optional=True, default=0.00, doc=doc_start_pref_ae),
        Argument("limit_pref_ae", [float, int], optional=True, default=0.00, doc=doc_limit_pref_ae),
        Argument("relative_f", [float, None], optional=True, doc=doc_relative_f)
    ]


# YWolfeee: Modified to support tensor type of loss args.
def loss_tensor():
    # doc_global_weight = "The prefactor of the weight of global loss. It should be larger than or equal to 0. If only `pref` is provided or both are not provided, training will be global mode, i.e. the shape of 'polarizability.npy` or `dipole.npy` should be #frams x [9 or 3]."
    # doc_local_weight =  "The prefactor of the weight of atomic loss. It should be larger than or equal to 0. If only `pref_atomic` is provided, training will be atomic mode, i.e. the shape of `polarizability.npy` or `dipole.npy` should be #frames x ([9 or 3] x #selected atoms). If both `pref` and `pref_atomic` are provided, training will be combined mode, and atomic label should be provided as well."
    doc_global_weight = "The prefactor of the weight of global loss. It should be larger than or equal to 0. If controls the weight of loss corresponding to global label, i.e. 'polarizability.npy` or `dipole.npy`, whose shape should be #frames x [9 or 3]. If it's larger than 0.0, this npy should be included."
    doc_local_weight = "The prefactor of the weight of atomic loss. It should be larger than or equal to 0. If controls the weight of loss corresponding to atomic label, i.e. `atomic_polarizability.npy` or `atomic_dipole.npy`, whose shape should be #frames x ([9 or 3] x #selected atoms). If it's larger than 0.0, this npy should be included. Both `pref` and `pref_atomic` should be provided, and either can be set to 0.0."
    return [
        Argument("name", str, optional=True, default='', doc="loss name."),
        Argument("pref", [float, int], optional=False, default=None, doc=doc_global_weight),
        Argument("pref_atomic", [float, int], optional=False, default=None, doc=doc_local_weight),
    ]

def loss_variant_type_args():
    doc_loss = 'The type of the loss. When the fitting type is `ener`, the loss type should be set to `ener` or left unset. When the fitting type is `dipole` or `polar`, the loss type should be set to `tensor`. \n\.'

    return Variant("type",
                   [Argument("ener", dict, loss_ener()),
                    Argument("tensor", dict, loss_tensor()),
                    # Argument("polar", dict, loss_tensor()),
                    # Argument("global_polar", dict, loss_tensor("global"))
                    ],
                   optional=True,
                   default_tag='ener',
                   doc=doc_loss)

def loss_args():
    doc_loss = 'A list of losses identified by key arg `name`.'
    return Argument('loss', list, doc=doc_loss)


#  --- Training configurations: --- #

def task_args():
    doc_name = 'name of task'
    doc_descrpt = 'descriptor name for this task'
    doc_fitting = 'fitting net name for this task'
    doc_learning_rate = 'learning rate name for this task'
    doc_loss = 'loss name for this task'
    args =[
        Argument("name", str, optional = False, doc = doc_name),
        Argument("descriptor", str, optional = False, doc = doc_descrpt),
        Argument("fitting_net", str, optional = False, doc = doc_fitting),
        Argument("learning_rate", str, optional = False, doc = doc_learning_rate),
        Argument("loss", str, optional = False, doc = doc_loss),
    ]
    return Argument("task", dict, args)


def training_args():  # ! modified by Ziyao: data configuration isolated.
    doc_tasks = 'A list of tasks to be trained.'
    doc_numb_steps = 'Number of training batch. Each training uses one batch of data.'
    doc_seed = 'The random seed for getting frames from the training data set.'
    doc_disp_file = 'The file for printing learning curve.'
    doc_disp_freq = 'The frequency of printing learning curve.'
    doc_numb_test = 'Number of frames used for the test during training.'
    doc_save_freq = 'The frequency of saving check point.'
    doc_save_ckpt = 'The file name of saving check point.'
    doc_disp_training = 'Displaying verbose information during training.'
    doc_time_training = 'Timing durining training.'
    doc_profiling = 'Profiling during training.'
    doc_profiling_file = 'Output file for profiling.'
    doc_tensorboard = 'Enable tensorboard'
    doc_tensorboard_log_dir = 'The log directory of tensorboard outputs'

    arg_training_data = training_data_args()
    arg_validation_data = validation_data_args()

    args = [
        arg_training_data,
        arg_validation_data,
        Argument("tasks", list, optional=False, doc=doc_tasks, alias=["sub_models"]),
        Argument("numb_steps", int, optional=False, doc=doc_numb_steps, alias=["stop_batch"]),
        Argument("seed", [int, None], optional=True, doc=doc_seed),
        Argument("disp_file", str, optional=True, default='lcueve.out', doc=doc_disp_file),
        Argument("disp_freq", int, optional=True, default=1000, doc=doc_disp_freq),
        Argument("numb_test", [list, int, str], optional=True, default=1, doc=doc_numb_test),
        Argument("save_freq", int, optional=True, default=1000, doc=doc_save_freq),
        Argument("save_ckpt", str, optional=True, default='model.ckpt', doc=doc_save_ckpt),
        Argument("disp_training", bool, optional=True, default=True, doc=doc_disp_training),
        Argument("time_training", bool, optional=True, default=True, doc=doc_time_training),
        Argument("profiling", bool, optional=True, default=False, doc=doc_profiling),
        Argument("profiling_file", str, optional=True, default='timeline.json', doc=doc_profiling_file),
        Argument("tensorboard", bool, optional=True, default=False, doc=doc_tensorboard),
        Argument("tensorboard_log_dir", str, optional=True, default='log', doc=doc_tensorboard_log_dir),
    ]

    doc_training = 'The training options.'
    return Argument("training", dict, args, [], doc=doc_training)



def normalize_list_of_args(old_list, pattern_arg):
    if isinstance(pattern_arg, Variant):
        pattern = Argument("pattern", dict, [], [pattern_arg])
    elif isinstance(pattern_arg, Argument):
        pattern = pattern_arg
    else:
        raise AssertionError("Wrong type of input pattern argument: %s" % str(pattern_arg))
    new_list = [pattern.normalize_value(ii, trim_pattern="_*") for ii in old_list]
    [pattern.check_value(ii, strict=True) for ii in new_list]
    return new_list


def normalize_hybrid_list(hy_list):
    new_list = []
    base = Argument("base", dict, [], [descrpt_variant_type_args()], doc="")
    for ii in range(len(hy_list)):
        data = base.normalize_value(hy_list[ii], trim_pattern="_*")
        base.check_value(data, strict=True)
        new_list.append(data)
    return new_list


def normalize_mt(data):
    ma = model_args()
    lra = learning_rate_args()
    la = loss_args()
    ta = training_args()

    base = Argument("base", dict, [ma, lra, la, ta])
    data = base.normalize_value(data, trim_pattern="_*")
    base.check_value(data, strict=True)

    data["model"]["descriptor"] = normalize_list_of_args(data["model"]["descriptor"], descrpt_variant_type_args())
    data["model"]["fitting_net"] = normalize_list_of_args(data["model"]["fitting_net"], fitting_variant_type_args())
    data["loss"] = normalize_list_of_args(data["loss"], loss_variant_type_args())
    data["learning_rate"] = normalize_list_of_args(data["learning_rate"], learning_rate_variant_type_args())
    data["training"]["tasks"] = normalize_list_of_args(data["training"]["tasks"], task_args())

    # normalize hybrid descriptors
    descrpts = data["model"]["descriptor"]
    for ii in range(len(descrpts)):
        if descrpts[ii]["type"] == "hybrid":
            descrpts[ii] = normalize_hybrid_list(descrpts[ii]["list"])
    data["model"]["descriptor"] = descrpts

    return data


if __name__ == '__main__':
    gen_doc()

