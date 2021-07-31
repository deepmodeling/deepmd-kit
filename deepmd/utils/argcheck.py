from dargs import dargs, Argument, Variant, ArgumentEncoder
from deepmd.common import ACTIVATION_FN_DICT, PRECISION_DICT
import json


def list_to_doc(xx):
    items = []
    for ii in xx:
        if len(items) == 0:
            items.append(f'"{ii}"')
        else:
            items.append(f', "{ii}"')
    items.append('.')
    return ''.join(items)


def make_link(content, ref_key):
    return f'`{content} <{ref_key}_>`_' if not dargs.RAW_ANCHOR \
        else f'`{content} <#{ref_key}>`_'


def type_embedding_args():
    doc_neuron = 'Number of neurons in each hidden layers of the embedding net. When two layers are of the same size or one layer is twice as large as the previous layer, a skip connection is built.'
    doc_resnet_dt = 'Whether to use a "Timestep" in the skip connection'
    doc_seed = 'Random seed for parameter initialization'
    doc_activation_function = f'The activation function in the embedding net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())}'
    doc_precision = f'The precision of the embedding net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())}'
    doc_trainable = 'If the parameters in the embedding net are trainable'
    
    return [
        Argument("neuron", list, optional = True, default = [2, 4, 8], doc = doc_neuron),
        Argument("activation_function", str, optional = True, default = 'tanh', doc = doc_activation_function),
        Argument("resnet_dt", bool, optional = True, default = False, doc = doc_resnet_dt),
        Argument("precision", str, optional = True, default = "float64", doc = doc_precision),
        Argument("trainable", bool, optional = True, default = True, doc = doc_trainable),
        Argument("seed", [int,None], optional = True, doc = doc_seed),
    ]        


#  --- Descriptor configurations: --- #
def descrpt_local_frame_args ():
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
        Argument("sel_a", list, optional = False, doc = doc_sel_a),
        Argument("sel_r", list, optional = False, doc = doc_sel_r),
        Argument("rcut", float, optional = True, default = 6.0, doc = doc_rcut),
        Argument("axis_rule", list, optional = False, doc = doc_axis_rule)
    ]


def descrpt_se_a_args():
    doc_sel = 'This parameter set the number of selected neighbors for each type of atom. It can be:\n\n\
    - `List[int]`. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius. It is noted that the total sel value must be less than 4096 in a GPU environment.\n\n\
    - `str`. Can be "auto:factor" or "auto". "factor" is a float number larger than 1. This option will automatically determine the `sel`. In detail it counts the maximal number of neighbors with in the cutoff radius for each type of neighbor, then multiply the maximum by the "factor". Finally the number is wraped up to 4 divisible. The option "auto" is equivalent to "auto:1.1".'
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
        Argument("sel", [list,str], optional = True, default = "auto", doc = doc_sel),
        Argument("rcut", float, optional = True, default = 6.0, doc = doc_rcut),
        Argument("rcut_smth", float, optional = True, default = 0.5, doc = doc_rcut_smth),
        Argument("neuron", list, optional = True, default = [10,20,40], doc = doc_neuron),
        Argument("axis_neuron", int, optional = True, default = 4, doc = doc_axis_neuron),
        Argument("activation_function", str, optional = True, default = 'tanh', doc = doc_activation_function),
        Argument("resnet_dt", bool, optional = True, default = False, doc = doc_resnet_dt),
        Argument("type_one_side", bool, optional = True, default = False, doc = doc_type_one_side),
        Argument("precision", str, optional = True, default = "float64", doc = doc_precision),
        Argument("trainable", bool, optional = True, default = True, doc = doc_trainable),
        Argument("seed", [int,None], optional = True, doc = doc_seed),
        Argument("exclude_types", list, optional = True, default = [], doc = doc_exclude_types),
        Argument("set_davg_zero", bool, optional = True, default = False, doc = doc_set_davg_zero)
    ]


def descrpt_se_t_args():
    doc_sel = 'This parameter set the number of selected neighbors for each type of atom. It can be:\n\n\
    - `List[int]`. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius. It is noted that the total sel value must be less than 4096 in a GPU environment.\n\n\
    - `str`. Can be "auto:factor" or "auto". "factor" is a float number larger than 1. This option will automatically determine the `sel`. In detail it counts the maximal number of neighbors with in the cutoff radius for each type of neighbor, then multiply the maximum by the "factor". Finally the number is wraped up to 4 divisible. The option "auto" is equivalent to "auto:1.1".'
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
        Argument("sel", [list,str], optional = True, default = "auto", doc = doc_sel),
        Argument("rcut", float, optional = True, default = 6.0, doc = doc_rcut),
        Argument("rcut_smth", float, optional = True, default = 0.5, doc = doc_rcut_smth),
        Argument("neuron", list, optional = True, default = [10,20,40], doc = doc_neuron),
        Argument("activation_function", str, optional = True, default = 'tanh', doc = doc_activation_function),
        Argument("resnet_dt", bool, optional = True, default = False, doc = doc_resnet_dt),
        Argument("precision", str, optional = True, default = "float64", doc = doc_precision),
        Argument("trainable", bool, optional = True, default = True, doc = doc_trainable),
        Argument("seed", [int,None], optional = True, doc = doc_seed),
        Argument("set_davg_zero", bool, optional = True, default = False, doc = doc_set_davg_zero)
    ]



def descrpt_se_a_tpe_args():
    doc_type_nchanl = 'number of channels for type embedding'
    doc_type_nlayer = 'number of hidden layers of type embedding net'
    doc_numb_aparam = 'dimension of atomic parameter. if set to a value > 0, the atomic parameters are embedded.'

    return descrpt_se_a_args() + [        
        Argument("type_nchanl", int, optional = True, default = 4, doc = doc_type_nchanl),
        Argument("type_nlayer", int, optional = True, default = 2, doc = doc_type_nlayer),
        Argument("numb_aparam", int, optional = True, default = 0, doc = doc_numb_aparam)
    ]


def descrpt_se_r_args():
    doc_sel = 'This parameter set the number of selected neighbors for each type of atom. It can be:\n\n\
    - `List[int]`. The length of the list should be the same as the number of atom types in the system. `sel[i]` gives the selected number of type-i neighbors. `sel[i]` is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius. It is noted that the total sel value must be less than 4096 in a GPU environment.\n\n\
    - `str`. Can be "auto:factor" or "auto". "factor" is a float number larger than 1. This option will automatically determine the `sel`. In detail it counts the maximal number of neighbors with in the cutoff radius for each type of neighbor, then multiply the maximum by the "factor". Finally the number is wraped up to 4 divisible. The option "auto" is equivalent to "auto:1.1".'
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
        Argument("sel", [list,str], optional = True, default = "auto", doc = doc_sel),
        Argument("rcut", float, optional = True, default = 6.0, doc = doc_rcut),
        Argument("rcut_smth", float, optional = True, default = 0.5, doc = doc_rcut_smth),
        Argument("neuron", list, optional = True, default = [10,20,40], doc = doc_neuron),
        Argument("activation_function", str, optional = True, default = 'tanh', doc = doc_activation_function),
        Argument("resnet_dt", bool, optional = True, default = False, doc = doc_resnet_dt),
        Argument("type_one_side", bool, optional = True, default = False, doc = doc_type_one_side),
        Argument("precision", str, optional = True, default = "float64", doc = doc_precision),
        Argument("trainable", bool, optional = True, default = True, doc = doc_trainable),
        Argument("seed", [int,None], optional = True, doc = doc_seed),
        Argument("exclude_types", list, optional = True, default = [], doc = doc_exclude_types),
        Argument("set_davg_zero", bool, optional = True, default = False, doc = doc_set_davg_zero)
    ]


def descrpt_se_ar_args():
    link = make_link('se_a', 'model/descriptor[se_a]')
    doc_a = f'The parameters of descriptor {link}'
    link = make_link('se_r', 'model/descriptor[se_r]')
    doc_r = f'The parameters of descriptor {link}'
    
    return [
        Argument("a", dict, optional = False, doc = doc_a),
        Argument("r", dict, optional = False, doc = doc_r),
    ]


def descrpt_hybrid_args():
    doc_list = f'A list of descriptor definitions'
    
    return [
        Argument("list", list, optional = False, doc = doc_list)
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
        Argument("se_e2_a", dict, descrpt_se_a_args(), alias = ['se_a']),
        Argument("se_e2_r", dict, descrpt_se_r_args(), alias = ['se_r']),
        Argument("se_e3", dict, descrpt_se_t_args(), alias = ['se_at', 'se_a_3be', 'se_t']),
        Argument("se_a_tpe", dict, descrpt_se_a_tpe_args(), alias = ['se_a_ebd']),
        Argument("hybrid", dict, descrpt_hybrid_args()),
    ], doc = doc_descrpt_type)


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
        Argument("numb_fparam", int, optional = True, default = 0, doc = doc_numb_fparam),
        Argument("numb_aparam", int, optional = True, default = 0, doc = doc_numb_aparam),
        Argument("neuron", list, optional = True, default = [120,120,120], doc = doc_neuron),
        Argument("activation_function", str, optional = True, default = 'tanh', doc = doc_activation_function),
        Argument("precision", str, optional = True, default = 'float64', doc = doc_precision),
        Argument("resnet_dt", bool, optional = True, default = True, doc = doc_resnet_dt),
        Argument("trainable", [list,bool], optional = True, default = True, doc = doc_trainable),
        Argument("rcond", float, optional = True, default = 1e-3, doc = doc_rcond),
        Argument("seed", [int,None], optional = True, doc = doc_seed),
        Argument("atom_ener", list, optional = True, default = [], doc = doc_atom_ener)
    ]


def fitting_polar():
    doc_neuron = 'The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built.'
    doc_activation_function = f'The activation function in the fitting net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())}'
    doc_resnet_dt = 'Whether to use a "Timestep" in the skip connection'
    doc_precision = f'The precision of the fitting net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())}'
    doc_scale = 'The output of the fitting net (polarizability matrix) will be scaled by ``scale``'
    #doc_diag_shift = 'The diagonal part of the polarizability matrix  will be shifted by ``diag_shift``. The shift operation is carried out after ``scale``.'
    doc_fit_diag = 'Fit the diagonal part of the rotational invariant polarizability matrix, which will be converted to normal polarizability matrix by contracting with the rotation matrix.'
    doc_sel_type = 'The atom types for which the atomic polarizability will be provided. If not set, all types will be selected.'
    doc_seed = 'Random seed for parameter initialization of the fitting net'
    
    # YWolfeee: user can decide whether to use shift diag
    doc_shift_diag = 'Whether to shift the diagonal of polar, which is beneficial to training. Default is true.'

    return [
        Argument("neuron", list, optional = True, default = [120,120,120], doc = doc_neuron),
        Argument("activation_function", str, optional = True, default = 'tanh', doc = doc_activation_function),
        Argument("resnet_dt", bool, optional = True, default = True, doc = doc_resnet_dt),
        Argument("precision", str, optional = True, default = 'float64', doc = doc_precision),
        Argument("fit_diag", bool, optional = True, default = True, doc = doc_fit_diag),
        Argument("scale", [list,float], optional = True, default = 1.0, doc = doc_scale),
        #Argument("diag_shift", [list,float], optional = True, default = 0.0, doc = doc_diag_shift),
        Argument("shift_diag", bool, optional = True, default = True, doc = doc_shift_diag),
        Argument("sel_type", [list,int,None], optional = True, doc = doc_sel_type),
        Argument("seed", [int,None], optional = True, doc = doc_seed)
    ]


#def fitting_global_polar():
#    return fitting_polar()


def fitting_dipole():
    doc_neuron = 'The number of neurons in each hidden layers of the fitting net. When two hidden layers are of the same size, a skip connection is built.'
    doc_activation_function = f'The activation function in the fitting net. Supported activation functions are {list_to_doc(ACTIVATION_FN_DICT.keys())}'
    doc_resnet_dt = 'Whether to use a "Timestep" in the skip connection'
    doc_precision = f'The precision of the fitting net parameters, supported options are {list_to_doc(PRECISION_DICT.keys())}'
    doc_sel_type = 'The atom types for which the atomic dipole will be provided. If not set, all types will be selected.'
    doc_seed = 'Random seed for parameter initialization of the fitting net'
    return [
        Argument("neuron", list, optional = True, default = [120,120,120], doc = doc_neuron),
        Argument("activation_function", str, optional = True, default = 'tanh', doc = doc_activation_function),
        Argument("resnet_dt", bool, optional = True, default = True, doc = doc_resnet_dt),
        Argument("precision", str, optional = True, default = 'float64', doc = doc_precision),
        Argument("sel_type", [list,int,None], optional = True, doc = doc_sel_type),
        Argument("seed", [int,None], optional = True, doc = doc_seed)
    ]    

#   YWolfeee: Delete global polar mode, merge it into polar mode and use loss setting to support.
def fitting_variant_type_args():
    doc_descrpt_type = 'The type of the fitting. See explanation below. \n\n\
- `ener`: Fit an energy model (potential energy surface).\n\n\
- `dipole`: Fit an atomic dipole model. Global dipole labels or atomic dipole labels for all the selected atoms (see `sel_type`) should be provided by `dipole.npy` in each data system. The file either has number of frames lines and 3 times of number of selected atoms columns, or has number of frames lines and 3 columns. See `loss` parameter.\n\n\
- `polar`: Fit an atomic polarizability model. Global polarizazbility labels or atomic polarizability labels for all the selected atoms (see `sel_type`) should be provided by `polarizability.npy` in each data system. The file eith has number of frames lines and 9 times of number of selected atoms columns, or has number of frames lines and 9 columns. See `loss` parameter.\n\n'

    return Variant("type", [Argument("ener", dict, fitting_ener()),
                            Argument("dipole", dict, fitting_dipole()),
                            Argument("polar", dict, fitting_polar()),
                            ], 
                   optional = True,
                   default_tag = 'ener',
                   doc = doc_descrpt_type)


#  --- Modifier configurations: --- #
def modifier_dipole_charge():
    doc_model_name = "The name of the frozen dipole model file."
    doc_model_charge_map = f"The charge of the WFCC. The list length should be the same as the {make_link('sel_type', 'model/fitting_net[dipole]/sel_type')}. "
    doc_sys_charge_map = f"The charge of real atoms. The list length should be the same as the {make_link('type_map', 'model/type_map')}"
    doc_ewald_h = f"The grid spacing of the FFT grid. Unit is A"
    doc_ewald_beta = f"The splitting parameter of Ewald sum. Unit is A^{-1}"
    
    return [
        Argument("model_name", str, optional = False, doc = doc_model_name),
        Argument("model_charge_map", list, optional = False, doc = doc_model_charge_map),
        Argument("sys_charge_map", list, optional = False, doc = doc_sys_charge_map),
        Argument("ewald_beta", float, optional = True, default = 0.4, doc = doc_ewald_beta),
        Argument("ewald_h", float, optional = True, default = 1.0, doc = doc_ewald_h),        
    ]


def modifier_variant_type_args():
    doc_modifier_type = "The type of modifier. See explanation below.\n\n\
-`dipole_charge`: Use WFCC to model the electronic structure of the system. Correct the long-range interaction"
    return Variant("type", 
                   [
                       Argument("dipole_charge", dict, modifier_dipole_charge()),
                   ],
                   optional = False,
                   doc = doc_modifier_type)

#  --- model compression configurations: --- #
def model_compression():
    doc_compress = "The name of the frozen model file."
    doc_model_file = f"The input model file, which will be compressed by the DeePMD-kit."
    doc_table_config = f"The arguments of model compression, including extrapolate(scale of model extrapolation), stride(uniform stride of tabulation's first and second table), and frequency(frequency of tabulation overflow check)."
    
    return [
        Argument("compress", bool, optional = False, default = True, doc = doc_compress),
        Argument("model_file", str, optional = False, default = 'frozen_model.pb', doc = doc_model_file),
        Argument("table_config", list, optional = False, default = [5, 0.01, 0.1, -1], doc = doc_table_config),
    ]

#  --- model compression configurations: --- #
def model_compression_type_args():
    doc_compress_type = "The type of model compression, which should be consistent with the descriptor type."
    
    return Variant("type", [
            Argument("se_e2_a", dict, model_compression(), alias = ['se_a'])
        ],
        optional = True,
        default_tag = 'se_e2_a',
        doc = doc_compress_type)


def model_args ():    
    doc_type_map = 'A list of strings. Give the name to each type of atoms. It is noted that the number of atom type of training system must be less than 128 in a GPU environment.'
    doc_data_stat_nbatch = 'The model determines the normalization from the statistics of the data. This key specifies the number of `frames` in each `system` used for statistics.'
    doc_data_stat_protect = 'Protect parameter for atomic energy regression.'
    doc_type_embedding = "The type embedding."
    doc_descrpt = 'The descriptor of atomic environment.'
    doc_fitting = 'The fitting of physical properties.'
    doc_modifier = 'The modifier of model output.'
    doc_use_srtab = 'The table for the short-range pairwise interaction added on top of DP. The table is a text data file with (N_t + 1) * N_t / 2 + 1 columes. The first colume is the distance between atoms. The second to the last columes are energies for pairs of certain types. For example we have two atom types, 0 and 1. The columes from 2nd to 4th are for 0-0, 0-1 and 1-1 correspondingly.'
    doc_smin_alpha = 'The short-range tabulated interaction will be swithed according to the distance of the nearest neighbor. This distance is calculated by softmin. This parameter is the decaying parameter in the softmin. It is only required when `use_srtab` is provided.'
    doc_sw_rmin = 'The lower boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided.'
    doc_sw_rmax = 'The upper boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided.'
    doc_compress_config = 'Model compression configurations'

    ca = Argument("model", dict, 
                  [Argument("type_map", list, optional = True, doc = doc_type_map),
                   Argument("data_stat_nbatch", int, optional = True, default = 10, doc = doc_data_stat_nbatch),
                   Argument("data_stat_protect", float, optional = True, default = 1e-2, doc = doc_data_stat_protect),
                   Argument("use_srtab", str, optional = True, doc = doc_use_srtab),
                   Argument("smin_alpha", float, optional = True, doc = doc_smin_alpha),
                   Argument("sw_rmin", float, optional = True, doc = doc_sw_rmin),
                   Argument("sw_rmax", float, optional = True, doc = doc_sw_rmax),
                   Argument("type_embedding", dict, type_embedding_args(), [], optional = True, doc = doc_type_embedding),
                   Argument("descriptor", dict, [], [descrpt_variant_type_args()], doc = doc_descrpt),
                   Argument("fitting_net", dict, [], [fitting_variant_type_args()], doc = doc_fitting),
                   Argument("modifier", dict, [], [modifier_variant_type_args()], optional = True, doc = doc_modifier),
                   Argument("compress", dict, [], [model_compression_type_args()], optional = True, doc = doc_compress_config)
                  ])
    # print(ca.gen_doc())
    return ca


#  --- Learning rate configurations: --- #
def learning_rate_exp():
    doc_start_lr = 'The learning rate the start of the training.'
    doc_stop_lr = 'The desired learning rate at the end of the training.'
    doc_decay_steps = 'The learning rate is decaying every this number of training steps.'
    
    args =  [
        Argument("start_lr", float, optional = True, default = 1e-3, doc = doc_start_lr),
        Argument("stop_lr", float, optional = True, default = 1e-8, doc = doc_stop_lr),
        Argument("decay_steps", int, optional = True, default = 5000, doc = doc_decay_steps)
    ]
    return args
    

def learning_rate_variant_type_args():
    doc_lr = 'The type of the learning rate.'

    return Variant("type", 
                   [Argument("exp", dict, learning_rate_exp())],
                   optional = True,
                   default_tag = 'exp',
                   doc = doc_lr)


def learning_rate_args():
    doc_lr = "The definitio of learning rate" 
    return Argument("learning_rate", dict, [], 
                    [learning_rate_variant_type_args()],
                    doc = doc_lr)


#  --- Loss configurations: --- #
def start_pref(item):
    return f'The prefactor of {item} loss at the start of the training. Should be larger than or equal to 0. If set to none-zero value, the {item} label should be provided by file {item}.npy in each data system. If both start_pref_{item} and limit_pref_{item} are set to 0, then the {item} will be ignored.'


def limit_pref(item):
    return f'The prefactor of {item} loss at the limit of the training, Should be larger than or equal to 0. i.e. the training step goes to infinity.'


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
        Argument("start_pref_e", [float,int], optional = True, default = 0.02, doc = doc_start_pref_e),
        Argument("limit_pref_e", [float,int], optional = True, default = 1.00, doc = doc_limit_pref_e),
        Argument("start_pref_f", [float,int], optional = True, default = 1000, doc = doc_start_pref_f),
        Argument("limit_pref_f", [float,int], optional = True, default = 1.00, doc = doc_limit_pref_f),
        Argument("start_pref_v", [float,int], optional = True, default = 0.00, doc = doc_start_pref_v),
        Argument("limit_pref_v", [float,int], optional = True, default = 0.00, doc = doc_limit_pref_v),
        Argument("start_pref_ae", [float,int], optional = True, default = 0.00, doc = doc_start_pref_ae),
        Argument("limit_pref_ae", [float,int], optional = True, default = 0.00, doc = doc_limit_pref_ae),
        Argument("relative_f", [float,None], optional = True, doc = doc_relative_f)
    ]

# YWolfeee: Modified to support tensor type of loss args.
def loss_tensor():
    #doc_global_weight = "The prefactor of the weight of global loss. It should be larger than or equal to 0. If only `pref` is provided or both are not provided, training will be global mode, i.e. the shape of 'polarizability.npy` or `dipole.npy` should be #frams x [9 or 3]." 
    #doc_local_weight =  "The prefactor of the weight of atomic loss. It should be larger than or equal to 0. If only `pref_atomic` is provided, training will be atomic mode, i.e. the shape of `polarizability.npy` or `dipole.npy` should be #frames x ([9 or 3] x #selected atoms). If both `pref` and `pref_atomic` are provided, training will be combined mode, and atomic label should be provided as well." 
    doc_global_weight = "The prefactor of the weight of global loss. It should be larger than or equal to 0. If controls the weight of loss corresponding to global label, i.e. 'polarizability.npy` or `dipole.npy`, whose shape should be #frames x [9 or 3]. If it's larger than 0.0, this npy should be included." 
    doc_local_weight =  "The prefactor of the weight of atomic loss. It should be larger than or equal to 0. If controls the weight of loss corresponding to atomic label, i.e. `atomic_polarizability.npy` or `atomic_dipole.npy`, whose shape should be #frames x ([9 or 3] x #selected atoms). If it's larger than 0.0, this npy should be included. Both `pref` and `pref_atomic` should be provided, and either can be set to 0.0." 
    return [
        Argument("pref", [float,int], optional = False, default = None, doc = doc_global_weight),
        Argument("pref_atomic", [float,int], optional = False, default = None, doc = doc_local_weight),
    ]


def loss_variant_type_args():
    doc_loss = 'The type of the loss. When the fitting type is `ener`, the loss type should be set to `ener` or left unset. When the fitting type is `dipole` or `polar`, the loss type should be set to `tensor`. \n\.'

    
    return Variant("type", 
                   [Argument("ener", dict, loss_ener()),
                    Argument("tensor", dict, loss_tensor()),
                    #Argument("polar", dict, loss_tensor()),
                    #Argument("global_polar", dict, loss_tensor("global"))
                    ],
                   optional = True,
                   default_tag = 'ener',
                   doc = doc_loss)


def loss_args():
    doc_loss = 'The definition of loss function. The loss type should be set to `tensor`, `ener` or left unset.\n\.'
    ca = Argument('loss', dict, [], 
                  [loss_variant_type_args()],
                  optional = True,
                  doc = doc_loss)
    return ca


#  --- Training configurations: --- #
def training_data_args():  # ! added by Ziyao: new specification style for data systems.
    link_sys = make_link("systems", "training/training_data/systems")
    doc_systems = 'The data systems for training. ' \
        'This key can be provided with a list that specifies the systems, or be provided with a string ' \
        'by which the prefix of all systems are given and the list of the systems is automatically generated.'
    doc_set_prefix = f'The prefix of the sets in the {link_sys}.'
    doc_batch_size = f'This key can be \n\n\
- list: the length of which is the same as the {link_sys}. The batch size of each system is given by the elements of the list.\n\n\
- int: all {link_sys} use the same batch size.\n\n\
- string "auto": automatically determines the batch size so that the batch_size times the number of atoms in the system is no less than 32.\n\n\
- string "auto:N": automatically determines the batch size so that the batch_size times the number of atoms in the system is no less than N.'
    doc_auto_prob_style = 'Determine the probability of systems automatically. The method is assigned by this key and can be\n\n\
- "prob_uniform"  : the probability all the systems are equal, namely 1.0/self.get_nsystems()\n\n\
- "prob_sys_size" : the probability of a system is proportional to the number of batches in the system\n\n\
- "prob_sys_size;stt_idx:end_idx:weight;stt_idx:end_idx:weight;..." : the list of systems is devided into blocks. A block is specified by `stt_idx:end_idx:weight`, where `stt_idx` is the starting index of the system, `end_idx` is then ending (not including) index of the system, the probabilities of the systems in this block sums up to `weight`, and the relatively probabilities within this block is proportional to the number of batches in the system.'
    doc_sys_probs = "A list of float if specified. " \
        "Should be of the same length as `systems`, " \
        "specifying the probability of each system."


    args = [
        Argument("systems", [list, str], optional=False, default=".", doc=doc_systems),
        Argument("set_prefix", str, optional=True, default='set', doc=doc_set_prefix),
        Argument("batch_size", [list, int, str], optional=True, default='auto', doc=doc_batch_size),
        Argument("auto_prob", str, optional=True, default="prob_sys_size",
                 doc=doc_auto_prob_style, alias=["auto_prob_style",]),
        Argument("sys_probs", list, optional=True, default=None, doc=doc_sys_probs, alias=["sys_weights"]),
    ]

    doc_training_data = "Configurations of training data."
    return Argument("training_data", dict, optional=False,
                    sub_fields=args, sub_variants=[], doc=doc_training_data)


def validation_data_args():  # ! added by Ziyao: new specification style for data systems.
    link_sys = make_link("systems", "training/validation_data/systems")
    doc_systems = 'The data systems for validation. ' \
                  'This key can be provided with a list that specifies the systems, or be provided with a string ' \
                  'by which the prefix of all systems are given and the list of the systems is automatically generated.'
    doc_set_prefix = f'The prefix of the sets in the {link_sys}.'
    doc_batch_size = f'This key can be \n\n\
- list: the length of which is the same as the {link_sys}. The batch size of each system is given by the elements of the list.\n\n\
- int: all {link_sys} use the same batch size.\n\n\
- string "auto": automatically determines the batch size so that the batch_size times the number of atoms in the system is no less than 32.\n\n\
- string "auto:N": automatically determines the batch size so that the batch_size times the number of atoms in the system is no less than N.'
    doc_auto_prob_style = 'Determine the probability of systems automatically. The method is assigned by this key and can be\n\n\
- "prob_uniform"  : the probability all the systems are equal, namely 1.0/self.get_nsystems()\n\n\
- "prob_sys_size" : the probability of a system is proportional to the number of batches in the system\n\n\
- "prob_sys_size;stt_idx:end_idx:weight;stt_idx:end_idx:weight;..." : the list of systems is devided into blocks. A block is specified by `stt_idx:end_idx:weight`, where `stt_idx` is the starting index of the system, `end_idx` is then ending (not including) index of the system, the probabilities of the systems in this block sums up to `weight`, and the relatively probabilities within this block is proportional to the number of batches in the system.'
    doc_sys_probs = "A list of float if specified. " \
                    "Should be of the same length as `systems`, " \
                    "specifying the probability of each system."
    doc_numb_btch = "An integer that specifies the number of systems to be sampled for each validation period."

    args = [
        Argument("systems", [list, str], optional=False, default=".", doc=doc_systems),
        Argument("set_prefix", str, optional=True, default='set', doc=doc_set_prefix),
        Argument("batch_size", [list, int, str], optional=True, default='auto', doc=doc_batch_size),
        Argument("auto_prob", str, optional=True, default="prob_sys_size",
                 doc=doc_auto_prob_style, alias=["auto_prob_style", ]),
        Argument("sys_probs", list, optional=True, default=None, doc=doc_sys_probs, alias=["sys_weights"]),
        Argument("numb_btch", int, optional=True, default=1, doc=doc_numb_btch, alias=["numb_batch", ])
    ]

    doc_validation_data = "Configurations of validation data. Similar to that of training data, " \
                        "except that a `numb_btch` argument may be configured"
    return Argument("validation_data", dict, optional=True, default=None,
                    sub_fields=args, sub_variants=[], doc=doc_validation_data)


def training_args():  # ! modified by Ziyao: data configuration isolated.
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
        Argument("numb_steps", int, optional=False, doc=doc_numb_steps, alias=["stop_batch"]),
        Argument("seed", [int,None], optional=True, doc=doc_seed),
        Argument("disp_file", str, optional=True, default='lcueve.out', doc=doc_disp_file),
        Argument("disp_freq", int, optional=True, default=1000, doc=doc_disp_freq),
        Argument("numb_test", [list,int,str], optional=True, default=1, doc=doc_numb_test),
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
    return Argument("training", dict, args, [], doc = doc_training)


def make_index(keys):
    ret = []
    for ii in keys:
        ret.append(make_link(ii, ii))
    return ', '.join(ret)


def gen_doc(*, make_anchor=True, make_link=True, **kwargs):
    if make_link:
        make_anchor = True
    ma = model_args()
    lra = learning_rate_args()
    la = loss_args()
    ta = training_args()
    ptr = []
    ptr.append(ma.gen_doc(make_anchor=make_anchor, make_link=make_link, **kwargs))
    ptr.append(la.gen_doc(make_anchor=make_anchor, make_link=make_link, **kwargs))
    ptr.append(lra.gen_doc(make_anchor=make_anchor, make_link=make_link, **kwargs))
    ptr.append(ta.gen_doc(make_anchor=make_anchor, make_link=make_link, **kwargs))

    key_words = []
    for ii in "\n\n".join(ptr).split('\n'):
        if 'argument path' in ii:
            key_words.append(ii.split(':')[1].replace('`','').strip())
    #ptr.insert(0, make_index(key_words))

    return "\n\n".join(ptr)

def gen_json(**kwargs):
    return json.dumps((
        model_args(),
        learning_rate_args(),
        loss_args(),
        training_args(),
    ), cls=ArgumentEncoder)

def normalize_hybrid_list(hy_list):
    new_list = []
    base = Argument("base", dict, [], [descrpt_variant_type_args()], doc = "")
    for ii in range(len(hy_list)):
        data = base.normalize_value(hy_list[ii], trim_pattern="_*")
        base.check_value(data, strict=True)
        new_list.append(data)
    return new_list


def normalize(data):
    if "hybrid" == data["model"]["descriptor"]["type"]:
        data["model"]["descriptor"]["list"] \
            = normalize_hybrid_list(data["model"]["descriptor"]["list"])

    ma = model_args()
    lra = learning_rate_args()
    la = loss_args()
    ta = training_args()

    base = Argument("base", dict, [ma, lra, la, ta])
    data = base.normalize_value(data, trim_pattern="_*")
    base.check_value(data, strict=True)

    return data


if __name__ == '__main__':
    gen_doc()

