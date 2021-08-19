from dargs import dargs, Argument, Variant
from .argcheck import list_to_doc, make_link
from .argcheck import type_embedding_args, descrpt_se_a_tpe_args, modifier_dipole_charge, modifier_variant_type_args
from .argcheck import model_compression, model_compression_type_args
from .argcheck import start_pref, limit_pref
from .argcheck import training_data_args, validation_data_args, make_index, gen_doc
from .argcheck import descrpt_local_frame_args, descrpt_se_a_args, descrpt_se_t_args, descrpt_se_r_args, descrpt_se_ar_args, descrpt_hybrid_args
from .argcheck import fitting_ener, fitting_polar, fitting_dipole
from .argcheck import learning_rate_exp, loss_ener, loss_tensor
from .argcheck import loss_variant_type_args
from .argcheck import fitting_variant_type_args
from .argcheck import learning_rate_variant_type_args
from .argcheck import descrpt_variant_type_args
# from deepmd.common import ACTIVATION_FN_DICT, PRECISION_DICT
ACTIVATION_FN_DICT = {}
PRECISION_DICT = {}



#  --- model configurations: --- #


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

def learning_rate_args():
    doc_lr = "A list of learning rates identified by key arg `name`."
    return Argument("learning_rate", list, doc=doc_lr)


#  --- Loss configurations: --- #

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


def training_data_args_mt():  
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
        Argument("auto_prob_method", str, optional=True, default="prob_uniform", doc=doc_auto_prob_style, alias=["auto_prob_style_method",]),
        Argument("sys_probs", list, optional=True, default=None, doc=doc_sys_probs, alias=["sys_weights"]),
    ]

    doc_training_data = "Configurations of training data."
    return Argument("training_data", dict, optional=False,
                    sub_fields=args, sub_variants=[], doc=doc_training_data)

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

    arg_training_data = training_data_args_mt()
    arg_validation_data = validation_data_args()

    args = [
        arg_training_data,
        arg_validation_data,
        Argument("tasks", list, optional=False, doc=doc_tasks, alias=["sub_models"]),
        Argument("numb_steps", int, optional=False, doc=doc_numb_steps, alias=["stop_batch"]),
        Argument("seed", [int, None], optional=True, doc=doc_seed),
        Argument("disp_file", str, optional=True, default='lcurve.out', doc=doc_disp_file),
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
        pattern = Argument("base", dict, [Argument("name", str, optional=True, default='', doc="name of the component."),
    ], [pattern_arg], doc="")
    elif isinstance(pattern_arg, Argument):
        pattern = pattern_arg
    else:
        raise AssertionError("Wrong type of input pattern argument: %s" % str(pattern_arg))
    new_list = [pattern.normalize_value(ii, trim_pattern="_*") for ii in old_list]
    [pattern.check_value(ii, strict=True) for ii in new_list]
    return new_list


def normalize_hybrid_list(hy_list):
    new_list = []
    base = Argument("base", dict, [
        Argument("name", str, optional=True, default='', doc="Descriptor name."),
    ], [descrpt_variant_type_args()], doc="")
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

