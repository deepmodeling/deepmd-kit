
import os 

from deepmd.env import tf
from deepmd.entrypoints.train import train
from deepmd.nvnmd.entrypoints.map import map
from deepmd.nvnmd.entrypoints.wrap import wrap
from deepmd.nvnmd.entrypoints.freeze import freeze_nvnmd as freeze

from deepmd.nvnmd.utils.fio import FioDic
from deepmd.nvnmd.utils.config import nvnmd_cfg
from deepmd.nvnmd.data.data import jdata_deepmd_input


jdata_cmd_train = {
    "INPUT": "train.json",
    "init_model": None,
    "restart": None,
    "output": "out.json",
    "init_frz_model": None,
    "mpi_log": "master",
    "log_level": 2,
    "log_path": None,
    "is_compress": False
}

jdata_cmd_freeze = {
    "checkpoint_folder": '.',
    "output": 'frozen_model.pb',
    "node_names": None,
    "nvnmd_weight": "nvnmd/weight.npy"
}


def replace_path(p, p2):
    pars = p.split(os.sep)
    pars[-2] = p2
    return os.path.join(*pars)


def add_path(p, p2):
    pars = p.split('/')
    pars.insert(-1, p2)
    return os.path.join(*pars)


def normalized_input(fn, PATH_CNN):
    f = FioDic()
    jdata = f.load(fn, jdata_deepmd_input)
    # nvnmd
    jdata_nvnmd = jdata_deepmd_input['nvnmd']
    jdata_nvnmd['enable'] = True
    jdata_nvnmd_ = f.get(jdata, 'nvnmd', jdata_nvnmd)
    jdata_nvnmd = f.update(jdata_nvnmd_, jdata_nvnmd)
    # model
    jdata_model = {
        "descriptor": {
            "seed": 1,
            "sel": jdata_nvnmd_["sel"],
            "rcut": jdata_nvnmd_['rcut'],
            "rcut_smth": jdata_nvnmd_['rcut_smth']
        },
        "fitting_net": {
            "seed": 1
        }}
    nvnmd_cfg.init_from_jdata(jdata_nvnmd)
    nvnmd_cfg.init_from_deepmd_input(jdata_model)
    nvnmd_cfg.init_train_mode('cnn')
    # training
    jdata_train = f.get(jdata, 'training', {})
    jdata_train['disp_training'] = True
    jdata_train['time_training'] = True
    jdata_train['profiling'] = False
    jdata_train['disp_file'] = add_path(jdata_train['disp_file'], PATH_CNN)
    jdata_train['save_ckpt'] = add_path(jdata_train['save_ckpt'], PATH_CNN)
    #
    jdata['model'] = nvnmd_cfg.get_model_jdata()
    jdata['nvnmd'] = nvnmd_cfg.get_nvnmd_jdata()
    return jdata


def normalized_input_qnn(jdata, PATH_QNN, CONFIG_CNN, WEIGHT_CNN, MAP_CNN):
    #
    jdata_nvnmd = jdata_deepmd_input['nvnmd']
    jdata_nvnmd['enable'] = True
    jdata_nvnmd['config_file'] = CONFIG_CNN
    jdata_nvnmd['weight_file'] = WEIGHT_CNN
    jdata_nvnmd['map_file'] = MAP_CNN
    nvnmd_cfg.init_from_jdata(jdata_nvnmd)
    nvnmd_cfg.init_train_mode('qnn')
    jdata['nvnmd'] = nvnmd_cfg.get_nvnmd_jdata()
    # learning_rate
    jdata1 = jdata['learning_rate']
    jdata2 = jdata1.copy()
    jdata2['start_lr'] = 1e-8
    jdata2['stop_lr'] = 1e-9
    jdata['learning_rate'] = jdata2
    # loss
    jdata1 = jdata['loss']
    jdata2 = jdata1.copy()
    # jdata2['start_pref_e'] = jdata1['limit_pref_e']
    # jdata2['start_pref_f'] = jdata1['limit_pref_f']
    # jdata2['start_pref_v'] = jdata1['limit_pref_v']
    jdata['loss'] = jdata2
    # training
    jdata1 = jdata['training']
    jdata2 = jdata1.copy()
    jdata2['stop_batch'] = 100000
    jdata2['save_freq'] = 10000
    jdata2['disp_freq'] = 1000
    jdata2['disp_file'] = replace_path(jdata2['disp_file'], PATH_QNN)
    jdata2['save_ckpt'] = replace_path(jdata2['save_ckpt'], PATH_QNN)
    jdata['training'] = jdata2
    return jdata


def train_nvnmd(
    *,
    INPUT: str,
    step: str,
    **kwargs,
):
    #
    PATH_CNN = 'nvnmd_cnn'
    CONFIG_CNN = os.path.join(PATH_CNN, 'config.npy')
    INPUT_CNN = os.path.join(PATH_CNN, 'train.json')
    WEIGHT_CNN = os.path.join(PATH_CNN, 'weight.npy')
    FRZ_MODEL_CNN = os.path.join(PATH_CNN, 'frozen_model.pb')
    MAP_CNN = os.path.join(PATH_CNN, 'map.npy')
    #
    PATH_QNN = 'nvnmd_qnn'
    CONFIG_QNN = os.path.join(PATH_QNN, 'config.npy')
    INPUT_QNN = os.path.join(PATH_QNN, 'train.json')
    WEIGHT_QNN = os.path.join(PATH_QNN, 'weight.npy')
    FRZ_MODEL_QNN = os.path.join(PATH_QNN, 'frozen_model.pb')
    MODEL_QNN = os.path.join(PATH_QNN, 'model.pb')
    # normailize input file
    jdata = normalized_input(INPUT, PATH_CNN)
    FioDic().save(INPUT_CNN, jdata)
    nvnmd_cfg.save(CONFIG_CNN)
    if step == "s1":
        # train cnn
        jdata = jdata_cmd_train.copy()
        jdata['INPUT'] = INPUT_CNN
        train(**jdata)
        tf.reset_default_graph()
        # freeze
        jdata = jdata_cmd_freeze.copy()
        jdata['checkpoint_folder'] = PATH_CNN
        jdata['output'] = FRZ_MODEL_CNN
        jdata['nvnmd_weight'] = WEIGHT_CNN
        freeze(**jdata)
        tf.reset_default_graph()
        # map
        jdata = {
            "nvnmd_config": CONFIG_CNN,
            "nvnmd_weight": WEIGHT_CNN,
            "nvnmd_map": MAP_CNN
        }
        map(**jdata)
        tf.reset_default_graph()
        #
        step = "s2"
    # normailize input file
    jdata = FioDic().load(INPUT_CNN)
    jdata = normalized_input_qnn(jdata, PATH_QNN, CONFIG_CNN, WEIGHT_CNN, MAP_CNN)
    FioDic().save(INPUT_QNN, jdata)
    nvnmd_cfg.save(CONFIG_QNN)

    if step == "s2":
        # train qnn
        jdata = jdata_cmd_train.copy()
        jdata['INPUT'] = INPUT_QNN
        train(**jdata)
        tf.reset_default_graph()
        # freeze
        jdata = jdata_cmd_freeze.copy()
        jdata['checkpoint_folder'] = PATH_QNN
        jdata['output'] = FRZ_MODEL_QNN
        jdata['nvnmd_weight'] = WEIGHT_QNN
        freeze(**jdata)
        tf.reset_default_graph()
        # wrap
        jdata = {
            "nvnmd_config": CONFIG_QNN,
            "nvnmd_weight": WEIGHT_QNN,
            "nvnmd_map": MAP_CNN,
            "nvnmd_model": MODEL_QNN
        }
        wrap(**jdata)
        tf.reset_default_graph()
        #
        step = "s3"
