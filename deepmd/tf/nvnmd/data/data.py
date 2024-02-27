# SPDX-License-Identifier: LGPL-3.0-or-later
jdata_sys = {"debug": False}

jdata_config_v0 = {
    "dscp": {
        # basic config from deepmd model
        "sel": [60, 60],
        "rcut": 6.0,
        "rcut_smth": 0.5,
        "neuron": [8, 16, 32],
        "resnet_dt": False,
        "axis_neuron": 4,
        "type_one_side": True,
        # rcut range
        "rc_lim": 0.5,
        "rc_max": 8.0,
        # embedding net size
        "M1": "neuron[-1]",
        "M2": "axis_neuron",
        "SEL": [60, 60, 0, 0],
        "NNODE_FEAS": "(1, neuron)",
        "nlayer_fea": "len(neuron)",
        "same_net": "type_one_side",
        # num neighbor
        "NI": 128,
        "NIDP": "sum(sel)",
        "NIX": "2^ceil(ln2(NIDP/1.5))",
        # num type
        "ntype": "len(sel)",
        "ntypex": "same_net ? 1: ntype",
        "ntypex_max": 1,
        "ntype_max": 4,
        # mapping table
        "dmin": 0,
        "smin": -2,
        "smax": 14,
    },
    "fitn": {
        # basic config from deepmd model
        "neuron": [128, 128, 128],
        "resnet_dt": False,
        "NNODE_FITS": "(M1*M2, neuron, 1)",
        "nlayer_fit": "len(neuron)+1",
        "NLAYER": "nlayer_fit",
    },
    # other input for generate input file
    "dpin": {"type_map": []},
    "size": {
        # atom system size for simulation
        "Na": 4096,
        "NaX": 32768,
        "NAEXT": "Na:nloc+nglb",
        # ntype in build model
        "NTYPE": "ntype_max",
        "NTYPEX": "ntypex_max",
        # model size
        "NH_DATA": [0, 0, 0, 0, 0, 0, 0],
        "NW_DATA": [0, 0, 0, 0, 0, 0, 0],
        "NH_SIM": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "ctrl": {
        # NSTDM
        "MAX_NNEI": 128,
        "NSTDM": 64,
        "NSTDM_M1": 32,
        "NSTDM_M2": 2,
        "NSTDM_M1X": 8,
        "NSEL": "NSTDM*NTYPE_MAX",
        "NSADV": "NSTDM+1",
        "VERSION": 0,
        "SUB_VERSION": 1,
    },
    "nbit": {
        # general
        "NBIT_FLTD": 29,
        "NBIT_FLTS": 1,
        "NBIT_FLTE": 8,
        "NBIT_FLTF": 20,
        "NBIT_FLTM": "1+NBIT_FLTF",
        "NBIT_FLTH": "1+NBIT_FLTM",
        "NBIT_FIXD": 32,
        "NBIT_FIXD_FL": 23,
        # atomic data
        "NBIT_SPE": "ln2(NTYPE_MAX)",
        "NBIT_LST": "ln2(NaX)",
        "NBIT_CRD": 32,
        "NBIT_CRD_FL": 24,
        "NBIT_CRD3": "NBIT_CRD*3",
        "NBIT_ATOM": "NBIT_SPE+NBIT_CRD3",
        "NBIT_ENE": 32,
        "NBIT_ENE_FL": "NBIT_FIT_DATA_FL",
        "NBIT_FRC": 32,
        "NBIT_FRC_FL": 19,
        "NBIT_VRL": 32,
        "NBIT_VRL_FL": 19,
        # network
        "NBIT_FIT_DATA": 27,
        "NBIT_FIT_DATA_FL": 23,
        "NBIT_FIT_SHORT_FL": 19,
        "NBIT_FIT_WEIGHT": 18,
        "NBIT_FIT_DISP": 3,
        "NBIT_FIT_WXDB": 29,
        # middle result
        "NBIT_SEL": "ln2(NSEL)",
        # communication
        "NBIT_SPE_MAX": 8,
        "NBIT_LST_MAX": 16,
        "NBIT_ADDR": 32,
        "NBIT_SYS": 32,
        "NBIT_BYPASS_DATA": 32,
        "NBIT_CFG": 64,
        "NBIT_NET": 72,
        "NBIT_MODEL_HEAD": 32,
        # nbit for mapt-version
        "NBIT_IDX_S2G": 9,
        "NBIT_NEIB": 8,
    },
    "end": "",
}

# change the configuration accordng to the max_nnei
jdata_config_v0_ni128 = jdata_config_v0.copy()
jdata_config_v0_ni256 = jdata_config_v0.copy()
jdata_config_v0_ni256["ctrl"] = {
    "MAX_NNEI": 256,
    "NSTDM": 128,
    "NSTDM_M1": 32,
    "NSTDM_M2": 4,
    "NSTDM_M1X": 8,
    "NSEL": "NSTDM*NTYPE_MAX",
    "NSADV": "NSTDM+1",
    "VERSION": 0,
    "SUB_VERSION": 1,
}
jdata_config_v0_ni256["nbit"]["NBIT_NEIB"] = 9

jdata_config_v1 = {
    "dscp": {
        # basic config from deepmd model
        "sel": 128,
        "rcut": 6.0,
        "rcut_smth": 0.5,
        "neuron": [8, 16, 32],
        "resnet_dt": False,
        "axis_neuron": 4,
        "type_one_side": True,
        # rcut range
        "rc_lim": 0.5,
        "rc_max": 8.0,
        # embedding net size
        "M1": "neuron[-1]",
        "M2": "axis_neuron",
        "SEL": 128,
        "NNODE_FEAS": "(1, neuron)",
        "nlayer_fea": "len(neuron)",
        "same_net": "type_one_side",
        # num neighbor
        "NI": 128,
        "NIDP": "sel",
        "NIX": "2^ceil(ln2(NIDP/1.5))",
        # num type
        "ntype": None,
        "ntype_max": 32,
        # mapping table
        "dmin": 0,
        "smin": -2,
        "smax": 14,
    },
    "fitn": {
        # basic config from deepmd model
        "neuron": [128, 128, 128],
        "resnet_dt": False,
        "NNODE_FITS": "(M1*M2, neuron, 1)",
        "nlayer_fit": "len(neuron)+1",
        "NLAYER": "nlayer_fit",
        "NTAVC": 8,
    },
    # other input for generate input file
    "dpin": {"type_map": []},
    "size": {
        # atom system size for simulation
        "Na": 4096,
        "NaX": 32768,
        "NAEXT": "Na:nloc+nglb",
        # ntype in build model
        "NTYPE": "ntype_max",
        "NTYPEX": "ntypex_max",
        # model size
        "NH_DATA": [0, 0, 0, 0, 0, 0, 0],
        "NW_DATA": [0, 0, 0, 0, 0, 0, 0],
        "NH_SIM": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "ctrl": {
        # NSTDM
        "MAX_NNEI": 128,
        "NSTDM": 64,
        "NSTDM_M1": 32,
        "NSTDM_M2": 2,
        "NSTDM_M1X": 8,
        "NSEL": "NSTDM",
        "NSADV": "NSTDM+1",
        "VERSION": 1,
        "SUB_VERSION": 1,
    },
    "nbit": {
        # general
        "NBIT_FLTD": 29,
        "NBIT_FLTS": 1,
        "NBIT_FLTE": 8,
        "NBIT_FLTF": 20,
        "NBIT_FLTM": "1+NBIT_FLTF",
        "NBIT_FLTH": "1+NBIT_FLTM",
        "NBIT_FIXD": 32,
        "NBIT_FIXD_FL": 23,
        # atomic data
        "NBIT_SPE": "ln2(NTYPE_MAX)",
        "NBIT_LST": "ln2(NaX)",
        "NBIT_CRD": 32,
        "NBIT_CRD_FL": 24,
        "NBIT_CRD3": "NBIT_CRD*3",
        "NBIT_ATOM": "NBIT_SPE+NBIT_CRD3",
        "NBIT_ENE": 32,
        "NBIT_ENE_FL": "NBIT_FIT_DATA_FL",
        "NBIT_FRC": 32,
        "NBIT_FRC_FL": 19,
        "NBIT_VRL": 32,
        "NBIT_VRL_FL": 19,
        # network
        "NBIT_FIT_DATA": 27,
        "NBIT_FIT_DATA_FL": 23,
        "NBIT_FIT_SHORT_FL": 19,
        "NBIT_FIT_WEIGHT": 18,
        "NBIT_FIT_DISP": 3,
        "NBIT_FIT_WXDB": 29,
        # middle result
        "NBIT_SEL": "ln2(NSEL)",
        # communication
        "NBIT_SPE_MAX": 8,
        "NBIT_LST_MAX": 16,
        "NBIT_ADDR": 32,
        "NBIT_SYS": 32,
        "NBIT_BYPASS_DATA": 32,
        "NBIT_CFG": 64,
        "NBIT_NET": 72,
        "NBIT_MODEL_HEAD": 32,
        # nbit for mapt-version
        "NBIT_IDX_S2G": 9,
        "NBIT_NEIB": 8,
    },
    "end": "",
}

# change the configuration accordng to the max_nnei
jdata_config_v1_ni128 = jdata_config_v1.copy()
jdata_config_v1_ni256 = jdata_config_v1.copy()
jdata_config_v1_ni256["ctrl"] = {
    "MAX_NNEI": 256,
    "NSTDM": 128,
    "NSTDM_M1": 32,
    "NSTDM_M2": 4,
    "NSTDM_M1X": 8,
    "NSEL": "NSTDM",
    "NSADV": "NSTDM+1",
    "VERSION": 1,
    "SUB_VERSION": 1,
}
jdata_config_v1_ni256["nbit"]["NBIT_NEIB"] = 9

jdata_deepmd_input_v0 = {
    "model": {
        "descriptor": {
            "seed": 1,
            "type": "se_a",
            "sel": [60, 60],
            "rcut": 7.0,
            "rcut_smth": 0.5,
            "neuron": [8, 16, 32],
            "type_one_side": True,
            "axis_neuron": 4,
            "resnet_dt": False,
        },
        "fitting_net": {"seed": 1, "neuron": [128, 128, 128], "resnet_dt": False},
    },
    "nvnmd": {
        "version": 0,
        "max_nnei": 128,  # 128 or 256
        "net_size": 128,
        "config_file": "none",
        "weight_file": "none",
        "map_file": "none",
        "enable": False,
        "restore_descriptor": False,
        "restore_fitting_net": False,
        "quantize_descriptor": False,
        "quantize_fitting_net": False,
    },
    "learning_rate": {
        "type": "exp",
        "decay_steps": 1,
        "start_lr": 1e-10,
        "stop_lr": 1e-10,
    },
    "loss": {
        "start_pref_e": 1,
        "limit_pref_e": 1,
        "start_pref_f": 1,
        "limit_pref_f": 1,
        "start_pref_v": 0,
        "limit_pref_v": 0,
    },
    "training": {
        "seed": 1,
        "stop_batch": 1,
        "disp_file": "lcurve.out",
        "disp_freq": 1,
        "numb_test": 1,
        "save_freq": 1,
        "save_ckpt": "model.ckpt",
        "disp_training": True,
        "time_training": True,
        "profiling": False,
        "training_data": {"systems": "dataset", "set_prefix": "set", "batch_size": 1},
    },
}

jdata_deepmd_input_v0_ni128 = jdata_deepmd_input_v0.copy()
jdata_deepmd_input_v0_ni256 = jdata_deepmd_input_v0.copy()
jdata_deepmd_input_v0_ni256["nvnmd"]["max_nnei"] = 256

jdata_deepmd_input_v1 = {
    "model": {
        "descriptor": {
            "seed": 1,
            "type": "se_atten",
            "stripped_type_embedding": True,
            "sel": 128,
            "rcut": 7.0,
            "rcut_smth": 0.5,
            "neuron": [8, 16, 32],
            "type_one_side": True,
            "axis_neuron": 4,
            "resnet_dt": False,
            "attn": 128,
            "attn_layer": 0,
            "attn_dotr": True,
            "attn_mask": False,
        },
        "fitting_net": {"seed": 1, "neuron": [128, 128, 128], "resnet_dt": False},
    },
    "nvnmd": {
        "version": 1,
        "max_nnei": 128,  # 128 or 256
        "net_size": 128,
        "config_file": "none",
        "weight_file": "none",
        "map_file": "none",
        "enable": False,
        "restore_descriptor": False,
        "restore_fitting_net": False,
        "quantize_descriptor": False,
        "quantize_fitting_net": False,
    },
    "learning_rate": {
        "type": "exp",
        "decay_steps": 1,
        "start_lr": 1e-10,
        "stop_lr": 1e-10,
    },
    "loss": {
        "start_pref_e": 1,
        "limit_pref_e": 1,
        "start_pref_f": 1,
        "limit_pref_f": 1,
        "start_pref_v": 0,
        "limit_pref_v": 0,
    },
    "training": {
        "seed": 1,
        "stop_batch": 1,
        "disp_file": "lcurve.out",
        "disp_freq": 1,
        "numb_test": 1,
        "save_freq": 1,
        "save_ckpt": "model.ckpt",
        "disp_training": True,
        "time_training": True,
        "profiling": False,
        "training_data": {"systems": "dataset", "set_prefix": "set", "batch_size": 1},
    },
}

jdata_deepmd_input_v1_ni128 = jdata_deepmd_input_v1.copy()
jdata_deepmd_input_v1_ni256 = jdata_deepmd_input_v1.copy()
jdata_deepmd_input_v1_ni256["nvnmd"]["max_nnei"] = 256

NVNMD_WELCOME = (
    r" _   _  __     __  _   _   __  __   ____  ",
    r"| \ | | \ \   / / | \ | | |  \/  | |  _ \ ",
    r"|  \| |  \ \ / /  |  \| | | |\/| | | | | |",
    r"| |\  |   \ V /   | |\  | | |  | | | |_| |",
    r"|_| \_|    \_/    |_| \_| |_|  |_| |____/ ",
)

NVNMD_CITATION = (
    "Please read and cite:",
    "Mo et al., npj Comput Mater 8, 107 (2022)",
)
