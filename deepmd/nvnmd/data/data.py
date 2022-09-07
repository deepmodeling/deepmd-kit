
jdata_sys = {
    "debug": False
}

jdata_config = {
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
        "smin": -2,
        "smax": 14,
        "NUM_MAPT": 1024,
        "NUM_U2S": "rc_max^2",
        "NUM_S2G": "rng_s_max - rng_s_min = 14 - (-2)"
    },

    "fitn": {
        # basic config from deepmd model
        "neuron": [32, 32, 32],
        "resnet_dt": False,

        "NNODE_FITS": "(M1*M2, neuron, 1)",
        "nlayer_fit": "len(neuron)+1",
        "NLAYER": "nlayer_fit"
    },

    # other input for generate input file
    "dpin": {
        "type_map" : []
    },

    "size": {
        # atom system size for simulation
        "Na": 4096,
        "NaX": 32768,
        "NAEXT": "Na:nloc+nglb",
        # ntype in build model
        "NTYPE": "ntype_max",
        "NTYPEX": "ntypex_max",
        # model size
        "NCFG": 35,
        "NFEA": 8192,
        "NNET": 4920
    },

    "ctrl": {
        # NSTDM
        "NSTDM": 16,
        "NSTDM_M1": 16,
        "NSTDM_M2": 1,
        "NSTDM_M1X": 4,
        "NSEL": "NSTDM*NTYPE_MAX",
        "NSADV": "NSTDM+1",
        "MAX_FANOUT": 30,
        # delay
        "NSTEP_DELAY": 20,
    },

    "nbit": {
        # general
        "NBIT_SHORT": 18,
        "NBIT_SHORT_FL": 10,
        "NBIT_DATA": 21,
        "NBIT_DATA_FL": 13,
        "NBIT_LONG": 24,
        "NBIT_LONG_FL": 16,
        "NBIT_LLONG": 27,
        "NBIT_LLONG_FL": 19,
        # atomic data
        "NBIT_SPE": "ln2(NTYPE_MAX)",
        "NBIT_LST": "ln2(NaX)",
        "NBIT_CRD": 32,
        "NBIT_CRD_FL": 24,
        "NBIT_CRD3": "NBIT_CRD*3",
        "NBIT_ATOM": "NBIT_SPE+NBIT_CRD3",
        "NBIT_FRC": 32,
        "NBIT_FRC_FL": "NBIT_LLONG_FL",
        "NBIT_VRL": 32,
        "NBIT_VRL_FL": "NBIT_LLONG_FL",
        # network
        "NBIT_WEIGHT": 18,
        "NBIT_WEIGHT_FL": 13,
        "NBIT_CBSV": 18,
        "NBIT_CBSV_FL": 16,
        # middle result
        "NBIT_SEL": "ln2(NSEL)",
        "NBIT_RIJ": "NBIT_DATA_FL+5",

        "NBIT_DATA2": "NBIT_DATA+NBIT_DATA_FL",
        "NBIT_DATA2_FL": "NBIT_DATA_FL*2",
        "NBIT_DATA_SHORT": "NBIT_DATA+NBIT_SHORT_FL",
        "NBIT_DATA_SHORT_FL": "NBIT_DATA_FL+NBIT_SHORT_FL",
        # communication
        "NBIT_SPE_MAX": 8,
        "NBIT_LST_MAX": 16,

        "NBIT_ADDR": 32,
        "NBIT_SYS": 32,

        "NBIT_BYPASS_DATA": 32,
        "NBIT_CFG": 64,
        "NBIT_NET": 72,
        
        "NBTI_MODEL_HEAD": 32,
        # nbit for mapt-version
        "NBIT_MAPT_XK": "ln2(NUM_MAPT)",
        "NBIT_MAPT_XK_U2S_FL": "ln2(NUM_MAPT/NUM_U2S)",
        "NBIT_MAPT_XK_S2G_FL": "ln2(NUM_MAPT/NUM_S2G)",
        "NBIT_SHIFT": 4
    },

    "end": ""
}

jdata_config_16 = {
    "dscp": {
        "neuron": [8, 16, 32],
        "axis_neuron": 4,
        "NI": 128
    },

    "fitn": {
        "neuron": [16, 16, 16]
    },

    "ctrl": {
        "NSTDM": 16,
        "NSTDM_M1": 16,
        "NSTDM_M2": 1,
        "NSTDM_M1X": 4
    }
}

jdata_config_32 = {
    "dscp": {
        "neuron": [8, 16, 32],
        "axis_neuron": 4,
        "NI": 128
    },

    "fitn": {
        "neuron": [32, 32, 32]
    },

    "ctrl": {
        "NSTDM": 16,
        "NSTDM_M1": 16,
        "NSTDM_M2": 1,
        "NSTDM_M1X": 4
    }
}

jdata_config_64 = {
    "dscp": {
        "neuron": [8, 16, 32],
        "axis_neuron": 4,
        "NI": 128
    },

    "fitn": {
        "neuron": [64, 64, 64]
    },

    "ctrl": {
        "NSTDM": 32,
        "NSTDM_M1": 32,
        "NSTDM_M2": 1,
        "NSTDM_M1X": 4
    }
}

jdata_config_128 = {
    "dscp": {
        "neuron": [8, 16, 32],
        "axis_neuron": 4,
        "NI": 128
    },

    "fitn": {
        "neuron": [128, 128, 128]
    },

    "ctrl": {
        "NSTDM": 32,
        "NSTDM_M1": 32,
        "NSTDM_M2": 1,
        "NSTDM_M1X": 4
    }
}

jdata_configs = {
    "_16": jdata_config_16,
    "_32": jdata_config_32,
    "_64": jdata_config_64,
    "128": jdata_config_128
}

jdata_deepmd_input = {
    "model": {
        "descriptor": {
            "seed": 1,
            "type": "se_a",
            "sel": [
                60,
                60
            ],
            "rcut": 7.0,
            "rcut_smth": 0.5,
            "neuron": [
                8,
                16,
                32
            ],
            "type_one_side": False,
            "axis_neuron": 4,
            "resnet_dt": False
        },
        "fitting_net": {
            "seed": 1,
            "neuron": [
                128,
                128,
                128
            ],
            "resnet_dt": False
        }
    },
    "nvnmd": {
        "net_size": 128,
        "config_file": "none",
        "weight_file": "none",
        "map_file": "none",
        "enable": False,
        "restore_descriptor": False,
        "restore_fitting_net": False,
        "quantize_descriptor": False,
        "quantize_fitting_net": False
    },
    "learning_rate": {
        "type": "exp",
        "decay_steps": 5000,
        "start_lr": 0.005,
        "stop_lr": 8.257687192506788e-05
    },
    "loss": {
        "start_pref_e": 0.02,
        "limit_pref_e": 1,
        "start_pref_f": 1000,
        "limit_pref_f": 1,
        "start_pref_v": 0,
        "limit_pref_v": 0
    },
    "training": {
        "seed": 1,
        "stop_batch": 10000,
        "disp_file": "lcurve.out",
        "disp_freq": 100,
        "numb_test": 10,
        "save_freq": 1000,
        "save_ckpt": "model.ckpt",
        "disp_training": True,
        "time_training": True,
        "profiling": False,
        "training_data": {
            "systems": "dataset",
            "set_prefix": "set",
            "batch_size": 1
        }
    }
}
NVNMD_WELCOME = (
    " _   _  __     __  _   _   __  __   ____  ",
    "| \ | | \ \   / / | \ | | |  \/  | |  _ \ ",
    "|  \| |  \ \ / /  |  \| | | |\/| | | | | |",
    "| |\  |   \ V /   | |\  | | |  | | | |_| |",
    "|_| \_|    \_/    |_| \_| |_|  |_| |____/ ",
)

NVNMD_CITATION = (
    "Please read and cite:",
    "Mo et al., npj Comput Mater 8, 107 (2022)",
)
