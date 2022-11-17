
jdata_sys = {
    "debug": False
}

jdata_config = {
    "dscp": {
        "sel": [60, 60],
        "rcut": 6.0,
        "rcut_smth": 0.5,
        "neuron": [8, 16, 32],
        "resnet_dt": False,
        "axis_neuron": 4,
        "type_one_side": True,

        "NI": 128,
        "rc_lim": 0.5,
        "M1": "neuron[-1]",
        "M2": "axis_neuron",
        "SEL": [60, 60, 0, 0],
        "NNODE_FEAS": "(1, neuron)",
        "nlayer_fea": "len(neuron)",
        "same_net": "type_one_side",
        "NIDP": "sum(sel)",
        "NIX": "2^ceil(ln2(NIDP/1.5))",
        "ntype": "len(sel)",
        "ntypex": "same_net ? 1: ntype",
        "ntypex_max": 1,
        "ntype_max": 4
    },

    "fitn": {
        "neuron": [32, 32, 32],
        "resnet_dt": False,

        "NNODE_FITS": "(M1*M2, neuron, 1)",
        "nlayer_fit": "len(neuron)+1",
        "NLAYER": "nlayer_fit"
    },

    "size": {
        "NTYPE_MAX": 4,
        "NSPU": 4096,
        "MSPU": 32768,
        "Na": "NSPU",
        "NaX": "MSPU"
    },

    "ctrl": {
        "NSTDM": 16,
        "NSTDM_M1": 16,
        "NSTDM_M2": 1,
        "NSADV": "NSTDM+1",
        "NSEL": "NSTDM*ntype_max",
        "NSTDM_M1X": 4,
        "NSTEP_DELAY": 20,
        "MAX_FANOUT": 30
    },

    "nbit": {
        "NBIT_DATA": 21,
        "NBIT_DATA_FL": 13,
        "NBIT_LONG_DATA": 32,
        "NBIT_LONG_DATA_FL": 24,
        "NBIT_DIFF_DATA": 24,

        "NBIT_SPE": 2,
        "NBIT_CRD": "NBIT_DATA*3",
        "NBIT_LST": "ln2(NaX)",

        "NBIT_SPE_MAX": 8,
        "NBIT_LST_MAX": 16,

        "NBIT_ATOM": "NBIT_SPE+NBIT_CRD",
        "NBIT_LONG_ATOM": "NBIT_SPE+NBIT_LONG_DATA*3",

        "NBIT_RIJ": "NBIT_DATA_FL+5",
        "NBIT_FEA_X": 10,
        "NBIT_FEA_X_FL": 4,
        "NBIT_FEA_X2_FL": 6,
        "NBIT_FEA": 18,
        "NBIT_FEA_FL": 10,
        "NBIT_SHIFT": 4,

        "NBIT_DATA2": "NBIT_DATA+NBIT_DATA_FL",
        "NBIT_DATA2_FL": "2*NBIT_DATA_FL",
        "NBIT_DATA_FEA": "NBIT_DATA+NBIT_FEA_FL",
        "NBIT_DATA_FEA_FL": "NBIT_DATA_FL+NBIT_FEA_FL",

        "NBIT_FORCE": 32,
        "NBIT_FORCE_FL": "2*NBIT_DATA_FL-1",

        "NBIT_SUM": "NBIT_DATA_FL+8",
        "NBIT_WEIGHT": 18,
        "NBIT_WEIGHT_FL": 13,

        "NBIT_RAM": 72,
        "NBIT_ADDR": 32,

        "NBTI_MODEL_HEAD": 32,

        "NBIT_TH_LONG_ADD": 30,
        "NBIT_ADD": 15,

        "RANGE_B": [-100, 100],
        "RANGE_W": [-20, 20],

        "NCFG": 35,
        "NNET": 4920,
        "NFEA": 8192
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
