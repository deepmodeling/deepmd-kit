# SPDX-License-Identifier: LGPL-3.0-or-later
from collections import (
    OrderedDict,
)

from ....consistent.common import (
    parameterize_func,
)


def LossParamEnergy(
    starter_learning_rate=1.0,
    pref_e=1.0,
    pref_f=1.0,
    pref_v=1.0,
    pref_ae=1.0,
    use_huber=False,
):
    key_to_pref_map = {
        "energy": pref_e,
        "force": pref_f,
        "virial": pref_v,
        "atom_ener": pref_ae,
    }
    input_dict = {
        "key_to_pref_map": key_to_pref_map,
        "starter_learning_rate": starter_learning_rate,
        "start_pref_e": pref_e,
        "limit_pref_e": pref_e / 2,
        "start_pref_f": pref_f,
        "limit_pref_f": pref_f / 2,
        "start_pref_v": pref_v,
        "limit_pref_v": pref_v / 2,
        "start_pref_ae": pref_ae,
        "limit_pref_ae": pref_ae / 2,
        "use_huber": use_huber,
    }
    return input_dict


LossParamEnergyList = parameterize_func(
    LossParamEnergy,
    OrderedDict(
        {
            "pref_e": (1.0, 0.0),
            "pref_f": (1.0, 0.0),
            "pref_v": (1.0, 0.0),
            "pref_ae": (1.0, 0.0),
            "use_huber": (False, True),
        }
    ),
)
# to get name for the default function
LossParamEnergy = LossParamEnergyList[0]


def LossParamEnergySpin(
    starter_learning_rate=1.0,
    pref_e=1.0,
    pref_fr=1.0,
    pref_fm=1.0,
    pref_v=1.0,
    pref_ae=1.0,
):
    key_to_pref_map = {
        "energy": pref_e,
        "force": pref_fr,
        "force_mag": pref_fm,
        "virial": pref_v,
        "atom_ener": pref_ae,
    }
    input_dict = {
        "key_to_pref_map": key_to_pref_map,
        "starter_learning_rate": starter_learning_rate,
        "start_pref_e": pref_e,
        "limit_pref_e": pref_e / 2,
        "start_pref_fr": pref_fr,
        "limit_pref_fr": pref_fr / 2,
        "start_pref_fm": pref_fm,
        "limit_pref_fm": pref_fm / 2,
        "start_pref_v": pref_v,
        "limit_pref_v": pref_v / 2,
        "start_pref_ae": pref_ae,
        "limit_pref_ae": pref_ae / 2,
    }
    return input_dict


LossParamEnergySpinList = parameterize_func(
    LossParamEnergySpin,
    OrderedDict(
        {
            "pref_e": (1.0, 0.0),
            "pref_fr": (1.0, 0.0),
            "pref_fm": (1.0, 0.0),
            "pref_v": (1.0, 0.0),
            "pref_ae": (1.0, 0.0),
        }
    ),
)
# to get name for the default function
LossParamEnergySpin = LossParamEnergySpinList[0]


def LossParamDos(
    starter_learning_rate=1.0,
    pref_dos=1.0,
    pref_ados=1.0,
):
    key_to_pref_map = {
        "dos": pref_dos,
        "atom_dos": pref_ados,
    }
    input_dict = {
        "key_to_pref_map": key_to_pref_map,
        "starter_learning_rate": starter_learning_rate,
        "numb_dos": 2,
        "start_pref_dos": pref_dos,
        "limit_pref_dos": pref_dos / 2,
        "start_pref_ados": pref_ados,
        "limit_pref_ados": pref_ados / 2,
        "start_pref_cdf": 0.0,
        "limit_pref_cdf": 0.0,
        "start_pref_acdf": 0.0,
        "limit_pref_acdf": 0.0,
    }
    return input_dict


LossParamDosList = parameterize_func(
    LossParamDos,
    OrderedDict(
        {
            "pref_dos": (1.0,),
            "pref_ados": (1.0, 0.0),
        }
    ),
) + parameterize_func(
    LossParamDos,
    OrderedDict(
        {
            "pref_dos": (0.0,),
            "pref_ados": (1.0,),
        }
    ),
)

# to get name for the default function
LossParamDos = LossParamDosList[0]


def LossParamTensor(
    pref=1.0,
    pref_atomic=1.0,
):
    tensor_name = "test_tensor"
    key_to_pref_map = {
        tensor_name: pref,
        f"atomic_{tensor_name}": pref_atomic,
    }
    input_dict = {
        "key_to_pref_map": key_to_pref_map,
        "tensor_name": tensor_name,
        "tensor_size": 2,
        "label_name": tensor_name,
        "pref": pref,
        "pref_atomic": pref_atomic,
    }
    return input_dict


LossParamTensorList = parameterize_func(
    LossParamTensor,
    OrderedDict(
        {
            "pref": (1.0,),
            "pref_atomic": (1.0, 0.0),
        }
    ),
) + parameterize_func(
    LossParamTensor,
    OrderedDict(
        {
            "pref": (0.0,),
            "pref_atomic": (1.0,),
        }
    ),
)
# to get name for the default function
LossParamTensor = LossParamTensorList[0]


def LossParamProperty():
    key_to_pref_map = {
        "foo": 1.0,
    }
    input_dict = {
        "key_to_pref_map": key_to_pref_map,
        "var_name": "foo",
        "out_bias": [0.1, 0.5, 1.2, -0.1, -10],
        "out_std": [8, 10, 0.001, -0.2, -10],
        "task_dim": 5,
    }
    return input_dict


LossParamPropertyList = [LossParamProperty]
# to get name for the default function
LossParamProperty = LossParamPropertyList[0]
