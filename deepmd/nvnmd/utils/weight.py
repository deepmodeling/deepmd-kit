
import numpy as np

from deepmd.env import tf
from deepmd.nvnmd.utils.fio import FioHead


def get_weight(weights, key):
    if key in weights.keys():
        return weights[key]
    else:
        head = FioHead().warning()
        print(f"{head}: There is not {key} in weights.")
        return None


def get_normalize(weights: dict):
    key = f"descrpt_attr.t_avg"
    avg = get_weight(weights, key)
    key = f"descrpt_attr.t_std"
    std = get_weight(weights, key)
    return avg, std


def get_rng_s(weights: dict):
    avg, std = get_normalize(weights)
    smin = np.min(-avg[:, 0] / std[:, 0])
    smax = np.max(2.0 / std[:, 0])
    return smin, smax


def get_filter_weight(weights: dict, spe_i: int, spe_j: int, layer_l: int):
    """:
    spe_i(int): 0~ntype-1
    spe_j(int): 0~ntype-1
    layer_l: 1~nlayer
    """
    # key = f"filter_type_{spe_i}.matrix_{layer_l}_{spe_j}" # type_one_side = false
    key = f"filter_type_all.matrix_{layer_l}_{spe_j}"  # type_one_side = true
    weight = get_weight(weights, key)
    # key = f"filter_type_{spe_i}.bias_{layer_l}_{spe_j}" # type_one_side = false
    key = f"filter_type_all.bias_{layer_l}_{spe_j}"  # type_one_side = true
    bias = get_weight(weights, key)
    return weight, bias


def get_fitnet_weight(weights: dict, spe_i: int, layer_l: int, nlayer: int = 10):
    """:
    spe_i(int): 0~ntype-1
    layer_l(int): 0~nlayer-1
    """
    if layer_l == nlayer - 1:
        key = f"final_layer_type_{spe_i}.matrix"
        weight = get_weight(weights, key)
        key = f"final_layer_type_{spe_i}.bias"
        bias = get_weight(weights, key)
    else:
        key = f"layer_{layer_l}_type_{spe_i}.matrix"
        weight = get_weight(weights, key)
        key = f"layer_{layer_l}_type_{spe_i}.bias"
        bias = get_weight(weights, key)

    return weight, bias


def get_constant_initializer(weights, name):
    scope = tf.get_variable_scope().name
    name = scope + '.' + name
    value = get_weight(weights, name)
    return tf.constant_initializer(value)
