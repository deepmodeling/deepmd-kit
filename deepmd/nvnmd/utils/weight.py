import logging

from deepmd.env import (
    tf,
)

log = logging.getLogger(__name__)


def get_weight(weights, key):
    r"""Get weight value according to key."""
    if key in weights.keys():
        return weights[key]
    else:
        log.warning(f"There is not {key} in weights.")
        return None


def get_normalize(weights: dict):
    r"""Get normalize parameter (avg and std) of :math:`s_{ji}`."""
    key = "descrpt_attr.t_avg"
    avg = get_weight(weights, key)
    key = "descrpt_attr.t_std"
    std = get_weight(weights, key)
    return avg, std


def get_filter_weight(weights: dict, spe_i: int, spe_j: int, layer_l: int):
    r"""Get weight and bias of embedding network.

    Parameters
    ----------
    weights : dict
        weights
    spe_i : int
        special order of central atom i
        0~ntype-1
    spe_j : int
        special order of neighbor atom j
        0~ntype-1
    layer_l
        layer order in embedding network
        1~nlayer
    """
    key = f"filter_type_all.matrix_{layer_l}_{spe_j}"  # type_one_side = true
    weight = get_weight(weights, key)
    key = f"filter_type_all.bias_{layer_l}_{spe_j}"  # type_one_side = true
    bias = get_weight(weights, key)
    return weight, bias


def get_fitnet_weight(weights: dict, spe_i: int, layer_l: int, nlayer: int = 10):
    r"""Get weight and bias of fitting network.

    Parameters
    ----------
    weights : dict
        weights
    spe_i : int
        special order of central atom i
        0~ntype-1
    layer_l : int
        layer order in embedding network
        0~nlayer-1
    nlayer : int
        number of layers
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
    r"""Get initial value by name and create a initializer."""
    scope = tf.get_variable_scope().name
    name = scope + "." + name
    value = get_weight(weights, name)
    return tf.constant_initializer(value)
