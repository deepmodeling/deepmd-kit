"""Module providing compatibility between `0.x.x` and `1.x.x` input versions."""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from deepmd.common import j_must_have


def convert_input_v0_v1(
    jdata: Dict[str, Any], warning: bool = True, dump: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Convert input from v0 format to v1.

    Parameters
    ----------
    jdata : Dict[str, Any]
        loaded json/yaml file
    warning : bool, optional
        whether to show deprecation warning, by default True
    dump : Optional[Union[str, Path]], optional
        whether to dump converted file, by default None

    Returns
    -------
    Dict[str, Any]
        converted output
    """
    output = {}
    if "with_distrib" in jdata:
        output["with_distrib"] = jdata["with_distrib"]
    output["model"] = _model(jdata, jdata["use_smooth"])
    output["learning_rate"] = _learning_rate(jdata)
    output["loss"] = _loss(jdata)
    output["training"] = _training(jdata)
    if warning:
        _warnning_input_v0_v1(dump)
    if dump is not None:
        with open(dump, "w") as fp:
            json.dump(output, fp, indent=4)
    return output


def _warnning_input_v0_v1(fname: Optional[Union[str, Path]]):
    msg = (
        "It seems that you are using a deepmd-kit input of version 0.x.x, "
        "which is deprecated. we have converted the input to >1.0.0 compatible"
    )
    if fname is not None:
        msg += f", and output it to file {fname}"
    warnings.warn(msg)


def _model(jdata: Dict[str, Any], smooth: bool) -> Dict[str, Dict[str, Any]]:
    """Convert data to v1 input for non-smooth model.

    Parameters
    ----------
    jdata : Dict[str, Any]
        parsed input json/yaml data
    smooth : bool
        whether to use smooth or non-smooth descriptor version

    Returns
    -------
    Dict[str, Dict[str, Any]]
        dictionary with model input parameters and sub-dictionaries for descriptor and
        fitting net
    """
    model = {}
    model["descriptor"] = (
        _smth_descriptor(jdata) if smooth else _nonsmth_descriptor(jdata)
    )
    model["fitting_net"] = _fitting_net(jdata)
    return model


def _nonsmth_descriptor(jdata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert data to v1 input for non-smooth descriptor.

    Parameters
    ----------
    jdata : Dict[str, Any]
        parsed input json/yaml data

    Returns
    -------
    Dict[str, Any]
        dict with descriptor parameters
    """
    descriptor = {}
    descriptor["type"] = "loc_frame"
    _jcopy(jdata, descriptor, ("sel_a", "sel_r", "rcut", "axis_rule"))
    return descriptor


def _smth_descriptor(jdata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert data to v1 input for smooth descriptor.

    Parameters
    ----------
    jdata : Dict[str, Any]
        parsed input json/yaml data

    Returns
    -------
    Dict[str, Any]
        dict with descriptor parameters
    """
    descriptor = {}
    seed = jdata.get("seed", None)
    if seed is not None:
        descriptor["seed"] = seed
    descriptor["type"] = "se_a"
    descriptor["sel"] = jdata["sel_a"]
    _jcopy(jdata, descriptor, ("rcut", ))
    descriptor["rcut_smth"] = jdata.get("rcut_smth", descriptor["rcut"])
    descriptor["neuron"] = j_must_have(jdata, "filter_neuron")
    descriptor["axis_neuron"] = j_must_have(jdata, "axis_neuron", ["n_axis_neuron"])
    descriptor["resnet_dt"] = False
    if "resnet_dt" in jdata:
        descriptor["resnet_dt"] = jdata["filter_resnet_dt"]

    return descriptor


def _fitting_net(jdata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert data to v1 input for fitting net.

    Parameters
    ----------
    jdata : Dict[str, Any]
        parsed input json/yaml data

    Returns
    -------
    Dict[str, Any]
        dict with fitting net parameters
    """
    fitting_net = {}

    seed = jdata.get("seed", None)
    if seed is not None:
        fitting_net["seed"] = seed
    fitting_net["neuron"] = j_must_have(jdata, "fitting_neuron", ["n_neuron"])
    fitting_net["resnet_dt"] = True
    if "resnet_dt" in jdata:
        fitting_net["resnet_dt"] = jdata["resnet_dt"]
    if "fitting_resnet_dt" in jdata:
        fitting_net["resnet_dt"] = jdata["fitting_resnet_dt"]
    return fitting_net


def _learning_rate(jdata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert data to v1 input for learning rate section.

    Parameters
    ----------
    jdata : Dict[str, Any]
        parsed input json/yaml data

    Returns
    -------
    Dict[str, Any]
        dict with learning rate parameters
    """
    learning_rate = {}
    learning_rate["type"] = "exp"
    _jcopy(jdata, learning_rate, ("decay_steps", "decay_rate", "start_lr"))
    return learning_rate


def _loss(jdata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert data to v1 input for loss function.

    Parameters
    ----------
    jdata : Dict[str, Any]
        parsed input json/yaml data

    Returns
    -------
    Dict[str, Any]
        dict with loss function parameters
    """
    loss: Dict[str, Any] = {}
    _jcopy(
        jdata,
        loss,
        (
            "start_pref_e",
            "limit_pref_e",
            "start_pref_f",
            "limit_pref_f",
            "start_pref_v",
            "limit_pref_v",
        ),
    )
    if "start_pref_ae" in jdata:
        loss["start_pref_ae"] = jdata["start_pref_ae"]
    if "limit_pref_ae" in jdata:
        loss["limit_pref_ae"] = jdata["limit_pref_ae"]
    return loss


def _training(jdata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert data to v1 input for training.

    Parameters
    ----------
    jdata : Dict[str, Any]
        parsed input json/yaml data

    Returns
    -------
    Dict[str, Any]
        dict with training parameters
    """
    training = {}
    seed = jdata.get("seed", None)
    if seed is not None:
        training["seed"] = seed

    _jcopy(jdata, training, ("systems", "set_prefix", "stop_batch", "batch_size"))
    training["disp_file"] = "lcurve.out"
    if "disp_file" in jdata:
        training["disp_file"] = jdata["disp_file"]
    training["disp_freq"] = j_must_have(jdata, "disp_freq")
    training["numb_test"] = j_must_have(jdata, "numb_test")
    training["save_freq"] = j_must_have(jdata, "save_freq")
    training["save_ckpt"] = j_must_have(jdata, "save_ckpt")
    training["disp_training"] = j_must_have(jdata, "disp_training")
    training["time_training"] = j_must_have(jdata, "time_training")
    if "profiling" in jdata:
        training["profiling"] = jdata["profiling"]
        if training["profiling"]:
            training["profiling_file"] = j_must_have(jdata, "profiling_file")
    return training


def _jcopy(src: Dict[str, Any], dst: Dict[str, Any], keys: Sequence[str]):
    """Copy specified keys from one dict to another.

    Parameters
    ----------
    src : Dict[str, Any]
        source dictionary
    dst : Dict[str, Any]
        destination dictionary, will be modified in place
    keys : Sequence[str]
        list of keys to copy
    must_have : bool
        ensure that the source dictionary contains the copyyied keys
    """
    for k in keys:
        dst[k] = src[k]
