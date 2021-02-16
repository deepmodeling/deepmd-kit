"""Submodule containing all the implemented potentials."""

from typing import Union
from .deep_dipole import DeepDipole
from .deep_eval import DeepEval
from .deep_polar import DeepGlobalPolar, DeepPolar
from .deep_pot import DeepPot
from .deep_wfc import DeepWFC
from .data_modifier import DipoleChargeModifier
from .ewald_recp import EwaldRecp

__all__ = [
    "DeepPotential",
    "DeepDipole",
    "DeepEval",
    "DeepGlobalPolar",
    "DeepPolar",
    "DeepPot",
    "DeepWFC",
    "DipoleChargeModifier",
    "EwaldRecp",
]


def DeepPotential(
    model_file: str, load_prefix: str = "load", default_tf_graph: bool = False
) -> Union[DeepDipole, DeepGlobalPolar, DeepPolar, DeepPot, DeepWFC]:
    """Factory function that will inialize appropriate potential read from `model_file`.

    Parameters
    ----------
    model_file: str
        The name of the frozen model file.
    load_prefix: str
        The prefix in the load computational graph
    default_tf_graph : bool
        If uses the default tf graph, otherwise build a new tf graph for evaluation

    Returns
    -------
    Union[DeepDipole, DeepGlobalPolar, DeepPolar, DeepPot, DeepWFC]
        one of the available potentials

    Raises
    ------
    RuntimeError
        if model file does not correspond to any implementd potential
    """
    model_type = DeepEval(
        model_file, load_prefix=load_prefix, default_tf_graph=default_tf_graph
    ).model_type

    if model_type == "ener":
        dp = DeepPot(model_file, prefix=load_prefix, default_tf_graph=default_tf_graph)
    elif model_type == "dipole":
        dp = DeepDipole(
            model_file, prefix=load_prefix, default_tf_graph=default_tf_graph
        )
    elif model_type == "polar":
        dp = DeepPolar(
            model_file, prefix=load_prefix, default_tf_graph=default_tf_graph
        )
    elif model_type == "global_polar":
        dp = DeepGlobalPolar(
            model_file, prefix=load_prefix, default_tf_graph=default_tf_graph
        )
    elif model_type == "wfc":
        dp = DeepWFC(model_file, prefix=load_prefix, default_tf_graph=default_tf_graph)
    else:
        raise RuntimeError(f"unknow model type {model_type}")

    return dp
