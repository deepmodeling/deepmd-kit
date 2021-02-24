"""Submodule containing all the implemented potentials."""

from pathlib import Path
from typing import Union

from .data_modifier import DipoleChargeModifier
from .deep_dipole import DeepDipole
from .deep_eval import DeepEval
from .deep_polar import DeepGlobalPolar, DeepPolar
from .deep_pot import DeepPot
from .deep_wfc import DeepWFC
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
    model_file: Union[str, Path],
    load_prefix: str = "load",
    default_tf_graph: bool = False,
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
    mf = Path(model_file)

    model_type = DeepEval(
        mf, load_prefix=load_prefix, default_tf_graph=default_tf_graph
    ).model_type

    if model_type == "ener":
        dp = DeepPot(mf, load_prefix=load_prefix, default_tf_graph=default_tf_graph)
    elif model_type == "dipole":
        dp = DeepDipole(mf, load_prefix=load_prefix, default_tf_graph=default_tf_graph)
    elif model_type == "polar":
        dp = DeepPolar(mf, load_prefix=load_prefix, default_tf_graph=default_tf_graph)
    elif model_type == "global_polar":
        dp = DeepGlobalPolar(
            mf, load_prefix=load_prefix, default_tf_graph=default_tf_graph
        )
    elif model_type == "wfc":
        dp = DeepWFC(mf, load_prefix=load_prefix, default_tf_graph=default_tf_graph)
    else:
        raise RuntimeError(f"unknow model type {model_type}")

    return dp
