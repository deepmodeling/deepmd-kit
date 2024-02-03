# SPDX-License-Identifier: LGPL-3.0-or-later
"""Submodule containing all the implemented potentials."""

from pathlib import (
    Path,
)
from typing import (
    Optional,
    Union,
)

from .data_modifier import (
    DipoleChargeModifier,
)
from .deep_dipole import (
    DeepDipole,
)
from .deep_dos import (
    DeepDOS,
)
from .deep_eval import (
    DeepEval,
)
from .deep_polar import (
    DeepGlobalPolar,
    DeepPolar,
)
from .deep_pot import (
    DeepPot,
)
from .deep_wfc import (
    DeepWFC,
)
from .ewald_recp import (
    EwaldRecp,
)
from .model_devi import (
    calc_model_devi,
)

__all__ = [
    "DeepPotential",
    "DeepDipole",
    "DeepEval",
    "DeepGlobalPolar",
    "DeepPolar",
    "DeepPot",
    "DeepDOS",
    "DeepWFC",
    "DipoleChargeModifier",
    "EwaldRecp",
    "calc_model_devi",
]


def DeepPotential(
    model_file: Union[str, Path],
    load_prefix: str = "load",
    default_tf_graph: bool = False,
    input_map: Optional[dict] = None,
    neighbor_list=None,
) -> Union[DeepDipole, DeepGlobalPolar, DeepPolar, DeepPot, DeepDOS, DeepWFC]:
    """Factory function that will inialize appropriate potential read from `model_file`.

    Parameters
    ----------
    model_file : str
        The name of the frozen model file.
    load_prefix : str
        The prefix in the load computational graph
    default_tf_graph : bool
        If uses the default tf graph, otherwise build a new tf graph for evaluation
    input_map : dict, optional
        The input map for tf.import_graph_def. Only work with default tf graph
    neighbor_list : ase.neighborlist.NeighborList, optional
        The neighbor list object. If None, then build the native neighbor list.

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
        mf,
        load_prefix=load_prefix,
        default_tf_graph=default_tf_graph,
        input_map=input_map,
    ).model_type

    if model_type == "ener":
        dp = DeepPot(
            mf,
            load_prefix=load_prefix,
            default_tf_graph=default_tf_graph,
            input_map=input_map,
            neighbor_list=neighbor_list,
        )
    elif model_type == "dos":
        dp = DeepDOS(
            mf,
            load_prefix=load_prefix,
            default_tf_graph=default_tf_graph,
            input_map=input_map,
        )
    elif model_type == "dipole":
        dp = DeepDipole(
            mf,
            load_prefix=load_prefix,
            default_tf_graph=default_tf_graph,
            input_map=input_map,
            neighbor_list=neighbor_list,
        )
    elif model_type == "polar":
        dp = DeepPolar(
            mf,
            load_prefix=load_prefix,
            default_tf_graph=default_tf_graph,
            input_map=input_map,
            neighbor_list=neighbor_list,
        )
    elif model_type == "global_polar":
        dp = DeepGlobalPolar(
            mf,
            load_prefix=load_prefix,
            default_tf_graph=default_tf_graph,
            input_map=input_map,
            neighbor_list=neighbor_list,
        )
    elif model_type == "wfc":
        dp = DeepWFC(
            mf,
            load_prefix=load_prefix,
            default_tf_graph=default_tf_graph,
            input_map=input_map,
        )
    else:
        raise RuntimeError(f"unknown model type {model_type}")

    return dp
