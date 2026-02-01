# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Any,
)

from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.utils.econf_embd import (
    sort_element_type,
)
from deepmd.utils.model_branch_dict import (
    OrderedDictTableWrapper,
    get_model_dict,
)

log = logging.getLogger(__name__)


def show(
    *,
    INPUT: str,
    ATTRIBUTES: list[str],
    **kwargs: Any,
) -> None:
    model = DeepEval(INPUT, head=0)
    model_params = model.get_model_def_script()
    model_is_multi_task = "model_dict" in model_params
    log.info("This is a multitask model") if model_is_multi_task else log.info(
        "This is a singletask model"
    )

    if "model-branch" in ATTRIBUTES:
        #  The model must be multitask mode
        if not model_is_multi_task:
            raise RuntimeError(
                "The 'model-branch' option requires a multitask model."
                " The provided model does not meet this criterion."
            )
        model_branches = list(model_params["model_dict"].keys())
        model_branches += ["RANDOM"]
        _, model_branch_dict = get_model_dict(model_params["model_dict"])
        log.info(
            f"Available model branches are {model_branches}, "
            f"where 'RANDOM' means using a randomly initialized fitting net."
        )
        log.info(
            "Detailed information: \n"
            + OrderedDictTableWrapper(model_branch_dict).as_table()
        )
    if "type-map" in ATTRIBUTES:
        if model_is_multi_task:
            model_branches = list(model_params["model_dict"].keys())
            for branch in model_branches:
                type_map = model_params["model_dict"][branch]["type_map"]
                log.info(f"The type_map of branch {branch} is {type_map}")
        else:
            type_map = model_params["type_map"]
            log.info(f"The type_map is {type_map}")
    if "descriptor" in ATTRIBUTES:
        if model_is_multi_task:
            model_branches = list(model_params["model_dict"].keys())
            for branch in model_branches:
                descriptor = model_params["model_dict"][branch]["descriptor"]
                log.info(f"The descriptor parameter of branch {branch} is {descriptor}")
        else:
            descriptor = model_params["descriptor"]
            log.info(f"The descriptor parameter is {descriptor}")
    if "fitting-net" in ATTRIBUTES:
        if model_is_multi_task:
            model_branches = list(model_params["model_dict"].keys())
            for branch in model_branches:
                fitting_net = model_params["model_dict"][branch]["fitting_net"]
                log.info(
                    f"The fitting_net parameter of branch {branch} is {fitting_net}"
                )
        else:
            fitting_net = model_params["fitting_net"]
            log.info(f"The fitting_net parameter is {fitting_net}")
    if "size" in ATTRIBUTES:
        size_dict = model.get_model_size()
        log_prefix = " for a single branch model" if model_is_multi_task else ""
        log.info(f"Parameter counts{log_prefix}:")
        for k in sorted(size_dict):
            log.info(f"Parameters in {k}: {size_dict[k]:,}")

    if "observed-type" in ATTRIBUTES:
        if model_is_multi_task:
            log.info("The observed types for each branch: ")
            total_observed_types_list = []
            model_branches = list(model_params["model_dict"].keys())
            for branch in model_branches:
                if (
                    model_params["model_dict"][branch]
                    .get("info", {})
                    .get("observed_type", None)
                    is not None
                ):
                    observed_type_list = model_params["model_dict"][branch]["info"][
                        "observed_type"
                    ]
                    observed_types = {
                        "type_num": len(observed_type_list),
                        "observed_type": observed_type_list,
                    }
                else:
                    tmp_model = DeepEval(INPUT, head=branch, no_jit=True)
                    observed_types = tmp_model.get_observed_types()
                log.info(
                    f"{branch}: Number of observed types: {observed_types['type_num']} "
                )
                log.info(
                    f"{branch}: Observed types: {observed_types['observed_type']} "
                )
                total_observed_types_list += [
                    tt
                    for tt in observed_types["observed_type"]
                    if tt not in total_observed_types_list
                ]
            log.info(
                f"TOTAL number of observed types in the model: {len(total_observed_types_list)} "
            )
            log.info(
                f"TOTAL observed types in the model: {sort_element_type(total_observed_types_list)} "
            )
        else:
            log.info("The observed types for this model: ")
            observed_types = model.get_observed_types()
            log.info(f"Number of observed types: {observed_types['type_num']} ")
            log.info(f"Observed types: {observed_types['observed_type']} ")
