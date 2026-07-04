# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compress a pt_expt model (.pte) by tabulating embedding nets."""

import logging

from deepmd.pt_expt.utils.serialization import (
    deserialize_to_file,
    serialize_from_file,
)

log = logging.getLogger(__name__)


def enable_compression(
    input_file: str,
    output: str,
    stride: float = 0.01,
    extrapolate: int = 5,
    check_frequency: int = -1,
    training_script: str | None = None,
) -> None:
    """Compress a .pte model by tabulating embedding nets.

    Parameters
    ----------
    input_file : str
        Path to the input .pte model file.
    output : str
        Path to the output compressed .pte model file.
    stride : float
        The uniform stride of the first table.
    extrapolate : int
        The scale of model extrapolation.
    check_frequency : int
        The overflow check frequency.
    training_script : str or None
        Path to training script, used to compute min_nbor_dist if not
        stored in the model.
    """
    from deepmd.pt_expt.model.model import (
        BaseModel,
    )

    # 1. Load the .pte model
    model_dict = serialize_from_file(input_file)
    model = BaseModel.deserialize(model_dict["model"])

    # 2. Get or compute min_nbor_dist
    min_nbor_dist = model.get_min_nbor_dist()
    if min_nbor_dist is None:
        min_nbor_dist = model_dict.get("min_nbor_dist")
    if min_nbor_dist is None:
        log.info(
            "Minimal neighbor distance is not saved in the model, "
            "compute it from the training data."
        )
        if training_script is None:
            raise ValueError(
                "The model does not have a minimum neighbor distance, "
                "so the training script and data must be provided "
                "(via -t,--training-script)."
            )
        from deepmd.common import (
            j_loader,
        )
        from deepmd.pt_expt.utils.update_sel import (
            UpdateSel,
        )
        from deepmd.utils.compat import (
            update_deepmd_input,
        )
        from deepmd.utils.data_system import (
            get_data,
        )

        jdata = j_loader(training_script)
        jdata = update_deepmd_input(jdata)
        type_map = jdata["model"].get("type_map", None)
        train_data = get_data(
            jdata["training"]["training_data"],
            0,
            type_map,
            None,
        )
        update_sel = UpdateSel()
        min_nbor_dist = update_sel.get_min_nbor_dist(train_data)

    model.min_nbor_dist = min_nbor_dist

    # 3. Enable compression (also ensures fake ops are registered now that
    #    the C++ custom op library is loaded via enable_compression imports)
    from deepmd.pt_expt.utils.tabulate_ops import (
        ensure_fake_registered,
    )

    ensure_fake_registered()

    log.info("Enabling compression...")
    model.enable_compression(
        extrapolate,
        stride,
        stride * 10,
        check_frequency,
    )

    # 4. Serialize the compressed model dict (includes tabulated data)
    compressed_model_dict = model.serialize()

    # 5. Re-export the compressed model.
    #
    # A geometrically compressed graph-lower descriptor (DPA1 / se_atten strip,
    # attn_layer == 0) evaluates its tabulated embedding through fused CUDA
    # operators that make_fx can trace, so its compressed graph exports directly
    # to a graph-lower ``.pt2``: ``deserialize_to_file`` bakes the end-to-end
    # fused table operator and the mandatory per-atom virial (see its docstring).
    # Every other compressed descriptor keeps tabulated operators make_fx cannot
    # trace: those export the UNCOMPRESSED graph and carry the compressed dict in
    # ``model.json`` so ``deserialize()`` restores the compression state for the
    # Python inference path.
    from deepmd.pt_expt.train.training import (
        _model_uses_graph_lower,
    )

    model_def_script = model_dict.get("model_def_script")
    if output.endswith(".pt2") and _model_uses_graph_lower(model):
        log.info("Re-exporting compressed graph...")
        deserialize_to_file(
            output,
            {"model": compressed_model_dict, "model_def_script": model_def_script},
            lower_kind="graph",
        )
    else:
        log.info("Re-exporting compressed model...")
        deserialize_to_file(
            output,
            {"model": model_dict["model"], "model_def_script": model_def_script},
            model_json_override={
                "model": compressed_model_dict,
                "model_def_script": model_def_script,
                "min_nbor_dist": float(min_nbor_dist),
            },
        )
    log.info("Compressed model saved to %s", output)
