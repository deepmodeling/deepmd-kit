# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compress a pt_expt model (.pte) by tabulating embedding nets."""

import logging
from typing import (
    Any,
)

from deepmd.pt_expt.utils.serialization import (
    deserialize_to_file,
    serialize_from_file,
)

log = logging.getLogger(__name__)


def _read_saved_min_nbor_dist(model: Any, model_dict: dict) -> tuple[float | None, str]:
    """Read the stored minimal neighbor distance and where it was read from.

    ``@variables`` is the cross-backend location of this value: it is written
    by :mod:`deepmd.pt.utils.serialization` and read back by the PyTorch and
    Paddle backends, so a ``.pt2`` produced by ``dp convert-backend`` carries
    the value there rather than inside the serialized model dict.

    Returns
    -------
    float or None
        The stored minimal neighbor distance, None if the model has none.
    str
        Human-readable description of where the value was read from.
    """
    min_nbor_dist = model.get_min_nbor_dist()
    if min_nbor_dist is not None:
        return float(min_nbor_dist), "the model"
    min_nbor_dist = model_dict.get("min_nbor_dist")
    if min_nbor_dist is not None:
        return float(min_nbor_dist), "the model file"
    min_nbor_dist = (model_dict.get("@variables") or {}).get("min_nbor_dist")
    if min_nbor_dist is not None:
        return float(min_nbor_dist), "the model file (@variables)"
    return None, ""


def enable_compression(
    input_file: str,
    output: str,
    stride: float = 0.01,
    extrapolate: int = 5,
    check_frequency: int = -1,
    training_script: str | None = None,
    recompute_min_nbor_dist: bool = False,
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
    recompute_min_nbor_dist : bool
        Ignore the min_nbor_dist stored in the model and recompute it from
        the training data. Requires training_script.
    """
    from deepmd.pt_expt.model.model import (
        BaseModel,
    )

    # 1. Load the .pte model
    model_dict = serialize_from_file(input_file)
    model = BaseModel.deserialize(model_dict["model"])

    # 2. Get or compute min_nbor_dist
    if recompute_min_nbor_dist:
        min_nbor_dist, source = None, ""
    else:
        min_nbor_dist, source = _read_saved_min_nbor_dist(model, model_dict)
    if min_nbor_dist is not None:
        log.info(f"Minimal neighbor distance read from {source}: {min_nbor_dist:f}")
    else:
        if recompute_min_nbor_dist:
            log.info(
                "Recompute the minimal neighbor distance from the training data, "
                "ignoring the one saved in the model."
            )
            if training_script is None:
                raise ValueError(
                    "Recomputing the minimal neighbor distance requires the "
                    "training script and data (via -t,--training-script)."
                )
        else:
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
    from deepmd.pt_expt.model.graph_lower import (
        model_uses_graph_lower,
    )

    model_def_script = model_dict.get("model_def_script")
    if output.endswith(".pt2") and model_uses_graph_lower(model):
        log.info("Re-exporting compressed graph...")
        deserialize_to_file(
            output,
            {"model": compressed_model_dict, "model_def_script": model_def_script},
            lower_kind="auto",
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
