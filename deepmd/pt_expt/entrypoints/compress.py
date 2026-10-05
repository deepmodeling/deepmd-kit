# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build and export compressed pt_expt inference models."""

import logging
from typing import (
    TYPE_CHECKING,
)

from deepmd.pt_expt.utils.serialization import (
    deserialize_to_file,
    serialize_from_file,
)

if TYPE_CHECKING:
    from deepmd.pt_expt.model.model import (
        BaseModel,
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
    """Compress a .pte or .pt2 model by tabulating embedding nets.

    Parameters
    ----------
    input_file : str
        Path to the input .pte or .pt2 model file.
    output : str
        Path to the output compressed .pte or .pt2 model file.
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

    model_dict = serialize_from_file(input_file)
    model = BaseModel.deserialize(model_dict["model"])
    compress_model(
        model,
        output,
        stride=stride,
        extrapolate=extrapolate,
        check_frequency=check_frequency,
        training_script=training_script,
        min_nbor_dist=model_dict.get("min_nbor_dist"),
        model_def_script=model_dict.get("model_def_script"),
    )


def compress_model(
    model: "BaseModel",
    output: str,
    *,
    stride: float = 0.01,
    extrapolate: int = 5,
    check_frequency: int = -1,
    training_script: str | None = None,
    min_nbor_dist: float | None = None,
    model_def_script: dict | None = None,
    with_comm: bool = True,
) -> None:
    """Compress a loaded model and export its inference artifact.

    Parameters
    ----------
    model
        Model whose parameters and output statistics are loaded.
    output
        Destination ``.pte`` or ``.pt2`` file.
    stride
        Uniform spacing of the first compression table.
    extrapolate
        Scale of the extrapolation region for models that require it.
    check_frequency
        Frequency of table overflow checks.
    training_script
        Training configuration used when neighbor statistics are required
        but absent from the model and ``min_nbor_dist``.
    min_nbor_dist
        Fallback minimum neighbor distance from artifact metadata.
    model_def_script
        Training model configuration to preserve in the exported artifact.
    with_comm
        Whether to include a communication artifact for cross-rank models.
    """
    # === Step 1. Resolve the domain required by the compression tables. ===
    model_min_nbor_dist = model.get_min_nbor_dist()
    if model_min_nbor_dist is not None:
        min_nbor_dist = model_min_nbor_dist
    if min_nbor_dist is None and not model.compression_needs_min_nbor_dist():
        log.info(
            "The model tabulates over an analytically bounded domain; "
            "skipping the neighbor statistics."
        )
        min_nbor_dist = 0.0
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

    # === Step 2. Fold deployment constants and build the compressed snapshot. ===
    # The neutral vacuum reference must be resolved before a charge-conditioned
    # descriptor specializes its tables to the deployment charge state.
    model.eval()
    model.fold_vacuum_reference()
    uncompressed_model_dict = model.serialize()
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

    compressed_model_dict = model.serialize()

    # === Step 3. Export the graph supported by the compressed model. ===
    #
    # Compressed graph descriptors use fused operators with traceable fake
    # implementations. Their graphs export directly to ``.pt2``; the exporter
    # selects a compact canonical schema when the target supports it and
    # preserves the per-atom virial alongside the total energy and forces.
    # Every other compressed descriptor keeps tabulated operators make_fx cannot
    # trace: those export the UNCOMPRESSED graph and carry the compressed dict in
    # ``model.json`` so ``deserialize()`` restores the compression state for the
    # Python inference path.
    from deepmd.pt_expt.model.graph_lower import (
        model_uses_graph_lower,
    )

    if output.endswith(".pt2") and model_uses_graph_lower(model):
        log.info("Re-exporting compressed graph...")
        deserialize_to_file(
            output,
            {"model": compressed_model_dict, "model_def_script": model_def_script},
            lower_kind="auto",
            with_comm=with_comm,
        )
    else:
        log.info("Re-exporting compressed model...")
        deserialize_to_file(
            output,
            {"model": uncompressed_model_dict, "model_def_script": model_def_script},
            model_json_override={
                "model": compressed_model_dict,
                "model_def_script": model_def_script,
                "min_nbor_dist": float(min_nbor_dist),
            },
            with_comm=with_comm,
        )
    log.info("Compressed model saved to %s", output)
