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


def _read_saved_min_nbor_dist(
    model: "BaseModel", model_dict: dict
) -> tuple[float | None, str]:
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


def _measure_min_nbor_dist(training_script: str) -> float:
    """Measure the minimal neighbor distance from the training data.

    This is a dense all-pairs pass over every frame, so callers run it only
    when the tabulation actually consumes the statistic.

    Returns
    -------
    float
        The shortest neighbor distance in the training data.
    """
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

    jdata = update_deepmd_input(j_loader(training_script))
    train_data = get_data(
        jdata["training"]["training_data"],
        0,
        jdata["model"].get("type_map", None),
        None,
    )
    return UpdateSel().get_min_nbor_dist(train_data)


def enable_compression(
    input_file: str,
    output: str,
    stride: float = 0.01,
    extrapolate: int = 5,
    check_frequency: int = -1,
    training_script: str | None = None,
    recompute_min_nbor_dist: bool = False,
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
    recompute_min_nbor_dist : bool
        Ignore the min_nbor_dist stored in the model and recompute it from
        the training data. Requires training_script. Only applies to models
        whose compression consumes the statistic; it is ignored, with a
        warning, for a model that tabulates over an analytically bounded
        domain.
    """
    from deepmd.pt_expt.model.model import (
        BaseModel,
    )

    model_dict = serialize_from_file(input_file)
    model = BaseModel.deserialize(model_dict["model"])
    saved_min_nbor_dist, source = _read_saved_min_nbor_dist(model, model_dict)
    if saved_min_nbor_dist is not None and not recompute_min_nbor_dist:
        log.info(
            f"Minimal neighbor distance read from {source}: {saved_min_nbor_dist:f}"
        )
    compress_model(
        model,
        output,
        stride=stride,
        extrapolate=extrapolate,
        check_frequency=check_frequency,
        training_script=training_script,
        min_nbor_dist=saved_min_nbor_dist,
        model_def_script=model_dict.get("model_def_script"),
        recompute_min_nbor_dist=recompute_min_nbor_dist,
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
    recompute_min_nbor_dist: bool = False,
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
    recompute_min_nbor_dist
        Ignore any stored distance and measure it from the training data.
        Requires ``training_script``. Only applies to a model whose
        compression consumes the statistic; it is ignored, with a warning,
        for one that tabulates over an analytically bounded domain.
    """
    # === Step 1. Resolve the domain required by the compression tables. ===
    # Branch on whether compression consumes the statistic: measuring it is a
    # dense all-pairs pass over the training data, so a model whose tabulation
    # domain is fixed analytically never scans.
    # A distance on the model wins over one recovered from artifact metadata.
    saved_min_nbor_dist = model.get_min_nbor_dist()
    if saved_min_nbor_dist is None:
        saved_min_nbor_dist = min_nbor_dist

    if model.compression_needs_min_nbor_dist():
        if recompute_min_nbor_dist:
            log.info(
                "Recompute the minimal neighbor distance from the training "
                "data, ignoring the one saved in the model."
            )
            if training_script is None:
                raise ValueError(
                    "Recomputing the minimal neighbor distance requires the "
                    "training script and data (via -t,--training-script)."
                )
            min_nbor_dist = _measure_min_nbor_dist(training_script)
        elif saved_min_nbor_dist is not None:
            min_nbor_dist = saved_min_nbor_dist
        else:
            log.info(
                "Minimal neighbor distance is not saved in the model, "
                "compute it from the training data."
            )
            if training_script is None:
                raise ValueError(
                    "The model does not have a minimum neighbor distance. "
                    "Provide training_script to compression, or freeze without "
                    "--compress and then run 'dp --pt-expt compress -i model.pte "
                    "-o compressed.pte -t INPUT.json' with the training script "
                    "and data."
                )
            min_nbor_dist = _measure_min_nbor_dist(training_script)
    else:
        if recompute_min_nbor_dist:
            log.warning(
                "--recompute-min-nbor-dist does not apply to this model: its "
                "tabulation domain is fixed analytically, so no neighbor "
                "statistics are measured. Ignoring the flag."
            )
        else:
            log.info(
                "The model tabulates over an analytically bounded domain; "
                "skipping the neighbor statistics."
            )
        # Carry a measured distance through untouched; leave it absent when
        # the model has none, rather than inventing one.
        min_nbor_dist = saved_min_nbor_dist

    # ``enable_compression`` reads the value off the model, so a descriptor
    # that ignores it still needs a number there. That placeholder stays local
    # to the call: afterwards the model carries a measured distance or none.
    model.min_nbor_dist = 0.0 if min_nbor_dist is None else min_nbor_dist

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
    model.min_nbor_dist = min_nbor_dist

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

    # Only a measured distance is metadata; a model that never measured one
    # leaves the field out rather than carrying a placeholder.
    measured = {} if min_nbor_dist is None else {"min_nbor_dist": float(min_nbor_dist)}
    if output.endswith(".pt2") and model_uses_graph_lower(model):
        log.info("Re-exporting compressed graph...")
        deserialize_to_file(
            output,
            {
                "model": compressed_model_dict,
                "model_def_script": model_def_script,
                **measured,
            },
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
                **measured,
            },
            with_comm=with_comm,
        )
    log.info("Compressed model saved to %s", output)
