# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import logging

import torch

from deepmd.common import (
    j_loader,
)
from deepmd.pt.cxx_op import (
    ENABLE_CUSTOMIZED_OP,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.data_system import (
    get_data,
)

log = logging.getLogger(__name__)


CUSTOMIZED_OP_COMPRESSION_ERROR = (
    "The PyTorch customized OP library (libdeepmd_op_pt) is not loaded. "
    "`dp --pt compress` needs these custom tabulation OPs to create a "
    "compressed model that can run during inference. Please install a "
    "deepmd-kit package with PyTorch customized OP support, or make sure "
    "libdeepmd_op_pt is available in the installed deepmd/lib directory."
)


def assert_customized_op_available_for_compression() -> None:
    """Fail early when PyTorch model compression cannot produce a usable model.

    Compression stores tabulated descriptor data in the scripted model.  The
    corresponding compressed forward paths call custom ``torch.ops.deepmd``
    kernels such as ``tabulate_fusion_se_a`` at inference time.  When
    ``libdeepmd_op_pt`` is missing, descriptor modules install Python fallback
    stubs only so the model can still be scripted; those stubs raise
    ``NotImplementedError`` later and make the saved artifact unusable.  Raising
    here keeps ``dp --pt compress`` from silently exporting such a broken model.
    """
    if not ENABLE_CUSTOMIZED_OP:
        raise RuntimeError(CUSTOMIZED_OP_COMPRESSION_ERROR)


def enable_compression(
    input_file: str,
    output: str,
    stride: float = 0.01,
    extrapolate: int = 5,
    check_frequency: int = -1,
    training_script: str | None = None,
) -> None:
    assert_customized_op_available_for_compression()

    saved_model = torch.jit.load(input_file, map_location="cpu")
    model_def_script = json.loads(saved_model.model_def_script)
    model = get_model(model_def_script)
    model.load_state_dict(saved_model.state_dict())

    if model.get_min_nbor_dist() is None:
        log.info(
            "Minimal neighbor distance is not saved in the model, compute it from the training data."
        )
        if training_script is None:
            raise ValueError(
                "The model does not have a minimum neighbor distance, "
                "so the training script and data must be provided "
                "(via -t,--training-script)."
            )

        jdata = j_loader(training_script)
        jdata = update_deepmd_input(jdata)

        type_map = jdata["model"].get("type_map", None)
        train_data = get_data(
            jdata["training"]["training_data"],
            0,  # not used
            type_map,
            None,
        )
        update_sel = UpdateSel()
        t_min_nbor_dist = update_sel.get_min_nbor_dist(
            train_data,
        )
        model.min_nbor_dist = torch.tensor(
            t_min_nbor_dist,
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )

    model.enable_compression(
        extrapolate,
        stride,
        stride * 10,
        check_frequency,
    )

    model = torch.jit.script(model)
    torch.jit.save(model, output)
