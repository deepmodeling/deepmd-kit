# SPDX-License-Identifier: LGPL-3.0-or-later
"""Import released Uni-Mol v1 checkpoints into deepmd.

The released files (``mol_pre_all_h_220816.pt`` and ``mol_pre_no_h_220816.pt``,
MIT licensed, from the Uni-Mol repository and its Hugging Face mirror) hold a
single ``model`` entry with 211 float32 tensors and no training state, so they
load with ``weights_only=True``.

The parameter names line up one to one with the ported modules, but the arrays
do not: deepmd stores a linear weight as ``(num_in, num_out)`` and applies it as
``x @ w``, which is the transpose of ``torch.nn.Linear.weight``, and it calls
layer-norm parameters ``w``/``b`` rather than ``weight``/``bias``. Every weight
is therefore renamed and transposed here rather than loaded directly.
"""

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

if TYPE_CHECKING:
    from deepmd.dpmodel.descriptor.unimol import (
        DescrptUniMol,
    )

__all__ = [
    "UNIMOL_V1_BASE_ARCHITECTURE",
    "descriptor_from_unimol_checkpoint",
    "load_unimol_state_dict",
    "split_unimol_state_dict",
]

# Uni-Mol's ``unimol_base`` architecture (unimol/models/unimol.py:423-442).
UNIMOL_V1_BASE_ARCHITECTURE = {
    "encoder_layers": 15,
    "encoder_embed_dim": 512,
    "encoder_ffn_embed_dim": 2048,
    "encoder_attention_heads": 64,
    "activation_function": "gelu_erf",
    "dropout": 0.1,
    "emb_dropout": 0.1,
    "attention_dropout": 0.1,
    "activation_dropout": 0.0,
    "max_seq_len": 512,
}

_BACKBONE_PREFIXES = ("embed_tokens", "gbf.", "gbf_proj.", "encoder.")
_HEAD_PREFIXES = ("lm_head.", "pair2coord_proj.", "dist_head.")


def load_unimol_state_dict(path: str) -> dict[str, np.ndarray]:
    """Read a released checkpoint into NumPy arrays.

    Parameters
    ----------
    path : str
        Local path to the ``.pt`` file. Nothing is downloaded here; see
        :func:`descriptor_from_unimol_checkpoint` for the optional download.

    Returns
    -------
    dict[str, np.ndarray]
        The ``model`` entry of the checkpoint.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "reading a Uni-Mol checkpoint needs PyTorch; install it with "
            "`pip install torch`"
        ) from e
    obj = torch.load(path, map_location="cpu", weights_only=True)
    state = obj["model"] if "model" in obj else obj
    return {k: v.numpy() for k, v in state.items()}


def split_unimol_state_dict(
    state_dict: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split a checkpoint into backbone and pretraining-head parameters.

    Returns
    -------
    tuple
        ``(backbone, heads)``. The heads are the element, coordinate and
        distance heads, which belong to the fitting rather than the descriptor.
    """
    backbone = {k: v for k, v in state_dict.items() if k.startswith(_BACKBONE_PREFIXES)}
    heads = {k: v for k, v in state_dict.items() if k.startswith(_HEAD_PREFIXES)}
    unknown = set(state_dict) - set(backbone) - set(heads)
    if unknown:
        raise ValueError(f"unexpected parameters in the checkpoint: {sorted(unknown)}")
    return backbone, heads


def _set_linear(layer: Any, state: dict[str, np.ndarray], prefix: str) -> None:
    """Assign a torch Linear onto a deepmd layer, transposing the weight."""
    layer.w = np.ascontiguousarray(state[prefix + ".weight"].T).astype(layer.w.dtype)
    bias = state.get(prefix + ".bias")
    if bias is not None and layer.b is not None:
        layer.b = bias.astype(layer.b.dtype)


def _set_layer_norm(layer: Any, state: dict[str, np.ndarray], prefix: str) -> None:
    """Assign a torch LayerNorm onto a deepmd layer norm."""
    layer.w = state[prefix + ".weight"].astype(layer.w.dtype)
    layer.b = state[prefix + ".bias"].astype(layer.b.dtype)


def apply_unimol_backbone(
    descriptor: "DescrptUniMol", state_dict: dict[str, np.ndarray]
) -> None:
    """Load backbone parameters into a descriptor, in place."""
    dtype = descriptor.embed_tokens.dtype
    descriptor.embed_tokens = state_dict["embed_tokens.weight"].astype(dtype)
    descriptor.gbf.means = state_dict["gbf.means.weight"].astype(dtype)
    descriptor.gbf.stds = state_dict["gbf.stds.weight"].astype(dtype)
    descriptor.gbf.mul = state_dict["gbf.mul.weight"].astype(dtype)
    descriptor.gbf.bias = state_dict["gbf.bias.weight"].astype(dtype)
    _set_linear(descriptor.gbf_proj.linear1, state_dict, "gbf_proj.linear1")
    _set_linear(descriptor.gbf_proj.linear2, state_dict, "gbf_proj.linear2")

    encoder = descriptor.encoder
    _set_layer_norm(encoder.emb_layer_norm, state_dict, "encoder.emb_layer_norm")
    _set_layer_norm(encoder.final_layer_norm, state_dict, "encoder.final_layer_norm")
    if encoder.final_head_layer_norm is not None:
        _set_layer_norm(
            encoder.final_head_layer_norm, state_dict, "encoder.final_head_layer_norm"
        )
    for i, layer in enumerate(encoder.layers):
        prefix = f"encoder.layers.{i}."
        _set_linear(layer.self_attn.in_proj, state_dict, prefix + "self_attn.in_proj")
        _set_linear(layer.self_attn.out_proj, state_dict, prefix + "self_attn.out_proj")
        _set_layer_norm(
            layer.self_attn_layer_norm, state_dict, prefix + "self_attn_layer_norm"
        )
        _set_linear(layer.fc1, state_dict, prefix + "fc1")
        _set_linear(layer.fc2, state_dict, prefix + "fc2")
        _set_layer_norm(layer.final_layer_norm, state_dict, prefix + "final_layer_norm")


def descriptor_from_unimol_checkpoint(
    path: str,
    type_map: list[str] | None = None,
    precision: str = "float64",
    **overrides: Any,
) -> "DescrptUniMol":
    """Build a descriptor carrying the released Uni-Mol v1 weights.

    Parameters
    ----------
    path : str
        Local path to the checkpoint.
    type_map : list[str], optional
        Element names for the model. Defaults to Uni-Mol's own 26 elements.
    precision : str
        Precision to hold the parameters in.
    **overrides
        Architecture overrides on top of ``UNIMOL_V1_BASE_ARCHITECTURE``.

    Returns
    -------
    DescrptUniMol
        A descriptor whose backbone reproduces upstream.
    """
    from deepmd.dpmodel.descriptor.unimol import (
        UNIMOL_ELEMENTS,
        DescrptUniMol,
    )

    state_dict = load_unimol_state_dict(path)
    backbone, _ = split_unimol_state_dict(state_dict)
    arch = {**UNIMOL_V1_BASE_ARCHITECTURE, **overrides}
    descriptor = DescrptUniMol(
        type_map=type_map if type_map is not None else list(UNIMOL_ELEMENTS),
        precision=precision,
        **arch,
    )
    apply_unimol_backbone(descriptor, backbone)
    return descriptor
