# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Public building blocks for the DPA4/SeZM descriptor.

This package re-exports the helper functions, embeddings, equivariant layers,
and quaternion-based Wigner-D utilities used by the DPA4/SeZM descriptor and model.

This package is the dpmodel (array-API) port of
``deepmd.pt.model.descriptor.sezm_nn``.
"""

from .activation import (
    GatedActivation,
    SwiGLU,
)
from .attention import (
    segment_envelope_gated_softmax,
)
from .attn_res import (
    DepthAttnRes,
)
from .block import (
    SeZMInteractionBlock,
)
from .cartesian import (
    EdgeCartesianTensorProduct,
    NodeCartesianTensorProduct,
    build_cartesian_basis,
    build_edge_cartesian_tensors,
)
from .edge_cache import (
    EdgeCache,
    build_edge_type_feat,
    compute_edge_src_gate,
    edge_cache_to_dtype,
)
from .embedding import (
    ChargeSpinEmbedding,
    EnvironmentInitialEmbedding,
    GeometricInitialEmbedding,
    SeZMTypeEmbedding,
    SpinEmbedding,
)
from .ffn import (
    EquivariantFFN,
)
from .grid_net import (
    BaseGridNet,
    GridBranch,
    GridMLP,
    S2GridNet,
    SO3GridNet,
)
from .indexing import (
    build_gie_zonal_index,
    build_l_major_index,
    build_m_major_index,
    build_m_major_l_index,
    build_rotate_inv_rescale,
    get_so3_dim_of_lmax,
    map_degree_idx,
    project_D_to_m,
    project_Dt_from_m,
    so3_packed_index,
)
from .lora import (
    LoRASO2,
    LoRASO3,
    apply_lora_to_sezm,
    build_merged_state_dict,
    fold_lora_state_dict_keys,
    has_lora,
    merge_lora_into_base,
    strip_lora_from_extra_state,
)
from .norm import (
    EquivariantRMSNorm,
    ReducedEquivariantRMSNorm,
    RMSNorm,
    ScalarRMSNorm,
)
from .projection import (
    BaseGridProjector,
    S2GridProjector,
    SO3GridProjector,
    resolve_s2_grid_resolution,
    resolve_so3_grid,
)
from .radial import (
    BridgingSwitch,
    C3CutoffEnvelope,
    InnerClamp,
    RadialBasis,
    RadialMLP,
)
from .so2 import (
    DynamicRadialDegreeMixer,
    SO2Convolution,
    SO2Linear,
)
from .so3 import (
    ChannelLinear,
    FocusLinear,
    SO3Linear,
)
from .utils import (
    ATTN_RES_MODES,
    get_promoted_dtype,
    init_trunc_normal_fan_in_out,
    safe_norm,
)
from .wignerd import (
    WignerDCalculator,
    build_edge_quaternion,
    quaternion_multiply,
    quaternion_nlerp,
    quaternion_normalize,
    quaternion_to_rotation_matrix,
    quaternion_z_rotation,
)

__all__ = [
    "ATTN_RES_MODES",
    "BaseGridNet",
    "BaseGridProjector",
    "BridgingSwitch",
    "C3CutoffEnvelope",
    "ChannelLinear",
    "ChargeSpinEmbedding",
    "DepthAttnRes",
    "DynamicRadialDegreeMixer",
    "EdgeCache",
    "EdgeCartesianTensorProduct",
    "EnvironmentInitialEmbedding",
    "EquivariantFFN",
    "EquivariantRMSNorm",
    "FocusLinear",
    "GatedActivation",
    "GeometricInitialEmbedding",
    "GridBranch",
    "GridMLP",
    "InnerClamp",
    "LoRASO2",
    "LoRASO3",
    "NodeCartesianTensorProduct",
    "RMSNorm",
    "RadialBasis",
    "RadialMLP",
    "ReducedEquivariantRMSNorm",
    "S2GridNet",
    "S2GridProjector",
    "SO2Convolution",
    "SO2Linear",
    "SO3GridNet",
    "SO3GridProjector",
    "SO3Linear",
    "ScalarRMSNorm",
    "SeZMInteractionBlock",
    "SeZMTypeEmbedding",
    "SpinEmbedding",
    "SwiGLU",
    "WignerDCalculator",
    "apply_lora_to_sezm",
    "build_cartesian_basis",
    "build_edge_cartesian_tensors",
    "build_edge_quaternion",
    "build_edge_type_feat",
    "build_gie_zonal_index",
    "build_l_major_index",
    "build_m_major_index",
    "build_m_major_l_index",
    "build_merged_state_dict",
    "build_rotate_inv_rescale",
    "compute_edge_src_gate",
    "edge_cache_to_dtype",
    "fold_lora_state_dict_keys",
    "get_promoted_dtype",
    "get_so3_dim_of_lmax",
    "has_lora",
    "init_trunc_normal_fan_in_out",
    "map_degree_idx",
    "merge_lora_into_base",
    "project_D_to_m",
    "project_Dt_from_m",
    "quaternion_multiply",
    "quaternion_nlerp",
    "quaternion_normalize",
    "quaternion_to_rotation_matrix",
    "quaternion_z_rotation",
    "resolve_s2_grid_resolution",
    "resolve_so3_grid",
    "safe_norm",
    "segment_envelope_gated_softmax",
    "so3_packed_index",
    "strip_lora_from_extra_state",
]
