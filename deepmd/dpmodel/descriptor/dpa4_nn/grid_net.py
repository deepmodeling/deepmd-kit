# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Grid-space nonlinearities for DPA4/SeZM coefficient tensors.

This module is the dpmodel port of
``deepmd.pt.model.descriptor.sezm_nn.grid_net``, restricted to the S2/Lebedev
path used by the core DPA4 configuration. A grid net receives coefficient
tensors, converts them to quadrature values, applies one point-wise grid
operation, and projects the result back to coefficients. The public shapes
are:

* ``mode='self'``: one input ``(N, D, F, 2*C)`` or ``(N, F, D, 2*C)``.
* grid values: ``(N, G, F, C)`` after S2 projection.

Ported names: ``BaseGridNet`` (``mode='self'``; ``op_type`` 'glu'/'mlp'/
'branch'), ``S2GridNet``, ``GridBranch``, ``GridMLP``.

Skipped names, with consumer evidence from the pt sources:

- ``SO3GridNet``: only constructed by ``so2.py`` (``node_wise_so3``,
  ``message_node_so3``) and ``ffn.py`` (``ffn_so3_grid``) — all disabled in
  the core DPA4 config.

Guarded (routable from the shared ``S2GridNet`` entry point but only used by
the disabled ``node_wise_s2``/``message_node_s2`` grid products in
``so2.py``): ``mode='cross'`` (and with it ``layout='flat'``) and
``residual_scale_init is not None`` raise ``NotImplementedError``.

Serialization contract: the pt ``S2GridNet`` and ``GridBranch`` define no
``serialize()`` (they only appear nested inside larger modules'
state-dicts); the dpmodel ``serialize()``/``deserialize()`` use
``@variables`` keys equal to the pt ``state_dict`` key names
(``scalar_gate.weight``, ``grid_op.left_proj.weight``, ...) so that pt
state-dict fragments load directly. The fixed projector matrices are
non-persistent buffers in pt (not in the state dict) and are rebuilt from
the config on deserialization.
"""

from __future__ import (
    annotations,
)

import math
from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    xp_asarray_nodetach,
    xp_sigmoid,
)
from deepmd.dpmodel.common import (
    get_xp_precision,
    to_numpy_array,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .activation import (
    SwiGLU,
)
from .indexing import (
    build_l_major_index,
    build_m_major_l_index,
    map_degree_idx,
)
from .projection import (
    BaseGridProjector,
    S2GridProjector,
)
from .so3 import (
    ChannelLinear,
    FocusLinear,
)


def _softmax_last_axis(x: Any) -> Any:
    """Numerically stable softmax on the last axis (matches torch.softmax)."""
    xp = array_api_compat.array_namespace(x)
    e_x = xp.exp(x - xp.max(x, axis=-1, keepdims=True))
    return e_x / xp.sum(e_x, axis=-1, keepdims=True)


def _build_frame_degree_index(
    *,
    lmax: int,
    mmax: int,
    coefficient_layout: str,
) -> np.ndarray:
    """Build the per-coefficient degree index used by frame channel mixers.

    The torch version's ``device`` parameter is dropped: the output is a static
    ``np.int64`` table mapping each coefficient row to its degree ``l``.
    """
    coefficient_layout = str(coefficient_layout).lower()
    if coefficient_layout == "m_major":
        return build_m_major_l_index(lmax, mmax)
    if coefficient_layout == "packed":
        degree_index = map_degree_idx(lmax)
        if int(mmax) == int(lmax):
            return degree_index
        coeff_index = build_l_major_index(lmax, mmax)
        return degree_index[coeff_index]
    raise ValueError("`coefficient_layout` must be either 'packed' or 'm_major'")


class GridMLP(NativeOP):
    """
    Polynomial point-wise MLP applied independently at every grid point.

    The op is a pure quadratic channel product with no nonlinearity: two
    channel-linear projections of the grid fields are multiplied and projected
    back. In ``self`` mode both projections see ``concat(query, context)`` and
    can form self and cross quadratic channel terms; in ``cross`` mode the
    query and context roles stay separate (``(W_q query) * (W_c context)``).

    Parameters
    ----------
    channels : int
        Number of channels per grid point.
    mode : str
        Pairing mode; ``"self"`` or ``"cross"``.
    precision : str
        Parameter precision.
    mlp_bias : bool
        Whether to use bias in the channel-linear projections. The pt
        ``GridMLP`` is always bias-free; this flag is threaded through so the
        net's ``mlp_bias`` setting reaches the grid op, but pt parity only
        holds when ``mlp_bias=False``.
    trainable : bool
        Whether parameters are trainable.
    seed : int | list[int] | None
        Random seed for weight initialization.
    """

    def __init__(
        self,
        *,
        channels: int,
        mode: str,
        precision: str = DEFAULT_PRECISION,
        mlp_bias: bool = False,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        self.channels = int(channels)
        self.mode = str(mode).lower()
        if self.mode not in {"self", "cross"}:
            raise ValueError("`mode` must be either 'self' or 'cross'")
        self.precision = precision
        self.mlp_bias = bool(mlp_bias)
        self.trainable = bool(trainable)
        self.input_channels = (
            2 * self.channels if self.mode == "self" else self.channels
        )
        self.hidden_channels = 2 * self.channels
        self.left_proj = ChannelLinear(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels,
            precision=precision,
            bias=self.mlp_bias,
            trainable=trainable,
            seed=child_seed(seed, 0),
        )
        self.right_proj = ChannelLinear(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels,
            precision=precision,
            bias=self.mlp_bias,
            trainable=trainable,
            seed=child_seed(seed, 1),
        )
        self.out_proj = ChannelLinear(
            in_channels=self.hidden_channels,
            out_channels=self.channels,
            precision=precision,
            bias=self.mlp_bias,
            trainable=trainable,
            seed=child_seed(seed, 2),
        )

    def call(self, query_grid: Any, context_grid: Any) -> Any:
        """
        Apply the point-wise polynomial MLP to ``(N, G, F, C)`` grid fields.

        Parameters
        ----------
        query_grid
            First grid source with shape ``(N, G, F, C)``.
        context_grid
            Second grid source with shape ``(N, G, F, C)``.
        """
        xp = array_api_compat.array_namespace(query_grid)
        if self.mode == "self":
            grid = xp.concat([query_grid, context_grid], axis=-1)
            left = self.left_proj(grid)
            right = self.right_proj(grid)
        else:
            left = self.left_proj(query_grid)
            right = self.right_proj(context_grid)
        return self.out_proj(left * right)  # (N, G, F, C)

    def serialize(self) -> dict[str, Any]:
        """Serialize the GridMLP to a dict.

        The pt ``GridMLP`` has no ``serialize()``; the ``@variables`` keys
        here match the pt ``state_dict`` key names.
        """
        variables = {
            "left_proj.weight": to_numpy_array(self.left_proj.weight),
            "right_proj.weight": to_numpy_array(self.right_proj.weight),
            "out_proj.weight": to_numpy_array(self.out_proj.weight),
        }
        if self.mlp_bias:
            variables["left_proj.bias"] = to_numpy_array(self.left_proj.bias)
            variables["right_proj.bias"] = to_numpy_array(self.right_proj.bias)
            variables["out_proj.bias"] = to_numpy_array(self.out_proj.bias)
        return {
            "@class": "GridMLP",
            "@version": 1,
            "config": {
                "channels": self.channels,
                "mode": self.mode,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "mlp_bias": self.mlp_bias,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": variables,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> GridMLP:
        """Deserialize a GridMLP from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "GridMLP":
            raise ValueError(f"Invalid class for GridMLP: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(
            channels=int(config["channels"]),
            mode=str(config["mode"]),
            precision=str(config["precision"]),
            mlp_bias=bool(config["mlp_bias"]),
            trainable=bool(config["trainable"]),
            seed=config.get("seed"),
        )
        obj._load_variables(variables)
        return obj

    def _load_variables(self, variables: dict[str, Any]) -> None:
        prec = PRECISION_DICT[self.precision.lower()]
        for name, proj in (
            ("left_proj", self.left_proj),
            ("right_proj", self.right_proj),
            ("out_proj", self.out_proj),
        ):
            weight = np.asarray(variables[f"{name}.weight"], dtype=prec)
            if weight.shape != proj.weight.shape:
                raise ValueError(
                    f"{name}.weight shape {weight.shape} does not match "
                    f"the expected shape {proj.weight.shape}"
                )
            proj.weight = weight
            if self.mlp_bias:
                proj.bias = np.asarray(variables[f"{name}.bias"], dtype=prec).reshape(
                    proj.bias.shape
                )


class GridBranch(NativeOP):
    """
    Scalar-routed polynomial mixer over grid product branches.

    The softmax sees only invariant scalar inputs. Each branch is a
    quadratic product of grid fields, so rotations only act through the grid
    argument and the operation remains as band-limited as the product path.

    Parameters
    ----------
    channels : int
        Number of channels per grid point.
    n_branches : int
        Number of scalar-routed product branches.
    precision : str
        Parameter precision.
    trainable : bool
        Whether parameters are trainable.
    seed : int | list[int] | None
        Random seed for weight initialization.
    """

    def __init__(
        self,
        *,
        channels: int,
        n_branches: int,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        self.channels = int(channels)
        self.n_branches = int(n_branches)
        if self.n_branches < 1:
            raise ValueError("`n_branches` must be positive")
        self.precision = precision
        self.trainable = bool(trainable)
        self.left_proj = ChannelLinear(
            in_channels=self.channels,
            out_channels=self.n_branches * self.channels,
            precision=precision,
            bias=False,
            trainable=trainable,
            seed=child_seed(seed, 0),
        )
        self.right_proj = ChannelLinear(
            in_channels=self.channels,
            out_channels=self.n_branches * self.channels,
            precision=precision,
            bias=False,
            trainable=trainable,
            seed=child_seed(seed, 1),
        )
        self.router = ChannelLinear(
            in_channels=2 * self.channels,
            out_channels=self.n_branches,
            precision=precision,
            bias=False,
            trainable=trainable,
            seed=child_seed(seed, 2),
        )
        self.out_proj = ChannelLinear(
            in_channels=self.channels,
            out_channels=self.channels,
            precision=precision,
            bias=False,
            trainable=trainable,
            seed=child_seed(seed, 3),
        )

    def call(
        self,
        query_grid: Any,
        context_grid: Any,
        scalar_pair: Any,
    ) -> Any:
        """
        Apply scalar-routed grid branch mixing.

        Parameters
        ----------
        query_grid
            First grid source with shape ``(N, G, F, C)``.
        context_grid
            Second grid source with shape ``(N, G, F, C)``.
        scalar_pair
            Invariant router source with shape ``(N, F, 2*C)``.
        """
        xp = array_api_compat.array_namespace(query_grid)
        n_batch, n_grid, n_focus, _ = query_grid.shape
        left = self.left_proj(query_grid)
        right = self.right_proj(context_grid)
        value = xp.reshape(
            left * right,
            (n_batch, n_grid, n_focus, self.n_branches, self.channels),
        )  # (N, G, F, N_branches, C)
        router = _softmax_last_axis(self.router(scalar_pair))  # (N, F, N_branches)
        # einsum "ngfhc,nfh->ngfc" as a broadcast sum over the branch axis
        out = xp.sum(value * router[:, None, :, :, None], axis=3)  # (N, G, F, C)
        return self.out_proj(out)

    def serialize(self) -> dict[str, Any]:
        """Serialize the GridBranch to a dict.

        The pt ``GridBranch`` has no ``serialize()``; the ``@variables`` keys
        here match the pt ``state_dict`` key names.
        """
        return {
            "@class": "GridBranch",
            "@version": 1,
            "config": {
                "channels": self.channels,
                "n_branches": self.n_branches,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": {
                "left_proj.weight": to_numpy_array(self.left_proj.weight),
                "right_proj.weight": to_numpy_array(self.right_proj.weight),
                "router.weight": to_numpy_array(self.router.weight),
                "out_proj.weight": to_numpy_array(self.out_proj.weight),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> GridBranch:
        """Deserialize a GridBranch from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "GridBranch":
            raise ValueError(f"Invalid class for GridBranch: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(
            channels=int(config["channels"]),
            n_branches=int(config["n_branches"]),
            precision=str(config["precision"]),
            trainable=bool(config["trainable"]),
            seed=config.get("seed"),
        )
        obj._load_variables(variables)
        return obj

    def _load_variables(self, variables: dict[str, Any]) -> None:
        prec = PRECISION_DICT[self.precision.lower()]
        for name, proj in (
            ("left_proj", self.left_proj),
            ("right_proj", self.right_proj),
            ("router", self.router),
            ("out_proj", self.out_proj),
        ):
            weight = np.asarray(variables[f"{name}.weight"], dtype=prec)
            if weight.shape != proj.weight.shape:
                raise ValueError(
                    f"{name}.weight shape {weight.shape} does not match "
                    f"the expected shape {proj.weight.shape}"
                )
            proj.weight = weight


class _FrameMixer(NativeOP):
    """Shared base for the per-degree frame channel mixers.

    The pt ``FrameContract``/``FrameExpand`` are unparameterised ``nn.Module``
    wrappers around a per-degree weight of shape ``(lmax + 1, in_ch, out_ch)``
    selected by a static degree-index buffer.  ``mode='self'`` S2 grid nets
    have ``n_frames == 1`` and never construct these; they back the SO(3)
    cross-mode grid products only.  Subclasses set ``in_channels`` /
    ``out_channels`` and the init ``bound`` (the pt weight init).
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int,
        coefficient_layout: str,
        n_frames: int,
        channels: int,
        in_channels: int,
        out_channels: int,
        init_bound: float,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        self.lmax = int(lmax)
        self.mmax = int(mmax)
        self.coefficient_layout = str(coefficient_layout).lower()
        self.n_frames = int(n_frames)
        self.channels = int(channels)
        self.precision = precision
        self.trainable = bool(trainable)
        self.degree_index = _build_frame_degree_index(
            lmax=self.lmax,
            mmax=self.mmax,
            coefficient_layout=self.coefficient_layout,
        )
        prec = PRECISION_DICT[self.precision.lower()]
        rng = np.random.default_rng(seed)
        shape = (self.lmax + 1, int(in_channels), int(out_channels))
        self.weight = rng.uniform(-init_bound, init_bound, size=shape).astype(prec)

    def call(self, coeff: Any) -> Any:
        """Apply the per-degree frame/channel map preserving the order index.

        ``einsum("ndfi,dio->ndfo", coeff, weight[degree_index])`` is realised as
        a broadcast batched matmul (the gathered weight ``(D, i, o)`` broadcasts
        over the leading frame batch dim of ``coeff``).
        """
        xp = array_api_compat.array_namespace(coeff)
        device = array_api_compat.device(coeff)
        weight = xp_asarray_nodetach(xp, self.weight[...], device=device)
        if weight.dtype != coeff.dtype:
            weight = xp.astype(weight, coeff.dtype)
        degree_index = xp_asarray_nodetach(xp, self.degree_index, device=device)
        weight = xp.take(weight, degree_index, axis=0)  # (D, i, o)
        # (N, D, F, i) @ (1, D, i, o) -> (N, D, F, o)
        return xp.matmul(coeff, weight[None, ...])

    def _serialize_config(self) -> dict[str, Any]:
        return {
            "lmax": self.lmax,
            "mmax": self.mmax,
            "coefficient_layout": self.coefficient_layout,
            "n_frames": self.n_frames,
            "channels": self.channels,
            "precision": np.dtype(PRECISION_DICT[self.precision]).name,
            "trainable": self.trainable,
            "seed": None,
        }

    @classmethod
    def _deserialize(cls, data: dict[str, Any]) -> Any:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != cls.__name__:
            raise ValueError(f"Invalid class for {cls.__name__}: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(
            lmax=int(config["lmax"]),
            mmax=int(config["mmax"]),
            coefficient_layout=str(config["coefficient_layout"]),
            n_frames=int(config["n_frames"]),
            channels=int(config["channels"]),
            precision=str(config["precision"]),
            trainable=bool(config["trainable"]),
            seed=config.get("seed"),
        )
        prec = PRECISION_DICT[obj.precision.lower()]
        weight = np.asarray(variables["weight"], dtype=prec)
        if weight.shape != obj.weight.shape:
            raise ValueError(
                f"weight shape {weight.shape} does not match "
                f"the expected shape {obj.weight.shape}"
            )
        obj.weight = weight
        return obj


class FrameContract(_FrameMixer):
    """Per-degree frame/channel contraction that preserves the order index.

    Maps ``(N, D, F, K*C) -> (N, D, F, C)`` with a per-degree weight of shape
    ``(lmax + 1, K*C, C)`` where ``K`` is ``n_frames``.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int,
        coefficient_layout: str,
        n_frames: int,
        channels: int,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        n_frames = int(n_frames)
        channels = int(channels)
        super().__init__(
            lmax=lmax,
            mmax=mmax,
            coefficient_layout=coefficient_layout,
            n_frames=n_frames,
            channels=channels,
            in_channels=n_frames * channels,
            out_channels=channels,
            init_bound=1.0 / math.sqrt(n_frames * channels),
            precision=precision,
            trainable=trainable,
            seed=seed,
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize the FrameContract to a dict.

        The pt ``FrameContract`` has no ``serialize()``; the ``@variables``
        key (``weight``) matches the pt ``state_dict`` key name. ``degree_index``
        is a non-persistent buffer in pt and is rebuilt from the config.
        """
        return {
            "@class": "FrameContract",
            "@version": 1,
            "config": self._serialize_config(),
            "@variables": {"weight": to_numpy_array(self.weight)},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> FrameContract:
        """Deserialize a FrameContract from a dict."""
        return cls._deserialize(data)


class FrameExpand(_FrameMixer):
    """Per-degree frame/channel expansion that preserves the order index.

    Maps ``(N, D, F, C) -> (N, D, F, K*C)`` with a per-degree weight of shape
    ``(lmax + 1, C, K*C)`` where ``K`` is ``n_frames``.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int,
        coefficient_layout: str,
        n_frames: int,
        channels: int,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        n_frames = int(n_frames)
        channels = int(channels)
        super().__init__(
            lmax=lmax,
            mmax=mmax,
            coefficient_layout=coefficient_layout,
            n_frames=n_frames,
            channels=channels,
            in_channels=channels,
            out_channels=n_frames * channels,
            init_bound=1.0 / math.sqrt(channels),
            precision=precision,
            trainable=trainable,
            seed=seed,
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize the FrameExpand to a dict.

        The pt ``FrameExpand`` has no ``serialize()``; the ``@variables``
        key (``weight``) matches the pt ``state_dict`` key name. ``degree_index``
        is a non-persistent buffer in pt and is rebuilt from the config.
        """
        return {
            "@class": "FrameExpand",
            "@version": 1,
            "config": self._serialize_config(),
            "@variables": {"weight": to_numpy_array(self.weight)},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> FrameExpand:
        """Deserialize a FrameExpand from a dict."""
        return cls._deserialize(data)


class BaseGridNet(NativeOP):
    """
    Shared implementation for S2 grid nets (``mode='self'`` only).

    ``mode='self'`` expects one input whose last channel axis contains two
    branches; the first half supplies the SwiGLU gates of the scalar path.

    The pt ``mode='cross'`` path (with ``layout='flat'``,
    ``residual_scale_init``, and the SO(3) frame machinery) backs the
    ``node_wise_s2``/``message_node_s2`` grid products only, which are
    disabled in the core DPA4 config; it is not ported.
    """

    def __init__(
        self,
        *,
        projector: BaseGridProjector,
        channels: int,
        n_focus: int,
        mode: str,
        op_type: str,
        precision: str = DEFAULT_PRECISION,
        layout: str,
        mlp_bias: bool,
        trainable: bool = True,
        grid_branches: int = 1,
        residual_scale_init: float | None = None,
        seed: int | list[int] | None = None,
    ) -> None:
        self.projector = projector
        self.lmax = int(projector.lmax)
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.n_frames = int(projector.n_frames)
        if self.n_frames != 1:
            raise ValueError(
                "dpmodel BaseGridNet only supports S2 projectors (n_frames == 1)"
            )
        self.mode = str(mode).lower()
        if self.mode not in {"self", "cross"}:
            raise ValueError("`mode` must be either 'self' or 'cross'")
        if self.mode == "cross":
            raise NotImplementedError(
                "mode='cross' (node_wise_s2/message_node_s2 grid products) "
                "is not ported to dpmodel"
            )
        self.op_type = str(op_type).lower()
        if self.op_type not in {"glu", "mlp", "branch"}:
            raise ValueError("`op_type` must be one of 'glu', 'mlp', or 'branch'")
        self.precision = precision
        self.layout = str(layout).lower()
        if self.layout not in {"ndfc", "nfdc", "flat"}:
            raise ValueError("`layout` must be one of 'ndfc', 'nfdc', or 'flat'")
        if self.mode == "self" and self.layout == "flat":
            raise ValueError("`layout='flat'` is only supported for cross grid nets")
        self.mlp_bias = bool(mlp_bias)
        self.trainable = bool(trainable)
        self.expanded_channels = self.n_frames * self.channels
        self.query_channels = 2 * self.expanded_channels
        self.output_channels = self.expanded_channels
        self.frame_zero_index = 0
        if residual_scale_init is not None:
            raise NotImplementedError(
                "`residual_scale_init` is only used by the cross-mode "
                "node_wise_s2/message_node_s2 grid products, which are not "
                "ported to dpmodel"
            )
        self.residual_scale = None

        self.scalar_act = SwiGLU()
        self.scalar_gate = FocusLinear(
            in_channels=2 * self.channels,
            out_channels=self.channels,
            n_focus=self.n_focus,
            precision=self.precision,
            bias=self.mlp_bias,
            trainable=trainable,
            seed=child_seed(seed, 0),
            init_std=0.01,
        )
        if self.op_type == "mlp":
            self.grid_op: GridMLP | GridBranch | None = GridMLP(
                channels=self.channels,
                mode=self.mode,
                precision=self.precision,
                mlp_bias=self.mlp_bias,
                trainable=trainable,
                seed=child_seed(seed, 1),
            )
        elif self.op_type == "branch":
            self.grid_op = GridBranch(
                channels=self.channels,
                n_branches=grid_branches,
                precision=self.precision,
                trainable=trainable,
                seed=child_seed(seed, 1),
            )
        else:
            # pt uses nn.Identity() here (parameter-free, no state-dict keys)
            self.grid_op = None

    def call(self, query: Any, context: Any = None) -> Any:
        """Apply the configured grid net and restore the input layout."""
        xp = array_api_compat.array_namespace(query)
        input_dtype = query.dtype
        compute_dtype = get_xp_precision(xp, self.precision)
        query_ndfc = self._to_ndfc(query)
        left, right = self._split_self_query(query_ndfc)
        scalar_pair = self._make_scalar_pair(left, right, compute_dtype)
        grid_out = self._apply_grid_op(left, right, scalar_pair, compute_dtype)
        coeff_out = self._from_grid(grid_out)
        coeff_out = self._apply_scalar_path(coeff_out, scalar_pair)
        if coeff_out.dtype != input_dtype:
            coeff_out = xp.astype(coeff_out, input_dtype)
        return self._restore_layout(coeff_out)

    def _apply_grid_op(
        self,
        left: Any,
        right: Any,
        scalar_pair: Any,
        compute_dtype: Any,
    ) -> Any:
        xp = array_api_compat.array_namespace(left)
        if left.dtype != compute_dtype:
            left = xp.astype(left, compute_dtype)
        if right.dtype != compute_dtype:
            right = xp.astype(right, compute_dtype)
        left_grid = self._to_grid(left)
        right_grid = self._to_grid(right)
        if self.op_type == "glu":
            return left_grid * right_grid
        if self.op_type == "mlp":
            return self.grid_op(left_grid, right_grid)
        return self.grid_op(left_grid, right_grid, scalar_pair)

    def _apply_scalar_path(self, coeff: Any, scalar_pair: Any) -> Any:
        xp = array_api_compat.array_namespace(coeff)
        scalar_out = self.scalar_act(scalar_pair)  # (N, F, C)
        scalar_gate = xp_sigmoid(self.scalar_gate(scalar_pair))  # (N, F, C)
        coeff = coeff * scalar_gate[:, None, :, :]
        # gradient-safe equivalent of the pt in-place
        # ``coeff_view[:, 0, :, 0, :].add_(scalar_out)`` (n_frames == 1)
        head = coeff[:, :1, :, :] + scalar_out[:, None, :, :]
        return xp.concat([head, coeff[:, 1:, :, :]], axis=1)

    def _split_self_query(self, query: Any) -> tuple[Any, Any]:
        self._check_last_dim(query, self.query_channels, "query")
        # torch.chunk(query, 2, dim=-1) with an even channel count
        return (
            query[..., : self.expanded_channels],
            query[..., self.expanded_channels :],
        )

    def _make_scalar_pair(self, left: Any, right: Any, compute_dtype: Any) -> Any:
        xp = array_api_compat.array_namespace(left)
        scalar_pair = xp.concat(
            [
                self._extract_scalar(left),
                self._extract_scalar(right),
            ],
            axis=-1,
        )
        if scalar_pair.dtype != compute_dtype:
            scalar_pair = xp.astype(scalar_pair, compute_dtype)
        return scalar_pair

    def _extract_scalar(self, coeff: Any) -> Any:
        # (N, D, F, C) -> the (l=0, m=0) scalar slice (N, F, C); n_frames == 1
        return coeff[:, 0, :, :]

    def _to_grid(self, coeff: Any) -> Any:
        # einsum "gd,ndfc->ngfc" (n_frames == 1) as a broadcast batched matmul
        xp = array_api_compat.array_namespace(coeff)
        n_batch, coeff_dim, n_focus, _ = coeff.shape
        to_grid_mat = xp_asarray_nodetach(
            xp, self.projector.to_grid_mat[...], device=array_api_compat.device(coeff)
        )
        if to_grid_mat.dtype != coeff.dtype:
            to_grid_mat = xp.astype(to_grid_mat, coeff.dtype)
        flat = xp.reshape(coeff, (n_batch, coeff_dim, n_focus * self.channels))
        out = xp.matmul(to_grid_mat[None, ...], flat)  # (N, G, F*C)
        return xp.reshape(
            out, (n_batch, self.projector.grid_size, n_focus, self.channels)
        )

    def _from_grid(self, grid: Any) -> Any:
        # einsum "dg,ngfc->ndfc" (n_frames == 1) as a broadcast batched matmul
        xp = array_api_compat.array_namespace(grid)
        n_batch, n_grid, n_focus, _ = grid.shape
        coeff_dim = self.projector.coeff_dim
        from_grid_mat = xp_asarray_nodetach(
            xp, self.projector.from_grid_mat[...], device=array_api_compat.device(grid)
        )
        if from_grid_mat.dtype != grid.dtype:
            from_grid_mat = xp.astype(from_grid_mat, grid.dtype)
        flat = xp.reshape(grid, (n_batch, n_grid, n_focus * self.channels))
        out = xp.matmul(from_grid_mat[None, ...], flat)  # (N, D, F*C)
        return xp.reshape(out, (n_batch, coeff_dim, n_focus, self.expanded_channels))

    def _to_ndfc(self, value: Any) -> Any:
        if self.layout == "ndfc":
            return value
        # "nfdc": (N, F, D, C) -> (N, D, F, C); "flat" is cross-only (blocked)
        xp = array_api_compat.array_namespace(value)
        return xp.permute_dims(value, (0, 2, 1, 3))

    def _restore_layout(self, value: Any) -> Any:
        if self.layout == "ndfc":
            return value
        xp = array_api_compat.array_namespace(value)
        return xp.permute_dims(value, (0, 2, 1, 3))

    def _check_last_dim(self, value: Any, expected: int, name: str) -> None:
        if value.shape[-1] != expected:
            raise ValueError(
                f"`{name}` last dimension must be {expected}, got {value.shape[-1]}"
            )


class S2GridNet(BaseGridNet):
    """Grid net using an S2 spherical-harmonic projector (Lebedev only).

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.
    mmax : int | None
        Maximum order kept in the coefficient layout. If None, use ``lmax``.
    channels : int
        Number of channels per (l, m) coefficient.
    n_focus : int
        Number of focus streams.
    mode : str
        Pairing mode; only ``"self"`` is ported.
    op_type : str
        Point-wise grid operation; ``"glu"``, ``"mlp"``, or ``"branch"``.
    precision : str
        Parameter precision.
    layout : str
        Tensor layout convention: ``"ndfc"`` or ``"nfdc"``.
    grid_resolution_list : list[int] | None
        Lebedev ``[precision, n_points]`` pair; resolved automatically if None.
    coefficient_layout : str
        ``"packed"`` or ``"m_major"`` coefficient ordering.
    grid_method : str
        S2 quadrature backend; only ``"lebedev"`` is ported.
    grid_branches : int
        Number of scalar-routed branches when ``op_type='branch'``.
    residual_scale_init : float | None
        Not ported (cross-mode only); must be None.
    mlp_bias : bool
        Whether to use bias in the scalar gate projection.
    trainable : bool
        Whether parameters are trainable.
    seed : int | list[int] | None
        Random seed for weight initialization.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        channels: int,
        n_focus: int = 1,
        mode: str,
        op_type: str,
        precision: str = DEFAULT_PRECISION,
        layout: str,
        grid_resolution_list: list[int] | None = None,
        coefficient_layout: str = "packed",
        # Deliberate divergence from pt's default ("e3nn"): the e3nn
        # product-grid branch is not ported to dpmodel and always raises, so
        # the only usable default here is "lebedev". Checkpoint compatibility
        # is unaffected because serialize always records the explicit value.
        grid_method: str = "lebedev",
        grid_branches: int = 1,
        residual_scale_init: float | None = None,
        mlp_bias: bool = False,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        projector = S2GridProjector(
            lmax=lmax,
            mmax=mmax,
            precision=precision,
            grid_resolution_list=grid_resolution_list,
            coefficient_layout=coefficient_layout,
            grid_method=grid_method,
        )
        self.grid_resolution_list = projector.grid_resolution_list
        self.grid_method = projector.grid_method
        self.grid_branches = int(grid_branches)
        super().__init__(
            projector=projector,
            channels=channels,
            n_focus=n_focus,
            mode=mode,
            op_type=op_type,
            precision=precision,
            layout=layout,
            mlp_bias=mlp_bias,
            trainable=trainable,
            grid_branches=grid_branches,
            residual_scale_init=residual_scale_init,
            seed=seed,
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize the S2GridNet to a dict.

        The pt ``S2GridNet`` has no ``serialize()``; the ``@variables`` keys
        here match the pt ``state_dict`` key names (the projector matrices
        are non-persistent buffers in pt and are rebuilt from the config).
        """
        variables = {"scalar_gate.weight": to_numpy_array(self.scalar_gate.weight)}
        if self.mlp_bias:
            variables["scalar_gate.bias"] = to_numpy_array(self.scalar_gate.bias)
        if self.op_type in {"branch", "mlp"}:
            grid_op_data = self.grid_op.serialize()["@variables"]
            for key, value in grid_op_data.items():
                variables[f"grid_op.{key}"] = value
        return {
            "@class": "S2GridNet",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.projector.mmax,
                "channels": self.channels,
                "n_focus": self.n_focus,
                "mode": self.mode,
                "op_type": self.op_type,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "layout": self.layout,
                "grid_resolution_list": self.grid_resolution_list,
                "coefficient_layout": self.projector.coefficient_layout,
                "grid_method": self.grid_method,
                "grid_branches": self.grid_branches,
                "mlp_bias": self.mlp_bias,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": variables,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> S2GridNet:
        """Deserialize an S2GridNet from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "S2GridNet":
            raise ValueError(f"Invalid class for S2GridNet: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(
            lmax=int(config["lmax"]),
            mmax=int(config["mmax"]),
            channels=int(config["channels"]),
            n_focus=int(config["n_focus"]),
            mode=str(config["mode"]),
            op_type=str(config["op_type"]),
            precision=str(config["precision"]),
            layout=str(config["layout"]),
            grid_resolution_list=config["grid_resolution_list"],
            coefficient_layout=str(config["coefficient_layout"]),
            grid_method=str(config["grid_method"]),
            grid_branches=int(config["grid_branches"]),
            mlp_bias=bool(config["mlp_bias"]),
            trainable=bool(config["trainable"]),
            seed=config.get("seed"),
        )
        prec = PRECISION_DICT[obj.precision.lower()]
        weight = np.asarray(variables["scalar_gate.weight"], dtype=prec)
        if weight.shape != obj.scalar_gate.weight.shape:
            raise ValueError(
                f"scalar_gate.weight shape {weight.shape} does not match "
                f"the expected shape {obj.scalar_gate.weight.shape}"
            )
        obj.scalar_gate.weight = weight
        if obj.mlp_bias:
            obj.scalar_gate.bias = np.asarray(
                variables["scalar_gate.bias"], dtype=prec
            ).reshape(obj.scalar_gate.bias.shape)
        if obj.op_type in {"branch", "mlp"}:
            obj.grid_op._load_variables(
                {
                    key[len("grid_op.") :]: value
                    for key, value in variables.items()
                    if key.startswith("grid_op.")
                }
            )
        return obj
