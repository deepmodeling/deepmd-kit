# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Grid-space nonlinearities for DPA4/SeZM coefficient tensors.

This module is the dpmodel port of
``deepmd.pt.model.descriptor.sezm_nn.grid_net``, covering the S2/Lebedev and
SO(3)/Wigner-D quadrature paths. A grid net receives coefficient tensors,
converts them to quadrature values, applies one point-wise grid operation, and
projects the result back to coefficients. The public shapes are:

* ``mode='self'``: one input ``(N, D, F, 2*C)`` or ``(N, F, D, 2*C)``.
* grid values: ``(N, G, F, C)`` after S2 or SO(3) projection.

Ported names: ``BaseGridNet`` (``mode='self'``/``'cross'``; ``op_type``
'glu'/'mlp'/'branch'), ``S2GridNet``, ``SO3GridNet``, ``GridBranch``,
``GridMLP``, ``FrameExpand``, ``FrameContract``.

``mode='cross'`` (with ``layout='flat'`` and ``residual_scale_init``) is
supported for both projectors. For the SO(3) projector (``n_frames > 1``) the
frame axis is created by ``FrameExpand`` (``channels -> n_frames*channels``)
and collapsed by ``FrameContract``; for the S2 projector (``n_frames == 1``)
there is no frame machinery and query/context stay separate.

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
    SO3GridProjector,
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
    trainable : bool
        Whether parameters are trainable.
    seed : int | list[int] | None
        Random seed for weight initialization.

    Notes
    -----
    Like the pt ``GridMLP``, the three channel-linear projections are always
    bias-free (the net-level ``mlp_bias`` flag only affects the scalar gate,
    not the grid op).
    """

    def __init__(
        self,
        *,
        channels: int,
        mode: str,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        self.channels = int(channels)
        self.mode = str(mode).lower()
        if self.mode not in {"self", "cross"}:
            raise ValueError("`mode` must be either 'self' or 'cross'")
        self.precision = precision
        self.trainable = bool(trainable)
        self.input_channels = (
            2 * self.channels if self.mode == "self" else self.channels
        )
        self.hidden_channels = 2 * self.channels
        self.left_proj = ChannelLinear(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels,
            precision=precision,
            bias=False,
            trainable=trainable,
            seed=child_seed(seed, 0),
        )
        self.right_proj = ChannelLinear(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels,
            precision=precision,
            bias=False,
            trainable=trainable,
            seed=child_seed(seed, 1),
        )
        self.out_proj = ChannelLinear(
            in_channels=self.hidden_channels,
            out_channels=self.channels,
            precision=precision,
            bias=False,
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
        return {
            "@class": "GridMLP",
            "@version": 1,
            "config": {
                "channels": self.channels,
                "mode": self.mode,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
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
    Shared implementation for S2 (``n_frames == 1``) and SO(3) grid nets.

    ``mode='self'`` expects one input whose last channel axis contains two
    branches; the first half supplies the SwiGLU gates of the scalar path.
    ``mode='cross'`` expects separate query and context inputs and supports
    ``layout='flat'`` and ``residual_scale_init``.

    The SO(3) frame machinery (FrameExpand/FrameContract, ``n_frames > 1``)
    that backs ``SO3GridNet`` cross-mode is wired through the optional
    ``frame_expand``/``frame_contract`` modules.  When both are ``None`` (the
    S2 path), the query/context/output widths collapse to
    ``expanded_channels`` and the ``n_frames == 1`` fast paths are used.
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
        frame_expand: _FrameMixer | None = None,
        frame_contract: _FrameMixer | None = None,
        residual_scale_init: float | None = None,
        seed: int | list[int] | None = None,
    ) -> None:
        self.projector = projector
        self.lmax = int(projector.lmax)
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.n_frames = int(projector.n_frames)
        self.mode = str(mode).lower()
        if self.mode not in {"self", "cross"}:
            raise ValueError("`mode` must be either 'self' or 'cross'")
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
        self.frame_expand = frame_expand
        self.frame_contract = frame_contract
        # With frame_expand present (SO(3) cross), the external query/context
        # widths are ``channels`` and the frame axis is created internally; with
        # frame_contract present, the output collapses back to ``channels``.
        # When both are None (the S2 path / SO(3) self path) all widths are
        # ``expanded_channels``.
        self.query_channels = (
            2 * self.expanded_channels
            if self.mode == "self"
            else (
                self.channels
                if self.frame_expand is not None
                else self.expanded_channels
            )
        )
        self.context_channels = (
            self.channels if self.frame_expand is not None else self.expanded_channels
        )
        self.output_channels = (
            self.channels if self.frame_contract is not None else self.expanded_channels
        )
        self.frame_zero_index = int(getattr(self.projector, "frame_zero_index", 0))
        self.residual_scale_init = (
            None if residual_scale_init is None else float(residual_scale_init)
        )
        if self.residual_scale_init is None:
            self.residual_scale = None
        else:
            prec = PRECISION_DICT[self.precision.lower()]
            self.residual_scale = (
                np.ones((self.n_focus, self.output_channels), dtype=prec)
                * self.residual_scale_init
            )

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
            # GridMLP projections are bias-free (mirrors pt); the net-level
            # mlp_bias only affects the scalar gate, not the grid op.
            self.grid_op: GridMLP | GridBranch | None = GridMLP(
                channels=self.channels,
                mode=self.mode,
                precision=self.precision,
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
        left, right, scalar_pair = self._prepare_pair(
            query_ndfc, context, compute_dtype
        )
        grid_out = self._apply_grid_op(left, right, scalar_pair, compute_dtype)
        coeff_out = self._from_grid(grid_out)
        coeff_out = self._apply_scalar_path(coeff_out, scalar_pair)
        coeff_out = self._contract_frames(coeff_out)
        coeff_out = self._apply_residual_scale(coeff_out)
        if coeff_out.dtype != input_dtype:
            coeff_out = xp.astype(coeff_out, input_dtype)
        return self._restore_layout(coeff_out)

    def _prepare_pair(
        self,
        query: Any,
        context: Any,
        compute_dtype: Any,
    ) -> tuple[Any, Any, Any]:
        if self.mode == "self":
            left, right = self._split_self_query(query)
            scalar_pair = self._make_scalar_pair(left, right, compute_dtype)
            return left, right, scalar_pair
        return self._prepare_cross_pair(query, context, compute_dtype)

    def _prepare_cross_pair(
        self,
        query: Any,
        context: Any,
        compute_dtype: Any,
    ) -> tuple[Any, Any, Any]:
        if context is None:
            raise ValueError("`context` is required when `mode='cross'`")
        xp = array_api_compat.array_namespace(query)
        context_ndfc = self._to_ndfc(context)
        self._check_last_dim(query, self.context_channels, "query")
        self._check_last_dim(context_ndfc, self.context_channels, "context")
        if self.frame_expand is None:
            # S2 path: query and context keep their incoming (N, D, F, C) shape.
            scalar_pair = self._make_scalar_pair(query, context_ndfc, compute_dtype)
            return query, context_ndfc, scalar_pair
        # SO(3) frame_expand path: the external query/context width is
        # ``channels``, so the (l=0) scalar slice is the full leading row; the
        # frame axis is created by frame_expand (channels -> n_frames*channels).
        scalar_pair = xp.concat([query[:, 0, :, :], context_ndfc[:, 0, :, :]], axis=-1)
        if scalar_pair.dtype != compute_dtype:
            scalar_pair = xp.astype(scalar_pair, compute_dtype)
        return (
            self.frame_expand(query),
            self.frame_expand(context_ndfc),
            scalar_pair,
        )

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

    def _contract_frames(self, coeff: Any) -> Any:
        # SO(3) cross-mode collapses the per-degree frame axis back to
        # ``channels``; the S2 path leaves the coefficient unchanged.
        if self.frame_contract is None:
            return coeff
        return self.frame_contract(coeff)

    def _apply_residual_scale(self, coeff: Any) -> Any:
        if self.residual_scale is None:
            return coeff
        xp = array_api_compat.array_namespace(coeff)
        scale = xp_asarray_nodetach(
            xp,
            self.residual_scale[...],
            device=array_api_compat.device(coeff),
        )
        if scale.dtype != coeff.dtype:
            scale = xp.astype(scale, coeff.dtype)
        # broadcast (n_focus, output_channels) over (N, D, F, C)
        return coeff * xp.reshape(scale, (1, 1, self.n_focus, self.output_channels))

    def _apply_scalar_path(self, coeff: Any, scalar_pair: Any) -> Any:
        xp = array_api_compat.array_namespace(coeff)
        scalar_out = self.scalar_act(scalar_pair)  # (N, F, C)
        scalar_gate = xp_sigmoid(self.scalar_gate(scalar_pair))  # (N, F, C)
        if self.n_frames == 1:
            coeff = coeff * scalar_gate[:, None, :, :]
            # gradient-safe equivalent of the pt in-place
            # ``coeff_view[:, 0, :, 0, :].add_(scalar_out)`` (n_frames == 1)
            head = coeff[:, :1, :, :] + scalar_out[:, None, :, :]
            return xp.concat([head, coeff[:, 1:, :, :]], axis=1)
        # n_frames > 1: reshape the channel axis into (n_frames, channels),
        # gate every frame, then add scalar_out only at (d=0, k=frame_zero).
        n_batch, coeff_dim, n_focus, _ = coeff.shape
        coeff_view = xp.reshape(
            coeff, (n_batch, coeff_dim, n_focus, self.n_frames, self.channels)
        )
        coeff_view = coeff_view * scalar_gate[:, None, :, None, :]
        # gradient-safe equivalent of the pt in-place
        # ``coeff_view[:, 0, :, frame_zero_index, :].add_(scalar_out)``: build a
        # constant (D, K) one-hot placement and broadcast scalar_out onto it.
        place = np.zeros((coeff_dim, self.n_frames), dtype=np.float64)
        place[0, self.frame_zero_index] = 1.0
        place_xp = xp_asarray_nodetach(xp, place, device=array_api_compat.device(coeff))
        if place_xp.dtype != coeff_view.dtype:
            place_xp = xp.astype(place_xp, coeff_view.dtype)
        add_term = scalar_out[:, None, :, None, :] * place_xp[None, :, None, :, None]
        coeff_view = coeff_view + add_term
        return xp.reshape(
            coeff_view, (n_batch, coeff_dim, n_focus, self.expanded_channels)
        )

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
        if self.n_frames == 1:
            return coeff[:, 0, :, :]
        # n_frames > 1: split the channel axis into (n_frames, channels) and
        # pick the zero-frame of the (l=0) row.
        xp = array_api_compat.array_namespace(coeff)
        n_batch, coeff_dim, n_focus, _ = coeff.shape
        coeff_view = xp.reshape(
            coeff, (n_batch, coeff_dim, n_focus, self.n_frames, self.channels)
        )
        return coeff_view[:, 0, :, self.frame_zero_index, :]

    def _to_grid(self, coeff: Any) -> Any:
        xp = array_api_compat.array_namespace(coeff)
        n_batch, coeff_dim, n_focus, _ = coeff.shape
        to_grid_mat = xp_asarray_nodetach(
            xp, self.projector.to_grid_mat[...], device=array_api_compat.device(coeff)
        )
        if to_grid_mat.dtype != coeff.dtype:
            to_grid_mat = xp.astype(to_grid_mat, coeff.dtype)
        if self.n_frames == 1:
            # einsum "gd,ndfc->ngfc" (n_frames == 1) as a broadcast batched matmul
            flat = xp.reshape(coeff, (n_batch, coeff_dim, n_focus * self.channels))
            out = xp.matmul(to_grid_mat[None, ...], flat)  # (N, G, F*C)
            return xp.reshape(
                out, (n_batch, self.projector.grid_size, n_focus, self.channels)
            )
        # einsum "gdk,ndfkc->ngfc": flatten (d, k) into the projector's J axis
        # (J = d*K + k matches the projector's flat_idx ordering) and matmul
        # against to_grid_mat (G, J).
        coeff_view = xp.reshape(
            coeff, (n_batch, coeff_dim, n_focus, self.n_frames, self.channels)
        )
        # (N, D, F, K, C) -> (N, D, K, F, C) -> (N, D*K, F*C)
        coeff_perm = xp.permute_dims(coeff_view, (0, 1, 3, 2, 4))
        flat = xp.reshape(
            coeff_perm,
            (n_batch, coeff_dim * self.n_frames, n_focus * self.channels),
        )
        out = xp.matmul(to_grid_mat[None, ...], flat)  # (N, G, F*C)
        return xp.reshape(
            out, (n_batch, self.projector.grid_size, n_focus, self.channels)
        )

    def _from_grid(self, grid: Any) -> Any:
        xp = array_api_compat.array_namespace(grid)
        n_batch, n_grid, n_focus, _ = grid.shape
        from_grid_mat = xp_asarray_nodetach(
            xp, self.projector.from_grid_mat[...], device=array_api_compat.device(grid)
        )
        if from_grid_mat.dtype != grid.dtype:
            from_grid_mat = xp.astype(from_grid_mat, grid.dtype)
        flat = xp.reshape(grid, (n_batch, n_grid, n_focus * self.channels))
        if self.n_frames == 1:
            # einsum "dg,ngfc->ndfc" (n_frames == 1) as a broadcast batched matmul
            out = xp.matmul(from_grid_mat[None, ...], flat)  # (N, D, F*C)
            return xp.reshape(
                out,
                (n_batch, self.projector.coeff_dim, n_focus, self.expanded_channels),
            )
        # einsum "dkg,ngfc->ndfkc": matmul against from_grid_mat (J=D*K, G) then
        # split J back into (D, K) and move the frame axis next to the channel.
        coeff_dim = self.projector.coeff_dim // self.n_frames
        out = xp.matmul(from_grid_mat[None, ...], flat)  # (N, D*K, F*C)
        out = xp.reshape(
            out, (n_batch, coeff_dim, self.n_frames, n_focus, self.channels)
        )
        # (N, D, K, F, C) -> (N, D, F, K, C)
        out = xp.permute_dims(out, (0, 1, 3, 2, 4))
        return xp.reshape(out, (n_batch, coeff_dim, n_focus, self.expanded_channels))

    def _to_ndfc(self, value: Any) -> Any:
        if self.layout == "ndfc":
            return value
        xp = array_api_compat.array_namespace(value)
        if self.layout == "nfdc":
            # (N, F, D, C) -> (N, D, F, C)
            return xp.permute_dims(value, (0, 2, 1, 3))
        # "flat" (cross-only): (N, D, F*C) -> (N, D, F, C)
        n_batch, coeff_dim, _ = value.shape
        return xp.reshape(value, (n_batch, coeff_dim, self.n_focus, -1))

    def _restore_layout(self, value: Any) -> Any:
        if self.layout == "ndfc":
            return value
        xp = array_api_compat.array_namespace(value)
        if self.layout == "nfdc":
            return xp.permute_dims(value, (0, 2, 1, 3))
        # "flat" (cross-only): (N, D, F, C) -> (N, D, F*C)
        n_batch, coeff_dim, _, _ = value.shape
        return xp.reshape(value, (n_batch, coeff_dim, -1))

    def _check_last_dim(self, value: Any, expected: int, name: str) -> None:
        if value.shape[-1] != expected:
            raise ValueError(
                f"`{name}` last dimension must be {expected}, got {value.shape[-1]}"
            )

    def _load_variables(self, variables: dict[str, Any]) -> None:
        """Load variables keyed by the pt ``state_dict`` key names.

        Handles ``scalar_gate``, the optional ``grid_op`` (branch/mlp), the
        optional SO(3) ``frame_expand``/``frame_contract`` per-degree mixers,
        and the optional ``residual_scale`` parameter. ``S2GridNet`` leaves the
        frame mixers as ``None`` so the corresponding keys are absent.
        """
        prec = PRECISION_DICT[self.precision.lower()]
        weight = np.asarray(variables["scalar_gate.weight"], dtype=prec)
        if weight.shape != self.scalar_gate.weight.shape:
            raise ValueError(
                f"scalar_gate.weight shape {weight.shape} does not match "
                f"the expected shape {self.scalar_gate.weight.shape}"
            )
        self.scalar_gate.weight = weight
        if self.mlp_bias:
            self.scalar_gate.bias = np.asarray(
                variables["scalar_gate.bias"], dtype=prec
            ).reshape(self.scalar_gate.bias.shape)
        if self.op_type in {"branch", "mlp"}:
            self.grid_op._load_variables(
                {
                    key[len("grid_op.") :]: value
                    for key, value in variables.items()
                    if key.startswith("grid_op.")
                }
            )
        for name, mixer in (
            ("frame_expand", self.frame_expand),
            ("frame_contract", self.frame_contract),
        ):
            if mixer is None:
                continue
            mixer_weight = np.asarray(variables[f"{name}.weight"], dtype=prec)
            if mixer_weight.shape != mixer.weight.shape:
                raise ValueError(
                    f"{name}.weight shape {mixer_weight.shape} does not match "
                    f"the expected shape {mixer.weight.shape}"
                )
            mixer.weight = mixer_weight
        if self.residual_scale is not None:
            residual_scale = np.asarray(variables["residual_scale"], dtype=prec)
            if residual_scale.shape != self.residual_scale.shape:
                raise ValueError(
                    f"residual_scale shape {residual_scale.shape} does not match "
                    f"the expected shape {self.residual_scale.shape}"
                )
            self.residual_scale = residual_scale


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
        Pairing mode; ``"self"`` or ``"cross"``.
    op_type : str
        Point-wise grid operation; ``"glu"``, ``"mlp"``, or ``"branch"``.
    precision : str
        Parameter precision.
    layout : str
        Tensor layout convention: ``"ndfc"``, ``"nfdc"``, or ``"flat"``
        (``"flat"`` is cross-mode only).
    grid_resolution_list : list[int] | None
        Lebedev ``[precision, n_points]`` pair; resolved automatically if None.
    coefficient_layout : str
        ``"packed"`` or ``"m_major"`` coefficient ordering.
    grid_method : str
        S2 quadrature backend; only ``"lebedev"`` is ported.
    grid_branches : int
        Number of scalar-routed branches when ``op_type='branch'``.
    residual_scale_init : float | None
        If set, scales the grid-net output by a per-(focus, channel) parameter
        initialised to this value (cross-mode use). ``None`` disables it.
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
        # ``residual_scale`` is an nn.Parameter directly on the pt module, so the
        # @variables key matches its pt state_dict key.
        if self.residual_scale is not None:
            variables["residual_scale"] = to_numpy_array(self.residual_scale)
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
                "residual_scale_init": self.residual_scale_init,
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
            residual_scale_init=config.get("residual_scale_init"),
            mlp_bias=bool(config["mlp_bias"]),
            trainable=bool(config["trainable"]),
            seed=config.get("seed"),
        )
        obj._load_variables(variables)
        return obj


class SO3GridNet(BaseGridNet):
    """Grid net using a Wigner-D SO(3) projector with frame indices.

    The dpmodel port of the pt ``SO3GridNet``.  ``mode='self'`` keeps the frame
    axis inside the channel (width ``n_frames * channels``); ``mode='cross'``
    expands the external ``channels``-wide query/context to the frame axis via
    ``FrameExpand`` and collapses the output back via ``FrameContract``.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.
    mmax : int | None
        Maximum order kept in the coefficient layout. If None, use ``lmax``.
    kmax : int
        Frame-index half-width; the frame set is ``{0, -1, 1, ..., -kmax, kmax}``.
    channels : int
        Number of channels per (l, m) coefficient.
    n_focus : int
        Number of focus streams.
    mode : str
        Pairing mode; ``"self"`` or ``"cross"``.
    op_type : str
        Point-wise grid operation; ``"glu"``, ``"mlp"``, or ``"branch"``.
    precision : str
        Parameter precision.
    layout : str
        Tensor layout convention: ``"ndfc"``, ``"nfdc"``, or ``"flat"``
        (``"flat"`` is cross-mode only).
    lebedev_precision : int | None
        Explicit Lebedev rule precision. If None, resolved automatically.
    coefficient_layout : str
        ``"packed"`` or ``"m_major"`` coefficient ordering.
    grid_branches : int
        Number of scalar-routed branches when ``op_type='branch'``.
    residual_scale_init : float | None
        If set, scales the grid-net output by a per-(focus, channel) parameter
        initialised to this value (cross-mode use). ``None`` disables it.
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
        kmax: int = 1,
        channels: int,
        n_focus: int = 1,
        mode: str,
        op_type: str,
        precision: str = DEFAULT_PRECISION,
        layout: str,
        lebedev_precision: int | None = None,
        coefficient_layout: str = "packed",
        grid_branches: int = 1,
        residual_scale_init: float | None = None,
        mlp_bias: bool = False,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        projector = SO3GridProjector(
            lmax=lmax,
            mmax=mmax,
            kmax=kmax,
            precision=precision,
            lebedev_precision=lebedev_precision,
            coefficient_layout=coefficient_layout,
        )
        self.frames = projector.frame_set
        self.kmax = projector.kmax
        self.lebedev_precision = projector.lebedev_precision
        self.n_gamma = projector.n_gamma
        self.grid_branches = int(grid_branches)
        frame_expand: FrameExpand | None = None
        frame_contract: FrameContract | None = None
        if str(mode).lower() == "cross":
            frame_expand = FrameExpand(
                lmax=lmax,
                mmax=projector.mmax,
                coefficient_layout=coefficient_layout,
                n_frames=projector.n_frames,
                channels=channels,
                precision=precision,
                trainable=trainable,
                seed=child_seed(seed, 4),
            )
            frame_contract = FrameContract(
                lmax=lmax,
                mmax=projector.mmax,
                coefficient_layout=coefficient_layout,
                n_frames=projector.n_frames,
                channels=channels,
                precision=precision,
                trainable=trainable,
                seed=child_seed(seed, 5),
            )
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
            frame_expand=frame_expand,
            frame_contract=frame_contract,
            residual_scale_init=residual_scale_init,
            seed=seed,
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize the SO3GridNet to a dict.

        The pt ``SO3GridNet`` has no ``serialize()``; the ``@variables`` keys
        here match the pt ``state_dict`` key names (the projector matrices are
        non-persistent buffers in pt and are rebuilt from the config). The
        ``frame_expand``/``frame_contract`` per-degree weights are emitted as
        ``frame_expand.weight``/``frame_contract.weight`` (their pt state_dict
        keys); their ``degree_index`` buffers are non-persistent and rebuilt.
        """
        variables = {"scalar_gate.weight": to_numpy_array(self.scalar_gate.weight)}
        if self.mlp_bias:
            variables["scalar_gate.bias"] = to_numpy_array(self.scalar_gate.bias)
        if self.op_type in {"branch", "mlp"}:
            grid_op_data = self.grid_op.serialize()["@variables"]
            for key, value in grid_op_data.items():
                variables[f"grid_op.{key}"] = value
        if self.frame_expand is not None:
            variables["frame_expand.weight"] = to_numpy_array(self.frame_expand.weight)
        if self.frame_contract is not None:
            variables["frame_contract.weight"] = to_numpy_array(
                self.frame_contract.weight
            )
        if self.residual_scale is not None:
            variables["residual_scale"] = to_numpy_array(self.residual_scale)
        return {
            "@class": "SO3GridNet",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.projector.mmax,
                "kmax": self.kmax,
                "channels": self.channels,
                "n_focus": self.n_focus,
                "mode": self.mode,
                "op_type": self.op_type,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "layout": self.layout,
                "lebedev_precision": self.lebedev_precision,
                "coefficient_layout": self.projector.coefficient_layout,
                "grid_branches": self.grid_branches,
                "residual_scale_init": self.residual_scale_init,
                "mlp_bias": self.mlp_bias,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": variables,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SO3GridNet:
        """Deserialize an SO3GridNet from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SO3GridNet":
            raise ValueError(f"Invalid class for SO3GridNet: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(
            lmax=int(config["lmax"]),
            mmax=int(config["mmax"]),
            kmax=int(config["kmax"]),
            channels=int(config["channels"]),
            n_focus=int(config["n_focus"]),
            mode=str(config["mode"]),
            op_type=str(config["op_type"]),
            precision=str(config["precision"]),
            layout=str(config["layout"]),
            lebedev_precision=int(config["lebedev_precision"]),
            coefficient_layout=str(config["coefficient_layout"]),
            grid_branches=int(config["grid_branches"]),
            residual_scale_init=config.get("residual_scale_init"),
            mlp_bias=bool(config["mlp_bias"]),
            trainable=bool(config["trainable"]),
            seed=config.get("seed"),
        )
        obj._load_variables(variables)
        return obj
