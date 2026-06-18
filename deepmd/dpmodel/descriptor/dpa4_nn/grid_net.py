# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Grid-space nonlinearities for DPA4/SeZM coefficient tensors.

This module is the dpmodel port of
``deepmd.pt.model.descriptor.sezm_nn.grid_net``. A grid net receives
coefficient tensors, converts them to quadrature values, applies one
point-wise grid operation, and projects the result back to coefficients. The
public shapes are:

* ``mode='self'``: one input ``(N, D, F, 2*C)`` or ``(N, F, D, 2*C)``.
* ``mode='cross'``: separate query/context inputs each with ``C`` channels.
* grid values: ``(N, G, F, C)`` after S2 or SO(3) projection.

Ported names: ``BaseGridNet`` (``mode`` 'self'/'cross'; ``op_type``
'glu'/'mlp'/'branch'; ``layout`` 'ndfc'/'nfdc'/'flat'; ``residual_scale_init``;
general ``n_frames``), ``S2GridNet``, ``GridProduct``, ``GridMLP``,
``GridBranch``.

``BaseGridNet`` mirrors the current pt ``BaseGridNet`` for arbitrary
``n_frames`` (the ``_to_grid``/``_from_grid`` frame-axis contraction). The S2
path (``n_frames == 1``, ``mode='self'``) keeps a dedicated fast branch that is
byte-identical to the previous S2-only specialization. The SO(3) frame
machinery (``SO3GridNet``, ``FrameContract``, ``FrameExpand``) is ported here;
``SO3GridNet`` builds an ``SO3GridProjector`` (``n_frames = 2 * kmax + 1``) and,
in ``mode='cross'``, plugs ``FrameExpand``/``FrameContract`` into the
``BaseGridNet`` ``frame_expand``/``frame_contract`` seams (kept ``None`` for S2).

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

from typing import (
    TYPE_CHECKING,
    Any,
)

import math

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

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


def _softmax_last_axis(x: Any) -> Any:
    """Numerically stable softmax on the last axis (matches torch.softmax)."""
    xp = array_api_compat.array_namespace(x)
    e_x = xp.exp(x - xp.max(x, axis=-1, keepdims=True))
    return e_x / xp.sum(e_x, axis=-1, keepdims=True)


def _project_frames(coeff: Any, proj: ChannelLinear, n_frames: int) -> Any:
    """
    Apply a channel-only linear map to each Wigner-D frame independently.

    Parameters
    ----------
    coeff
        Frame-packed coefficients with shape ``(N, D, F, n_frames * C_in)``.
    proj : ChannelLinear
        Linear map acting on the per-frame channel axis (``C_in -> C_out``).
    n_frames : int
        Number of Wigner-D frames packed along the trailing axis.

    Returns
    -------
    Array
        Projected coefficients with shape ``(N, D, F, n_frames * C_out)``.

    Notes
    -----
    ``to_grid`` and ``from_grid`` are frame-wise linear and commute with any
    channel map, so applying the map at coefficient resolution here is identical
    to applying it on the grid field while touching ``n_frames``-fold fewer rows
    than the ``G``-point grid.
    """
    xp = array_api_compat.array_namespace(coeff)
    n_batch, coeff_dim, n_focus, _ = coeff.shape
    projected = proj(xp.reshape(coeff, (n_batch, coeff_dim, n_focus, n_frames, -1)))
    return xp.reshape(projected, (n_batch, coeff_dim, n_focus, -1))


def _build_frame_degree_index(
    *,
    lmax: int,
    mmax: int,
    coefficient_layout: str,
) -> np.ndarray:
    """Build the per-coefficient degree index used by the frame channel mixers.

    The pt version's ``device`` parameter is dropped: the output is a static
    ``np.int64`` table mapping each coefficient row to its degree ``l`` for the
    packed / truncated / m-major layouts.
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


class _FrameMixer(NativeOP):
    """Shared base for the per-degree frame channel mixers.

    The pt ``FrameContract`` / ``FrameExpand`` are ``nn.Module`` wrappers around
    a per-degree weight of shape ``(lmax + 1, in_ch, out_ch)`` selected by a
    static degree-index buffer; they realise an
    ``einsum("ndfi,dio->ndfo", coeff, weight[degree_index])``. ``mode='self'``
    S2 grid nets have ``n_frames == 1`` and never construct these; they back the
    SO(3) cross-mode grid products only. Subclasses set ``in_channels`` /
    ``out_channels`` and the init ``bound`` (matching the pt weight init).
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
        # static np.int64 table; rebuilt from config on deserialize (the pt
        # degree_index is a non-persistent buffer, not in the state dict)
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
        a broadcast batched matmul: the gathered weight ``(D, i, o)`` broadcasts
        over the leading frame batch dim of ``coeff``.
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

        The pt ``FrameContract`` has no ``serialize()``; the ``@variables`` key
        (``weight``) matches the pt ``state_dict`` key name. ``degree_index`` is
        a non-persistent buffer in pt and is rebuilt from the config.
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

        The pt ``FrameExpand`` has no ``serialize()``; the ``@variables`` key
        (``weight``) matches the pt ``state_dict`` key name. ``degree_index`` is
        a non-persistent buffer in pt and is rebuilt from the config.
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


class GridProduct(NativeOP):
    """Parameter-free quadratic grid product ``u(g) * v(g)``."""

    def call(
        self,
        left: Any,
        right: Any,
        scalar_pair: Any,
        *,
        to_grid: Callable[[Any], Any],
        from_grid: Callable[[Any], Any],
    ) -> Any:
        """
        Combine two coefficient operands by a point-wise grid product.

        Parameters
        ----------
        left, right
            Coefficient operands with shape ``(N, D, F, C)``.
        scalar_pair
            Invariant routing signal; unused on this path.
        to_grid, from_grid
            Coefficient/grid projectors supplied by the owning grid net.
        """
        return from_grid(to_grid(left) * to_grid(right))

    def serialize(self) -> dict[str, Any]:
        """Serialize the parameter-free grid product to a dict."""
        return {
            "@class": "GridProduct",
            "@version": 1,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> GridProduct:
        """Deserialize a GridProduct from a dict."""
        data = data.copy()
        data_cls = data.pop("@class", "GridProduct")
        if data_cls != "GridProduct":
            raise ValueError(f"Invalid class for GridProduct: {data_cls}")
        check_version_compatibility(int(data.pop("@version", 1)), 1, 1)
        return cls()


class GridMLP(NativeOP):
    """
    Polynomial point-wise MLP applied independently at every grid point.

    Frame-aware port of the pt ``GridMLP``: operands are packed as
    ``(N, D, F, n_frames * C)`` and every channel projection is applied to each
    Wigner-D frame independently through :func:`_project_frames`. The S2 case
    (``n_frames == 1``) reduces to a plain per-channel projection, byte-for-byte
    identical to the previous S2-only specialization.

    Parameters
    ----------
    channels : int
        Number of channels per grid point (per frame).
    mode : str
        Pairing mode, either ``"self"`` or ``"cross"``.
    n_frames : int
        Number of Wigner-D frames packed along the trailing channel axis.
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
        mode: str,
        n_frames: int,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        self.channels = int(channels)
        self.mode = str(mode).lower()
        if self.mode not in {"self", "cross"}:
            raise ValueError("`mode` must be either 'self' or 'cross'")
        self.n_frames = int(n_frames)
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

    def call(
        self,
        left: Any,
        right: Any,
        scalar_pair: Any = None,
        *,
        to_grid: Callable[[Any], Any],
        from_grid: Callable[[Any], Any],
    ) -> Any:
        """
        Apply the polynomial point-wise MLP on coefficient operands.

        In self mode both projections see the per-frame concatenation of the
        two operands and can form self and cross quadratic channel terms. In
        cross mode the query and context roles stay separate:
        ``(W_q query) * (W_c context)``.

        Parameters
        ----------
        left, right
            Coefficient operands with shape ``(N, D, F, n_frames * C)``.
        scalar_pair
            Invariant routing signal; unused on this path.
        to_grid, from_grid
            Coefficient/grid projectors supplied by the owning grid net.
        """
        # === Step 1. Channel projections at coefficient resolution ===
        if self.mode == "self":
            xp = array_api_compat.array_namespace(left)
            left_shape = tuple(left.shape)
            shape = (*left_shape[:-1], self.n_frames, -1)
            fused = xp.concat(
                [xp.reshape(left, shape), xp.reshape(right, shape)], axis=-1
            )  # per-frame concat -> (N, D, F, n_frames, 2C)
            fused = xp.reshape(fused, (*left_shape[:-1], -1))  # (N, D, F, n_frames*2C)
            left = _project_frames(fused, self.left_proj, self.n_frames)
            right = _project_frames(fused, self.right_proj, self.n_frames)
        else:
            left = _project_frames(left, self.left_proj, self.n_frames)
            right = _project_frames(right, self.right_proj, self.n_frames)

        # === Step 2. Quadratic product on the grid, projected back ===
        coeff = from_grid(to_grid(left) * to_grid(right))
        return _project_frames(coeff, self.out_proj, self.n_frames)

    def serialize(self) -> dict[str, Any]:
        """Serialize the GridMLP to a dict.

        The pt ``GridMLP`` has no ``serialize()``; the ``@variables`` keys here
        match the pt ``state_dict`` key names.
        """
        return {
            "@class": "GridMLP",
            "@version": 1,
            "config": {
                "channels": self.channels,
                "mode": self.mode,
                "n_frames": self.n_frames,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": {
                "left_proj.weight": to_numpy_array(self.left_proj.weight),
                "right_proj.weight": to_numpy_array(self.right_proj.weight),
                "out_proj.weight": to_numpy_array(self.out_proj.weight),
            },
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
            n_frames=int(config["n_frames"]),
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

    Frame-aware port of the pt ``GridBranch``: operands are packed as
    ``(N, D, F, n_frames * C)`` and every channel projection is applied to each
    Wigner-D frame independently through :func:`_project_frames`. The S2 case
    (``n_frames == 1``) reduces to a plain per-channel projection, byte-for-byte
    identical to the previous S2-only specialization.

    Parameters
    ----------
    channels : int
        Number of channels per grid point (per frame).
    n_branches : int
        Number of scalar-routed product branches.
    n_frames : int
        Number of Wigner-D frames packed along the trailing channel axis.
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
        n_frames: int,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        self.channels = int(channels)
        self.n_branches = int(n_branches)
        if self.n_branches < 1:
            raise ValueError("`n_branches` must be positive")
        self.n_frames = int(n_frames)
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
        left: Any,
        right: Any,
        scalar_pair: Any,
        *,
        to_grid: Callable[[Any], Any],
        from_grid: Callable[[Any], Any],
    ) -> Any:
        """
        Apply scalar-routed grid branch mixing on coefficient operands.

        The channel maps are applied at coefficient resolution (per Wigner-D
        frame via :func:`_project_frames`) and the grid transform is deferred to
        the injected ``to_grid``/``from_grid`` callables, matching the pt
        ``GridBranch``. The router operates on invariant scalars only, so the
        softmax is frame-independent.

        Parameters
        ----------
        left, right
            Coefficient operands with shape ``(N, D, F, n_frames * C)``.
        scalar_pair
            Invariant router source with shape ``(N, F, 2*C)``.
        to_grid, from_grid
            Coefficient/grid projectors supplied by the owning grid net.
        """
        xp = array_api_compat.array_namespace(left)
        # === Step 1. Branch channel projections at coefficient resolution ===
        left = _project_frames(left, self.left_proj, self.n_frames)
        right = _project_frames(right, self.right_proj, self.n_frames)

        # === Step 2. Quadratic branches on the grid, routed by scalars ===
        value = to_grid(left) * to_grid(right)  # (N, G, F, N_branches * C)
        n_batch, n_grid, n_focus, _ = value.shape
        value = xp.reshape(
            value,
            (n_batch, n_grid, n_focus, self.n_branches, self.channels),
        )  # (N, G, F, N_branches, C)
        router = _softmax_last_axis(self.router(scalar_pair))  # (N, F, N_branches)
        # einsum "ngfhc,nfh->ngfc" as a broadcast sum over the branch axis
        out = xp.sum(value * router[:, None, :, :, None], axis=3)  # (N, G, F, C)

        # === Step 3. Project back to coefficients and mix output channels ===
        return _project_frames(from_grid(out), self.out_proj, self.n_frames)

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
                "n_frames": self.n_frames,
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
            n_frames=int(config["n_frames"]),
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


class BaseGridNet(NativeOP):
    """
    Shared implementation for S2 and SO(3) grid nets.

    ``mode='self'`` expects one input whose last channel axis contains two
    branches; the first half supplies the SwiGLU gates of the scalar path.
    ``mode='cross'`` expects separate query and context inputs.

    Mirrors the current pt ``BaseGridNet``: ``mode`` ('self'/'cross'),
    ``layout`` ('ndfc'/'nfdc'/'flat'), ``residual_scale_init`` and arbitrary
    ``n_frames`` are all supported. The S2 (``n_frames == 1``) path keeps a
    dedicated fast branch in ``_to_grid``/``_from_grid``/``_apply_scalar_path``
    that is byte-identical to the previous S2-only specialization. The SO(3)
    frame machinery (``frame_expand``/``frame_contract``) is built by the
    not-yet-ported ``SO3GridNet``; the seams here stay ``None`` for S2.
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
        frame_expand: NativeOP | None = None,
        frame_contract: NativeOP | None = None,
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
        # ``frame_expand``/``frame_contract`` are the SO(3) frame machinery
        # (built only by ``SO3GridNet`` in cross mode). They stay ``None`` for
        # S2 (``n_frames == 1``); the seam below lets a later SO(3) port plug
        # them in without touching the shared forward.
        self.frame_expand = frame_expand
        self.frame_contract = frame_contract
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
        self.frame_zero_index = int(getattr(projector, "frame_zero_index", 0))
        self.residual_scale_init = residual_scale_init
        if residual_scale_init is None:
            self.residual_scale: np.ndarray | None = None
        else:
            prec = PRECISION_DICT[self.precision.lower()]
            self.residual_scale = np.ones(
                (self.n_focus, self.output_channels), dtype=prec
            ) * float(residual_scale_init)

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
            self.grid_op: NativeOP = GridMLP(
                channels=self.channels,
                mode=self.mode,
                n_frames=self.n_frames,
                precision=self.precision,
                trainable=trainable,
                seed=child_seed(seed, 1),
            )
        elif self.op_type == "branch":
            self.grid_op = GridBranch(
                channels=self.channels,
                n_branches=grid_branches,
                n_frames=self.n_frames,
                precision=self.precision,
                trainable=trainable,
                seed=child_seed(seed, 1),
            )
        else:
            self.grid_op = GridProduct()

    def call(self, query: Any, context: Any = None) -> Any:
        """Apply the configured grid net and restore the input layout."""
        xp = array_api_compat.array_namespace(query)
        input_dtype = query.dtype
        compute_dtype = get_xp_precision(xp, self.precision)
        query_ndfc, shape_info = self._to_ndfc(query)
        left, right, scalar_pair = self._prepare_pair(
            query_ndfc, context, compute_dtype
        )
        coeff_out = self._apply_grid_op(left, right, scalar_pair, compute_dtype)
        coeff_out = self._apply_scalar_path(coeff_out, scalar_pair)
        coeff_out = self._contract_frames(coeff_out)
        coeff_out = self._apply_residual_scale(coeff_out)
        if coeff_out.dtype != input_dtype:
            coeff_out = xp.astype(coeff_out, input_dtype)
        return self._restore_layout(coeff_out, shape_info)

    def _prepare_pair(
        self, query: Any, context: Any, compute_dtype: Any
    ) -> tuple[Any, Any, Any]:
        if self.mode == "self":
            return self._prepare_self_pair(query, compute_dtype)
        return self._prepare_cross_pair(query, context, compute_dtype)

    def _prepare_self_pair(
        self, query: Any, compute_dtype: Any
    ) -> tuple[Any, Any, Any]:
        left, right = self._split_self_query(query)
        scalar_pair = self._make_scalar_pair(left, right, compute_dtype)
        return left, right, scalar_pair

    def _prepare_cross_pair(
        self, query: Any, context: Any, compute_dtype: Any
    ) -> tuple[Any, Any, Any]:
        if context is None:
            raise ValueError("`context` is required when `mode='cross'`")
        context_ndfc, _ = self._to_ndfc(context)
        self._check_last_dim(query, self.context_channels, "query")
        self._check_last_dim(context_ndfc, self.context_channels, "context")
        if self.frame_expand is None:
            scalar_pair = self._make_scalar_pair(query, context_ndfc, compute_dtype)
            return query, context_ndfc, scalar_pair
        # SO(3) frame-expansion seam (built only by a later SO3GridNet port):
        # the scalar pair is read from the d=0 slice before expansion, then
        # both operands are lifted to the frame-packed width.
        xp = array_api_compat.array_namespace(query)
        scalar_pair = xp.concat([query[:, 0, :, :], context_ndfc[:, 0, :, :]], axis=-1)
        if scalar_pair.dtype != compute_dtype:
            scalar_pair = xp.astype(scalar_pair, compute_dtype)
        return (
            self.frame_expand(query),
            self.frame_expand(context_ndfc),
            scalar_pair,
        )

    def _contract_frames(self, coeff: Any) -> Any:
        if self.frame_contract is None:
            return coeff
        return self.frame_contract(coeff)

    def _apply_residual_scale(self, coeff: Any) -> Any:
        if self.residual_scale is None:
            return coeff
        xp = array_api_compat.array_namespace(coeff)
        residual_scale = xp_asarray_nodetach(
            xp, self.residual_scale[...], device=array_api_compat.device(coeff)
        )
        if residual_scale.dtype != coeff.dtype:
            residual_scale = xp.astype(residual_scale, coeff.dtype)
        return coeff * xp.reshape(
            residual_scale, (1, 1, self.n_focus, self.output_channels)
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
        return self.grid_op(
            left,
            right,
            scalar_pair,
            to_grid=self._to_grid,
            from_grid=self._from_grid,
        )

    def _apply_scalar_path(self, coeff: Any, scalar_pair: Any) -> Any:
        xp = array_api_compat.array_namespace(coeff)
        scalar_out = self.scalar_act(scalar_pair)  # (N, F, C)
        scalar_gate = xp_sigmoid(self.scalar_gate(scalar_pair))  # (N, F, C)
        if self.n_frames == 1:
            # Fast S2 path (byte-identical to the previous specialization).
            coeff = coeff * scalar_gate[:, None, :, :]
            # gradient-safe equivalent of the pt in-place
            # ``coeff_view[:, 0, :, 0, :].add_(scalar_out)`` (n_frames == 1)
            head = coeff[:, :1, :, :] + scalar_out[:, None, :, :]
            return xp.concat([head, coeff[:, 1:, :, :]], axis=1)
        # General frame-packed path mirroring the pt
        # ``coeff_view = coeff.reshape(N, D, F, K, C)`` followed by a gated
        # multiply and an in-place add into ``[:, 0, :, frame_zero_index, :]``.
        n_batch, coeff_dim, n_focus, _ = coeff.shape
        coeff_view = xp.reshape(
            coeff, (n_batch, coeff_dim, n_focus, self.n_frames, self.channels)
        )
        coeff_view = coeff_view * scalar_gate[:, None, :, None, :]
        # gradient-safe in-place add into the d=0, frame_zero_index slice
        fzi = self.frame_zero_index
        head = coeff_view[:, :1, :, :, :]  # (N, 1, F, K, C)
        pre = head[:, :, :, :fzi, :]
        mid = head[:, :, :, fzi : fzi + 1, :] + scalar_out[:, None, :, None, :]
        post = head[:, :, :, fzi + 1 :, :]
        head = xp.concat([pre, mid, post], axis=3)
        coeff_view = xp.concat([head, coeff_view[:, 1:, :, :, :]], axis=1)
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
        # (N, D, F, K*C) -> the (l=0, m=0) scalar slice (N, F, C).
        if self.n_frames == 1:
            return coeff[:, 0, :, :]
        xp = array_api_compat.array_namespace(coeff)
        n_batch, coeff_dim, n_focus, _ = coeff.shape
        coeff_view = xp.reshape(
            coeff, (n_batch, coeff_dim, n_focus, self.n_frames, self.channels)
        )
        return coeff_view[:, 0, :, self.frame_zero_index, :]

    def _to_grid(self, coeff: Any) -> Any:
        xp = array_api_compat.array_namespace(coeff)
        to_grid_mat = xp_asarray_nodetach(
            xp, self.projector.to_grid_mat[...], device=array_api_compat.device(coeff)
        )
        if to_grid_mat.dtype != coeff.dtype:
            to_grid_mat = xp.astype(to_grid_mat, coeff.dtype)
        if self.n_frames == 1:
            # einsum "gd,ndfc->ngfc" as a broadcast batched matmul. The per-point
            # channel width is inferred so the projector also serves widened
            # operands (e.g. a branch hidden width ``n_branches * C``).
            n_batch, coeff_dim, n_focus, n_channels = coeff.shape
            flat = xp.reshape(coeff, (n_batch, coeff_dim, n_focus * n_channels))
            out = xp.matmul(to_grid_mat[None, ...], flat)  # (N, G, F*C)
            return xp.reshape(
                out, (n_batch, self.projector.grid_size, n_focus, n_channels)
            )
        # General SO(3) frame-packed path mirroring the pt
        # ``einsum("gdk,ndfkc->ngfc", to_grid.reshape(G, D, K), coeff_view)``.
        # ``to_grid_mat`` columns are ordered (d outer, k inner), so the operand
        # is permuted to the matching ``(d, k)`` flattening before the matmul.
        n_batch, coeff_dim, n_focus, last = coeff.shape
        n_channels = last // self.n_frames
        coeff_view = xp.reshape(
            coeff, (n_batch, coeff_dim, n_focus, self.n_frames, n_channels)
        )
        coeff_dk = xp.permute_dims(coeff_view, (0, 1, 3, 2, 4))  # (N, D, K, F, C)
        coeff_flat = xp.reshape(
            coeff_dk, (n_batch, coeff_dim * self.n_frames, n_focus * n_channels)
        )
        out = xp.matmul(to_grid_mat[None, ...], coeff_flat)  # (N, G, F*C)
        return xp.reshape(out, (n_batch, self.projector.grid_size, n_focus, n_channels))

    def _from_grid(self, grid: Any) -> Any:
        xp = array_api_compat.array_namespace(grid)
        from_grid_mat = xp_asarray_nodetach(
            xp, self.projector.from_grid_mat[...], device=array_api_compat.device(grid)
        )
        if from_grid_mat.dtype != grid.dtype:
            from_grid_mat = xp.astype(from_grid_mat, grid.dtype)
        if self.n_frames == 1:
            # einsum "dg,ngfc->ndfc" as a broadcast batched matmul. The channel
            # width is inferred to match the (possibly widened) grid field.
            n_batch, n_grid, n_focus, n_channels = grid.shape
            coeff_dim = self.projector.coeff_dim
            flat = xp.reshape(grid, (n_batch, n_grid, n_focus * n_channels))
            out = xp.matmul(from_grid_mat[None, ...], flat)  # (N, D, F*C)
            return xp.reshape(out, (n_batch, coeff_dim, n_focus, n_channels))
        # General SO(3) frame-packed path mirroring the pt
        # ``einsum("dkg,ngfc->ndfkc", from_grid.reshape(D, K, G), grid)`` then a
        # reshape to ``(N, D, F, K*C)``. ``from_grid_mat`` rows are ordered
        # (d outer, k inner); the matmul output is reshaped/permuted to match.
        n_batch, n_grid, n_focus, n_channels = grid.shape
        coeff_dim = self.projector.coeff_dim // self.n_frames
        flat = xp.reshape(grid, (n_batch, n_grid, n_focus * n_channels))
        out = xp.matmul(from_grid_mat[None, ...], flat)  # (N, D*K, F*C)
        out = xp.reshape(out, (n_batch, coeff_dim, self.n_frames, n_focus, n_channels))
        out = xp.permute_dims(out, (0, 1, 3, 2, 4))  # (N, D, F, K, C)
        return xp.reshape(
            out, (n_batch, coeff_dim, n_focus, self.n_frames * n_channels)
        )

    def _to_ndfc(self, value: Any) -> tuple[Any, tuple[int, ...]]:
        shape_info = tuple(value.shape)
        if self.layout == "ndfc":
            return value, shape_info
        if self.layout == "nfdc":
            # (N, F, D, C) -> (N, D, F, C)
            xp = array_api_compat.array_namespace(value)
            return xp.permute_dims(value, (0, 2, 1, 3)), shape_info
        # "flat": (N, D, F*k*C) -> (N, D, F, k*C)
        xp = array_api_compat.array_namespace(value)
        n_batch, coeff_dim, _ = value.shape
        return (
            xp.reshape(value, (n_batch, coeff_dim, self.n_focus, -1)),
            shape_info,
        )

    def _restore_layout(self, value: Any, shape_info: tuple[int, ...]) -> Any:
        if self.layout == "ndfc":
            return value
        xp = array_api_compat.array_namespace(value)
        if self.layout == "nfdc":
            return xp.permute_dims(value, (0, 2, 1, 3))
        # "flat": (N, D, F, k*C) -> (N, D, F*k*C)
        n_batch, coeff_dim, _ = shape_info
        return xp.reshape(value, (n_batch, coeff_dim, -1))

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
        Pairing mode; ``"self"`` or ``"cross"``.
    op_type : str
        Point-wise grid operation; ``"glu"``, ``"mlp"`` or ``"branch"``.
    precision : str
        Parameter precision.
    layout : str
        Tensor layout convention: ``"ndfc"``, ``"nfdc"`` or ``"flat"``
        (``"flat"`` is cross-only).
    grid_resolution_list : list[int] | None
        Lebedev ``[precision, n_points]`` pair; resolved automatically if None.
    coefficient_layout : str
        ``"packed"`` or ``"m_major"`` coefficient ordering.
    grid_method : str
        S2 quadrature backend; only ``"lebedev"`` is ported.
    grid_branches : int
        Number of scalar-routed branches when ``op_type='branch'``.
    residual_scale_init : float | None
        Initial value of the per-(focus, channel) residual scale; ``None``
        disables the residual scale.
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
        if self.op_type in {"mlp", "branch"}:
            grid_op_data = self.grid_op.serialize()["@variables"]
            for key, value in grid_op_data.items():
                variables[f"grid_op.{key}"] = value
        if self.residual_scale is not None:
            # pt state-dict key name for the (n_focus, output_channels) parameter
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
        if obj.residual_scale is not None:
            residual_scale = np.asarray(variables["residual_scale"], dtype=prec)
            if residual_scale.shape != obj.residual_scale.shape:
                raise ValueError(
                    f"residual_scale shape {residual_scale.shape} does not match "
                    f"the expected shape {obj.residual_scale.shape}"
                )
            obj.residual_scale = residual_scale
        if obj.op_type in {"mlp", "branch"}:
            obj.grid_op._load_variables(
                {
                    key[len("grid_op.") :]: value
                    for key, value in variables.items()
                    if key.startswith("grid_op.")
                }
            )
        return obj


class SO3GridNet(BaseGridNet):
    """Grid net using a Wigner-D SO(3) projector with frame indices.

    dpmodel port of the current pt
    ``deepmd.pt.model.descriptor.sezm_nn.grid_net.SO3GridNet``. Unlike
    ``S2GridNet`` (``n_frames == 1``), the SO(3) projector packs
    ``n_frames = 2 * kmax + 1`` Wigner-D frames along the trailing channel axis,
    exercising the general ``n_frames > 1`` ``_to_grid``/``_from_grid`` paths of
    ``BaseGridNet``. In ``mode='cross'`` it additionally builds the per-degree
    :class:`FrameExpand` / :class:`FrameContract` channel mixers and plugs them
    into the ``BaseGridNet`` ``frame_expand``/``frame_contract`` seam: the query
    and context are expanded ``C -> n_frames * C`` before the grid product and
    contracted ``n_frames * C -> C`` afterwards.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.
    mmax : int | None
        Maximum order kept in the coefficient layout. If None, use ``lmax``.
    kmax : int
        Frame band width; ``n_frames = 2 * kmax + 1`` Wigner-D frames.
    channels : int
        Number of channels per (l, m) coefficient (per frame).
    n_focus : int
        Number of focus streams.
    mode : str
        Pairing mode; ``"self"`` or ``"cross"``.
    op_type : str
        Point-wise grid operation; ``"glu"``, ``"mlp"`` or ``"branch"``.
    precision : str
        Parameter precision.
    layout : str
        Tensor layout convention: ``"ndfc"``, ``"nfdc"`` or ``"flat"``
        (``"flat"`` is cross-only).
    lebedev_precision : int | None
        Lebedev algebraic precision; resolved automatically if None.
    coefficient_layout : str
        ``"packed"`` or ``"m_major"`` coefficient ordering.
    grid_branches : int
        Number of scalar-routed branches when ``op_type='branch'``.
    residual_scale_init : float | None
        Initial value of the per-(focus, channel) residual scale; ``None``
        disables the residual scale.
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
            # pt builds the frame mixers with child_seed(seed, 4)/(seed, 5);
            # ``BaseGridNet`` uses child_seed(seed, 0)/(seed, 1) for scalar_gate
            # /grid_op, so these branches never collide.
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
        here match the pt ``state_dict`` key names (``scalar_gate.weight``,
        ``grid_op.*``, ``frame_expand.weight``, ``frame_contract.weight``,
        ``residual_scale``) so pt state-dict fragments load directly. The
        projector matrices are non-persistent buffers in pt and are rebuilt
        from the nested projector config on deserialization.
        """
        variables = {"scalar_gate.weight": to_numpy_array(self.scalar_gate.weight)}
        if self.mlp_bias:
            variables["scalar_gate.bias"] = to_numpy_array(self.scalar_gate.bias)
        if self.op_type in {"mlp", "branch"}:
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
                "channels": self.channels,
                "n_focus": self.n_focus,
                "mode": self.mode,
                "op_type": self.op_type,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "layout": self.layout,
                "grid_branches": self.grid_branches,
                "residual_scale_init": self.residual_scale_init,
                "mlp_bias": self.mlp_bias,
                "trainable": self.trainable,
                "seed": None,
                "projector": self.projector.serialize(),
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
        projector_config = config["projector"]["config"]
        obj = cls(
            lmax=int(projector_config["lmax"]),
            mmax=int(projector_config["mmax"]),
            kmax=int(projector_config["kmax"]),
            channels=int(config["channels"]),
            n_focus=int(config["n_focus"]),
            mode=str(config["mode"]),
            op_type=str(config["op_type"]),
            precision=str(config["precision"]),
            layout=str(config["layout"]),
            lebedev_precision=int(projector_config["lebedev_precision"]),
            coefficient_layout=str(projector_config["coefficient_layout"]),
            grid_branches=int(config["grid_branches"]),
            residual_scale_init=config.get("residual_scale_init"),
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
        if obj.residual_scale is not None:
            residual_scale = np.asarray(variables["residual_scale"], dtype=prec)
            if residual_scale.shape != obj.residual_scale.shape:
                raise ValueError(
                    f"residual_scale shape {residual_scale.shape} does not match "
                    f"the expected shape {obj.residual_scale.shape}"
                )
            obj.residual_scale = residual_scale
        if obj.frame_expand is not None:
            expand_weight = np.asarray(variables["frame_expand.weight"], dtype=prec)
            if expand_weight.shape != obj.frame_expand.weight.shape:
                raise ValueError(
                    f"frame_expand.weight shape {expand_weight.shape} does not "
                    f"match the expected shape {obj.frame_expand.weight.shape}"
                )
            obj.frame_expand.weight = expand_weight
        if obj.frame_contract is not None:
            contract_weight = np.asarray(variables["frame_contract.weight"], dtype=prec)
            if contract_weight.shape != obj.frame_contract.weight.shape:
                raise ValueError(
                    f"frame_contract.weight shape {contract_weight.shape} does "
                    f"not match the expected shape {obj.frame_contract.weight.shape}"
                )
            obj.frame_contract.weight = contract_weight
        if obj.op_type in {"mlp", "branch"}:
            obj.grid_op._load_variables(
                {
                    key[len("grid_op.") :]: value
                    for key, value in variables.items()
                    if key.startswith("grid_op.")
                }
            )
        return obj
