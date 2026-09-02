# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Grid-space nonlinearities for DPA4/SeZM coefficient tensors.

A grid net receives coefficient tensors, converts them to quadrature values,
applies one point-wise grid operation, and projects the result back to
coefficients.  The public shapes are:

* ``mode='self'``: one input ``(N, D, F, 2*C)`` or ``(N, F, D, 2*C)``.
* ``mode='cross'``: query and context inputs with separate ``C`` channels.
* grid values: ``(N, G, F, C)`` after S2 or SO3 projection.

The only nonlinear scalar functions are SwiGLU, sigmoid, and softmax on the
``l=0`` scalar branch.  Non-scalar grid values use channel-linear maps and
point-wise products so equivariance is governed by the projector quadrature.

This module is the dpmodel (array-API) port of
``deepmd.pt.model.descriptor.sezm_nn.grid_net``.
"""

from __future__ import (
    annotations,
)

import math
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
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
    xp_einsum,
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

GridNetLayout = Literal["ndfc", "nfdc", "fndc", "flat"]
GridNetMode = Literal["self", "cross"]
GridNetOp = Literal["glu", "mlp", "branch"]


def _build_frame_degree_index(
    *,
    lmax: int,
    mmax: int,
    coefficient_layout: str,
) -> np.ndarray:
    """Build the per-coefficient degree index used by frame channel mixers."""
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


def _build_so3_scalar_product_weight(
    projector: SO3GridProjector,
) -> np.ndarray:
    """Build the Haar inner-product weight for every ``(l, m, k)`` slot.

    Real Wigner-D coefficients obey
    ``integral D_lmk D_l'm'k' dR = delta_ll' delta_mm' delta_kk' / (2*l+1)``.
    Frame slots with ``abs(k) > l`` are structural zeros in the regular SeZM
    layout and therefore receive zero weight.
    """
    degree_index = _build_frame_degree_index(
        lmax=projector.lmax,
        mmax=projector.mmax,
        coefficient_layout=projector.coefficient_layout,
    ).reshape(-1, 1)
    frame_values = projector.frame_values.reshape(1, -1)
    valid_frame = np.abs(frame_values) <= degree_index
    dtype = PRECISION_DICT[projector.precision.lower()]
    degree_weight = (1.0 / (2 * degree_index + 1)).astype(dtype)
    return valid_frame.astype(dtype) * degree_weight


def _project_frames(coeff: Any, proj: ChannelLinear, n_frames: int) -> Any:
    """
    Apply a channel-only linear map to each Wigner-D frame independently.

    Parameters
    ----------
    coeff : Array
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


def _project_pair_in_one_transform(
    left: Any,
    right: Any,
    *,
    n_frames: int,
    to_grid: Callable[[Any], Any],
) -> tuple[Any, Any]:
    """Project two equally shaped coefficient operands in one linear transform."""
    xp = array_api_compat.array_namespace(left, right)
    n_batch, coeff_dim, n_focus, _ = left.shape
    frame_shape = (n_batch, coeff_dim, n_focus, n_frames, -1)
    pair = xp.reshape(
        xp.concat(
            [xp.reshape(left, frame_shape), xp.reshape(right, frame_shape)],
            axis=-1,
        ),
        (n_batch, coeff_dim, n_focus, -1),
    )
    pair_grid = to_grid(pair)
    split = pair_grid.shape[-1] // 2
    return pair_grid[..., :split], pair_grid[..., split:]


def _project_pair(
    left: Any,
    right: Any,
    *,
    to_grid: Callable[[Any], Any],
    project_pair: Callable[[Any, Any], tuple[Any, Any]] | None,
) -> tuple[Any, Any]:
    """Project two operands through the selected projector composition."""
    if project_pair is not None:
        return project_pair(left, right)
    return to_grid(left), to_grid(right)


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
        pair_grid: Callable[[Any, Any], Any | None] | None = None,
        project_pair: Callable[[Any, Any], tuple[Any, Any]] | None = None,
        scalar_product: Callable[[Any, Any], Any] | None = None,
    ) -> Any:
        """
        Combine two coefficient operands by a point-wise grid product.

        Parameters
        ----------
        left, right : Array
            Coefficient operands with shape ``(N, D, F, n_frames * C)``.
        scalar_pair : Array
            Invariant routing signal; unused on this path.
        to_grid, from_grid : Callable
            Coefficient/grid projectors supplied by the owning grid net.
        pair_grid, project_pair, scalar_product : Callable, optional
            Optional fused full composition, paired forward projection, and
            direct scalar coefficient contraction.

        Returns
        -------
        Array
            Coefficient result. A direct scalar contraction has shape
            ``(N, 1, F, C)``; other paths retain ``n_frames * C`` channels.
        """
        if scalar_product is not None:
            return scalar_product(left, right)
        fused = pair_grid(left, right) if pair_grid is not None else None
        if fused is not None:
            return fused
        left_grid, right_grid = _project_pair(
            left,
            right,
            to_grid=to_grid,
            project_pair=project_pair,
        )
        return from_grid(left_grid * right_grid)


class GridMLP(NativeOP):
    """Polynomial point-wise MLP applied independently at every grid point."""

    def __init__(
        self,
        *,
        channels: int,
        mode: GridNetMode,
        n_frames: int,
        precision: str = DEFAULT_PRECISION,
        trainable: bool,
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
        scalar_pair: Any,
        *,
        to_grid: Callable[[Any], Any],
        from_grid: Callable[[Any], Any],
        pair_grid: Callable[[Any, Any], Any | None] | None = None,
        project_pair: Callable[[Any, Any], tuple[Any, Any]] | None = None,
        scalar_product: Callable[[Any, Any], Any] | None = None,
    ) -> Any:
        """
        Apply the polynomial point-wise MLP on coefficient operands.

        In self mode, both projections see the per-frame concatenation of the
        two operands and can form self and cross quadratic channel terms.  In
        cross mode the query and context roles stay separate:
        ``(W_q query) * (W_c context)``.

        Parameters
        ----------
        left, right : Array
            Coefficient operands with shape ``(N, D, F, n_frames * C)``.
        scalar_pair : Array
            Invariant routing signal; unused on this path.
        to_grid, from_grid : Callable
            Coefficient/grid projectors supplied by the owning grid net.
        pair_grid, project_pair, scalar_product : Callable, optional
            Optional fused full composition, paired forward projection, and
            direct scalar coefficient contraction.

        Returns
        -------
        Array
            Coefficient result. A direct scalar contraction has shape
            ``(N, 1, F, C)``; other paths retain ``n_frames * C`` channels.
        """
        # === Step 1. Channel projections at coefficient resolution ===
        left, right = self._project_operands(left, right)

        # === Step 2. Quadratic product on the grid, projected back ===
        if scalar_product is not None:
            coeff = scalar_product(left, right)
        else:
            coeff = pair_grid(left, right) if pair_grid is not None else None
            if coeff is None:
                left_grid, right_grid = _project_pair(
                    left,
                    right,
                    to_grid=to_grid,
                    project_pair=project_pair,
                )
                coeff = from_grid(left_grid * right_grid)
        if scalar_product is not None:
            return self.out_proj(coeff)
        return _project_frames(coeff, self.out_proj, self.n_frames)

    def _project_operands(
        self,
        left: Any,
        right: Any,
    ) -> tuple[Any, Any]:
        """Apply the two coefficient-space channel projections."""
        xp = array_api_compat.array_namespace(left)
        if self.mode == "self":
            shape = (*left.shape[:-1], self.n_frames, -1)
            fused = xp.reshape(
                xp.concat([xp.reshape(left, shape), xp.reshape(right, shape)], axis=-1),
                (*left.shape[:-1], -1),
            )  # per-frame concat -> (N, D, F, K*2C)
            left = _project_frames(fused, self.left_proj, self.n_frames)
            right = _project_frames(fused, self.right_proj, self.n_frames)
        else:
            left = _project_frames(left, self.left_proj, self.n_frames)
            right = _project_frames(right, self.right_proj, self.n_frames)
        return left, right

    def serialize(self) -> dict[str, Any]:
        """Serialize the GridMLP to a dict."""
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
        check_version_compatibility(int(data.pop("@version")), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj._load_variables(variables)
        return obj

    def _load_variables(self, variables: dict[str, Any]) -> None:
        prec = PRECISION_DICT[self.precision.lower()]
        self.left_proj.weight = np.asarray(variables["left_proj.weight"], dtype=prec)
        self.right_proj.weight = np.asarray(variables["right_proj.weight"], dtype=prec)
        self.out_proj.weight = np.asarray(variables["out_proj.weight"], dtype=prec)


class GridBranch(NativeOP):
    """
    Scalar-routed polynomial mixer over grid product branches.

    The softmax sees only invariant scalar inputs.  Each branch is a
    quadratic product of grid fields, so rotations only act through the grid
    argument and the operation remains as band-limited as the product path.
    """

    def __init__(
        self,
        *,
        channels: int,
        n_branches: int,
        n_frames: int,
        precision: str = DEFAULT_PRECISION,
        trainable: bool,
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
        # A single branch makes the routing softmax identically one, so the
        # router carries no degree of freedom: its gradient is exactly zero on
        # every path, and the fused grid product skips it altogether. Freezing
        # it states that, and keeps DDP from waiting for a gradient that never
        # arrives -- which aborts the step unless ``find_unused_parameters`` is
        # paid for. The parameter itself stays, so checkpoints round-trip.
        self.router = ChannelLinear(
            in_channels=2 * self.channels,
            out_channels=self.n_branches,
            precision=precision,
            bias=False,
            trainable=trainable and self.n_branches > 1,
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
        pair_grid: Callable[[Any, Any], Any | None] | None = None,
        project_pair: Callable[[Any, Any], tuple[Any, Any]] | None = None,
        scalar_product: Callable[[Any, Any], Any] | None = None,
    ) -> Any:
        """
        Apply scalar-routed grid branch mixing on coefficient operands.

        Parameters
        ----------
        left, right : Array
            Coefficient operands with shape ``(N, D, F, n_frames * C)``.
        scalar_pair : Array
            Invariant router source with shape ``(N, F, 2*C)``.
        to_grid, from_grid : Callable
            Coefficient/grid projectors supplied by the owning grid net.
        pair_grid, project_pair, scalar_product : Callable, optional
            Optional fused full composition, paired forward projection, and
            direct scalar coefficient contraction.

        Returns
        -------
        Array
            Coefficient result. A direct scalar contraction has shape
            ``(N, 1, F, C)``; other paths retain ``n_frames * C`` channels.
        """
        xp = array_api_compat.array_namespace(left)
        # === Step 1. Branch channel projections at coefficient resolution ===
        left = _project_frames(left, self.left_proj, self.n_frames)
        right = _project_frames(right, self.right_proj, self.n_frames)

        # === Step 2. Quadratic branches on the grid, routed by scalars ===
        # A single branch makes the router softmax identically one, which
        # reduces the routed product to the plain grid product the fused
        # operator evaluates.
        if scalar_product is not None:
            coeff = scalar_product(left, right)
            n_batch, coeff_dim, n_focus, _ = coeff.shape
            value = xp.reshape(
                coeff,
                (
                    n_batch,
                    coeff_dim,
                    n_focus,
                    self.n_branches,
                    self.channels,
                ),
            )
            router = self.router(scalar_pair)
            router = xp.exp(router - xp.max(router, axis=-1, keepdims=True))
            router = router / xp.sum(router, axis=-1, keepdims=True)
            coeff = xp.sum(value * router[:, None, :, :, None], axis=3)
        else:
            coeff = (
                pair_grid(left, right)
                if self.n_branches == 1 and pair_grid is not None
                else None
            )
        if coeff is None:
            left_grid, right_grid = _project_pair(
                left,
                right,
                to_grid=to_grid,
                project_pair=project_pair,
            )
            value = left_grid * right_grid  # (N, G, F, N_branches * C)
            n_batch, n_grid, n_focus, _ = value.shape
            value = xp.reshape(
                value, (n_batch, n_grid, n_focus, self.n_branches, self.channels)
            )
            router = self.router(scalar_pair)
            router = xp.exp(router - xp.max(router, axis=-1, keepdims=True))
            router = router / xp.sum(router, axis=-1, keepdims=True)
            out = xp.sum(value * router[:, None, :, :, None], axis=3)  # (N, G, F, C)
            coeff = from_grid(out)

        # === Step 3. Project back to coefficients and mix output channels ===
        if scalar_product is not None:
            return self.out_proj(coeff)
        return _project_frames(coeff, self.out_proj, self.n_frames)

    def serialize(self) -> dict[str, Any]:
        """Serialize the GridBranch to a dict."""
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
        check_version_compatibility(int(data.pop("@version")), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj._load_variables(variables)
        return obj

    def _load_variables(self, variables: dict[str, Any]) -> None:
        prec = PRECISION_DICT[self.precision.lower()]
        self.left_proj.weight = np.asarray(variables["left_proj.weight"], dtype=prec)
        self.right_proj.weight = np.asarray(variables["right_proj.weight"], dtype=prec)
        self.router.weight = np.asarray(variables["router.weight"], dtype=prec)
        self.out_proj.weight = np.asarray(variables["out_proj.weight"], dtype=prec)


class FrameContract(NativeOP):
    """Per-degree frame/channel contraction that preserves the order index."""

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int,
        coefficient_layout: str,
        n_frames: int,
        channels: int,
        precision: str = DEFAULT_PRECISION,
        trainable: bool,
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
        bound = 1.0 / math.sqrt(self.n_frames * self.channels)
        self.weight = rng.uniform(
            -bound,
            bound,
            size=(self.lmax + 1, self.n_frames * self.channels, self.channels),
        ).astype(prec)

    def call(self, coeff: Any) -> Any:
        """Contract ``(N, D, F, K*C)`` frame coefficients to ``(N, D, F, C)``."""
        xp = array_api_compat.array_namespace(coeff)
        device = array_api_compat.device(coeff)
        weight = xp_asarray_nodetach(xp, self.weight[...], device=device)
        degree_index = xp_asarray_nodetach(xp, self.degree_index, device=device)
        weight = xp.take(weight, degree_index, axis=0)
        return xp_einsum("ndfi,dio->ndfo", coeff, weight)

    def call_scalar(self, coeff: Any) -> Any:
        """Contract the single ``l=0`` coefficient with its frame weights.

        Parameters
        ----------
        coeff : Array
            Scalar coefficient with shape ``(N, 1, F, K*C)``.

        Returns
        -------
        Array
            Contracted scalar with shape ``(N, 1, F, C)``.
        """
        xp = array_api_compat.array_namespace(coeff)
        weight = xp_asarray_nodetach(
            xp, self.weight[0:1], device=array_api_compat.device(coeff)
        )
        return xp_einsum("ndfi,dio->ndfo", coeff, weight)

    def serialize(self) -> dict[str, Any]:
        """Serialize the FrameContract to a dict."""
        return {
            "@class": "FrameContract",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "coefficient_layout": self.coefficient_layout,
                "n_frames": self.n_frames,
                "channels": self.channels,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": {"weight": to_numpy_array(self.weight)},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> FrameContract:
        """Deserialize a FrameContract from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "FrameContract":
            raise ValueError(f"Invalid class for FrameContract: {data_cls}")
        check_version_compatibility(int(data.pop("@version")), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj.weight = np.asarray(
            variables["weight"], dtype=PRECISION_DICT[obj.precision.lower()]
        )
        return obj


class FrameExpand(NativeOP):
    """Per-degree frame/channel expansion that preserves the order index."""

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int,
        coefficient_layout: str,
        n_frames: int,
        channels: int,
        precision: str = DEFAULT_PRECISION,
        trainable: bool,
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
        bound = 1.0 / math.sqrt(self.channels)
        self.weight = rng.uniform(
            -bound,
            bound,
            size=(self.lmax + 1, self.channels, self.n_frames * self.channels),
        ).astype(prec)

    def call(self, coeff: Any) -> Any:
        """Expand ``(N, D, F, C)`` coefficients to ``(N, D, F, K*C)``."""
        xp = array_api_compat.array_namespace(coeff)
        device = array_api_compat.device(coeff)
        weight = xp_asarray_nodetach(xp, self.weight[...], device=device)
        degree_index = xp_asarray_nodetach(xp, self.degree_index, device=device)
        weight = xp.take(weight, degree_index, axis=0)
        return xp_einsum("ndfi,dio->ndfo", coeff, weight)

    def serialize(self) -> dict[str, Any]:
        """Serialize the FrameExpand to a dict."""
        return {
            "@class": "FrameExpand",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "coefficient_layout": self.coefficient_layout,
                "n_frames": self.n_frames,
                "channels": self.channels,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": {"weight": to_numpy_array(self.weight)},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> FrameExpand:
        """Deserialize a FrameExpand from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "FrameExpand":
            raise ValueError(f"Invalid class for FrameExpand: {data_cls}")
        check_version_compatibility(int(data.pop("@version")), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj.weight = np.asarray(
            variables["weight"], dtype=PRECISION_DICT[obj.precision.lower()]
        )
        return obj


class BaseGridNet(NativeOP):
    """
    Shared implementation for S2 and SO(3) grid nets.

    ``mode='self'`` expects one input whose last channel axis contains two
    branches.  ``mode='cross'`` expects query and context inputs; the query side
    is the source of attention queries and SwiGLU gates, while the context side
    is the key/value or second product branch.
    """

    CONFIG_DERIVED_ARRAYS = ("_from_grid_t", "_scalar_product_weight")

    def __init__(
        self,
        *,
        projector: BaseGridProjector,
        channels: int,
        n_focus: int,
        mode: GridNetMode,
        op_type: GridNetOp,
        precision: str = DEFAULT_PRECISION,
        layout: GridNetLayout,
        mlp_bias: bool,
        trainable: bool,
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
        coefficient_rows = int(projector.coeff_dim) // self.n_frames
        # One wider projection reduces launch overhead for at most 25 coefficient
        # rows. Larger operands retain independent projections to bound the
        # short-lived concatenated tensor in the compiled training graph.
        self._combine_grid_projection = coefficient_rows <= 25
        self.mode = str(mode).lower()
        if self.mode not in {"self", "cross"}:
            raise ValueError("`mode` must be either 'self' or 'cross'")
        self.op_type = str(op_type).lower()
        if self.op_type not in {"glu", "mlp", "branch"}:
            raise ValueError("`op_type` must be one of 'glu', 'mlp', or 'branch'")
        self.precision = precision
        self.layout = str(layout).lower()
        if self.layout not in {"ndfc", "nfdc", "fndc", "flat"}:
            raise ValueError(
                "`layout` must be one of 'ndfc', 'nfdc', 'fndc', or 'flat'"
            )
        if self.mode == "self" and self.layout == "flat":
            raise ValueError("`layout='flat'` is only supported for cross grid nets")
        self.mlp_bias = bool(mlp_bias)
        self.trainable = bool(trainable)
        self.expanded_channels = self.n_frames * self.channels
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
        scalar_product_weight = (
            _build_so3_scalar_product_weight(projector)
            if isinstance(projector, SO3GridProjector)
            else None
        )
        self._scalar_product_weight = scalar_product_weight

        # The fused grid pair product needs the grid-to-coefficient projector
        # transposed so both matrices are read row-major by grid point.
        # The operator is instantiated per coefficient-slot count, which this
        # projector fixes, so the choice is made once here rather than per call.
        # The array-API reference leaves the backend hooks unbound; ``pt_expt``
        # binds them at construction when an operator serves the slot count.
        # The inference hook serves the compact ``(N, P, C)`` layout; the
        # training hook differentiates the same expression inside the force
        # graph on the frame-packed operands, with analytic first and second
        # order.
        self._grid_pair_fn = None
        self._grid_pair_train_fn = None
        self._from_grid_t = np.ascontiguousarray(projector.from_grid_mat.T)

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

        self.residual_scale_init = residual_scale_init
        if residual_scale_init is None:
            self.residual_scale: np.ndarray | None = None
        else:
            prec = PRECISION_DICT[self.precision.lower()]
            self.residual_scale = np.ones(
                (self.n_focus, self.output_channels), dtype=prec
            ) * float(residual_scale_init)

    def call(self, query: Any, context: Any = None) -> Any:
        """Apply the configured grid net and restore the input layout."""
        return self._forward(query, context, scalar_only=False)

    def call_scalar(self, query: Any, context: Any = None) -> Any:
        """Apply the grid net and return only the scalar coefficient.

        Parameters
        ----------
        query : Array
            Query coefficient tensor in the configured layout.
        context : Array, optional
            Optional context coefficient tensor for cross mode.

        Returns
        -------
        Array
            Grid-net output with the degree axis restricted to ``l=0``.

        Notes
        -----
        The final SeZM readout consumes only ``l=0``. SO(3) Haar orthogonality
        reduces its quadratic grid projection to a weighted coefficient inner
        product. Other projectors restrict the inverse grid projection to the
        scalar row. Accelerated inference keeps the full fused pair projection
        because materializing a scalar-only fallback grid would be slower than
        that fused operator.
        """
        if self._grid_pair_fn is not None and not getattr(self, "training", False):
            return self._slice_scalar_layout(self.call(query, context))
        return self._forward(query, context, scalar_only=True)

    def _forward(
        self,
        query: Any,
        context: Any,
        *,
        scalar_only: bool,
    ) -> Any:
        """Run the shared full or scalar-only grid path."""
        # === Step 1. Normalize the input layout and build product operands ===
        xp = array_api_compat.array_namespace(query)
        input_dtype = query.dtype
        query_ndfc, shape_info = self._to_ndfc(query)
        left, right, scalar_pair = self._prepare_pair(query_ndfc, context)

        # === Step 2. Select the static projection plan and apply the grid op ===
        direct_scalar = scalar_only and self._scalar_product_weight is not None
        coeff_out = self.grid_op(
            xp.astype(left, get_xp_precision(xp, self.precision)),
            xp.astype(right, get_xp_precision(xp, self.precision)),
            scalar_pair,
            to_grid=self._to_grid,
            project_pair=(
                self._project_pair_in_one_transform
                if not direct_scalar
                and (
                    scalar_only
                    or (
                        getattr(self, "training", False)
                        and self._combine_grid_projection
                    )
                )
                else None
            ),
            from_grid=self._from_grid_scalar if scalar_only else self._from_grid,
            pair_grid=None if scalar_only else self._pair_grid,
            scalar_product=self._scalar_so3_product if direct_scalar else None,
        )

        # === Step 3. Apply scalar gating and contract Wigner-D frames ===
        coeff_out = self._apply_scalar_path(
            coeff_out,
            scalar_pair,
            compact_scalar=direct_scalar,
        )
        coeff_out = self._contract_frames(coeff_out, scalar_only=scalar_only)
        coeff_out = self._apply_residual_scale(coeff_out)

        # === Step 4. Restore the caller layout and dtype ===
        return self._restore_layout(
            xp.astype(coeff_out, input_dtype),
            shape_info,
            scalar_only=scalar_only,
        )

    def _prepare_pair(
        self,
        query: Any,
        context: Any,
    ) -> tuple[Any, Any, Any]:
        if self.mode == "self":
            return self._prepare_self_pair(query)
        return self._prepare_cross_pair(query, context)

    def _prepare_self_pair(
        self,
        query: Any,
    ) -> tuple[Any, Any, Any]:
        left, right = self._split_self_query(query)
        scalar_pair = self._make_scalar_pair(left, right)
        return left, right, scalar_pair

    def _prepare_cross_pair(
        self,
        query: Any,
        context: Any,
    ) -> tuple[Any, Any, Any]:
        if context is None:
            raise ValueError("`context` is required when `mode='cross'`")
        context_ndfc, _ = self._to_ndfc(context)
        self._check_last_dim(query, self.context_channels, "query")
        self._check_last_dim(context_ndfc, self.context_channels, "context")
        if self.frame_expand is None:
            scalar_pair = self._make_scalar_pair(query, context_ndfc)
            return query, context_ndfc, scalar_pair

        xp = array_api_compat.array_namespace(query)
        scalar_pair = xp.astype(
            xp.concat(
                [
                    query[:, 0, :, :],
                    context_ndfc[:, 0, :, :],
                ],
                axis=-1,
            ),
            get_xp_precision(xp, self.precision),
        )
        return (
            self.frame_expand(query),
            self.frame_expand(context_ndfc),
            scalar_pair,
        )

    def _contract_frames(
        self,
        coeff: Any,
        *,
        scalar_only: bool,
    ) -> Any:
        if self.frame_contract is None:
            return coeff
        if scalar_only:
            return self.frame_contract.call_scalar(coeff)
        return self.frame_contract(coeff)

    def _apply_residual_scale(self, coeff: Any) -> Any:
        if self.residual_scale is None:
            return coeff
        xp = array_api_compat.array_namespace(coeff)
        residual_scale = xp_asarray_nodetach(
            xp, self.residual_scale[...], device=array_api_compat.device(coeff)
        )
        residual_scale = xp.astype(residual_scale, coeff.dtype)
        return coeff * xp.reshape(
            residual_scale,
            (1, 1, self.n_focus, self.output_channels),
        )

    def _apply_scalar_path(
        self,
        coeff: Any,
        scalar_pair: Any,
        *,
        compact_scalar: bool,
    ) -> Any:
        xp = array_api_compat.array_namespace(coeff)
        scalar_out = self.scalar_act(scalar_pair)
        scalar_gate = xp_sigmoid(self.scalar_gate(scalar_pair))
        if compact_scalar:
            scalar_coeff = coeff * scalar_gate[:, None, :, :]
            scalar_coeff = scalar_coeff + scalar_out[:, None, :, :]
            return self._pack_scalar_frame(scalar_coeff)
        n_batch, coeff_dim, n_focus, _ = coeff.shape
        coeff_view = xp.reshape(
            coeff,
            (
                n_batch,
                coeff_dim,
                n_focus,
                self.n_frames,
                self.channels,
            ),
        )
        coeff_view = coeff_view * scalar_gate[:, None, :, None, :]
        # torch in-place ``coeff_view[:, 0, :, frame_zero_index, :].add_(scalar_out)``:
        # array-API has no in-place item assignment, so the d=0 slab is rebuilt by
        # functional concat with the scalar update added to the frame_zero_index frame.
        fzi = self.frame_zero_index
        head = coeff_view[:, :1, :, :, :]
        head = xp.concat(
            [
                head[:, :, :, :fzi, :],
                head[:, :, :, fzi : fzi + 1, :] + scalar_out[:, None, :, None, :],
                head[:, :, :, fzi + 1 :, :],
            ],
            axis=3,
        )
        coeff_view = xp.concat([head, coeff_view[:, 1:, :, :, :]], axis=1)
        return xp.reshape(
            coeff_view, (n_batch, coeff_dim, n_focus, self.expanded_channels)
        )

    def _pack_scalar_frame(self, scalar: Any) -> Any:
        """Embed ``(N, 1, F, C)`` scalars in the ``k=0`` slot of ``K*C``."""
        xp = array_api_compat.array_namespace(scalar)
        device = array_api_compat.device(scalar)
        n_batch, _, n_focus, channels = scalar.shape
        before = xp.zeros(
            (
                n_batch,
                1,
                n_focus,
                self.frame_zero_index,
                channels,
            ),
            dtype=scalar.dtype,
            device=device,
        )
        after = xp.zeros(
            (
                n_batch,
                1,
                n_focus,
                self.n_frames - self.frame_zero_index - 1,
                channels,
            ),
            dtype=scalar.dtype,
            device=device,
        )
        coeff = xp.concat(
            [before, scalar[:, :, :, None, :], after],
            axis=3,
        )
        return xp.reshape(coeff, (n_batch, 1, n_focus, self.expanded_channels))

    def _split_self_query(self, query: Any) -> tuple[Any, Any]:
        self._check_last_dim(query, self.query_channels, "query")
        # torch.chunk(query, chunks=2, dim=-1) with an even channel count
        return (
            query[..., : self.expanded_channels],
            query[..., self.expanded_channels :],
        )

    def _make_scalar_pair(self, left: Any, right: Any) -> Any:
        xp = array_api_compat.array_namespace(left)
        return xp.astype(
            xp.concat(
                [
                    self._extract_scalar(left),
                    self._extract_scalar(right),
                ],
                axis=-1,
            ),
            get_xp_precision(xp, self.precision),
        )

    def _extract_scalar(self, coeff: Any) -> Any:
        xp = array_api_compat.array_namespace(coeff)
        n_batch, _, n_focus, _ = coeff.shape
        coeff_view = xp.reshape(
            coeff,
            (
                n_batch,
                coeff.shape[1],
                n_focus,
                self.n_frames,
                self.channels,
            ),
        )
        return coeff_view[:, 0, :, self.frame_zero_index, :]

    def _pair_grid(self, left: Any, right: Any) -> Any | None:
        """
        Evaluate ``from_grid(to_grid(left) * to_grid(right))`` in one operator.

        The grid field is 39 times larger than its coefficient operand at the
        production SO(3) shape, so keeping it off device memory is worth a
        dedicated kernel. Returns ``None`` when the fused operator does not
        serve this shape, and the caller keeps the projector composition.

        Parameters
        ----------
        left, right : Array
            Coefficient operands with shape ``(N, D, F, n_frames * C)``.

        Returns
        -------
        Array or None
            Coefficient result with shape ``(N, D, F, n_frames * C)``.
        """
        c_wide = left.shape[3] // self.n_frames
        if c_wide % 32 != 0 or left.shape != right.shape:
            return None
        if getattr(self, "training", False):
            # Training form: frame-packed operands ride through unreshaped,
            # with analytic first and second order behind the call; under
            # autocast it runs the same reduced-precision regime as the dense
            # einsum composition it replaces.
            if self._grid_pair_train_fn is None:
                return None
            return self._grid_pair_train_fn(
                left,
                right,
                self.projector.to_grid_mat,
                self._from_grid_t,
                self.n_frames,
            )
        if self._grid_pair_fn is None or left.shape[2] != 1:
            return None
        n_batch, coeff_dim = left.shape[0], left.shape[1]
        flat_p = coeff_dim * self.n_frames
        xp = array_api_compat.array_namespace(left, right)
        out = self._grid_pair_fn(
            xp.reshape(left, (n_batch, flat_p, c_wide)),
            xp.reshape(right, (n_batch, flat_p, c_wide)),
            self.projector.to_grid_mat,
            self._from_grid_t,
        )
        return xp.reshape(out, (n_batch, coeff_dim, 1, self.n_frames * c_wide))

    def _project_pair_in_one_transform(
        self,
        left: Any,
        right: Any,
    ) -> tuple[Any, Any]:
        """Project scalar-output operands with one shared linear transform."""
        return _project_pair_in_one_transform(
            left,
            right,
            n_frames=self.n_frames,
            to_grid=self._to_grid,
        )

    def _to_grid(self, coeff: Any) -> Any:
        # The per-frame channel width is inferred so the projector also serves
        # widened operands (e.g. a branch hidden width ``n_branches * C``).
        xp = array_api_compat.array_namespace(coeff)
        n_batch, coeff_dim, n_focus, _ = coeff.shape
        coeff_view = xp.reshape(coeff, (n_batch, coeff_dim, n_focus, self.n_frames, -1))
        to_grid = xp_asarray_nodetach(
            xp, self.projector.to_grid_mat[...], device=array_api_compat.device(coeff)
        )
        to_grid = xp.astype(to_grid, coeff.dtype)
        # einsum "gdk,ndfkc->ngfc" (with to_grid reshaped (G, D, K)) as a
        # broadcast batched matmul: the contracted (d, k) axes are flattened
        # (d outer, k inner) and to_grid is already stored as (G, D*K).
        #
        # Two alternatives were measured on the Pro shape and both lost, so
        # this spelling is deliberate: folding the node axis into the GEMM
        # (`xp_einsum("gj,njc->ngc")`) costs 5 ms per step, and stating the
        # five-axis contraction the way the pt backend does
        # (`xp_einsum("gdk,ndfkc->ngfc")`) costs 3.6 ms. The same contraction
        # is faster there and slower here, so the choice belongs to the graph
        # around it rather than to the contraction itself.
        n_channels = coeff_view.shape[-1]
        coeff_dk = xp.permute_dims(coeff_view, (0, 1, 3, 2, 4))  # (N, D, K, F, C)
        coeff_flat = xp.reshape(
            coeff_dk, (n_batch, coeff_dim * self.n_frames, n_focus * n_channels)
        )
        out = xp.matmul(to_grid[None, ...], coeff_flat)  # (N, G, F*C)
        return xp.reshape(out, (n_batch, self.projector.grid_size, n_focus, n_channels))

    def _from_grid(self, grid: Any) -> Any:
        # Channel width is inferred to match the (possibly widened) grid field.
        xp = array_api_compat.array_namespace(grid)
        n_batch, _, n_focus, _ = grid.shape
        coeff_dim = self.projector.coeff_dim // self.n_frames
        from_grid = xp_asarray_nodetach(
            xp, self.projector.from_grid_mat[...], device=array_api_compat.device(grid)
        )
        from_grid = xp.astype(from_grid, grid.dtype)
        # einsum "dkg,ngfc->ndfkc" (with from_grid reshaped (D, K, G)) as a
        # broadcast batched matmul, then a reshape to (N, D, F, K*C). from_grid
        # is already stored as (D*K, G); the matmul output is reshaped/permuted.
        n_channels = grid.shape[-1]
        grid_flat = xp.reshape(
            grid, (n_batch, self.projector.grid_size, n_focus * n_channels)
        )
        coeff = xp.matmul(from_grid[None, ...], grid_flat)  # (N, D*K, F*C)
        coeff = xp.reshape(
            coeff, (n_batch, coeff_dim, self.n_frames, n_focus, n_channels)
        )
        coeff = xp.permute_dims(coeff, (0, 1, 3, 2, 4))  # (N, D, F, K, C)
        return xp.reshape(
            coeff, (n_batch, coeff_dim, n_focus, self.n_frames * n_channels)
        )

    def _from_grid_scalar(self, grid: Any) -> Any:
        """Project a grid field to the ``l=0`` coefficient only."""
        xp = array_api_compat.array_namespace(grid)
        n_batch, _, n_focus, _ = grid.shape
        from_grid = xp_asarray_nodetach(
            xp, self.projector.from_grid_mat[...], device=array_api_compat.device(grid)
        )
        from_grid = xp.astype(from_grid[: self.n_frames], grid.dtype)
        n_channels = grid.shape[-1]
        grid_flat = xp.reshape(
            grid, (n_batch, self.projector.grid_size, n_focus * n_channels)
        )
        coeff = xp.matmul(from_grid[None, ...], grid_flat)  # (N, K, F*C)
        coeff = xp.reshape(coeff, (n_batch, 1, self.n_frames, n_focus, n_channels))
        coeff = xp.permute_dims(coeff, (0, 1, 3, 2, 4))  # (N, 1, F, K, C)
        return xp.reshape(coeff, (n_batch, 1, n_focus, self.n_frames * n_channels))

    def _scalar_so3_product(self, left: Any, right: Any) -> Any:
        """Contract a quadratic SO(3) product directly to ``l=0, k=0``."""
        weight = self._scalar_product_weight
        if weight is None:
            raise RuntimeError("SO(3) scalar product weights are unavailable")
        xp = array_api_compat.array_namespace(left, right)
        weight = xp_asarray_nodetach(
            xp, weight[...], device=array_api_compat.device(left)
        )
        weight = xp.astype(weight, left.dtype)
        n_batch, coeff_dim, n_focus, _ = left.shape
        left_view = xp.reshape(left, (n_batch, coeff_dim, n_focus, self.n_frames, -1))
        right_view = xp.reshape(right, (n_batch, coeff_dim, n_focus, self.n_frames, -1))
        scalar = xp.sum(
            left_view * weight[None, :, None, :, None] * right_view,
            axis=(1, 3),
        )
        return scalar[:, None, :, :]

    def _to_ndfc(self, value: Any) -> tuple[Any, tuple[int, ...]]:
        # All grid operations run in the canonical ``(N, D, F, C)`` layout; the
        # ``fndc`` re-orientation folds the focus-major SO(2) mixing layout into the
        # same transpose the ``nfdc`` path performs, so the grid compute below is
        # identical regardless of the caller's layout.
        xp = array_api_compat.array_namespace(value)
        if self.layout == "ndfc":
            return value, tuple(value.shape)
        if self.layout == "nfdc":
            return xp.permute_dims(value, (0, 2, 1, 3)), tuple(value.shape)
        if self.layout == "fndc":
            return xp.permute_dims(value, (1, 2, 0, 3)), tuple(value.shape)
        n_batch, coeff_dim, _ = value.shape
        return (
            xp.reshape(value, (n_batch, coeff_dim, self.n_focus, -1)),
            tuple(value.shape),
        )

    def _restore_layout(
        self,
        value: Any,
        shape_info: tuple[int, ...],
        *,
        scalar_only: bool = False,
    ) -> Any:
        xp = array_api_compat.array_namespace(value)
        if self.layout == "ndfc":
            return value
        if self.layout == "nfdc":
            return xp.permute_dims(value, (0, 2, 1, 3))
        if self.layout == "fndc":
            return xp.permute_dims(value, (2, 0, 1, 3))
        n_batch, input_coeff_dim, _ = shape_info
        coeff_dim = 1 if scalar_only else input_coeff_dim
        return xp.reshape(value, (n_batch, coeff_dim, -1))

    def _slice_scalar_layout(self, value: Any) -> Any:
        """Select the degree axis from a restored full-layout tensor."""
        if self.layout == "ndfc":
            return value[:, 0:1, :, :]
        if self.layout in {"nfdc", "fndc"}:
            return value[:, :, 0:1, :]
        return value[:, 0:1, :]

    def _check_last_dim(
        self,
        value: Any,
        expected: int,
        name: str,
    ) -> None:
        if value.shape[-1] != expected:
            raise ValueError(
                f"`{name}` last dimension must be {expected}, got {value.shape[-1]}"
            )

    def _variables(self) -> dict[str, Any]:
        """Collect weights keyed by the pt ``state_dict`` key names."""
        variables: dict[str, Any] = {
            "scalar_gate.weight": to_numpy_array(self.scalar_gate.weight)
        }
        if self.mlp_bias:
            variables["scalar_gate.bias"] = to_numpy_array(self.scalar_gate.bias)
        if self.op_type in {"mlp", "branch"}:
            for key, value in self.grid_op.serialize()["@variables"].items():
                variables[f"grid_op.{key}"] = value
        if self.frame_expand is not None:
            variables["frame_expand.weight"] = to_numpy_array(self.frame_expand.weight)
        if self.frame_contract is not None:
            variables["frame_contract.weight"] = to_numpy_array(
                self.frame_contract.weight
            )
        if self.residual_scale is not None:
            variables["residual_scale"] = to_numpy_array(self.residual_scale)
        return variables

    def _load_variables(self, variables: dict[str, Any]) -> None:
        """Load weights keyed by the pt ``state_dict`` key names."""
        prec = PRECISION_DICT[self.precision.lower()]
        self.scalar_gate.weight = np.asarray(
            variables["scalar_gate.weight"], dtype=prec
        )
        if self.mlp_bias:
            self.scalar_gate.bias = np.asarray(
                variables["scalar_gate.bias"], dtype=prec
            )
        if self.op_type in {"mlp", "branch"}:
            self.grid_op._load_variables(
                {
                    key[len("grid_op.") :]: value
                    for key, value in variables.items()
                    if key.startswith("grid_op.")
                }
            )
        if self.frame_expand is not None:
            self.frame_expand.weight = np.asarray(
                variables["frame_expand.weight"], dtype=prec
            )
        if self.frame_contract is not None:
            self.frame_contract.weight = np.asarray(
                variables["frame_contract.weight"], dtype=prec
            )
        if self.residual_scale is not None:
            self.residual_scale = np.asarray(variables["residual_scale"], dtype=prec)


class S2GridNet(BaseGridNet):
    """Grid net using an S2 spherical-harmonic projector."""

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        channels: int,
        n_focus: int = 1,
        mode: GridNetMode,
        op_type: GridNetOp,
        precision: str = DEFAULT_PRECISION,
        layout: GridNetLayout,
        grid_resolution_list: list[int] | None = None,
        coefficient_layout: str = "packed",
        grid_method: str = "e3nn",
        grid_branches: int = 1,
        residual_scale_init: float | None = None,
        mlp_bias: bool = False,
        trainable: bool,
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
        """Serialize the S2GridNet to a dict."""
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
            "@variables": self._variables(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> S2GridNet:
        """Deserialize an S2GridNet from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "S2GridNet":
            raise ValueError(f"Invalid class for S2GridNet: {data_cls}")
        check_version_compatibility(int(data.pop("@version")), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj._load_variables(variables)
        return obj


class SO3GridNet(BaseGridNet):
    """Grid net using a Wigner-D SO(3) projector with frame indices."""

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        kmax: int = 1,
        channels: int,
        n_focus: int = 1,
        mode: GridNetMode,
        op_type: GridNetOp,
        precision: str = DEFAULT_PRECISION,
        layout: GridNetLayout,
        lebedev_precision: int | None = None,
        coefficient_layout: str = "packed",
        grid_branches: int = 1,
        residual_scale_init: float | None = None,
        mlp_bias: bool = False,
        trainable: bool,
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
        frame_expand = None
        frame_contract = None
        if mode == "cross":
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
        """Serialize the SO3GridNet to a dict."""
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
            "@variables": self._variables(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SO3GridNet:
        """Deserialize an SO3GridNet from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SO3GridNet":
            raise ValueError(f"Invalid class for SO3GridNet: {data_cls}")
        check_version_compatibility(int(data.pop("@version")), 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj._load_variables(variables)
        return obj
