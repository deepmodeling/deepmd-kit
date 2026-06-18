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

Ported names: ``BaseGridNet`` (``mode='self'``; ``op_type``
'glu'/'mlp'/'branch'), ``S2GridNet``, ``GridProduct``, ``GridMLP``,
``GridBranch``.

Skipped names, with consumer evidence from the pt sources:

- ``SO3GridNet``: only constructed by ``so2.py`` (``node_wise_so3``,
  ``message_node_so3``) and ``ffn.py`` (``ffn_so3_grid``) — all disabled in
  the core DPA4 config.
- ``FrameContract``, ``FrameExpand``, ``_build_frame_degree_index``: only
  constructed by ``SO3GridNet`` (``mode='cross'``); the S2 projector always
  has ``n_frames == 1``, so the frame machinery is unreachable here.

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

from typing import (
    TYPE_CHECKING,
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
from .projection import (
    BaseGridProjector,
    S2GridProjector,
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
        left: Any,
        right: Any,
        scalar_pair: Any,
        *,
        to_grid: Callable[[Any], Any],
        from_grid: Callable[[Any], Any],
    ) -> Any:
        """
        Apply scalar-routed grid branch mixing on coefficient operands.

        The channel maps are applied at coefficient resolution and the grid
        transform is deferred to the injected ``to_grid``/``from_grid``
        callables, matching the pt ``GridBranch`` specialized to the S2
        ``n_frames == 1`` case (so no per-frame packing is needed).

        Parameters
        ----------
        left, right
            Coefficient operands with shape ``(N, D, F, C)``.
        scalar_pair
            Invariant router source with shape ``(N, F, 2*C)``.
        to_grid, from_grid
            Coefficient/grid projectors supplied by the owning grid net.
        """
        xp = array_api_compat.array_namespace(left)
        left = self.left_proj(left)  # (N, D, F, N_branches * C)
        right = self.right_proj(right)  # (N, D, F, N_branches * C)
        value = to_grid(left) * to_grid(right)  # (N, G, F, N_branches * C)
        n_batch, n_grid, n_focus, _ = value.shape
        value = xp.reshape(
            value,
            (n_batch, n_grid, n_focus, self.n_branches, self.channels),
        )  # (N, G, F, N_branches, C)
        router = _softmax_last_axis(self.router(scalar_pair))  # (N, F, N_branches)
        # einsum "ngfhc,nfh->ngfc" as a broadcast sum over the branch axis
        out = xp.sum(value * router[:, None, :, :, None], axis=3)  # (N, G, F, C)
        return self.out_proj(from_grid(out))  # (N, D, F, C)

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
        query_ndfc = self._to_ndfc(query)
        left, right = self._split_self_query(query_ndfc)
        scalar_pair = self._make_scalar_pair(left, right, compute_dtype)
        coeff_out = self._apply_grid_op(left, right, scalar_pair, compute_dtype)
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
        # einsum "gd,ndfc->ngfc" (n_frames == 1) as a broadcast batched matmul.
        # The per-point channel width is inferred so the projector also serves
        # widened operands (e.g. a branch hidden width ``n_branches * C``).
        xp = array_api_compat.array_namespace(coeff)
        n_batch, coeff_dim, n_focus, n_channels = coeff.shape
        to_grid_mat = xp_asarray_nodetach(
            xp, self.projector.to_grid_mat[...], device=array_api_compat.device(coeff)
        )
        if to_grid_mat.dtype != coeff.dtype:
            to_grid_mat = xp.astype(to_grid_mat, coeff.dtype)
        flat = xp.reshape(coeff, (n_batch, coeff_dim, n_focus * n_channels))
        out = xp.matmul(to_grid_mat[None, ...], flat)  # (N, G, F*C)
        return xp.reshape(out, (n_batch, self.projector.grid_size, n_focus, n_channels))

    def _from_grid(self, grid: Any) -> Any:
        # einsum "dg,ngfc->ndfc" (n_frames == 1) as a broadcast batched matmul.
        # The channel width is inferred to match the (possibly widened) grid.
        xp = array_api_compat.array_namespace(grid)
        n_batch, n_grid, n_focus, n_channels = grid.shape
        coeff_dim = self.projector.coeff_dim
        from_grid_mat = xp_asarray_nodetach(
            xp, self.projector.from_grid_mat[...], device=array_api_compat.device(grid)
        )
        if from_grid_mat.dtype != grid.dtype:
            from_grid_mat = xp.astype(from_grid_mat, grid.dtype)
        flat = xp.reshape(grid, (n_batch, n_grid, n_focus * n_channels))
        out = xp.matmul(from_grid_mat[None, ...], flat)  # (N, D, F*C)
        return xp.reshape(out, (n_batch, coeff_dim, n_focus, n_channels))

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
        Point-wise grid operation; ``"glu"`` or ``"branch"`` (``"mlp"`` is
        not ported).
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
        if self.op_type in {"mlp", "branch"}:
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
        if obj.op_type in {"mlp", "branch"}:
            obj.grid_op._load_variables(
                {
                    key[len("grid_op.") :]: value
                    for key, value in variables.items()
                    if key.startswith("grid_op.")
                }
            )
        return obj
