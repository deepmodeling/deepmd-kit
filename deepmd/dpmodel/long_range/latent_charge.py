# SPDX-License-Identifier: LGPL-3.0-or-later
import math
from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.array_api import (
    Array,
    xp_add_at,
)
from deepmd.dpmodel.common import (
    GLOBAL_NP_FLOAT_PRECISION,
    NativeOP,
    to_numpy_array,
)
from deepmd.dpmodel.fitting.latent_charge import (
    LatentChargeFitting,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    ALL_PAIRS,
    K_GRID,
    PME_MESH,
    KGrid,
    MeshGrid,
    build_topologies,
    enumerate_k_grid_bounds,
    frame_id_from_n_node,
    k_grid_bounds_of_cells,
    segment_sum,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_long_range import (
    BaseLongRange,
)
from .kernel import (
    COULOMB_CONSTANT,
    BaseInteractionKernel,
)

CHARGE_TARGET_SOURCES = ("neutral", "charge_spin", "fparam")
"""Sources of the total charge a constrained channel sums to."""

CHARGE_NET_NEURON = [64, 64]
"""Default hidden widths of the charge network."""

CHARGE_NET_SEED_INDEX = 10_000
"""Child index of the fitting seed that seeds a charge network configured without a seed.

The networks of the fitting and of the descriptor draw from small child
indices of their seeds, so the charge network draws a stream of its own even
when the fitting and descriptor seeds coincide.
"""


def _fft_smooth(n: int) -> int:
    """The smallest integer >= ``n`` whose prime factors are among 2, 3, 5, 7."""

    def smooth(m: int) -> bool:
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        return m == 1

    m = max(1, n)
    while not smooth(m):
        m += 1
    return m


def _bspline_integer_weights(p: int) -> np.ndarray:
    r"""The order-``p`` cardinal B-spline at :math:`M_p(l+1)`, ``l = 0..p-2``.

    The weights of the SPME denominator
    :math:`D(\theta) = \sum_l M_p(l+1) e^{i \theta l}`.
    """
    out = []
    for l in range(p - 1):
        x = float(l + 1)
        out.append(
            sum(
                (-1.0) ** j * math.comb(p, j) * max(x - j, 0.0) ** (p - 1)
                for j in range(p + 1)
            )
            / math.factorial(p - 1)
        )
    return np.asarray(out)


def _bspline_eval(xp: Any, x: Array, p: int) -> Array:
    r"""The order-``p`` cardinal B-spline :math:`M_p(x)` elementwise.

    :math:`M_p(x) = \frac{1}{(p-1)!} \sum_j (-1)^j \binom{p}{j}
    \mathrm{relu}(x - j)^{p-1}` — a piecewise polynomial, differentiable
    (twice for :math:`p \ge 4`) with respect to ``x``.
    """
    w = None
    for j in range(p + 1):
        term = ((-1.0) ** j * math.comb(p, j)) * xp.maximum(
            x - j, xp.zeros_like(x)
        ) ** (p - 1)
        w = term if w is None else w + term
    return w / math.factorial(p - 1)


def _inv3(xp: Any, m: Array) -> Array:
    """Inverse of a batch of 3x3 matrices, adjugate over determinant.

    Rows a, b, c: the columns of the inverse are (b x c)/det, (c x a)/det,
    (a x b)/det. The closed form is cheap and forward-only here — the cell is
    not an autograd leaf; the strain leaf keeps ``linalg.inv`` whose
    registered backward handles a symbolic frame count.
    """
    a, b, c = m[..., 0, :], m[..., 1, :], m[..., 2, :]
    bc = xp.linalg.cross(b, c)
    ca = xp.linalg.cross(c, a)
    ab = xp.linalg.cross(a, b)
    det = xp.sum(a * bc, axis=-1)
    inv = xp.stack([bc, ca, ab], axis=-1)  # columns
    return inv / det[..., None, None]


@BaseLongRange.register("latent_charge")
class LatentChargeLongRange(NativeOP, BaseLongRange):
    r"""Energy of interacting latent charges, periodic or not.

    A charge network predicts :math:`C` latent charge channels
    :math:`q_i^c` per atom, in units of :math:`e`, from the descriptor shared
    with the energy fitting; padding and atoms owned by another rank carry no
    charge. For a non-periodic batch the energy is the direct sum over the
    pairs of atoms of each frame listed by the
    :data:`~deepmd.dpmodel.utils.neighbor_graph.ALL_PAIRS` topology, each
    unordered pair once; for a periodic batch (with ``periodic=True``) it is
    the reciprocal-space lattice sum over the wave numbers of the
    :data:`~deepmd.dpmodel.utils.neighbor_graph.K_GRID` topology,

    .. math::
        E_{\rm non-periodic} = k_e \sum_{i<j} \sum_{c=1}^{C} q_i^c q_j^c K(r_{ij}),
        \qquad
        E_{\rm periodic} = k_e \Big[
        \frac{1}{2V} \sum_{\mathbf{k} \neq \mathbf{0}} \sum_{c=1}^{C}
        \hat K(\mathbf{k}) \, |\hat\rho_c(\mathbf{k})|^2
        - \frac{1}{2} K(0) \sum_{i,c} (q_i^c)^2 \Big],

    with the Coulomb constant :math:`k_e`, the kernel :math:`K` in Å⁻¹ and its
    Fourier transform :math:`\hat K`; every channel interacts through the same
    kernel and channels do not mix. The periodic sum uses the tin-foil boundary
    convention (the :math:`\mathbf{k} = \mathbf{0}` mode is dropped) and
    requires every channel to be neutral: a constrained channel must sum to
    zero, so set ``charge_constraint`` to ``neutral`` or to a source holding a
    neutral total charge.

    An optional constraint shifts one channel uniformly over the atoms of
    every frame so that it sums to the total charge of the frame; this channel
    is then the physical charge, while the other channels remain latent.

    Parameters
    ----------
    charge_net
        The network predicting the latent charges; its output dimension is
        the number of channels.
    kernel
        The pair interaction kernel.
    charge_constraint
        ``None`` for unconstrained charges, or a dict with the ``source`` of
        the total charge (``"neutral"`` for zero, ``"charge_spin"`` for the
        first column of the frame charge/spin condition, or ``"fparam"`` for
        the frame-parameter column ``fparam_index``) and the constrained
        ``channel``.
    periodic
        Whether the module serves periodic systems: it then declares the
        :data:`~deepmd.dpmodel.utils.neighbor_graph.K_GRID` topology beside
        :data:`~deepmd.dpmodel.utils.neighbor_graph.ALL_PAIRS` and evaluates
        batches by their cells. Non-periodic batches keep working.
    kspace_tolerance
        The relative weight of the slowest-decaying kernel mode at the
        wave-number cutoff of the k grid of a periodic batch.
    k_grid_mode
        How the reciprocal sum of a periodic batch is evaluated: ``"ragged"``
        enumerates the wave numbers of every frame eagerly, so the k axis is
        a dynamic dimension; ``"fixed"`` shares one integer grid over the
        frames — sized by the largest cell of the data statistics, or by
        ``k_grid_bounds`` — evaluates every frame on it by a static-shaped
        transform, and masks the points a frame does not reach; ``"pme"``
        spreads the charges to a static real-space mesh with a B-spline and
        evaluates the sum by an FFT (SPME), sized by the largest cell of the
        data statistics or by ``pme_mesh``. The fixed grid is hard: a frame
        needing an integer index beyond it (a larger cell, or a kernel that
        moved the cutoff out) is served by the grid alone, so its k-sum is
        truncated to the grid with a warning; on the mesh such a frame is
        evaluated with a coarser effective spacing (a smooth accuracy
        degradation; no warning — sizing the mesh is the caller's
        responsibility).
    k_grid_bounds
        The per-axis integer bounds :math:`n_d^{max}` of the fixed grid, in
        place of the data statistics; only meaningful with
        ``k_grid_mode="fixed"``.
    decouple_k_grid
        Decouple the k-point selection of a periodic batch from the kernel
        parameters: the wave-number cutoff is derived once from the initial
        parameters and kept for the life of the model, so training the
        kernel no longer moves the k grid. The kernel values the grid is
        evaluated with still follow the live parameters. The frozen cutoff
        is serialized with the model, so a restarted run keeps the grid the
        model was built with. Requires ``periodic=True``.
    pme_spacing
        The target real-space mesh spacing of the ``"pme"`` mode, in Å; the
        default is the Nyquist spacing of the kernel's wave-number cutoff,
        :math:`\pi / k_{max}`.
    pme_order
        The B-spline order of the ``"pme"`` charge assignment.
    pme_mesh
        The per-axis mesh dimensions :math:`(n_1, n_2, n_3)` of the
        ``"pme"`` mode, in place of the data statistics.
    """

    def __init__(
        self,
        charge_net: LatentChargeFitting,
        kernel: BaseInteractionKernel,
        charge_constraint: dict | None = None,
        periodic: bool = False,
        kspace_tolerance: float = 1e-8,
        k_grid_mode: str = "ragged",
        k_grid_bounds: list[int] | None = None,
        decouple_k_grid: bool = False,
        pme_spacing: float | None = None,
        pme_order: int = 4,
        pme_mesh: list[int] | None = None,
    ) -> None:
        if charge_net.vacuum_ref:
            raise ValueError(
                "the charge network reads the descriptor of the real atoms only "
                "and cannot reference isolated atoms (`vacuum_ref`)"
            )
        self.charge_net = charge_net
        self.kernel = kernel
        self.charge_constraint = self._normalize_constraint(
            charge_constraint, charge_net
        )
        self.periodic = bool(periodic)
        self.kspace_tolerance = float(kspace_tolerance)
        if (
            not np.isfinite(self.kspace_tolerance)
            or not 0.0 < self.kspace_tolerance < 1.0
        ):
            raise ValueError(
                f"`kspace_tolerance` must be in (0, 1), got {self.kspace_tolerance}"
            )
        # The frame charge/spin condition a compressed model is specialized
        # to; see :meth:`fold_charge_state`.
        self.charge_state = None
        self.k_grid_mode = str(k_grid_mode)
        if self.k_grid_mode not in ("ragged", "fixed", "pme"):
            raise ValueError(
                f"`k_grid_mode` must be 'ragged', 'fixed' or 'pme', got {self.k_grid_mode!r}"
            )
        if self.k_grid_mode in ("fixed", "pme") and not self.periodic:
            raise ValueError(
                f"`k_grid_mode='{self.k_grid_mode}'` requires `periodic=True`"
            )
        if k_grid_bounds is not None and self.k_grid_mode != "fixed":
            raise ValueError(
                "`k_grid_bounds` is only meaningful with `k_grid_mode='fixed'`"
            )
        self.decouple_k_grid = bool(decouple_k_grid)
        if self.decouple_k_grid and not self.periodic:
            raise ValueError("`decouple_k_grid` requires `periodic=True`")
        # A decoupled grid reads the kernel once, at construction: the k-point
        # selection stays at the initial kernel however training moves it,
        # while the kernel values in the graph still use the live parameters.
        self.frozen_k_max = (
            float(self.kernel.kspace_cutoff(self.kspace_tolerance))
            if self.decouple_k_grid
            else None
        )
        # The shared integer grid of the fixed mode, empty until the data
        # statistics size it (see :meth:`compute_input_stats`) — an array even
        # then, so a backend keeping arrays in its state dict sees a stable
        # set of keys and a checkpoint restores the sized grid.
        if k_grid_bounds is not None:
            self.k_grid_n = enumerate_k_grid_bounds(k_grid_bounds)
        elif self.k_grid_mode == "fixed":
            self.k_grid_n = np.zeros((0, 3), dtype=np.int64)
        else:
            self.k_grid_n = None
        # The PME mesh parameters. The mesh dims are an array even while
        # unsized, so a backend keeping arrays in its state dict sees a
        # stable set of keys and a checkpoint restores the sized mesh; the
        # plain-integer copy is what a trace reads as static shapes.
        self.pme_order = int(pme_order)
        if self.pme_order < 2:
            raise ValueError(f"`pme_order` must be at least 2, got {self.pme_order}")
        if pme_spacing is not None:
            pme_spacing = float(pme_spacing)
            if not np.isfinite(pme_spacing) or pme_spacing <= 0.0:
                raise ValueError(
                    f"`pme_spacing` must be finite and positive, got {pme_spacing}"
                )
        self.pme_spacing = pme_spacing
        if self.k_grid_mode != "pme":
            if pme_spacing is not None or pme_mesh is not None:
                raise ValueError(
                    "`pme_spacing`/`pme_mesh` are only meaningful with "
                    "`k_grid_mode='pme'`"
                )
            self.pme_mesh = None
            self._pme_n = None
        elif pme_mesh is not None:
            mesh = np.asarray(pme_mesh, dtype=np.int64).reshape(-1)
            if mesh.shape != (3,) or np.any(mesh < 1):
                raise ValueError(
                    f"`pme_mesh` must be three positive integers, got {pme_mesh}"
                )
            self.pme_mesh = mesh
            self._pme_n = (int(mesh[0]), int(mesh[1]), int(mesh[2]))
        else:
            # Unsized sentinel: the data statistics size it (see
            # :meth:`compute_input_stats`), or a checkpoint restores the
            # sized mesh through the state dict.
            self.pme_mesh = np.zeros((3,), dtype=np.int64)
            self._pme_n = None

    def _pme_dims(self) -> tuple[int, int, int] | None:
        """The mesh dimensions as plain integers, or ``None`` when unsized.

        The plain-integer cache is what a trace reads as static shapes; it is
        set at construction and at statistics time. A backend restoring the
        sized mesh through its state dict (a checkpoint does not run
        :meth:`compute_input_stats`) is picked up lazily from the mesh array.
        """
        if self._pme_n is not None:
            return self._pme_n
        if self.pme_mesh is None or int(self.pme_mesh[0]) < 1:
            return None
        mesh = self.pme_mesh
        self._pme_n = (int(mesh[0]), int(mesh[1]), int(mesh[2]))
        return self._pme_n

    @staticmethod
    def _normalize_constraint(
        constraint: dict | None, charge_net: LatentChargeFitting
    ) -> dict | None:
        if constraint is None:
            return None
        source = str(constraint.get("source", "neutral"))
        channel = int(constraint.get("channel", 0))
        fparam_index = int(constraint.get("fparam_index", 0))
        if source not in CHARGE_TARGET_SOURCES:
            raise ValueError(
                f"charge constraint source must be one of {CHARGE_TARGET_SOURCES}, "
                f"got {source!r}"
            )
        if not 0 <= channel < charge_net.dim_out:
            raise ValueError(
                f"constrained channel {channel} is out of range for "
                f"{charge_net.dim_out} latent charge channel(s)"
            )
        if source == "fparam" and not 0 <= fparam_index < charge_net.numb_fparam:
            raise ValueError(
                f"`fparam_index` {fparam_index} is out of range for "
                f"{charge_net.numb_fparam} frame parameter(s)"
            )
        return {"source": source, "channel": channel, "fparam_index": fparam_index}

    @classmethod
    def from_config(
        cls,
        config: dict,
        descriptor: Any,
        fitting: Any,
        fitting_seed: int | list[int] | None = None,
    ) -> "LatentChargeLongRange":
        """Build the module from its configuration.

        The charge network reads the descriptor and takes the conditioning
        inputs of the energy fitting: its frame and atomic parameters, their
        defaults and its case embedding. It follows the activation function,
        skip connection and precision of the fitting unless its configuration
        sets them, and its seed defaults to a child of the fitting seed. The
        kernel parameters are stored in the precision of the charge network.

        Parameters
        ----------
        config
            The module configuration: ``channels``, ``charge_net``, ``kernel``
            and ``charge_constraint``.
        descriptor
            The descriptor of the model.
        fitting
            The energy fitting of the model.
        fitting_seed
            The seed of the energy fitting.
        """
        charge_net_config = {
            "neuron": CHARGE_NET_NEURON,
            "activation_function": fitting.activation_function,
            "resnet_dt": fitting.resnet_dt,
            "precision": fitting.precision,
            "seed": child_seed(fitting_seed, CHARGE_NET_SEED_INDEX),
            **config.get("charge_net", {}),
        }
        kernel_config = {
            "precision": charge_net_config["precision"],
            **config["kernel"],
        }
        kernel_cls = BaseInteractionKernel.get_class_by_type(kernel_config.pop("type"))
        charge_net = LatentChargeFitting(
            var_name="latent_charge",
            ntypes=descriptor.get_ntypes(),
            dim_descrpt=descriptor.get_dim_out(),
            dim_out=int(config.get("channels", 1)),
            numb_fparam=fitting.get_dim_fparam(),
            numb_aparam=fitting.get_dim_aparam(),
            dim_case_embd=fitting.dim_case_embd,
            default_fparam=fitting.get_default_fparam(),
            use_aparam_as_mask=fitting.use_aparam_as_mask,
            mixed_types=descriptor.mixed_types(),
            type_map=fitting.get_type_map(),
            **charge_net_config,
        )
        return cls(
            charge_net,
            kernel_cls(**kernel_config),
            config.get("charge_constraint"),
            periodic=config.get("periodic", False),
            kspace_tolerance=config.get("kspace_tolerance", 1e-8),
            k_grid_mode=config.get("k_grid_mode", "ragged"),
            k_grid_bounds=config.get("k_grid_bounds"),
            decouple_k_grid=config.get("decouple_k_grid", False),
            pme_spacing=config.get("pme_spacing"),
            pme_order=config.get("pme_order", 4),
            pme_mesh=config.get("pme_mesh"),
        )

    def _k_max(self) -> float:
        """The wave-number cutoff selecting the k points of a periodic batch.

        A coupled grid reads the current kernel parameters, so a trained
        kernel re-derives its grid every batch; a decoupled grid keeps the
        cutoff of the initial parameters.
        """
        if self.decouple_k_grid:
            return self.frozen_k_max
        return float(self.kernel.kspace_cutoff(self.kspace_tolerance))

    def _pme_target_spacing(self) -> float:
        r"""The target PME mesh spacing, in Å.

        The explicit ``pme_spacing``, or the Nyquist spacing of the kernel's
        wave-number cutoff :math:`\pi / k_{max}` — fine enough that the
        kernel's spectral weight at the mesh cutoff is negligible.
        """
        if self.pme_spacing is not None:
            return self.pme_spacing
        return math.pi / self._k_max()

    def build_topologies(
        self,
        coord: Array,
        n_node: Array,
        box: Array | None,
        *,
        atype: Array,
        node_mask: Array | None = None,
        pair_excl: Any = None,
    ) -> dict[str, Any]:
        """Build the topologies of the module for one batch.

        The k grid of a periodic batch is enumerated from the cells with the
        wave-number cutoff of the kernel (the trainable parameters are read at
        build time, so a drifting kernel re-derives its grid every batch,
        unless the grid is decoupled from them by ``decouple_k_grid``). In
        the fixed grid mode the batch is evaluated on the shared integer grid
        instead, whose mask the same cutoff derives.
        """
        params = {}
        if self.periodic:
            if self.k_grid_mode == "pme":
                if self._pme_dims() is None:
                    raise ValueError(
                        "the PME mesh is not sized yet: run the data "
                        "statistics (`compute_input_stats`) on periodic "
                        "frames, or configure `pme_mesh`"
                    )
                params[PME_MESH] = {"n_mesh": self.pme_mesh}
            else:
                params[K_GRID] = {"k_max": self._k_max()}
                if self.k_grid_mode == "fixed":
                    if self.k_grid_n is None or self.k_grid_n.shape[0] == 0:
                        raise ValueError(
                            "the fixed k grid is not sized yet: run the data "
                            "statistics (`compute_input_stats`) on periodic "
                            "frames, or configure `k_grid_bounds`"
                        )
                    params[K_GRID]["n_grid"] = self.k_grid_n
        return build_topologies(
            self.required_topologies(),
            coord,
            n_node,
            box,
            node_mask,
            atype=atype,
            pair_excl=pair_excl,
            topology_params=params,
        )

    def required_topologies(self) -> tuple[str, ...]:
        if self.periodic:
            if self.k_grid_mode == "pme":
                return (ALL_PAIRS, PME_MESH)
            return (ALL_PAIRS, K_GRID)
        return (ALL_PAIRS,)

    def uses_charge_spin(self) -> bool:
        # A folded state stands in for the condition.
        return (
            self.charge_constraint is not None
            and self.charge_constraint["source"] == "charge_spin"
            and self.charge_state is None
        )

    def fold_charge_state(self, charge_spin: list[float]) -> None:
        """Specialize the constraint to one frame charge/spin condition.

        A compressed descriptor folds the condition into its tables and reads
        none at run time, so a constraint that reads its total charge from the
        condition takes it from this state instead. The state is a constant
        of the deployed model that a deployment rewrites, together with the
        tables of the descriptor, when it serves another condition.

        Parameters
        ----------
        charge_spin
            The condition ``[charge, spin multiplicity]``.
        """
        self.charge_state = np.asarray(
            charge_spin, dtype=GLOBAL_NP_FLOAT_PRECISION
        ).reshape(1, 2)

    def charge_state_constants(self) -> list[Array]:
        return [] if self.charge_state is None else [self.charge_state]

    def rebuild_charge_state(self, charge_spin: Array) -> list[Array]:
        if self.charge_state is None:
            return []
        xp = array_api_compat.array_namespace(charge_spin)
        state = xp.astype(charge_spin, self.charge_state.dtype)
        return [xp.reshape(state, (1, 2))]

    def call(self, *args: Any, **kwargs: Any) -> Array:
        return self.call_graph(*args, **kwargs)

    def call_graph(
        self,
        descriptor: Array,
        atype: Array,
        node_mask: Array,
        n_node: Array,
        topologies: dict[str, Any],
        fparam: Array | None = None,
        aparam: Array | None = None,
        charge_spin: Array | None = None,
    ) -> Array:
        xp = array_api_compat.array_namespace(descriptor)
        n = descriptor.shape[0]
        pairs = topologies[ALL_PAIRS]
        frame_id = frame_id_from_n_node(n_node, n_total=n)
        fparam_node = None if fparam is None else xp.take(fparam, frame_id, axis=0)
        charge = self.charge_net.call_graph(
            descriptor, atype, fparam=fparam_node, aparam=aparam
        )["latent_charge"]
        # The pair sum cancels between charges of opposite sign, so it runs in
        # the precision of the geometry.
        charge = xp.astype(charge, pairs.vec.dtype)
        charge = xp.where(node_mask[:, None], charge, xp.zeros_like(charge))
        if self.charge_constraint is not None:
            charge = self._constrain(
                charge, node_mask, frame_id, n_node, fparam, charge_spin
            )
        energy = self._pair_branch(charge, pairs, n)
        if self.periodic:
            if self.k_grid_mode == "pme":
                energy = energy + self._pme_branch(
                    charge, topologies[PME_MESH], frame_id, node_mask, n_node
                )
            else:
                energy = energy + self._kspace_branch(
                    charge, topologies[K_GRID], frame_id, node_mask, n_node
                )
        return xp.reshape(energy, (n, 1))

    def _pair_branch(self, charge: Array, pairs: Any, n: int) -> Array:
        """Direct pair sum over the non-periodic frames, per atom."""
        xp = array_api_compat.array_namespace(charge)
        src = pairs.index[0]
        dst = pairs.index[1]
        pair_charge = xp.sum(
            xp.take(charge, src, axis=0) * xp.take(charge, dst, axis=0), axis=-1
        )
        r_sq = xp.sum(pairs.vec * pairs.vec, axis=-1)
        pair_energy = pair_charge * self._pair_kernel(r_sq, src, n)
        half_energy = (0.5 * COULOMB_CONSTANT) * xp.where(
            pairs.mask, pair_energy, xp.zeros_like(r_sq)
        )
        return segment_sum(half_energy, src, n) + segment_sum(half_energy, dst, n)

    def _kspace_branch(
        self,
        charge: Array,
        k_grid: KGrid,
        frame_id: Array,
        node_mask: Array,
        n_node: Array,
    ) -> Array:
        r"""Reciprocal-space energy of the periodic frames, per atom.

        Evaluates
        :math:`E_f = k_e [ \frac{1}{2V_f} \sum_{\mathbf{k}\neq\mathbf{0}}
        \hat K(\mathbf{k}) |\hat\rho(\mathbf{k})|^2
        - \frac{1}{2} K(0) \sum_{i \in f} |q_i|^2 ]`
        on the strained geometry of :attr:`KGrid.strain` (identity when no
        strain leaf is set): coordinates by ``I + strain``, wave numbers by
        ``(I + strain)^{-1}`` on their row layout and volumes by
        ``det(I + strain)``, so that the gradient of the energy with respect
        to the strain leaf is the negative k-space virial of the frame. The
        reciprocal energy is attributed to the charged atoms of the frame in
        equal shares; the self term is per atom by construction. Guard points
        carry a zero mask.
        """
        xp = array_api_compat.array_namespace(charge)
        nf = n_node.shape[0]
        coord = k_grid.coord
        if coord is None:
            raise ValueError(
                "the k-grid topology carries no coordinates; the graph lower "
                "re-leafs them before the evaluation"
            )
        coord = xp.astype(coord, charge.dtype)
        k_vec = xp.astype(k_grid.vec, charge.dtype)
        volume = xp.astype(k_grid.volume, charge.dtype)
        if k_grid.strain is not None:
            eye = xp.eye(3, dtype=charge.dtype, device=array_api_compat.device(charge))
            s = xp.astype(k_grid.strain, charge.dtype) + eye  # (nf, 3, 3)
            # r' = S r (columns)
            s_node = xp.take(s, frame_id, axis=0)  # (N, 3, 3)
            coord = xp.matmul(s_node, coord[:, :, None])[:, :, 0]
            # k' = S^{-T} k (columns), i.e. rows multiplied by S^{-1}
            s_inv_t = xp.matrix_transpose(xp.linalg.inv(s))
            s_inv_t_k = xp.take(s_inv_t, k_grid.frame, axis=0)  # (nk, 3, 3)
            k_vec = xp.matmul(s_inv_t_k, k_vec[:, :, None])[:, :, 0]
            volume = volume * xp.linalg.det(s)
        phase = xp.matmul(k_vec, xp.matrix_transpose(coord))  # (nk, N)
        same_frame = frame_id[None, :] == k_grid.frame[:, None]
        cos_phase = xp.cos(phase) * same_frame
        sin_phase = xp.sin(phase) * same_frame
        rho_real = xp.matmul(cos_phase, charge)  # (nk, C)
        rho_imag = xp.matmul(sin_phase, charge)  # (nk, C)
        s_sq = xp.sum(rho_real**2 + rho_imag**2, axis=-1)  # (nk,)
        k_sq = xp.sum(k_vec * k_vec, axis=-1)
        kernel_value = self.kernel.kspace(k_sq)  # (nk,)
        half_ke = 0.5 * COULOMB_CONSTANT
        volume_k = xp.take(volume, k_grid.frame, axis=0)
        energy_k = half_ke * kernel_value * s_sq / volume_k
        energy_k = xp.where(k_grid.mask, energy_k, xp.zeros_like(energy_k))
        e_rec_frame = segment_sum(energy_k, k_grid.frame, nf)  # (nf,)
        weight = xp.astype(node_mask, charge.dtype)
        count = segment_sum(weight, frame_id, nf)
        count = xp.maximum(count, xp.ones_like(count))
        e_rec = xp.take(e_rec_frame / count, frame_id, axis=0) * weight  # (N,)
        # The self term belongs to the periodic lattice sum alone: a frame
        # without real k points (a non-periodic batch on a periodic-enabled
        # model) keeps its non-periodic self-free energy.
        periodic_frame = (
            segment_sum(xp.astype(k_grid.mask, charge.dtype), k_grid.frame, nf) > 0
        )
        e_self = (
            -half_ke
            * self.kernel.contact_value(charge)
            * xp.sum(charge * charge, axis=-1)
            * xp.astype(xp.take(periodic_frame, frame_id, axis=0), charge.dtype)
        )
        return e_rec + e_self

    def _pme_b_factor(self, xp: Any, dtype: Any, device: Any) -> Array:
        r"""The SPME assignment correction :math:`|B(m)|^2` on the mesh.

        Per axis :math:`|D_d(m)|^2 = |\sum_l M_p(l+1) e^{2\pi i m l / n_d}|^2`
        (equal to :math:`\mathrm{sinc}(m/n_d)^{2p}`), the full factor the
        three-axis outer product, in the ``fftfreq`` index order.
        """
        w = _bspline_integer_weights(self.pme_order)
        axes = []
        for n_d in self._pme_dims():
            m = xp.arange(n_d, dtype=dtype, device=device)
            m = xp.where(m > n_d // 2, m - n_d, m)
            theta = 2.0 * math.pi * m / n_d
            re = xp.zeros_like(theta)
            im = xp.zeros_like(theta)
            for l, wl in enumerate(w):
                re = re + float(wl) * xp.cos(theta * l)
                im = im + float(wl) * xp.sin(theta * l)
            axes.append(1.0 / (re * re + im * im))
        b = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
        return xp.reshape(b, (-1,))

    def _pme_branch(
        self,
        charge: Array,
        mesh: MeshGrid,
        frame_id: Array,
        node_mask: Array,
        n_node: Array,
    ) -> Array:
        r"""Reciprocal-space energy of the periodic frames on a mesh, per atom.

        Evaluates the same lattice sum as :meth:`_kspace_branch` by the SPME
        algorithm: the charges spread to a static real-space mesh with a
        B-spline of order :attr:`pme_order`, the mesh transforms by an FFT,
        and the energy convolves with the kernel transform corrected by the
        assignment factor (:meth:`_pme_b_factor`). The strain leaf enters the
        coordinates by ``I + strain`` and the cells by
        ``B (I + strain)^T`` — the fractional coordinates are strain-
        invariant, so the assignment is unstrained and the virial comes from
        the reciprocal vectors and the volume alone. Non-periodic frames
        (``mesh.mask`` false) carry no mesh energy and no self term.
        """
        xp = array_api_compat.array_namespace(charge)
        device = array_api_compat.device(charge)
        nf = n_node.shape[0]
        n_channel = charge.shape[-1]
        n1, n2, n3 = self._pme_dims()
        p = self.pme_order
        m_total = n1 * n2 * n3
        box = xp.astype(mesh.box, charge.dtype)
        coord = xp.astype(mesh.coord, charge.dtype)
        volume = xp.astype(mesh.volume, charge.dtype)
        b_inv = _inv3(xp, box)  # (nf, 3, 3); columns are g_d / 2pi
        b_inv_node = xp.take(b_inv, frame_id, axis=0)  # (N, 3, 3)
        # fractional coordinates, wrapped to [0, 1): frac = r B^-1
        frac = xp.matmul(coord[:, None, :], b_inv_node)[:, 0, :]
        frac = frac - xp.floor(frac)
        # per-axis assignment: weights (N, p) and mesh indices (N, p)
        weights = []
        indices = []
        for d, n_d in enumerate((n1, n2, n3)):
            u = frac[:, d] * n_d
            base = xp.floor(u)
            t = (u - base)[:, None]
            shift = xp.arange(p, dtype=charge.dtype, device=device)[None, :]
            # x_k = t + p - 1 - k lies in the support [0, p] of M_p
            weights.append(_bspline_eval(xp, t + (p - 1) - shift, p))
            idx = (
                xp.astype(base, xp.int64)[:, None]
                - (p - 1)
                + xp.arange(p, dtype=xp.int64, device=device)[None, :]
            )
            indices.append(idx % n_d)
        w1, w2, w3 = weights
        j1, j2, j3 = indices
        w = xp.reshape(
            w1[:, :, None, None] * w2[:, None, :, None] * w3[:, None, None, :],
            (frac.shape[0], p**3),
        )
        lin = xp.reshape(
            (j1[:, :, None, None] * n2 + j2[:, None, :, None]) * n3
            + j3[:, None, None, :],
            (frac.shape[0], p**3),
        )
        # spread the charges to the mesh, frame- and channel-major
        q_mesh = xp.zeros(
            (nf * n_channel * m_total,), dtype=charge.dtype, device=device
        )
        for c in range(n_channel):
            target = ((frame_id * n_channel + c) * m_total)[:, None] + lin
            value = charge[:, c][:, None] * w
            q_mesh = xp_add_at(
                q_mesh, xp.reshape(target, (-1,)), xp.reshape(value, (-1,))
            )
        q_mesh = xp.reshape(q_mesh, (nf * n_channel, n1, n2, n3))
        rho = xp.fft.fftn(q_mesh, axes=(-3, -2, -1))
        rho_sq = xp.real(rho) ** 2 + xp.imag(rho) ** 2
        rho_sq = xp.reshape(rho_sq, (nf, n_channel, m_total))
        # the reciprocal vectors of every frame on the mesh: mode m is the
        # integer Miller index, k(m) = 2 pi B^-1 m
        h_axes = []
        for n_d in (n1, n2, n3):
            m = xp.arange(n_d, dtype=charge.dtype, device=device)
            m = xp.where(m > n_d // 2, m - n_d, m)
            h_axes.append(m)
        g1, g2, g3 = xp.meshgrid(*h_axes, indexing="ij")
        h = xp.stack(
            [xp.reshape(g1, (-1,)), xp.reshape(g2, (-1,)), xp.reshape(g3, (-1,))],
            axis=-1,
        )  # (M, 3)
        k_vec = 2.0 * math.pi * xp.matmul(h[None, :, :], xp.matrix_transpose(b_inv))
        if mesh.strain is not None:
            # The strain family of the virial: B' = B S^T with S = I + strain.
            # The fractional coordinates are exactly strain-invariant, so the
            # assignment above needs no strain; the reciprocal vectors and the
            # volume pick it up: k' = S^-T k, V' = V det(S). The 3x3 inverse
            # and determinant stay on linalg — its registered backward handles
            # a symbolic frame count, unlike hand-written select/cross forms.
            eye = xp.eye(3, dtype=charge.dtype, device=device)
            s = xp.astype(mesh.strain, charge.dtype) + eye  # (nf, 3, 3)
            s_inv = xp.linalg.inv(s)
            # rows: k'_i = sum_j k_j (S^-1)[j, i]
            k_vec = xp.matmul(k_vec, s_inv)
            volume = volume * xp.linalg.det(s)
        k_sq = xp.reshape(xp.sum(k_vec * k_vec, axis=-1), (-1,))  # (nf * M,)
        # the zero mode sits at flat index 0; guard the singular kernel there
        safe = xp.where(k_sq == 0.0, xp.ones_like(k_sq), k_sq)
        influence = xp.reshape(self.kernel.kspace(safe), (nf, m_total))
        influence = influence * self._pme_b_factor(xp, charge.dtype, device)[None, :]
        # the zero mode sits at flat index 0 (tin-foil); keep everything else
        keep = xp.concat(
            [
                xp.zeros((1,), dtype=xp.bool, device=device),
                xp.ones((m_total - 1,), dtype=xp.bool, device=device),
            ]
        )
        influence = xp.where(keep[None, :], influence, xp.zeros_like(influence))
        influence = influence * xp.astype(mesh.mask, charge.dtype)[:, None]
        half_ke = 0.5 * COULOMB_CONSTANT
        e_frame = (
            half_ke * xp.sum(influence * xp.sum(rho_sq, axis=1), axis=-1) / volume
        )  # (nf,)
        weight = xp.astype(node_mask, charge.dtype)
        count = segment_sum(weight, frame_id, nf)
        count = xp.maximum(count, xp.ones_like(count))
        e_rec = xp.take(e_frame / count, frame_id, axis=0) * weight  # (N,)
        e_self = (
            -half_ke
            * self.kernel.contact_value(charge)
            * xp.sum(charge * charge, axis=-1)
            * xp.astype(xp.take(mesh.mask, frame_id, axis=0), charge.dtype)
        )
        return e_rec + e_self

    def _pair_kernel(self, r_sq: Array, src: Array, n: int) -> Array:
        """Kernel values of the pairs.

        The pairs are located on the node axis for a backend that brings the
        kernel coefficients to the pairs through it.

        Parameters
        ----------
        r_sq
            Squared pair distances with shape ``(P,)``, in Å².
        src
            First node of every pair with shape ``(P,)``.
        n
            The number of nodes.

        Returns
        -------
        Array
            The kernel values with shape ``(P,)``, in Å⁻¹.
        """
        return self.kernel(r_sq)

    def _constrain(
        self,
        charge: Array,
        node_mask: Array,
        frame_id: Array,
        n_node: Array,
        fparam: Array | None,
        charge_spin: Array | None,
    ) -> Array:
        """Shift the constrained channel so that it sums to the frame total charge."""
        xp = array_api_compat.array_namespace(charge)
        nf = n_node.shape[0]
        channel = self.charge_constraint["channel"]
        target = self._total_charge(nf, charge, fparam, charge_spin)
        weight = xp.astype(node_mask, charge.dtype)
        count = segment_sum(weight, frame_id, nf)
        total = segment_sum(charge[:, channel], frame_id, nf)
        # A frame without a charged atom has nothing to shift.
        shift = (total - target) / xp.maximum(count, xp.ones_like(count))
        constrained = charge[:, channel] - xp.take(shift, frame_id, axis=0) * weight
        return xp.concat(
            [charge[:, :channel], constrained[:, None], charge[:, channel + 1 :]],
            axis=-1,
        )

    def _total_charge(
        self,
        nf: int,
        like: Array,
        fparam: Array | None,
        charge_spin: Array | None,
    ) -> Array:
        """The total charge of every frame, which the constrained channel sums to.

        Parameters
        ----------
        nf
            The number of frames.
        like
            An array giving the namespace, dtype and device of the result.
        fparam, charge_spin
            As in :meth:`call_graph`.

        Returns
        -------
        Array
            The total charges with shape ``(nf,)``, in units of :math:`e`.
        """
        xp = array_api_compat.array_namespace(like)
        device = array_api_compat.device(like)
        source = self.charge_constraint["source"]
        if source == "charge_spin" and self.charge_state is not None:
            state = xp.asarray(self.charge_state, device=device)
            target = xp.broadcast_to(state[:, 0], (nf,))
        elif source == "charge_spin":
            if charge_spin is None:
                raise ValueError(
                    "the charge constraint reads the total charge from "
                    "`charge_spin`, which was not given"
                )
            target = charge_spin[:, 0]
        elif source == "fparam":
            if fparam is None:
                raise ValueError(
                    "the charge constraint reads the total charge from "
                    "`fparam`, which was not given"
                )
            target = fparam[:, self.charge_constraint["fparam_index"]]
        else:
            target = xp.zeros((nf,), dtype=like.dtype, device=device)
        return xp.astype(target, like.dtype)

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        protection: float = 1e-2,
        stat_file_path: DPPath | None = None,
    ) -> None:
        if self.k_grid_mode == "fixed" and (
            self.k_grid_n is None or self.k_grid_n.shape[0] == 0
        ):
            # Materialize the samples once for the grid sizing and the
            # networks; a lazy sampler is shared by the callers.
            sampled = merged() if callable(merged) else merged
            self._size_k_grid(sampled)
            merged = sampled
        elif self.k_grid_mode == "pme" and self._pme_dims() is None:
            sampled = merged() if callable(merged) else merged
            self._size_pme_mesh(sampled)
            merged = sampled
        self.charge_net.compute_input_stats(
            merged, protection=protection, stat_file_path=stat_file_path
        )

    def _size_k_grid(self, sampled: list[dict]) -> None:
        r"""Size the fixed integer k grid by the largest sampled cell.

        Every sampled frame needs :math:`|n_d| \le k_{max} |a_d| / 2\pi` per
        axis at the wave-number cutoff of the kernel; the grid takes the
        elementwise maximum of the bounds over the frames. The kernel
        parameters are read now, so a grid is sized for the kernel at
        statistics time — frames that later outgrow it (larger cells, or a
        kernel moving its cutoff out) are truncated to the grid.
        """
        k_max = self._k_max()
        bounds = None
        for frame in sampled:
            box = frame.get("box")
            if box is None:
                continue
            box = np.asarray(to_numpy_array(box), dtype=np.float64)
            if box.size == 0:
                continue
            frame_max = np.max(k_grid_bounds_of_cells(box, k_max), axis=0)
            bounds = frame_max if bounds is None else np.maximum(bounds, frame_max)
        if bounds is None:
            raise ValueError(
                "the fixed k grid takes its size from the data statistics, "
                "but the sampled frames hold no simulation cell; provide "
                "periodic training data or configure `k_grid_bounds`"
            )
        self.k_grid_n = enumerate_k_grid_bounds(bounds)

    def _size_pme_mesh(self, sampled: list[dict]) -> None:
        r"""Size the PME mesh by the largest sampled cell.

        Every frame is evaluated on the shared mesh with the effective
        spacing :math:`|a_d|/n_d` per axis; the mesh takes the largest sampled
        cell axis down to the target spacing, rounded up to an FFT-smooth
        dimension. A frame larger than the mesh (a larger cell than the
        statistics saw) is evaluated with a coarser effective spacing — a
        smooth accuracy degradation, by design without a warning.
        """
        spacing = self._pme_target_spacing()
        a_max = None
        for frame in sampled:
            box = frame.get("box")
            if box is None:
                continue
            box = np.asarray(to_numpy_array(box), dtype=np.float64)
            if box.size == 0:
                continue
            frame_max = np.max(np.linalg.norm(box.reshape(-1, 3, 3), axis=-1), axis=0)
            a_max = frame_max if a_max is None else np.maximum(a_max, frame_max)
        if a_max is None:
            raise ValueError(
                "the PME mesh takes its size from the data statistics, "
                "but the sampled frames hold no simulation cell; provide "
                "periodic training data or configure `pme_mesh`"
            )
        dims = [_fft_smooth(int(np.ceil(a / spacing))) for a in a_max]
        self.pme_mesh = np.asarray(dims, dtype=np.int64)
        self._pme_n = (dims[0], dims[1], dims[2])

    def change_type_map(self, type_map: list[str]) -> None:
        self.charge_net.change_type_map(type_map=type_map)

    def set_case_embd(self, case_idx: int) -> None:
        self.charge_net.set_case_embd(case_idx)

    def serialize(self) -> dict:
        data = {
            "@class": "LongRange",
            "type": "latent_charge",
            "@version": 1,
            "charge_net": self.charge_net.serialize(),
            "kernel": self.kernel.serialize(),
            "charge_constraint": self.charge_constraint,
        }
        if self.periodic:
            data["periodic"] = True
            data["kspace_tolerance"] = self.kspace_tolerance
        if self.k_grid_mode != "ragged":
            data["k_grid_mode"] = self.k_grid_mode
        if self.k_grid_n is not None and self.k_grid_n.shape[0] > 0:
            data["k_grid_n"] = to_numpy_array(self.k_grid_n).tolist()
        if self.decouple_k_grid:
            data["decouple_k_grid"] = True
            data["frozen_k_max"] = self.frozen_k_max
        if self.k_grid_mode == "pme":
            data["pme_order"] = self.pme_order
            if self.pme_spacing is not None:
                data["pme_spacing"] = self.pme_spacing
            if self._pme_dims() is not None:
                data["pme_mesh"] = to_numpy_array(self.pme_mesh).tolist()
        if self.charge_state is not None:
            data["charge_state"] = to_numpy_array(self.charge_state).tolist()
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "LatentChargeLongRange":
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        obj = cls(
            LatentChargeFitting.deserialize(data["charge_net"]),
            BaseInteractionKernel.deserialize(data["kernel"]),
            data["charge_constraint"],
            periodic=data.get("periodic", False),
            kspace_tolerance=data.get("kspace_tolerance", 1e-8),
            k_grid_mode=data.get("k_grid_mode", "ragged"),
            decouple_k_grid=data.get("decouple_k_grid", False),
            pme_spacing=data.get("pme_spacing"),
            pme_order=data.get("pme_order", 4),
            pme_mesh=data.get("pme_mesh"),
        )
        if data.get("frozen_k_max") is not None:
            # The constructor derives the frozen cutoff from the (possibly
            # trained) kernel; the record holds the value the grid was
            # actually built with.
            obj.frozen_k_max = float(data["frozen_k_max"])
        if data.get("k_grid_n") is not None:
            obj.k_grid_n = np.asarray(data["k_grid_n"], dtype=np.int64).reshape(-1, 3)
        if data.get("charge_state") is not None:
            obj.fold_charge_state(np.asarray(data["charge_state"]).reshape(-1).tolist())
        return obj
