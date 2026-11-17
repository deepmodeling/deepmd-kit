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
from .reciprocal import (
    kspace_energy,
    mesh_energy,
)

CHARGE_TARGET_SOURCES = ("neutral", "charge_spin", "fparam")
"""Sources of the total charge a constrained channel sums to."""

CHARGE_NET_NEURON = [64, 64]
"""Default hidden widths of the charge network."""

PME_ORDER = 8
"""Default order of the B-splines that spread the charges to the mesh."""

PME_SPACING_FRACTION = 0.75
"""Default mesh spacing, as a fraction of the Nyquist spacing of the wave-number cutoff.

At order 8 the mesh evaluation of a periodic frame then stays within a few
parts in a million of the sum over the wave numbers, in the reciprocal energy
and in the force.
"""

CHARGE_NET_SEED_INDEX = 10_000
"""Child index of the fitting seed that seeds a charge network configured without a seed.

The networks of the fitting and of the descriptor draw from small child
indices of their seeds, so the charge network draws a stream of its own even
when the fitting and descriptor seeds coincide.
"""


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
    the lattice sum of the kernel over the images of the cell, evaluated in
    reciprocal space (see :mod:`~deepmd.dpmodel.long_range.reciprocal`),

    .. math::
        E_{\rm non-periodic} = k_e \sum_{i<j} \sum_{c=1}^{C} q_i^c q_j^c K(r_{ij}),
        \qquad
        E_{\rm periodic} = k_e \Big[
        \frac{1}{2V} \sum_{\mathbf{k} \neq \mathbf{0}} \sum_{c=1}^{C}
        \hat K(\mathbf{k}) \, |\hat\rho_c(\mathbf{k})|^2
        - \frac{1}{2} K(0) \sum_{i,c} (q_i^c)^2 \Big],

    with the Coulomb constant :math:`k_e`, the kernel :math:`K` in Å⁻¹ and its
    Fourier transform :math:`\hat K`; every channel interacts through the same
    kernel and channels do not mix. The periodic sum drops the
    :math:`\mathbf{k} = \mathbf{0}` mode, the tin-foil convention, which
    places a uniform neutralizing background under a channel with a net
    charge. Either way an atom carries half of the interaction of its charges
    with all others, :math:`E_i = \frac{1}{2} \sum_c q_i^c \phi_i^c`.

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
        Whether the module serves periodic systems: it then declares a
        reciprocal-space topology beside
        :data:`~deepmd.dpmodel.utils.neighbor_graph.ALL_PAIRS` and evaluates
        a batch by its cells. Non-periodic batches keep working.
    kspace_tolerance
        The relative weight of the slowest-decaying kernel mode at the
        wave-number cutoff :math:`k_{max}` of a periodic batch.
    k_grid_mode
        How training evaluates the reciprocal sum of a periodic batch:
        ``"pme"``, the default, by the smooth particle-mesh Ewald method on a
        mesh of spacing ``pme_spacing`` sized by the cells of the batch;
        ``"ragged"`` over the wave numbers of every frame within
        :math:`k_{max}`, enumerated eagerly, so that the k axis is a dynamic
        dimension; or ``"fixed"`` over one integer grid shared by the frames,
        sized by the largest cell of the data statistics or by
        ``k_grid_bounds``, with the points a frame does not reach masked. A
        frame whose cell needs indices beyond the fixed grid is truncated to
        it, with a warning. A frozen model evaluates every periodic frame on
        the mesh, whatever the mode. A module that serves no periodic system
        evaluates no reciprocal sum and takes no mode.
    k_grid_bounds
        The per-axis integer bounds :math:`n_d^{max}` of the fixed grid, in
        place of the data statistics; only meaningful with
        ``k_grid_mode="fixed"``.
    decouple_k_grid
        Decouple the wave-number cutoff of a periodic batch from the kernel
        parameters: it is derived once from the initial parameters and kept
        for the life of the model, so that the k grid and the mesh stay put
        while the kernel trains. The kernel values still follow the
        live parameters. The frozen cutoff is serialized with the model.
        Requires ``periodic=True``.
    pme_spacing
        The largest spacing of the mesh of a periodic frame, in Å; the
        default is three quarters of the Nyquist spacing of the wave-number
        cutoff, :math:`0.75 \pi / k_{max}`. Requires ``periodic=True``.
    pme_order
        The order of the B-splines that spread the charges to the mesh.
        Requires ``periodic=True``.
    """

    def __init__(
        self,
        charge_net: LatentChargeFitting,
        kernel: BaseInteractionKernel,
        charge_constraint: dict | None = None,
        periodic: bool = False,
        kspace_tolerance: float = 1e-8,
        k_grid_mode: str | None = None,
        k_grid_bounds: list[int] | None = None,
        decouple_k_grid: bool = False,
        pme_spacing: float | None = None,
        pme_order: int | None = None,
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
        if k_grid_mode is None:
            k_grid_mode = "pme" if self.periodic else "ragged"
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
        # The per-axis bounds of the shared integer grid of the fixed mode,
        # zero until the data statistics size it (see
        # :meth:`compute_input_stats`): an array of fixed shape, so that a
        # backend keeping arrays in its state dict sees the same keys and
        # shapes before and after the sizing, and a checkpoint restores the
        # sized grid.
        if k_grid_bounds is not None:
            bounds = np.asarray(k_grid_bounds, dtype=np.int64).reshape(-1)
            if bounds.shape != (3,) or np.any(bounds < 0) or not np.any(bounds):
                raise ValueError(
                    "`k_grid_bounds` must be three non-negative integers, not "
                    f"all zero, got {list(k_grid_bounds)}"
                )
            self.k_grid_bounds = bounds
        elif self.k_grid_mode == "fixed":
            self.k_grid_bounds = np.zeros(3, dtype=np.int64)
        else:
            self.k_grid_bounds = None
        # The mesh of a periodic frame, on which training in the "pme" mode
        # and every frozen model evaluate the reciprocal sum.
        if not self.periodic and (pme_spacing is not None or pme_order is not None):
            raise ValueError("`pme_spacing` and `pme_order` require `periodic=True`")
        self.pme_order = PME_ORDER if pme_order is None else int(pme_order)
        if self.pme_order < 2:
            raise ValueError(f"`pme_order` must be at least 2, got {self.pme_order}")
        if pme_spacing is not None:
            pme_spacing = float(pme_spacing)
            if not np.isfinite(pme_spacing) or pme_spacing <= 0.0:
                raise ValueError(
                    f"`pme_spacing` must be finite and positive, got {pme_spacing}"
                )
        self.pme_spacing = pme_spacing

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
            (the ``les`` kernel when absent) and ``charge_constraint``.
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
            "type": "les",
            "precision": charge_net_config["precision"],
            **config.get("kernel", {}),
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
            k_grid_mode=config.get("k_grid_mode"),
            k_grid_bounds=config.get("k_grid_bounds"),
            decouple_k_grid=config.get("decouple_k_grid", False),
            pme_spacing=config.get("pme_spacing"),
            pme_order=config.get("pme_order"),
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

    def mesh_spacing(self) -> float:
        r"""The largest spacing of the mesh of a periodic frame, in Å.

        The configured ``pme_spacing``, or :data:`PME_SPACING_FRACTION` of the
        Nyquist spacing :math:`\pi / k_{max}` of the wave-number cutoff.
        """
        if self.pme_spacing is not None:
            return self.pme_spacing
        return PME_SPACING_FRACTION * math.pi / self._k_max()

    def use_mesh_evaluation(self) -> None:
        """Evaluate the reciprocal sum of every periodic batch on the mesh.

        A frozen model takes the mesh of every frame as an input of dynamic
        size, sized by the cell of the frame, so that it serves any cell at
        the accuracy of :meth:`mesh_spacing`; the explicit wave numbers of
        the other modes have a count that no input layout fixes. The spacing
        is fixed at the current kernel parameters, a constant of the frozen
        model.
        """
        if self.periodic:
            self.pme_spacing = self.mesh_spacing()
            self.k_grid_mode = "pme"

    def topology_params(self) -> dict[str, dict[str, Any]]:
        """The build parameters of the reciprocal-space topology of a periodic batch.

        The wave-number cutoff reads the kernel parameters at build time, so
        that a trained kernel moves the k grid and the mesh with it, unless
        ``decouple_k_grid`` keeps the initial cutoff. The k grid holds the
        wave numbers of every frame within the cutoff, or the points of the
        fixed integer grid with those beyond it masked; the mesh resolves
        every cell of the batch at :meth:`mesh_spacing`.

        Raises
        ------
        ValueError
            If the fixed grid has not been sized.
        """
        if not self.periodic:
            return {}
        if self.k_grid_mode == "pme":
            return {PME_MESH: {"spacing": self.mesh_spacing()}}
        params = {"k_max": self._k_max()}
        if self.k_grid_mode == "fixed":
            if not self._k_grid_sized():
                raise ValueError(
                    "the fixed k grid is not sized yet: run the data "
                    "statistics (`compute_input_stats`) on periodic "
                    "frames, or configure `k_grid_bounds`"
                )
            params["bounds"] = to_numpy_array(self.k_grid_bounds)
        return {K_GRID: params}

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
            # The lattice sum of a periodic frame and its self term cancel to
            # a small fraction of either, so the reciprocal-space energy is
            # formed in float64 whatever the precision of the geometry.
            charge_64 = xp.astype(charge, xp.float64)
            if self.k_grid_mode == "pme":
                reciprocal = mesh_energy(
                    charge_64,
                    topologies[PME_MESH],
                    frame_id,
                    self.kernel,
                    self.pme_order,
                )
            else:
                reciprocal = kspace_energy(charge_64, topologies[K_GRID], self.kernel)
            energy = energy + xp.astype(reciprocal, energy.dtype)
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
        if self.k_grid_mode == "fixed" and not self._k_grid_sized():
            # Materialize the samples once for the grid sizing and the
            # networks; a lazy sampler is shared by the callers.
            sampled = merged() if callable(merged) else merged
            self._size_k_grid(sampled)
            merged = sampled
        self.charge_net.compute_input_stats(
            merged, protection=protection, stat_file_path=stat_file_path
        )

    def _size_k_grid(self, sampled: list[dict]) -> None:
        r"""Size the fixed integer k grid by the largest sampled cell.

        Every sampled frame needs :math:`|n_d| \le k_{max} |a_d| / 2\pi` per
        axis at the wave-number cutoff of the kernel; the grid takes the
        elementwise maximum of the bounds over the frames. The kernel
        parameters are read at statistics time, so the grid is sized for the
        kernel of that moment; frames that outgrow it later, by larger cells
        or a kernel whose cutoff grows, are truncated to the grid.
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
        self.k_grid_bounds = bounds.astype(np.int64)

    def _k_grid_sized(self) -> bool:
        """Whether the fixed integer grid has its bounds."""
        return bool(np.any(to_numpy_array(self.k_grid_bounds)))

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
            data["k_grid_mode"] = self.k_grid_mode
        if self.k_grid_mode == "fixed" and self._k_grid_sized():
            data["k_grid_bounds"] = to_numpy_array(self.k_grid_bounds).tolist()
        if self.decouple_k_grid:
            data["decouple_k_grid"] = True
            data["frozen_k_max"] = self.frozen_k_max
        if self.periodic:
            data["pme_order"] = self.pme_order
            if self.pme_spacing is not None:
                data["pme_spacing"] = self.pme_spacing
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
            # A periodic record without a mode is one of the explicit wave
            # numbers.
            k_grid_mode=data.get("k_grid_mode", "ragged"),
            k_grid_bounds=data.get("k_grid_bounds"),
            decouple_k_grid=data.get("decouple_k_grid", False),
            pme_spacing=data.get("pme_spacing"),
            pme_order=data.get("pme_order"),
        )
        if data.get("frozen_k_max") is not None:
            # The constructor derives the frozen cutoff from the (possibly
            # trained) kernel; the record holds the value the grid was
            # actually built with.
            obj.frozen_k_max = float(data["frozen_k_max"])
        if data.get("charge_state") is not None:
            obj.fold_charge_state(np.asarray(data["charge_state"]).reshape(-1).tolist())
        return obj
