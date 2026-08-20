# SPDX-License-Identifier: LGPL-3.0-or-later
"""PyTorch-exportable execution backend for DPA4C.

``deepmd.dpmodel.descriptor.dpa4c`` defines the graph algorithm and tensor
contracts. This module implements its performance-critical primitives with
native PyTorch operations, promotes DPA4C trainable arrays to parameters,
owns the mixed-precision and compressed-inference policies, and registers the
descriptor with the pt_expt backend.
"""

from typing import (
    Any,
    ClassVar,
)

import torch

from deepmd.dpmodel.descriptor.dpa4c import DescrptDPA4C as DescrptDPA4CDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.kernels.utils import (
    fused_energy_force_enabled,
    fused_operators_enabled,
    use_amp_infer,
)
from deepmd.pt_expt.utils.update_sel import (
    UpdateSel,
)

#: Learned arrays that the dpmodel keeps as buffers, keyed by owning class.
_TRAINABLE_ATTRS: dict[str, tuple[str, ...]] = {
    "SeZMTypeEmbedding": ("adam_type_embedding",),
    "RadialBasis": ("adam_freqs",),
    "OrderedPairFiLM": (
        "adam_spin_scale_anchor",
        "adam_spin_shift_anchor",
    ),
    "SpinChannels": (
        "adam_spin_vector_weight",
        "adam_spin_quadrupole_weight",
        "spin_gate",
    ),
}


def _promote_trainable_tree(module: torch.nn.Module) -> torch.nn.Module:
    """Expose the dpmodel's learned buffers as PyTorch parameters.

    The two passes cannot be merged. Freezing recurses into children, so a
    frozen parent can only detach a trainable child's array once that array
    has become a parameter, which the first pass guarantees for the whole
    tree before the second pass runs.

    Parameters
    ----------
    module
        Descriptor tree to promote in place.

    Returns
    -------
    torch.nn.Module
        The same module, for use as an expression.
    """
    for submodule in module.modules():
        if not getattr(submodule, "trainable", True):
            continue
        for name in _TRAINABLE_ATTRS.get(type(submodule).__name__, ()):
            value = submodule._buffers.get(name)
            if value is None or not value.is_floating_point():
                continue
            del submodule._buffers[name]
            setattr(submodule, name, torch.nn.Parameter(value, requires_grad=True))

    for submodule in module.modules():
        if not getattr(submodule, "trainable", True):
            for parameter in submodule.parameters(recurse=True):
                parameter.requires_grad_(False)
    return module


@BaseDescriptor.register("dpa4c")
@torch_module
class DescrptDPA4C(DescrptDPA4CDP):
    """Execute the backend-neutral DPA4C equations with PyTorch tensors.

    Notes
    -----
    DPA4C components such as ``SeZMTypeEmbedding`` and ``RadialBasis`` store
    trainable arrays in the dpmodel representation. The wrapper promotes those
    arrays after construction and deserialization so they remain visible to
    PyTorch optimizers and force-loss double backward. Graph gathers and cutoff
    evaluation use native PyTorch primitives; the segment reductions and the
    shared geometry and invariant modules already dispatch to PyTorch tensor
    operations through the backend-neutral equations.

    This wrapper also owns the ``use_amp`` policy, because the array API has no
    autocast and the dpmodel therefore only records the flag.
    """

    _update_sel_cls = UpdateSel

    #: Artifacts whose element type is not the ``float32`` the kernel consumes.
    _COMPRESSION_BUFFER_DTYPES: ClassVar[dict[str, torch.dtype]] = {
        "info": torch.float64,
        "coupling_meta": torch.int32,
        "coupling_entry": torch.int32,
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Construct and parameterize a PyTorch DPA4C descriptor.

        Parameters
        ----------
        *args
            Positional arguments forwarded to the dpmodel DPA4C constructor.
        **kwargs
            Keyword arguments forwarded to the dpmodel DPA4C constructor.
        """
        super().__init__(*args, **kwargs)
        _promote_trainable_tree(self)
        self.compress = False
        # Eval-time AMP is opted into through the environment and captured
        # once, so a traced graph cannot depend on a later mutation.
        self.use_amp_infer = use_amp_infer()
        self._apply_autocast_policy()

    def call_graph(
        self,
        graph: Any,
        atype: torch.Tensor,
        type_embedding: torch.Tensor | None = None,
        comm_dict: dict | None = None,
        spin: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Evaluate the graph descriptor with compressed CUDA dispatch.

        Parameters
        ----------
        graph
            NeighborGraph over the flat node axis.
        atype
            Flat atom types with shape ``(N,)``.
        type_embedding
            Optional complete DPA4 type table.
        comm_dict
            Communication metadata accepted by the common graph ABI; unused.
        spin
            Per-node spin with shape ``(N, 3)``, mandatory for a
            spin-conditioned descriptor.
        charge_spin
            Frame-level charge state with shape ``(nf, 2)``. The compressed
            branch does not read it: its frozen tables already carry the
            charge state that compression baked in, which is why a compressed
            descriptor reports a zero runtime condition width.

        Returns
        -------
        descriptor
            Invariant descriptor with shape ``(N, get_dim_out())``.
        rot_mat
            ``None``.
        """
        if (
            self.compress
            and not self.training
            and not self.exclude_types
            and fused_operators_enabled()
            and graph.destination_order is not None
            and graph.destination_row_ptr is not None
        ):
            from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
                dpa4c_graph_compress,
                mega_eligible,
                op_available,
            )

            if op_available(self.spin is not None) and mega_eligible(self):
                # The operator conditions the moment on device from its frozen
                # per-type table, so it takes the raw input rather than the
                # output of ``SpinChannels.conditioned_spin``.
                return dpa4c_graph_compress(
                    self,
                    graph,
                    atype,
                    None if self.spin is None else self.require_spin(spin),
                ), None
        if type_embedding is None:
            type_embedding = self.type_embedding.call()
        return super().call_graph(
            graph,
            atype,
            type_embedding=type_embedding,
            comm_dict=comm_dict,
            spin=spin,
            charge_spin=charge_spin,
        )

    def build_edge_features(
        self,
        graph: Any,
        *args: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Build the edge features under the DPA4C mixed-precision policy.

        The per-edge stage is the only region DPA4C autocasts. It holds every
        tensor that scales with the edge count and every large matrix product,
        namely the radial network and the pair-conditioned mode mixing, so
        bfloat16 halves the dominant activation footprint. Autocast leaves the
        geometry, the cutoff envelope, and the harmonics in full precision
        because they are elementwise.

        The region ends at the returned features. The destination reduction
        accumulates over the whole neighborhood and the readout raises the
        moments to the fourth power, so both stay in the descriptor compute
        precision; the ordered pair cache is likewise evaluated outside, over
        the finite type table. A charge-conditioned descriptor is the
        exception: its conditioning heads run on the edge axis and therefore
        inside the region, producing their bounded outputs in bfloat16
        alongside the amplitude they scale.

        Training follows ``use_amp`` and evaluation follows ``DP_AMP_INFER``.
        The two are independent: mixed precision at inference is a throughput
        choice that must not require a model to have been trained with it.

        A spin-conditioned descriptor never autocasts. Its scalar and
        quadrupole families are quadratic in the magnetic moment and feed a
        fourth-order readout, and the magnetic force differentiates them
        twice, so the eight mantissa bits of bfloat16 are not an acceptable
        trade for a configuration whose throughput is not the binding
        constraint.

        Parameters
        ----------
        graph
            Neighbor graph in descriptor compute precision.
        *args
            Node types, the ordered pair cache, and the conditioned spin,
            forwarded unchanged to the backend-neutral implementation.

        Returns
        -------
        amplitude
            Masked edge amplitudes with shape ``(E, channels)``.
        basis
            Masked Cartesian harmonics with shape ``(E, (lmax + 1) ** 2)``.
        envelope
            Masked C³ envelope with shape ``(E,)``.
        spin_payload
            Masked per-edge spin payload, or ``None``.
        """
        autocast = (
            self.spin is None
            and graph.edge_vec.device.type == "cuda"
            and (self.use_amp if self.training else self.use_amp_infer)
        )
        if not autocast:
            return super().build_edge_features(graph, *args)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            features = super().build_edge_features(graph, *args)
        dtype = graph.edge_vec.dtype
        return tuple(
            None if feature is None else feature.to(dtype) for feature in features
        )

    def _apply_autocast_policy(self) -> None:
        """Let the layers inside the autocast region emit reduced precision.

        ``NativeLayer`` restores the dtype of its input after the affine map,
        which would undo autocast at every layer of the radial network. Only
        the radial trunk and the mode head sit inside the region, so only they
        are opted out of that restoration, and only when mixed precision can
        actually engage. A descriptor with neither switch set, and every
        spin-conditioned descriptor, keeps the default behavior exactly.
        """
        enabled = self.spin is None and (self.use_amp or self.use_amp_infer)
        for layer in self.radial_embedding.layers:
            layer.autocast_output = enabled
        if self.radial_mode_head is not None:
            self.radial_mode_head.autocast_output = enabled

    # === Backend primitives ===

    def gather_rows(
        self,
        values: torch.Tensor,
        index: torch.Tensor,
        xp: Any | None = None,
    ) -> torch.Tensor:
        """Gather rows along the leading axis.

        Parameters
        ----------
        values
            Source tensor with shape ``(N, ...)``.
        index
            Row indices with shape ``(M,)``. Every DPA4C gather addresses a
            flat node or edge axis, so a one-dimensional index suffices and
            ``index_select`` avoids the advanced-indexing path.
        xp
            Array namespace accepted by the backend-neutral hook; unused.

        Returns
        -------
        torch.Tensor
            Gathered values with shape ``(M, *values.shape[1:])``.
        """
        del xp
        return torch.index_select(values, 0, index)

    def evaluate_cutoff_envelope(self, distance: torch.Tensor) -> torch.Tensor:
        """Evaluate the fixed C³ cutoff envelope.

        Parameters
        ----------
        distance
            Regularized edge distances with shape ``(E, 1)`` in Å.

        Returns
        -------
        torch.Tensor
            Envelope values with shape ``(E, 1)``.
        """
        u = torch.clamp(
            (self.rcut - distance) / self.rcut,
            min=0.0,
            max=1.0,
        )
        x = 1.0 - u
        series = 1.0 + x * (4.0 + x * (10.0 + x * (20.0 + x * 35.0)))
        return u**4 * series

    # === Statistics and parameter sharing ===

    def compute_input_stats(
        self,
        merged: Any,
        path: Any | None = None,
    ) -> None:
        """Calibrate output features without retaining an autograd graph."""
        with torch.no_grad():
            super().compute_input_stats(merged, path)

    def share_params(
        self,
        base_class: Any,
        shared_level: int,
        model_prob: float = 1.0,
        resume: bool = False,
    ) -> None:
        """Reject compressed snapshots, then share as the base descriptor does.

        Assigning a submodule of ``base_class`` registers it in this module's
        submodule table, so the backend-neutral implementation already
        produces correct PyTorch sharing. Rebinding the whole submodule table
        instead would additionally capture the pair-exclusion mask, which is
        branch-local state.

        Parameters
        ----------
        base_class
            DPA4C descriptor that owns the shared parameters.
        shared_level
            Sharing level. Only level zero is supported.
        model_prob
            Model sampling probability accepted by the common ABI; unused.
        resume
            Checkpoint-restoration flag accepted by the common ABI; unused.

        Raises
        ------
        RuntimeError
            If either descriptor is a compressed inference snapshot.
        """
        if self.compress or bool(getattr(base_class, "compress", False)):
            raise RuntimeError(
                "Compressed DPA4C snapshots cannot participate in parameter sharing."
            )
        super().share_params(base_class, shared_level, model_prob, resume)

    # === Compressed-inference artifacts ===

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA4C":
        """Deserialize DPA4C and restore trainable PyTorch parameters.

        Parameters
        ----------
        data
            Versioned dpmodel descriptor dictionary.

        Returns
        -------
        DescrptDPA4C
            Reconstructed PyTorch descriptor.
        """
        data = data.copy()
        compression = data.pop("compress", None)
        obj = super().deserialize(data)
        obj = _promote_trainable_tree(obj)
        # Deserialization rebuilds the radial modules, so the autocast policy
        # has to be reapplied to the fresh layers.
        obj._apply_autocast_policy()
        obj.compress = False
        if compression is not None:
            obj._set_compression(
                {
                    name: torch.as_tensor(compression["@variables"][name])
                    for name in obj._COMPRESSION_BUFFER_NAMES
                }
            )
        return obj

    def _set_compression(
        self,
        artifacts: dict[str, torch.Tensor],
    ) -> None:
        """Store immutable compressed-inference artifacts as module buffers.

        The metadata block keeps ``float64`` so that the radial table stride
        and the cutoff survive without rounding, and the angular coupling
        layout keeps ``int32``; every other artifact is the ``float32`` the
        kernel consumes.
        """
        device = next(self.parameters()).device
        info = torch.as_tensor(artifacts["info"])
        self._compression_scalars = tuple(
            float(value) for value in info.detach().cpu().tolist()
        )
        for name in self._COMPRESSION_BUFFER_NAMES:
            dtype = self._COMPRESSION_BUFFER_DTYPES.get(name, torch.float32)
            value = artifacts[name].to(device=device, dtype=dtype).contiguous()
            buffer_name = f"compress_{name}"
            if buffer_name in self._buffers:
                self._buffers[buffer_name] = value
            else:
                self.register_buffer(buffer_name, value)
        self.compress = True

    def apply_charge_state(self, charge_spin: Any) -> None:
        """Re-specialize a compressed snapshot to a frame charge state.

        The condition reaches the compiled kernel only through the ordered
        pair encoder and the centre type table, so a charge state is fully
        described by four of the frozen artifacts. Rebuilding those four
        moves the snapshot to a different state at the cost of one evaluation
        over the finite type table, leaving the radial table, the readout
        projections, the angular couplings, the per-type spin scalars and the
        output calibration untouched.

        The state is a constant of a molecular-dynamics run, so this is a
        load-time operation. Applying it on every step would pay an
        evaluation over ``(T + 1) ** 2`` ordered pairs for a value that never
        changes, and that evaluation does not shrink with the system size.

        Parameters
        ----------
        charge_spin
            Frame condition ``[charge, multiplicity]``.

        Raises
        ------
        RuntimeError
            If the descriptor is not a compressed snapshot, or carries no
            charge conditioning.
        """
        if not self.compress:
            raise RuntimeError(
                "A charge state is applied to the frozen tables of a "
                "compressed DPA4C snapshot; an uncompressed descriptor reads "
                "the condition directly on every call."
            )
        if self.charge_spin_embedding is None:
            raise RuntimeError(
                "This DPA4C was not built with `add_chg_spin_ebd`, so it has "
                "no charge state to apply."
            )
        from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
            build_charge_state_artifacts,
        )

        artifacts = build_charge_state_artifacts(self, charge_spin)
        device = self.compress_pair_film.device
        for name, value in artifacts.items():
            self._buffers[f"compress_{name}"] = value.to(
                device=device,
                dtype=torch.float32,
            ).contiguous()

    def set_stat_mean_and_stddev(self, mean: Any, stddev: Any) -> None:
        """Update output calibration and its compressed snapshot."""
        super().set_stat_mean_and_stddev(mean, stddev)
        if self.compress:
            device = self.compress_output_mean.device
            self._buffers["compress_output_mean"] = torch.as_tensor(
                mean,
                dtype=torch.float32,
                device=device,
            ).contiguous()
            self._buffers["compress_output_inv_std"] = torch.reciprocal(
                torch.as_tensor(
                    stddev,
                    dtype=torch.float32,
                    device=device,
                )
            ).contiguous()

    def train(self, mode: bool = True) -> "DescrptDPA4C":
        """Set training mode while preserving compression immutability."""
        if mode and self.compress:
            raise RuntimeError(
                "A compressed DPA4C descriptor is an immutable inference "
                "snapshot and cannot re-enter training mode."
            )
        return super().train(mode)

    def compression_needs_min_nbor_dist(self) -> bool:
        """Return whether compression consumes the minimum neighbor distance.

        Returns
        -------
        bool
            Always ``False``. The radial table spans ``[0, rcut]``, a domain
            fixed by the cutoff rather than by the training data, so the
            caller can skip the neighbor-statistics pass.
        """
        return False

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 1.0,
        table_stride_1: float = 0.002,
        table_stride_2: float = 0.002,
        check_frequency: int = -1,
    ) -> None:
        """Build immutable artifacts for the current DPA4C mega kernel.

        A charge-conditioned descriptor folds ``default_chg_spin`` into the
        frozen type and ordered pair tables. The snapshot therefore evaluates
        that one charge state at no runtime cost, and moves to another through
        :meth:`apply_charge_state` rather than through a second compression.

        Parameters
        ----------
        min_nbor_dist
            Minimum neighbor distance accepted by the common compression ABI.
            DPA4C tabulates the finite DPA4 radial basis from zero to ``rcut``
            and therefore does not use this value.
        table_extrapolate
            Common compression parameter; unused because the C³ radial map is
            exactly zero beyond ``rcut``.
        table_stride_1
            Uniform radial spline spacing in Å.
        table_stride_2
            Common two-region table spacing; unused by the uniform DPA4C table.
        check_frequency
            Common overflow-check setting; unused because the radial domain is
            bounded analytically.

        Raises
        ------
        ValueError
            If compression is already enabled, the descriptor excludes type
            pairs, or the descriptor configuration has no compiled CUDA
            specialization.
        """
        del min_nbor_dist, table_extrapolate, table_stride_2, check_frequency
        if self.compress:
            raise ValueError("Compression is already enabled.")
        from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
            build_compression_artifacts,
        )

        self._set_compression(build_compression_artifacts(self, table_stride_1))

    def fused_energy_force_graph(
        self,
        fitting: Any,
        graph: Any,
        atype: torch.Tensor,
        ownership: torch.Tensor,
        atom_bias: torch.Tensor,
        do_atomic_virial: bool,
        spin: torch.Tensor | None = None,
    ) -> (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None
    ):
        """Evaluate the inference-only compressed energy-force composition.

        Returns ``None`` when the model or graph cannot use the level-two CUDA
        path, allowing the caller to retain the generic autograd lower. The
        trailing output is the magnetic force, empty for a spin-free model.
        """
        if (
            self.training
            or not self.compress
            or bool(self.exclude_types)
            or not fused_energy_force_enabled()
            or graph.destination_order is None
            or graph.destination_row_ptr is None
            or graph.source_order is None
            or graph.source_row_ptr is None
        ):
            return None
        from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
            dpa4c_graph_compress_energy_force,
            ef_op_available,
            mega_eligible,
        )
        from deepmd.pt_expt.kernels.graph_fitting import (
            fitting_eligible,
        )

        if (
            not ef_op_available(self.spin is not None)
            or not mega_eligible(self)
            or not fitting_eligible(fitting)
        ):
            return None
        return dpa4c_graph_compress_energy_force(
            self,
            fitting,
            graph,
            atype,
            ownership,
            atom_bias,
            atype.shape[0],
            do_atomic_virial,
            None if self.spin is None else self.require_spin(spin),
        )
