# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.common import (
    get_xp_precision,
)
from deepmd.dpmodel.descriptor.dpa4 import DescrptDPA4 as DescrptDPA4DP
from deepmd.dpmodel.descriptor.dpa4_nn.activation import SwiGLU as SwiGLUDP
from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import GridProduct as GridProductDP
from deepmd.dpmodel.descriptor.dpa4_nn.radial import BridgingSwitch as BridgingSwitchDP
from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
    C3CutoffEnvelope as C3CutoffEnvelopeDP,
)
from deepmd.dpmodel.descriptor.dpa4_nn.radial import InnerClamp as InnerClampDP
from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
    attach_edge_csr,
    compact_edges,
)
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.kernels.utils import (
    cuda_infer_level,
    use_amp_infer,
)
from deepmd.pt_expt.utils.update_sel import (
    UpdateSel,
)


@torch_module
class SwiGLU(SwiGLUDP):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)


# SwiGLU is parameter-free (no serialize); rebuild fresh.
register_dpmodel_mapping(SwiGLUDP, lambda v: SwiGLU())


@torch_module
class C3CutoffEnvelope(C3CutoffEnvelopeDP):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)


@torch_module
class InnerClamp(InnerClampDP):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)


# InnerClamp/BridgingSwitch are parameter-free (scalar bridging radii only,
# no serialize()); rebuild fresh from the stored constructor arguments.
register_dpmodel_mapping(
    InnerClampDP,
    lambda v: InnerClamp(v.r_inner, v.r_outer),
)


@torch_module
class BridgingSwitch(BridgingSwitchDP):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)


register_dpmodel_mapping(
    BridgingSwitchDP,
    lambda v: BridgingSwitch(v.r_inner, v.r_outer),
)


# C3CutoffEnvelope carries only scalar configuration (cutoff radius and
# polynomial exponent) and holds no trainable arrays, so it implements no
# serialize()/deserialize() that the generic auto-wrap path relies on; rebuild
# it directly from the stored constructor arguments (``p`` is the exponent).
register_dpmodel_mapping(
    C3CutoffEnvelopeDP,
    lambda v: C3CutoffEnvelope(v.rcut, v.p, precision=v.precision),
)


@torch_module
class GridProduct(GridProductDP):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)


# GridProduct is a parameter-free quadratic grid product with no constructor
# arguments and no serialize()/deserialize(); rebuild a fresh instance.
register_dpmodel_mapping(GridProductDP, lambda v: GridProduct())


# ---------------------------------------------------------------------------
# Trainable-weight promotion
#
# ``dpmodel_setattr`` registers every numpy attribute as a torch *buffer*, so
# the auto-wrapped dpa4_nn sub-modules would otherwise expose their trainable
# weights as non-trainable buffers (no autograd, invisible to the optimizer).
# The table below lists, per dpmodel class name, the attributes that are
# ``torch.nn.Parameter`` in the reference pt SeZM implementation
# (deepmd/pt/model/descriptor/sezm_nn).  ``_promote_trainable_tree`` walks the
# fully-built module tree and re-registers those buffers as Parameters.
#
# Constant float buffers (e.g. ``balance_weight``, ``rotate_inv_rescale_full``,
# ``mean``/``stddev``) are intentionally NOT listed: they are buffers in pt
# too.  Lists of weights (e.g. ``SO2Linear.weight_m``) are already converted
# to trainable ``ParameterList`` by ``_try_convert_list``.
# ---------------------------------------------------------------------------
_TRAINABLE_ATTRS: dict[str, tuple[str, ...]] = {
    # dpa4_nn.norm
    "RMSNorm": ("adam_scale",),
    "EquivariantRMSNorm": ("adam_scale", "bias"),
    "ReducedEquivariantRMSNorm": ("adam_scale", "bias0"),
    "ScalarRMSNorm": ("adam_scale",),
    # dpa4_nn.radial
    "RadialBasis": ("adam_freqs",),
    # dpa4_nn.so3
    "SO3Linear": ("weight", "bias"),
    "FocusLinear": ("weight", "bias"),
    "ChannelLinear": ("weight", "bias"),
    # dpa4_nn.so2
    "SO2Linear": ("weight_m0", "bias0"),
    "DynamicRadialDegreeMixer": ("weight", "channel_basis"),
    "SO2Convolution": (
        "adamw_attn_logit_w",
        "adamw_attn_z_bias_raw",
        "adamw_attn_gate_w",
        "adamw_focus_compete_w",
        "focus_compete_bias",
    ),
    # dpa4_nn.embedding
    "SeZMTypeEmbedding": ("adam_type_embedding",),
    # dpa4_nn.embedding (native spin): these are nn.Parameter in pt but land as
    # numpy->buffer in dpmodel; mag_layer1/2 are NativeLayer and auto-promote,
    # and _promote_trainable skips a missing buffer, so no-spin configs (where
    # spin_scale is absent) stay safe.
    "SpinEmbedding": ("adam_spin_vec_weight", "adam_spin_nbr_weight"),
    "EnvironmentInitialEmbedding": ("spin_scale",),
    # dpa4_nn.attn_res
    "DepthAttnRes": ("adamw_pseudo_query",),
    # dpa4_nn.grid_net (residual_scale is None when disabled; _promote_trainable
    # skips the missing buffer, so listing both concrete subclasses is safe)
    "S2GridNet": ("residual_scale",),
    "SO3GridNet": ("residual_scale",),
    # dpa4_nn.grid_net frame mixing, built only by ``mode="cross"`` grid nets.
    # Unlike the surrounding projections these are plain numpy arrays rather
    # than NativeLayer objects, so they need an explicit entry here.
    "FrameExpand": ("weight",),
    "FrameContract": ("weight",),
    # descriptor-level FiLM strengths
    "DescrptDPA4": ("film_scale_strength_log", "film_shift_strength_log"),
}


def _promote_trainable(module: torch.nn.Module, names: tuple[str, ...]) -> None:
    """Re-register the given float buffers of *module* as Parameters."""
    if not getattr(module, "trainable", True):
        return
    for name in names:
        buf = module._buffers.get(name)
        if buf is None or not buf.is_floating_point():
            continue
        del module._buffers[name]
        setattr(module, name, torch.nn.Parameter(buf, requires_grad=True))


def _promote_trainable_tree(module: torch.nn.Module) -> torch.nn.Module:
    """Promote trainable buffers to Parameters across the whole module tree.

    Must run after the tree is fully built (post ``__init__`` /
    ``deserialize``): dpmodel deserialize may assign numpy arrays onto nested
    attributes, which ``dpmodel_setattr`` would re-register as buffers.
    """
    for sub in module.modules():
        names = _TRAINABLE_ATTRS.get(type(sub).__name__)
        if names is not None:
            _promote_trainable(sub, names)
    # Freeze every Parameter under a ``trainable=False`` module.  This covers
    # parameters that exist regardless of the promotion table above, e.g. the
    # ``SO2Linear.weight_m`` list, which ``_try_convert_list`` converts to a
    # ParameterList with ``requires_grad=True`` unconditionally.
    for sub in module.modules():
        if getattr(sub, "trainable", True) is False:
            for p in sub.parameters(recurse=True):
                p.requires_grad_(False)
    return module


@BaseDescriptor.register("SeZM")
@BaseDescriptor.register("sezm")
@BaseDescriptor.register("DPA4")
@BaseDescriptor.register("dpa4")
@torch_module
class DescrptDPA4(DescrptDPA4DP):
    _update_sel_cls = UpdateSel

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The fused convolution paths consume only the three structural rows of
        # each Wigner degree block. Source-gated attention bypasses that fused
        # convolution, so its dense per-edge rotations remain available.
        self.cuda_infer_l_2_covers_all_blocks = (
            self.bridging_switch is None
            and bool(self.blocks)
            and all(
                getattr(block.so2_conv, "cuda_infer_l_2_conv", None) is not None
                and not block.so2_conv.cuda_infer_l_2_conv.focus_compete
                for block in self.blocks
            )
        )
        self.cuda_train_covers_all_blocks = bool(self.blocks) and all(
            getattr(block.so2_conv, "cuda_train_value", None) is not None
            and block.so2_conv.flash_attention is not None
            and block.so2_conv.flash_attention_supports_training
            for block in self.blocks
        )

        # The envelope and the radial basis are both functions of the pair
        # distance and are cheap enough that the compiler inlines them into
        # every consumer and re-evaluates them there. Behind an operator
        # boundary the chain runs once per step.
        self.cuda_infer_l_1_radial = None
        self.cuda_infer_l_1_wigner = None
        if cuda_infer_level() >= 1:
            from deepmd.pt_expt.kernels.cuda.dpa4.edge_radial import (
                make_cuda_edge_radial,
            )
            from deepmd.pt_expt.kernels.cuda.dpa4.wigner_dense import (
                make_cuda_wigner_dense,
            )

            self.cuda_infer_l_1_radial = make_cuda_edge_radial(
                self.edge_envelope, self.radial_basis
            )
            # The dense Wigner pair otherwise costs five full-size passes
            # over the (E, D, D) tensors; the fused build pays only the
            # output writes.
            self.cuda_infer_l_1_wigner = make_cuda_wigner_dense(
                self.mp_init_lmax,
                get_xp_precision(torch, self.compute_precision),
            )

        # Persisted graph-routing knob (first-class training configuration):
        # ``disable_graph_lower()`` used to flip only the plain dpmodel bool,
        # which a Trainer checkpoint restart silently reset (the fresh model
        # is rebuilt from config before ``load_state_dict``, and neither the
        # state-dict keys nor ``_extra_state.model_params`` carried the
        # choice) -- on a binding-sel system that switched the training
        # equation and gradients without warning.  A persistent buffer rides
        # every pt_expt state_dict, so save/restart round-trips it.
        torch.nn.Module.register_buffer(
            self,
            "graph_lower_disabled",
            torch.zeros((), dtype=torch.bool, device="cpu"),
        )
        # Persisted descriptor version, for the same reason: pt_expt rebuilds
        # the module from config before loading, so without a buffer every
        # checkpoint would come back claiming the semantics of the running
        # code and silently skip ``_migrate_variables``.
        torch.nn.Module.register_buffer(
            self,
            "version_tensor",
            torch.tensor(self.version, dtype=torch.float64, device="cpu"),
        )
        self.use_amp_infer = use_amp_infer()
        _promote_trainable_tree(self)

    def _shared_wigner_runs(self, edge_cache: Any, lmax: int) -> torch.Tensor | None:
        """
        Zonal coupling taken from the packed runs the convolution already builds.

        The fused convolution stages a packed block-diagonal Wigner run per
        edge whose degree-``l`` ``m = 0`` row occupies entries ``l ** 2`` to
        ``(l + 1) ** 2``. That is the same quantity as
        ``Dt_full[:, row(l, m), col(l, 0)]``, so degrees ``1..lmax`` are one
        contiguous slice and the rotation algebra runs once per step instead of
        twice. The runs are cached on the edge cache, so whichever consumer
        comes first pays for them.

        Parameters
        ----------
        edge_cache : EdgeCache
            The step's edge feature cache.
        lmax : int
            Highest degree the coupling must cover.

        Returns
        -------
        torch.Tensor or None
            Coupling with shape ``(E, (lmax + 1) ** 2 - 1)``, or ``None`` when
            no convolution supplies runs of at least this degree.
        """
        if edge_cache.csr_cache is None:
            return None
        if self.training:
            if not self.cuda_train_covers_all_blocks:
                return None
            fused = self.blocks[0].so2_conv.cuda_train_value
        else:
            if not self.cuda_infer_l_2_covers_all_blocks:
                return None
            fused = self.blocks[0].so2_conv.cuda_infer_l_2_conv
        if fused is None or lmax > self.lmax:
            return None
        return fused.edge_runs(edge_cache)[:, 1 : (lmax + 1) ** 2]

    def is_cute_infer_packed_wigner_candidate(
        self,
        device: torch.device,
        dtype: torch.dtype,
        geometry_dtype: torch.dtype,
    ) -> bool:
        """Return whether all blocks satisfy the packed CuTe SO2 contract.

        Parameters
        ----------
        device : torch.device
            Device that executes the descriptor blocks.
        dtype : torch.dtype
            Descriptor block dtype.
        geometry_dtype : torch.dtype
            Edge-geometry compute dtype.

        Returns
        -------
        bool
            Whether packed Wigner storage can replace dense storage for every
            interaction block.
        """
        from deepmd.pt_expt.kernels.cute.sezm.so2.operation import (
            is_packed_wigner_candidate,
        )

        return is_packed_wigner_candidate(
            blocks=self.blocks,
            training=self.training,
            device=device,
            dtype=dtype,
            producer_modules=(
                self.radial_basis,
                self.radial_embedding,
                self.edge_envelope,
                *((self.inner_clamp,) if self.inner_clamp is not None else ()),
                *((self.bridging_switch,) if self.bridging_switch is not None else ()),
            ),
            producer_dtypes=(dtype, geometry_dtype),
            has_edge_src_gate=self.bridging_switch is not None,
        )

    def prepare_packed_wigner_graph(
        self,
        graph: NeighborGraph,
        n_nodes: int,
    ) -> NeighborGraph | None:
        """Prepare a compact destination-major graph for CuTe packed SO2.

        Parameters
        ----------
        graph : NeighborGraph
            Edge graph supplied to the descriptor.
        n_nodes : int
            Number of nodes addressed by the graph.

        Returns
        -------
        NeighborGraph or None
            Canonical graph without masked edges, or ``None`` when the exact
            CuTe contract is not satisfied.
        """
        dtype = get_xp_precision(torch, self.precision)
        if not self.is_cute_infer_packed_wigner_candidate(
            graph.edge_vec.device,
            dtype,
            graph.edge_vec.dtype,
        ):
            return None
        # Graph-native inputs retain canonical CSR through compaction. Dense
        # adapters and exclusion transforms construct it once at this boundary.
        graph = compact_edges(graph)
        if not graph.destination_sorted:
            graph = attach_edge_csr(graph, n_nodes, canonicalize=True)
        from deepmd.pt_expt.kernels.cute.sezm.so2.operation import (
            packed_wigner_edges_eligible,
        )

        if not packed_wigner_edges_eligible(
            candidate=True,
            edge_count=graph.edge_index.shape[1],
            node_count=n_nodes,
            destinations_sorted=graph.destination_sorted,
            runtime_dtypes=(
                dtype,
                get_xp_precision(torch, self.compute_precision),
            ),
        ):
            return None
        return graph

    def build_packed_wigner(
        self,
        edge_quat: torch.Tensor,
        wigner_calc: Any,
    ) -> torch.Tensor | None:
        """Build the CuTe packed Wigner panel for an eligible graph.

        Parameters
        ----------
        edge_quat : torch.Tensor
            Global-to-local edge quaternions with shape ``(E, 4)``.
        wigner_calc : Any
            Wigner calculator carrying the degree and basis convention.

        Returns
        -------
        torch.Tensor or None
            CuTe packed Wigner storage with shape ``(E, 46)``, or ``None`` when
            the kernel declines the runtime input.
        """
        from deepmd.pt_expt.kernels.cute.sezm.wignerd import (
            run_cute_wignerd,
        )

        result = run_cute_wignerd(
            edge_quat,
            wigner_calc,
            packed_wigner=True,
        )
        return None if result is None else result[0]

    def prepare_cute_infer_so2_metadata(
        self,
        edge_cache: Any,
        n_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Adapt graph-owned CSR metadata for the CuTe SO2 implementation.

        Parameters
        ----------
        edge_cache : EdgeCache
            Per-forward cache carrying the destination-major edge payload.
        n_nodes : int
            Number of nodes addressed by the edge payload.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] or None
            Destination row pointers, source order, and source row pointers, or
            ``None`` when the exact CuTe contract is not satisfied.
        """
        if (
            self.training
            or not edge_cache.destinations_sorted
            or edge_cache.D_packed is None
            or edge_cache.edge_src_gate is not None
        ):
            return None
        from deepmd.pt_expt.kernels.cute.sezm import (
            runtime_policy,
        )

        if not runtime_policy.is_cute_infer_enabled():
            return None
        csr_cache = edge_cache.csr_cache
        if csr_cache is None or "dst" not in csr_cache or "src" not in csr_cache:
            raise RuntimeError("packed CuTe SO2 requires destination/source graph CSR")
        destination_row_ptr = csr_cache["dst"][1]
        source_order, source_row_ptr = csr_cache["src"]
        if (
            destination_row_ptr.shape != (n_nodes + 1,)
            or source_order.shape != edge_cache.src.shape
            or source_row_ptr.shape != (n_nodes + 1,)
        ):
            raise RuntimeError("packed CuTe SO2 received inconsistent graph CSR shapes")
        if runtime_policy.is_cute_strict_enabled() and edge_cache.dst.numel() > 1:
            torch._assert_async(
                torch.all(edge_cache.dst[1:] >= edge_cache.dst[:-1]),
                "Neo SO2 destinations_sorted=True requires monotonically "
                "nondecreasing destination indices",
            )
        return (
            destination_row_ptr.to(dtype=torch.int32).contiguous(),
            source_order.to(dtype=torch.int32).contiguous(),
            source_row_ptr.to(dtype=torch.int32).contiguous(),
        )

    def build_cute_infer_zonal_coupling(self, edge_cache: Any) -> torch.Tensor:
        """Extract the GIE zonal coupling from packed Wigner storage.

        Parameters
        ----------
        edge_cache : EdgeCache
            Per-forward cache carrying CuTe packed Wigner storage.

        Returns
        -------
        torch.Tensor
            Zonal coupling with shape ``(E, D_node - 1)``.

        Raises
        ------
        ValueError
            If the edge cache does not carry packed Wigner storage.
        """
        D_packed = edge_cache.D_packed
        if D_packed is None:
            raise ValueError("CuTe zonal coupling requires packed Wigner storage")
        mp_coupling = D_packed.index_select(1, self.gie.packed_zonal_offsets)
        if self.gie_zonal_wigner_calc is None:
            return mp_coupling
        extra_coupling = self.gie_zonal_wigner_calc.forward_zonal(
            self._edge_quaternion(edge_cache),
            lmin=self.lmax + 1,
        )
        return torch.cat([mp_coupling, extra_coupling], dim=1)

    def run_cute_infer_readout(
        self,
        ffn_in: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run the CuTe readout when its exact inference contract matches.

        Parameters
        ----------
        ffn_in : torch.Tensor
            Equivariant readout input with shape ``(N, D, 1, C)``.

        Returns
        -------
        torch.Tensor or None
            Residual-inclusive scalar output with shape ``(N, C)``, or ``None``
            when the exact CuTe contract is not satisfied.
        """
        from deepmd.pt_expt.kernels.cute.sezm import (
            runtime_policy,
        )

        if self.training or not runtime_policy.is_cute_infer_enabled():
            return None
        from deepmd.pt_expt.kernels.cute.sezm.output_grid.readout_l0 import (
            maybe_run_neo_readout_l0,
        )

        return maybe_run_neo_readout_l0(
            self.output_ffn,
            ffn_in,
        )

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA4":
        # deserialize assigns numpy arrays after __init__, which demotes
        # promoted Parameters back to buffers; re-promote at the end.
        obj = super().deserialize(data)
        # The buffer carries the version of the restored variables, not the
        # version the fresh construction started from.
        obj.version_tensor.fill_(obj.version)
        return _promote_trainable_tree(obj)

    def _in_training_mode(self) -> bool:
        """Torch runtime hook for the training-only random local-Z roll.

        Overrides the dpmodel default (``False``) with the torch module's
        ``training`` flag, restoring pt's ``random_gamma=self.random_gamma
        and self.training`` semantics: train-mode forwards draw a fresh
        gamma per call, eval/export forwards fix gamma (the export path
        calls ``model.eval()`` before tracing).
        """
        return bool(self.training)

    def _gate_partial_exchange(
        self,
        partials: torch.Tensor,
        comm_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Reverse-accumulate ghost partials to owners, then broadcast back.

        ``border_op_backward`` sums each ghost row into its owner across
        ranks and zeroes the ghost rows; ``border_op`` refills them with the
        completed owner values. Both ops carry autograd (the two are
        transposes), so gate gradients cross ranks (issue #5906).

        Parameters
        ----------
        partials
            (n_nodes, 2) float tensor of [log_eta, zero_count] partials.
        comm_dict
            The border-exchange control tensors.

        Returns
        -------
        torch.Tensor
            The globally completed (n_nodes, 2) tensor.
        """
        # border_op exchanges rows by raw pointer arithmetic; a strided
        # view would corrupt the exchange.
        p = partials.contiguous()
        comm_args = (
            comm_dict["send_list"],
            comm_dict["send_proc"],
            comm_dict["recv_proc"],
            comm_dict["send_num"],
            comm_dict["recv_num"],
        )
        tail = (
            comm_dict["communicator"],
            comm_dict["nlocal"],
            comm_dict["nghost"],
        )
        p = torch.ops.deepmd_export.border_op_backward(*comm_args, p, *tail)
        p = torch.ops.deepmd_export.border_op(*comm_args, p, *tail)
        return p

    def disable_graph_lower(self) -> None:
        """Persisted variant of the dpmodel escape hatch (see base class).

        The buffer (and the routing bool) are PER-TASK state: multi-task
        ``share_params`` shares network submodules, not this buffer, so
        disabling the graph lower on one task branch does not propagate to
        branches sharing the same descriptor weights -- each branch owns
        its routing decision.
        """
        super().disable_graph_lower()
        self.graph_lower_disabled.fill_(True)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Back-compat: checkpoints written before the knob was persisted lack
        # the buffer; default to the fresh module's value (graph enabled)
        # instead of failing the strict load.
        key = prefix + "graph_lower_disabled"
        if key not in state_dict:
            state_dict[key] = self.graph_lower_disabled.detach().clone()
        else:
            # Re-sync the dpmodel-side routing bool from the RESTORED value
            # here, at load time, where the incoming tensor is real.  The
            # routing predicate itself must stay a plain python bool:
            # ``uses_graph_lower()`` runs inside traced forwards (the dense
            # adapter gate), and reading the buffer there would emit a
            # data-dependent ``bool(FakeTensor)`` guard that breaks
            # torch.export (GuardOnDataDependentSymNode Eq(u0, 1)).
            self._graph_lower_disabled = bool(state_dict[key])

        # Back-compat: checkpoints predating the version buffer were written
        # under version 1.1, the last one released before it existed.
        version_key = prefix + "version_tensor"
        if version_key not in state_dict:
            state_dict[version_key] = self.version_tensor.new_tensor(1.1)
        state_dict[version_key] = self.version_tensor.new_tensor(
            self._migrate_variables(
                state_dict, float(state_dict[version_key].item()), prefix
            )
        )

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self.version = float(self.version_tensor.item())

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)

    def _forward_blocks(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        """Run the interaction blocks under the pt_expt AMP policy.

        This is the torch (pt_expt) implementation of the descriptor's
        ``use_amp`` switch, mirroring the reference pt ``_compute_mode_ctx``:
        bfloat16 autocast wraps only the interaction-block region, while the
        geometry, edge cache, radial, env-seed, GIE and output FFN stages stay
        in fp32 (or higher). The dpmodel base stores ``use_amp`` only as a
        config flag and never autocasts (array-API has no autocast), so the
        real automatic mixed precision lives here.

        Training follows ``use_amp`` and evaluation follows ``DP_AMP_INFER``
        (captured once at construction as ``use_amp_infer``). The two are
        independent: mixed precision at inference is a throughput choice that
        must not require a model to have been trained with it. ``x`` is the
        node-feature tensor entering the blocks, and its device is the working
        device.
        """
        enabled = self.use_amp if self.training else self.use_amp_infer
        if enabled and x.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                return super()._forward_blocks(x, *args, **kwargs)
        return super()._forward_blocks(x, *args, **kwargs)

    def share_params(
        self,
        base_class: "DescrptDPA4",
        shared_level: int,
        model_prob: float = 1.0,
        resume: bool = False,
    ) -> None:
        # Multi-task parameter sharing for DPA4 is out of scope for this PR.
        raise NotImplementedError("share_params is not yet implemented for DescrptDPA4")
