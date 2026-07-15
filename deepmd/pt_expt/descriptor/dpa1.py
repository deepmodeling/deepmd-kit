# SPDX-License-Identifier: LGPL-3.0-or-later
import dataclasses
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.common import (
    cast_precision,
)
from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1 as DescrptDPA1DP
from deepmd.dpmodel.utils.env_mat_stat import (
    merge_env_stat,
)
from deepmd.kernels.cuda.dpa1.graph_compress import (
    dpa1_graph_compress,
)
from deepmd.kernels.cuda.dpa1.graph_compress import (
    op_available as cuda_compress_available,
)
from deepmd.kernels.cuda.dpa1.graph_descriptor import (
    dpa1_graph_descriptor,
)
from deepmd.kernels.cuda.dpa1.graph_descriptor import (
    op_available as cuda_descriptor_available,
)
from deepmd.kernels.triton.dpa1.activation import (
    ACT_CODES,
    TRITON_AVAILABLE,
)
from deepmd.kernels.triton.dpa1.edge_conv import (
    concat_gate_placeholders as edge_concat_gate_placeholders,
)
from deepmd.kernels.triton.dpa1.edge_conv import (
    edge_conv,
)
from deepmd.kernels.triton.dpa1.gemm_fp16x3 import (
    embed_last_gemm,
)
from deepmd.kernels.triton.dpa1.se_conv import (
    concat_gate_placeholders,
    se_conv,
)
from deepmd.kernels.triton.env_mat import edge_env_mat as _edge_env_mat_triton
from deepmd.kernels.triton.env_mat import env_mat as _env_mat_triton
from deepmd.kernels.utils import (
    cuda_infer_level,
    triton_infer_level,
)
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.utils.update_sel import (
    UpdateSel,
)


def _has_graph_fields(graph: Any, fields: tuple[str, ...]) -> bool:
    return all(getattr(graph, field, None) is not None for field in fields)


_DESTINATION_CSR_FIELDS = ("destination_order", "destination_row_ptr")
_DUAL_CSR_FIELDS = (
    *_DESTINATION_CSR_FIELDS,
    "source_order",
    "source_row_ptr",
)


# ======================================================================
# Fused Triton environment convolution (attn-free inference path)
#
# The prologue / epilogue below are module-level free functions taking the
# descriptor as ``desc``: pure computation shared by the tabulated
# (:meth:`DescrptDPA1._call_compressed`) and Triton
# (:meth:`DescrptDPA1._call_triton`) entry methods, across both tebd-input
# modes. ``_env_mat`` is the shared environment-matrix prologue; strip adds the
# type-pair gate through ``_strip_pair_index`` while concat folds the type
# feature into the embedding input through ``_concat_embedding_input``. The
# entry methods and the eligibility test live on the descriptor (and are
# delegated by ``DescrptSeAttenV2``), keeping the acceleration paths symmetric.
# Routing is device-free (decided from ``triton_infer_level`` and the layer
# configuration, never a tensor's device), so a CPU ``make_fx`` trace bakes the
# ``se_conv`` operator into the exported graph for the pt_expt ``.pt2``.
# ======================================================================
def _env_mat(
    desc: Any,
    coord_ext: torch.Tensor,
    atype_ext: torch.Tensor,
    nlist: torch.Tensor,
) -> tuple:
    """Environment-matrix prologue shared by every fused path (strip / concat).

    Returns ``(nf, nloc, nnei, ng, nfnl, rr, ss, sw, nlist_masked,
    type_embedding)``: ``rr`` the ``(nfnl, nnei, 4)`` environment matrix
    (excluded edges zeroed), ``ss`` its radial channel ``(nfnl, nnei, 1)``,
    ``sw`` the ``(nfnl, nnei, 1)`` smooth cutoff (zeroed on excluded/padding
    edges), and ``nlist_masked`` the neighbor indices with excluded/padding
    entries mapped to ``0`` (for downstream gathers).
    """
    se = desc.se_atten
    nf, nloc, nnei = nlist.shape
    if triton_infer_level() >= 1:
        # Fused env-matrix operator, captured opaquely under the pt_expt trace and
        # resolving to the Triton kernel at CUDA runtime; identical outputs to the
        # array-API ``EnvMat.call`` below.
        rr, _diff, sw = _env_mat_triton(
            coord_ext,
            nlist,
            atype_ext[:, :nloc],
            se.mean[...],
            se.stddev[...],
            se.env_mat.rcut,
            se.env_mat.rcut_smth,
            radial_only=False,
            protection=se.env_mat.protection,
            use_exp_switch=se.env_mat.use_exp_switch,
        )
    else:
        rr, _diff, sw = se.env_mat.call(
            coord_ext, atype_ext, nlist, se.mean[...], se.stddev[...]
        )
    nf, nloc, nnei, _ = rr.shape
    ng = se.neuron[-1]
    nfnl = nf * nloc

    exclude_mask = (
        se.emask.build_type_exclude_mask(nlist, atype_ext)
        .view(nfnl, nnei)
        .to(torch.bool)
    )
    nlist = nlist.view(nfnl, nnei)
    nlist = torch.where(exclude_mask, nlist, torch.full_like(nlist, -1))
    nlist_mask = nlist != -1
    nlist_masked = torch.where(nlist_mask, nlist, torch.zeros_like(nlist))
    sw = torch.where(
        nlist_mask[:, :, None],
        sw.view(nfnl, nnei, 1),
        torch.zeros(nfnl, nnei, 1, dtype=sw.dtype, device=sw.device),
    )
    rr = rr.view(nfnl, nnei, 4) * exclude_mask[:, :, None].to(rr.dtype)
    ss = rr[:, :, :1]

    type_embedding = desc.type_embedding.call()
    return nf, nloc, nnei, ng, nfnl, rr, ss, sw, nlist_masked, type_embedding


def _strip_pair_index(
    desc: Any,
    atype_ext: torch.Tensor,
    nlist_masked: torch.Tensor,
    type_embedding: torch.Tensor,
    nf: int,
    nloc: int,
    nnei: int,
) -> torch.Tensor:
    """Strip-mode per-edge row index into the type-pair embedding table.

    One-side uses the neighbor type; two-side folds the ``(center, neighbor)``
    pair into a flat index, mirroring the dense reference. Shape ``(nfnl*nnei,)``.
    """
    se = desc.se_atten
    ntypes_with_padding = type_embedding.shape[0]
    nlist_index = nlist_masked.view(nf, nloc * nnei)
    nei_type = torch.gather(atype_ext, dim=1, index=nlist_index)
    if se.type_one_side:
        return nei_type.reshape(-1).to(torch.long)
    atype = atype_ext[:, :nloc]
    idx_i = torch.tile(atype.reshape(-1, 1) * ntypes_with_padding, [1, nnei]).view(-1)
    return (idx_i + nei_type.reshape(-1)).to(torch.long)


def _concat_embedding_input(
    desc: Any,
    ss: torch.Tensor,
    atype_ext: torch.Tensor,
    nlist_masked: torch.Tensor,
    type_embedding: torch.Tensor,
    nf: int,
    nloc: int,
    nnei: int,
) -> torch.Tensor:
    """Concat-mode embedding input: radial channel concatenated with the
    neighbor (and, two-side, center) type embeddings.

    Returns ``(nfnl, nnei, 1 + k * tebd_dim)`` with ``k in {1, 2}`` (one-side /
    two-side), matching the dense concat reference exactly.
    """
    se = desc.se_atten
    tebd_dim = desc.tebd_dim
    nfnl = nf * nloc
    atype_embd_ext = type_embedding[atype_ext.reshape(-1)].view(
        nf, atype_ext.shape[1], tebd_dim
    )
    # neighbor type embedding, gathered per edge
    nlist_index = nlist_masked.view(nf, nloc * nnei)
    idx = nlist_index[:, :, None].expand(-1, -1, tebd_dim)
    nlist_tebd = torch.gather(atype_embd_ext, dim=1, index=idx).view(
        nfnl, nnei, tebd_dim
    )
    if se.type_one_side:
        return torch.cat([ss, nlist_tebd], dim=-1)
    # center type embedding, broadcast over neighbors
    atype_embd = atype_embd_ext[:, :nloc, :].reshape(nfnl, 1, tebd_dim)
    atype_tebd = atype_embd.expand(-1, nnei, -1)
    return torch.cat([ss, nlist_tebd, atype_tebd], dim=-1)


def _grrg_from_moment(
    desc: Any,
    xyz_scatter: torch.Tensor,
    type_embedding: torch.Tensor,
    coord_ext: torch.Tensor,
    atype_ext: torch.Tensor,
    nf: int,
    nloc: int,
    nnei: int,
    ng: int,
    sw: torch.Tensor,
) -> Any:
    """Strip-mode epilogue: symmetry-invariant contraction of the moment.

    Consumes the unnormalized moment ``xyz_scatter`` (nfnl, 4, ng), applies the
    ``1 / nnei`` normalization, forms the ``G^T G`` descriptor and the rotation
    matrix, and appends the center type embedding when ``concat_output_tebd``.
    """
    se = desc.se_atten
    xyz_scatter = xyz_scatter / se.nnei
    xyz_scatter_1 = xyz_scatter.permute(0, 2, 1)
    rot_mat = xyz_scatter_1[:, :, 1:4]
    xyz_scatter_2 = xyz_scatter[:, :, 0 : se.axis_neuron]
    result = torch.matmul(xyz_scatter_1, xyz_scatter_2)
    result = result.view(nf, nloc, ng * se.axis_neuron)
    rot_mat = rot_mat.view(nf, nloc, ng, 3)
    if desc.concat_output_tebd:
        nall = coord_ext.view(nf, -1).shape[1] // 3
        atype_embd_ext = type_embedding[atype_ext.reshape(-1)].view(
            nf, nall, desc.tebd_dim
        )
        atype_embd = atype_embd_ext[:, :nloc, :]
        result = torch.cat([result, atype_embd.view(nf, nloc, desc.tebd_dim)], dim=-1)
    return result, rot_mat, None, None, sw.view(nf, nloc, nnei, 1)


def _type_pair_table(desc: Any, type_embedding: torch.Tensor) -> torch.Tensor:
    """Type-pair embedding table (P, ng) gathered per edge by ``tebd_idx``.

    One-side keeps the neighbor-type rows; two-side forms every
    ``(neighbor, center)`` pair, mirroring the dense reference.
    """
    se = desc.se_atten
    if se.type_one_side:
        return se.cal_g_strip(type_embedding, 0)
    nt = type_embedding.shape[1]
    ntypes_with_padding = type_embedding.shape[0]
    nei = type_embedding.view(1, ntypes_with_padding, nt).expand(
        ntypes_with_padding, ntypes_with_padding, nt
    )
    center = type_embedding.view(ntypes_with_padding, 1, nt).expand(
        ntypes_with_padding, ntypes_with_padding, nt
    )
    two_side = torch.cat([nei, center], dim=-1).reshape(-1, nt * 2)
    return se.cal_g_strip(two_side, 0)


@BaseDescriptor.register("se_atten")
@BaseDescriptor.register("dpa1")
@torch_module
class DescrptDPA1(DescrptDPA1DP):
    _update_sel_cls = UpdateSel

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Persisted graph-routing knob (first-class training configuration):
        # ``disable_graph_lower()`` used to flip only the plain dpmodel bool,
        # which a Trainer checkpoint restart silently reset (the fresh model
        # is rebuilt from config before ``load_state_dict``, and neither the
        # state-dict keys nor ``_extra_state.model_params`` carried the
        # choice) -- on a binding-sel system that switched the training
        # equation and gradients without warning.  A persistent buffer rides
        # every pt_expt state_dict, so save/restart round-trips it.
        torch.nn.Module.register_buffer(
            self, "graph_lower_disabled", torch.zeros((), dtype=torch.bool)
        )

    def disable_graph_lower(self) -> None:
        """Persisted variant of the dpmodel escape hatch (see base class)."""
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
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def share_params(
        self,
        base_class: Any,
        shared_level: int,
        model_prob: float = 1.0,
        resume: bool = False,
    ) -> None:
        """Share parameters with base_class for multi-task training.

        Level 0: share type_embedding and se_atten (all modules and buffers).
        Level 1: share type_embedding only.
        """
        assert self.__class__ == base_class.__class__, (
            "Only descriptors of the same type can share params!"
        )
        if shared_level == 0:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
            if not resume:
                merge_env_stat(base_class.se_atten, self.se_atten, model_prob)
            self._modules["se_atten"] = base_class._modules["se_atten"]
        elif shared_level == 1:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
        else:
            raise NotImplementedError

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Enable compression for the DPA1 descriptor.

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between atoms
        table_extrapolate
            The scale of model extrapolation
        table_stride_1
            The uniform stride of the first table
        table_stride_2
            The uniform stride of the second table
        check_frequency
            The overflow check frequency
        """
        from deepmd.pt_expt.utils.tabulate import (
            DPTabulate,
        )

        if self.compress:
            raise ValueError("Compression is already enabled.")
        if self.se_atten.tebd_input_mode != "strip":
            raise RuntimeError("Type embedding compression only works in strip mode")
        if self.se_atten.resnet_dt:
            raise RuntimeError(
                "Model compression error: descriptor resnet_dt must be false!"
            )

        data = self.serialize()
        self.table = DPTabulate(
            self,
            data["neuron"],
            data["type_one_side"],
            data["exclude_types"],
            data["activation_function"],
        )
        self.table_config = [
            table_extrapolate,
            table_stride_1,
            table_stride_2,
            check_frequency,
        ]

        # Precompute type embedding data
        self._store_type_embd_data()

        if self.se_atten.attn_layer == 0:
            # Build geometric embedding table
            self.lower, self.upper = self.table.build(
                min_nbor_dist, table_extrapolate, table_stride_1, table_stride_2
            )
            self._store_compress_data()
            self.geo_compress = True
        else:
            self.geo_compress = False

        self.compress = True

    def _store_compress_data(self) -> None:
        """Store tabulated data as buffers for the compressed geometric embedding."""
        table_data = self.table.data
        table_config = self.table_config
        lower = self.lower
        upper = self.upper
        prec = self.se_atten.mean.dtype

        net_key = "filter_net"
        info = torch.as_tensor(
            [
                lower[net_key],
                upper[net_key],
                upper[net_key] * table_config[0],
                table_config[1],
                table_config[2],
                table_config[3],
            ],
            dtype=prec,
            device="cpu",
        )
        tensor_data = table_data[net_key].to(dtype=prec)
        self.compress_data = torch.nn.ParameterList(
            [torch.nn.Parameter(tensor_data, requires_grad=False)]
        )
        self.compress_info = torch.nn.ParameterList(
            [torch.nn.Parameter(info, requires_grad=False)]
        )

    def _store_type_embd_data(self) -> None:
        """Precompute type embedding outputs and store as a buffer."""
        with torch.no_grad():
            # type_embedding.call() returns (ntypes+1) x tebd_dim (with padding)
            full_embd = self.type_embedding.call()
            nt, t_dim = full_embd.shape

            if self.se_atten.type_one_side:
                # One-side: only neighbor types
                # (ntypes+1) x tebd_dim -> (ntypes+1) x ng
                embd_tensor = self.se_atten.embeddings_strip[0].call(full_embd).detach()
            else:
                # Two-side: all (ntypes+1)^2 type pair combinations
                # Build [neighbor, center] combinations
                embd_nei = full_embd.view(1, nt, t_dim).expand(nt, nt, t_dim)
                embd_center = full_embd.view(nt, 1, t_dim).expand(nt, nt, t_dim)
                two_side_embd = torch.cat([embd_nei, embd_center], dim=-1).reshape(
                    -1, t_dim * 2
                )
                # ((ntypes+1)^2) x ng
                embd_tensor = (
                    self.se_atten.embeddings_strip[0].call(two_side_embd).detach()
                )

            torch.nn.Module.register_buffer(self, "type_embd_data", embd_tensor)

    @cast_precision
    def call(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        comm_dict: dict | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> Any:
        # Compressed descriptors take the tabulated dense path.
        if self.compress:
            return self._call_compressed(coord_ext, atype_ext, nlist)
        # Fused Triton convolution (strip / concat) for inference when eligible.
        if (
            triton_infer_level() >= 1
            and not self.training
            and self._fused_eligible("triton")
        ):
            return self._call_triton(coord_ext, atype_ext, nlist)
        # Array-API reference (dpmodel).
        return DescrptDPA1DP.call.__wrapped__(
            self, coord_ext, atype_ext, nlist, mapping
        )

    def call_graph(
        self,
        graph: Any,
        atype: torch.Tensor,
        type_embedding: torch.Tensor | None = None,
        static_nnei: int | None = None,
        comm_dict: dict | None = None,
    ) -> Any:
        """Graph-native forward, routed through the fused edge kernel when eligible.

        Concat and strip attention-free blocks are all graph-eligible. A
        geometrically compressed descriptor takes the tabulated route
        (:meth:`_call_graph_cuda_compress` at ``DP_CUDA_INFER >= 1``, the
        original fused table operator through
        :meth:`_call_graph_compress_reference` at level 0); an uncompressed
        descriptor takes the embedding route (:meth:`_call_graph_cuda` mega
        kernel, the Triton edge convolution for concat, or the dpmodel
        reference). ``static_nnei`` is the shape-static neighbor count the dense
        :meth:`call` adapter supplies for the attention edge-pair enumeration;
        it is forwarded to the dpmodel reference and unused by the
        attention-free fused paths.

        Unlike the dense :meth:`call`, the graph lower is not
        ``@cast_precision``-wrapped: ``edge_vec`` arrives in the model-agnostic
        fp64 ``.pt2`` ABI. The fused CUDA kernels consume it in that leaf dtype
        directly (casting to the model precision in-kernel); the Triton and
        reference paths need the Python-side alignment below so their
        environment matrix runs at the model precision. Force / virial precision
        is decided downstream by the model output transform, not here.
        """
        # Fused kernels apply to inference with a type embedding; backend
        # capability is decided by :meth:`_fused_eligible`, and this method only
        # routes among the eligible implementations.
        fused = not self.training and type_embedding is not None
        # CUDA consumes the fp64 leaf directly. Eligibility already encodes the
        # compressed / MLP rule split; dispatch selects the matching kernel.
        cuda_graph_eligible = not self.geo_compress or _has_graph_fields(
            graph, _DESTINATION_CSR_FIELDS
        )
        if (
            fused
            and cuda_graph_eligible
            and cuda_infer_level() >= 1
            and self._fused_eligible("cuda")
        ):
            if self.geo_compress:
                return self._call_graph_cuda_compress(graph, atype, type_embedding)
            return self._call_graph_cuda(graph, atype, type_embedding)
        # Every remaining path evaluates the environment matrix in the model
        # (statistics) precision, so align the fp64 leaf once.
        prec = self.se_atten.mean.dtype
        if graph.edge_vec.dtype != prec:
            graph = dataclasses.replace(graph, edge_vec=graph.edge_vec.to(prec))
        # Triton edge convolution: serves the uncompressed concat / strip lower;
        # a tabulated (geo_compress) descriptor stays on the table path below.
        if (
            fused
            and triton_infer_level() >= 1
            and not self.geo_compress
            and self._fused_eligible("triton")
        ):
            return self._call_graph_triton(graph, atype, type_embedding)
        # Reference: the tabulated table operator for a compressed descriptor
        # (level 0 / trace-time / CPU fallback), else the dpmodel array API.
        if self.geo_compress:
            return self._call_graph_compress_reference(graph, atype, type_embedding)
        return DescrptDPA1DP.call_graph(
            self,
            graph,
            atype,
            type_embedding=type_embedding,
            static_nnei=static_nnei,
            comm_dict=comm_dict,
        )

    def _fused_eligible(self, backend: str) -> bool:
        """Whether a fused descriptor kernel can serve this block.

        Parameters
        ----------
        backend : str
            ``"cuda"`` for the mega kernel (:mod:`.cuda.dpa1.graph_descriptor`)
            or ``"triton"`` for the environment convolution
            (:mod:`.triton.dpa1`). Both require the attention-free path, both
            tebd input modes (``concat`` / ``strip``) and an activation the
            kernel inlines (``tanh`` / ``silu``; see
            :data:`.activation.ACT_CODES`). ``cuda`` additionally requires no
            excluded types, fp32 statistics, the compiled operator library,
            and -- since its MLP runs entirely in registers with
            template-specialized widths -- a three-layer embedding where each
            layer keeps or doubles the previous width, with
            ``neuron[0] in {8, 16, 32, 64}``, ``neuron[1] <= 64`` and
            ``neuron[2] <= 128``, and one inlined activation on every layer
            (the strip gate network is unconstrained: its table is
            precomputed outside the kernel). ``triton`` serves any layer
            stack (the head layers run on cuBLAS), so only the last
            activation must be inlined.
        """
        se = self.se_atten
        if se.attn_layer != 0:
            return False
        layers = se.embeddings[0].layers
        if backend == "cuda":
            if self.geo_compress:
                # The tabulated kernel replaces the embedding MLP, so its width
                # stack is unconstrained; only the table width (``neuron[-1]``)
                # matters, served up to 256 by the padded bucket dispatch.
                return (
                    se.tebd_input_mode == "strip"
                    and not se.exclude_types
                    and se.mean.dtype == torch.float32
                    and self.compress_data[0].dtype == torch.float32
                    and self.type_embd_data.dtype == torch.float32
                    and int(se.neuron[-1]) <= 256
                    and 0 < int(se.axis_neuron) <= min(16, int(se.neuron[-1]))
                    and cuda_compress_available()
                )
            widths = [int(layer.w.shape[1]) for layer in layers]
            first = layers[0]
            first_has_residual = bool(first.resnet) and first.w.shape[1] in (
                first.w.shape[0],
                2 * first.w.shape[0],
            )
            parameters = [
                tensor
                for layer in layers
                for tensor in (layer.w, layer.b, layer.idt)
                if tensor is not None
            ]
            return (
                not self.compress
                and se.tebd_input_mode in ("concat", "strip")
                and not se.exclude_types
                and se.mean.dtype == torch.float32
                and all(tensor.dtype == torch.float32 for tensor in parameters)
                and not first_has_residual
                and len(widths) == 3
                and widths[0] in (8, 16, 32, 64)
                and widths[1] in (widths[0], 2 * widths[0])
                and widths[2] in (widths[1], 2 * widths[1])
                and widths[1] <= 64
                and widths[2] <= 128
                and len({str(layer.activation_function).lower() for layer in layers})
                == 1
                and str(layers[0].activation_function).lower() in ACT_CODES
                and cuda_descriptor_available()
            )
        return (
            TRITON_AVAILABLE
            and se.tebd_input_mode in ("strip", "concat")
            and str(layers[-1].activation_function).lower() in ACT_CODES
        )

    def _call_triton(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
    ) -> Any:
        """Fused Triton environment convolution (attn-free strip / concat path).

        The two embedding GEMMs stay on cuBLAS; the last layer's activation and
        residual, the smooth cutoff and the moment reduction collapse into
        :func:`se_conv`, which never materializes an ``(E, ng)`` tensor. Strip
        additionally folds the type-pair gate (gathered inline); concat instead
        feeds the type feature through the embedding input and applies no gate.
        Composes under ``make_fx`` / ``torch.export`` so the operator is baked
        into the pt_expt ``.pt2``.
        """
        nf, nloc, nnei, ng, nfnl, rr, ss, sw, nlist_masked, type_embedding = _env_mat(
            self, coord_ext, atype_ext, nlist
        )
        se = self.se_atten
        strip = se.tebd_input_mode == "strip"
        # Embedding-net input: the radial channel (strip) or the radial-plus-
        # type-embedding concatenation (concat).
        if strip:
            emb_in = ss
        else:
            emb_in = _concat_embedding_input(
                self, ss, atype_ext, nlist_masked, type_embedding, nf, nloc, nnei
            )

        # Evaluate the embedding net through its penultimate layer (cuBLAS) and
        # fold the final layer's pre-activation into se_conv.
        *head, last = se.embeddings[0].layers
        h = emb_in
        for layer in head:
            h = layer.call(h)
        z2 = embed_last_gemm(h, last.w, last.b)
        h1_dim, out_dim = last.w.shape
        if last.resnet and out_dim == 2 * h1_dim:
            resnet_mult = 2
        elif last.resnet and out_dim == h1_dim:
            resnet_mult = 1
        else:
            resnet_mult = 0
        idt = (
            last.idt
            if last.idt is not None
            else torch.ones(out_dim, dtype=z2.dtype, device=z2.device)
        )
        act = ACT_CODES[str(last.activation_function).lower()]
        if strip:
            gated = 1
            tt_full = _type_pair_table(self, type_embedding)
            tebd_idx = _strip_pair_index(
                self, atype_ext, nlist_masked, type_embedding, nf, nloc, nnei
            )
            sw_eff = (
                sw.view(nfnl, nnei)
                if se.smooth
                else torch.ones(nfnl, nnei, dtype=rr.dtype, device=rr.device)
            )
        else:
            # Concat folds the type feature into ``emb_in``; the gate inputs are
            # unused by the kernel (``gated == 0``).
            gated = 0
            tt_full, tebd_idx, sw_eff = concat_gate_placeholders(z2, ng)
        # Unnormalized moment (nfnl, 4, ng); _grrg_from_moment applies 1 / nnei.
        xyz_scatter = se_conv(
            z2.contiguous(),
            h.contiguous(),
            idt,
            tt_full,
            tebd_idx,
            sw_eff,
            rr,
            resnet_mult,
            act,
            gated,
        )
        return _grrg_from_moment(
            self,
            xyz_scatter,
            type_embedding,
            coord_ext,
            atype_ext,
            nf,
            nloc,
            nnei,
            ng,
            sw,
        )

    def _call_compressed(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
    ) -> Any:
        """Compressed forward for DPA1 descriptor (strip only)."""
        nf, nloc, nnei, ng, nfnl, rr, ss, sw, nlist_masked, type_embedding = _env_mat(
            self, coord_ext, atype_ext, nlist
        )
        tebd_idx = _strip_pair_index(
            self, atype_ext, nlist_masked, type_embedding, nf, nloc, nnei
        )
        # (nf x nl) x nnei x ng, gathered from the precomputed type-pair table.
        gg_t = self.type_embd_data[tebd_idx].view(nfnl, nnei, ng)
        if self.se_atten.smooth:
            gg_t = gg_t * sw.view(nfnl, self.se_atten.nnei, 1)

        if self.geo_compress:
            is_sorted = len(self.se_atten.exclude_types) == 0
            xyz_scatter = torch.ops.deepmd.tabulate_fusion_se_atten(
                self.compress_data[0].contiguous(),
                self.compress_info[0].cpu().contiguous(),
                ss.reshape(-1, 1).contiguous(),
                rr.contiguous(),
                gg_t.reshape(-1, gg_t.size(-1)).contiguous(),
                self.se_atten.neuron[-1],
                is_sorted,
            )[0]
        else:
            # No geometric compression: run embedding net + attention.
            gg_s = self.se_atten.embeddings[0].call(ss)
            gg = gg_s * gg_t + gg_s
            nlist_mask = (
                self.se_atten.emask.build_type_exclude_mask(nlist, atype_ext)
                .view(nfnl, nnei)
                .to(torch.bool)
            )
            input_r = torch.nn.functional.normalize(
                rr.view(-1, self.se_atten.nnei, 4)[:, :, 1:4], dim=-1
            )
            gg = self.se_atten.dpa1_attention(gg, nlist_mask, input_r=input_r, sw=sw)
            xyz_scatter = torch.matmul(rr.permute(0, 2, 1), gg)

        return _grrg_from_moment(
            self,
            xyz_scatter,
            type_embedding,
            coord_ext,
            atype_ext,
            nf,
            nloc,
            nnei,
            ng,
            sw,
        )

    def _call_graph_compress_reference(
        self,
        graph: Any,
        atype: torch.Tensor,
        type_embedding: torch.Tensor,
    ) -> Any:
        """Reference geo-compressed strip descriptor on the edge stream (attn-free).

        The ``DP_CUDA_INFER == 0`` graph path and the CPU / trace-time fallback.
        Evaluates the tabulated geometric embedding with the original fused
        table operator ``deepmd::tabulate_fusion_se_atten``, treating each edge
        as a one-neighbor block (``nloc = E``, ``nnei = 1``) so the operator
        returns the per-edge moment outer product ``(E, 4, ng)``; a
        ``segment_sum`` over edge centers then forms the per-node moment. This
        matches the dense :meth:`DescrptDPA1._call_compressed` (same table, same
        gate) to the fp32 summation-order floor, and composes under autograd so
        the force / virial assembly differentiates through it. Not
        ``make_fx``-traceable (the table operator has no meta kernel); the
        exportable graph is the level >= 1 CUDA kernel.

        Parameters
        ----------
        graph : NeighborGraph
            Lowered neighbor graph; ``edge_vec`` is in the model precision.
        atype : torch.Tensor
            Flat node atom types with shape (N,), int64.
        type_embedding : torch.Tensor
            Type embedding table with shape (ntypes + 1, tebd_dim).

        Returns
        -------
        grrg : torch.Tensor
            Descriptor with shape (N, ng * axis [+ tebd_dim]).
        rot_mat : torch.Tensor
            Equivariant rotation matrix with shape (N, ng, 3).
        """
        se = self.se_atten
        ng = int(se.neuron[-1])
        n_total = atype.shape[0]
        src, dst = graph.edge_index[0], graph.edge_index[1]
        center_type, nei_type = atype[dst], atype[src]

        # === Step 1. Per-edge environment matrix (canonical statistics slot) ===
        ev = graph.edge_vec
        length = ev.norm(dim=-1, keepdim=True)
        # Padding edges enter the switch at |r| + 1 so the 1/q factors stay
        # finite; their moment weight is zeroed by the edge mask below.
        length = length + (~graph.edge_mask[:, None]).to(length.dtype)
        q = length + se.env_protection
        u = ((length - se.rcut_smth) / (se.rcut - se.rcut_smth)).clamp(0.0, 1.0)
        sw = u**3 * (-6 * u**2 + 15 * u - 10) + 1.0
        em = torch.cat([sw / q, ev * (sw / q**2)], dim=-1)
        rr = (em - se.mean[:, 0, :][center_type]) / se.stddev[:, 0, :][center_type]

        # === Step 2. Strip type-pair gate from the precomputed table ===
        ntypes = type_embedding.shape[0]
        pair_idx = nei_type if se.type_one_side else center_type * ntypes + nei_type
        gate = self.type_embd_data[pair_idx]
        if se.smooth:
            gate = gate * sw

        # === Step 3. Tabulated embedding and per-edge moment outer product ===
        # Each edge is a one-neighbor block, so the operator's neighbor sum is
        # the identity and it returns the per-edge outer product em (x) g.
        is_sorted = len(se.exclude_types) == 0
        outer = torch.ops.deepmd.tabulate_fusion_se_atten(
            self.compress_data[0].contiguous(),
            self.compress_info[0].cpu().contiguous(),
            rr[:, 0:1].contiguous(),
            rr.reshape(-1, 1, 4).contiguous(),
            gate.contiguous(),
            ng,
            is_sorted,
        )[0]

        # === Step 4. Moment reduction and G^T G contraction ===
        outer = outer * graph.edge_mask[:, None, None].to(outer.dtype)
        gr = torch.zeros(n_total, 4, ng, dtype=outer.dtype, device=outer.device)
        gr.index_add_(0, dst, outer)
        gr = gr / se.nnei
        gr_perm = gr.permute(0, 2, 1)  # (N, ng, 4)
        rot_mat = gr_perm[:, :, 1:4]
        gr_sub = gr[:, :, : se.axis_neuron]  # (N, 4, axis)
        grrg = torch.matmul(gr_perm, gr_sub).reshape(n_total, ng * se.axis_neuron)
        grrg = grrg.to(graph.edge_vec.dtype)
        if self.concat_output_tebd:
            grrg = torch.cat([grrg, type_embedding[atype]], dim=-1)
        return grrg, rot_mat

    def _call_graph_cuda_compress(
        self,
        graph: Any,
        atype: torch.Tensor,
        type_embedding: torch.Tensor,
    ) -> Any:
        """Fused CUDA graph-native geo-compressed strip descriptor (attn-free).

        Numerically equivalent to :meth:`DescrptDPA1._call_compressed` through
        the :func:`~deepmd.kernels.cuda.dpa1.graph_compress.dpa1_graph_compress`
        operator: the environment matrix, quintic table lookup, strip type-pair
        gate, moment reduction and ``G^T G`` contraction collapse into one CUDA
        mega kernel whose registered backward exposes the ``edge_vec`` gradient
        for the analytic force / virial assembly.
        """
        return dpa1_graph_compress(self, graph, atype, type_embedding)

    def _call_graph_cuda(
        self,
        graph: Any,
        atype: torch.Tensor,
        type_embedding: torch.Tensor,
    ) -> Any:
        """Fused CUDA graph-native descriptor (concat, attn-free).

        Numerically equivalent to :meth:`DescrptDPA1DP.call_graph` through the
        :func:`~deepmd.kernels.cuda.dpa1.graph_descriptor.dpa1_graph_descriptor`
        operator: the environment matrix, embedding MLP, moment reduction and
        ``G^T G`` contraction collapse into one CUDA mega kernel whose
        registered backward exposes the ``edge_vec`` gradient for the analytic
        force / virial assembly. Composes under ``make_fx`` / ``torch.export``
        so the operator is baked into the pt_expt graph-form ``.pt2``.
        """
        return dpa1_graph_descriptor(self, graph, atype, type_embedding)

    def fused_energy_force_graph(
        self,
        fit: Any,
        graph: Any,
        atype: torch.Tensor,
        ownership: torch.Tensor,
        atom_bias: torch.Tensor,
        do_atomic_virial: bool,
    ) -> tuple[torch.Tensor, ...] | None:
        """End-to-end fused energy / force / virial from the edge stream.

        Collapses this descriptor, the energy fitting and the analytic force /
        virial assembly into one value-returning CUDA operator (no autograd
        tape). Returns ``(energy, atom_energy, force, virial, atom_virial)``, or
        ``None`` when the descriptor or fitting is not fused-eligible or the
        operator library is unavailable -- the caller then uses the autograd
        lower. The geo-compressed descriptor dispatches to its tabulated operator
        (:func:`~deepmd.kernels.cuda.dpa1.graph_compress.dpa1_graph_compress_energy_force`);
        the embedding-MLP descriptor to
        :func:`~deepmd.kernels.cuda.dpa1.graph_energy_force.dpa1_graph_energy_force`.

        Parameters
        ----------
        fit : EnergyFittingNet
            The energy fitting module fused with this descriptor.
        graph : NeighborGraph
            The lowered neighbor graph (``edge_vec``, ``edge_index``,
            ``edge_mask``, ``n_node``).
        atype : torch.Tensor
            Flat node atom types with shape (N,), int64.
        ownership : torch.Tensor
            Energy-contributing node mask with shape (N,), bool.
        atom_bias : torch.Tensor
            Combined fitting and atomic-model bias with shape (ntypes,).
        do_atomic_virial : bool
            Whether to also assemble the per-atom virial.

        Returns
        -------
        tuple[torch.Tensor, ...] or None
            ``(energy, atom_energy, force, virial, atom_virial)``, or ``None``.
        """
        from deepmd.kernels.cuda.graph_fitting import (
            fitting_eligible,
        )

        if not (
            self._fused_eligible("cuda")
            and fitting_eligible(fit)
            and _has_graph_fields(graph, _DUAL_CSR_FIELDS)
        ):
            return None
        type_embedding = self.type_embedding.call()
        node_capacity = atype.shape[0]
        if self.geo_compress:
            from deepmd.kernels.cuda.dpa1.graph_compress import (
                dpa1_graph_compress_energy_force,
                ef_op_available,
                mega_eligible,
            )

            if not (ef_op_available() and mega_eligible(self)):
                return None
            return dpa1_graph_compress_energy_force(
                self,
                fit,
                graph,
                atype,
                type_embedding,
                ownership,
                atom_bias,
                node_capacity=node_capacity,
                do_atomic_virial=do_atomic_virial,
            )
        from deepmd.kernels.cuda.dpa1.graph_energy_force import (
            dpa1_graph_energy_force,
            op_available,
        )

        if not op_available():
            return None
        return dpa1_graph_energy_force(
            self,
            fit,
            graph,
            atype,
            type_embedding,
            ownership,
            atom_bias,
            node_capacity=node_capacity,
            do_atomic_virial=do_atomic_virial,
        )

    def _call_graph_triton(
        self,
        graph: Any,
        atype: torch.Tensor,
        type_embedding: torch.Tensor,
    ) -> Any:
        """Fused graph-native environment convolution (concat / strip, attn-free).

        Bit-exact analogue of :meth:`DescrptDPA1DP.call_graph`: builds the
        per-edge environment matrix and embedding input, runs the two embedding
        GEMMs on cuBLAS, then folds the last layer, the type-pair gate (strip),
        the edge mask, the outer product and the ``segment_sum`` scatter into
        :func:`edge_conv`. The two tebd-input modes differ only in the embedding
        input (concat folds the type feature in; strip runs on the radial channel
        alone) and the gate (strip multiplies by the type-pair table, concat does
        not). Composes under ``make_fx`` / ``torch.export`` so the operator is
        baked into the pt_expt graph-form ``.pt2``.
        """
        se = self.se_atten
        strip = se.tebd_input_mode == "strip"
        n_total = atype.shape[0]
        src = graph.edge_index[0, :]
        dst = graph.edge_index[1, :]
        center_type = atype[dst]
        nei_type = atype[src]
        # Per-edge env-mat 4-vector, normalized by the center (dst) atom type;
        # mean/stddev are slot-independent, so slot 0 is the canonical vector.
        # The fused operator is captured opaquely under the pt_expt trace and
        # resolves to the Triton kernel at CUDA runtime (identical to the
        # array-API ``edge_env_mat`` reference it routes to at level 0). Strip
        # also needs the per-edge switch for the type-pair gate (differentiable
        # in ``edge_vec``); the same operator emits it when ``return_sw`` is set.
        rr, sw_e = _edge_env_mat_triton(
            graph.edge_vec,
            center_type,
            se.mean[:, 0, :],
            se.stddev[:, 0, :],
            se.rcut,
            se.rcut_smth,
            protection=se.env_protection,
            edge_mask=graph.edge_mask,
            return_sw=True,
        )  # (E, 4), (E, 1)
        ss = rr[:, 0:1]
        ng = se.neuron[-1]
        if strip:
            # Strip: the geometric net runs on the radial channel alone; the
            # type feature enters as the multiplicative gate ``1 + tt[idx] * sw``
            # applied inside edge_conv. ``tt`` is the type-pair table (built like
            # the dense strip reference); ``idx`` folds the (center, neighbor)
            # pair (two-side) or the neighbor type (one-side); ``sw`` is the
            # switch when smooth, else ones (matching the dense path).
            emb_in = ss
            tt = _type_pair_table(self, type_embedding)
            ntypes_with_padding = type_embedding.shape[0]
            if se.type_one_side:
                gate_idx = nei_type.to(torch.long)
            else:
                gate_idx = center_type.to(
                    torch.long
                ) * ntypes_with_padding + nei_type.to(torch.long)
            sw_gate = (
                sw_e.reshape(-1)
                if se.smooth
                else torch.ones(rr.shape[0], dtype=rr.dtype, device=rr.device)
            )
            gated = 1
        else:
            # concat embedding input: radial channel plus the neighbor (and, two-
            # side, center) type embeddings. Ghost type == owner type, so
            # gathering by the local owner reproduces the dense neighbor tebd.
            nlist_tebd = type_embedding[nei_type]  # (E, tebd_dim)
            if se.type_one_side:
                emb_in = torch.cat([ss, nlist_tebd], dim=-1)
            else:
                center_tebd = type_embedding[center_type]  # (E, tebd_dim)
                emb_in = torch.cat([ss, nlist_tebd, center_tebd], dim=-1)
            tt, gate_idx, sw_gate = edge_concat_gate_placeholders(rr, ng)
            gated = 0

        # Embedding net through the penultimate layer (cuBLAS); fold the final
        # layer's pre-activation into edge_conv.
        *head, last = se.embeddings[0].layers
        h = emb_in
        for layer in head:
            h = layer.call(h)
        z2 = embed_last_gemm(h, last.w, last.b)
        h1_dim, out_dim = last.w.shape
        if last.resnet and out_dim == 2 * h1_dim:
            resnet_mult = 2
        elif last.resnet and out_dim == h1_dim:
            resnet_mult = 1
        else:
            resnet_mult = 0
        idt = (
            last.idt
            if last.idt is not None
            else torch.ones(out_dim, dtype=z2.dtype, device=z2.device)
        )
        act = ACT_CODES[str(last.activation_function).lower()]
        # Unnormalized moment (N, 4, ng); apply the 1 / nnei normalization here.
        gr = (
            edge_conv(
                z2.contiguous(),
                h.contiguous(),
                idt,
                tt.contiguous(),
                gate_idx.contiguous(),
                sw_gate.contiguous(),
                rr.contiguous(),
                dst,
                graph.edge_mask,
                n_total,
                resnet_mult,
                act,
                gated,
            )
            / se.nnei
        )
        ng = se.neuron[-1]
        # G^T G contraction over the 4-axis (mirrors the dense _grrg_from_moment
        # on the (N, 4, ng) moment): permute to (N, ng, 4), contract the first
        # ``axis_neuron`` channels.
        gr_perm = gr.permute(0, 2, 1)  # (N, ng, 4)
        rot_mat = gr_perm[:, :, 1:4]  # (N, ng, 3)
        gr_sub = gr[:, :, : se.axis_neuron]  # (N, 4, axis)
        grrg = torch.matmul(gr_perm, gr_sub)  # (N, ng, axis)
        grrg = grrg.reshape(n_total, ng * se.axis_neuron).to(graph.edge_vec.dtype)
        if self.concat_output_tebd:
            atype_embd = type_embedding[atype]  # (N, tebd_dim)
            grrg = torch.cat([grrg, atype_embd], dim=-1)
        return grrg, rot_mat
