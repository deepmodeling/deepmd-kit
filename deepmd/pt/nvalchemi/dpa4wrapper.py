# SPDX-License-Identifier: LGPL-3.0-or-later
"""nvalchemi-toolkit wrapper for the SeZM / DPA-4 PyTorch energy model.

:class:`DPA4Wrapper` adapts a trained DeePMD-kit SeZM / DPA-4 model to the
``nvalchemi-toolkit`` model interface (:class:`~nvalchemi.models.base.BaseModelMixin`)
so it can be driven by any ``nvalchemi`` dynamics engine (NVE, NVT, NPT, FIRE).
The underlying model is used unchanged; the wrapper only translates between the
``nvalchemi`` graph batch and SeZM's sparse edge-list interface, and maps the
outputs back to ``nvalchemi``'s ``energy`` / ``forces`` / ``stress``.

Two backends are supported through :meth:`DPA4Wrapper.from_checkpoint`:

* a ``.pt`` training checkpoint, run eagerly as a :class:`SeZMModel`. Set
  ``DP_COMPILE_INFER=1`` (optionally ``DP_TRITON_INFER=1``) before loading to
  enable SeZM's compiled-inference path.
* a frozen ``.pt2`` AOTInductor package, run through its precompiled callable
  (float64 I/O, device-locked to the host it was frozen on).

Neighbour-list and geometry conventions
---------------------------------------
``nvalchemi`` supplies a COO neighbour list whose rows are ``[source, target]``
(``source`` is the centre atom, ``target`` the neighbour), with an integer image
``neighbor_list_shifts`` belonging to ``target``. The per-edge displacement is
``r = positions[target] - positions[source] + shifts @ cell``. SeZM consumes
``edge_index = [src, dst]`` with ``edge_vec = r_src - r_dst`` and aggregates
messages onto ``dst``, so the wrapper maps ``dst = source`` and ``src = target``.

The reported stress is the Cauchy stress (virial divided by the cell volume),
which matches ``nvalchemi``'s sign convention. A whole batch is presented to
SeZM as a single frame; per-graph energy and virial are recovered by
segment-summing the per-atom outputs with ``batch_idx``.
"""

from __future__ import (
    annotations,
)

from collections import (
    OrderedDict,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
from nvalchemi.data import (
    AtomicData,
    Batch,
)
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)
from torch import (
    nn,
)

from deepmd.pt.model.model.sezm_model import (
    ELEMENT_TO_Z,
    SeZMModel,
)

if TYPE_CHECKING:
    from pathlib import (
        Path,
    )

    from nvalchemi._typing import (
        ModelOutputs,
    )

__all__ = ["DPA4Wrapper", "SeZMWrapper"]

# Location of the metadata written into a SeZM/DPA-4 ``.pt2`` archive.
_PT2_METADATA_ENTRY = "model/extra/metadata.json"


class DPA4Wrapper(nn.Module, BaseModelMixin):
    """Wrap a trained SeZM / DPA-4 model as an ``nvalchemi`` model.

    Use :meth:`from_checkpoint` to load from a ``.pt`` checkpoint or a frozen
    ``.pt2`` package, or construct directly from an in-memory :class:`SeZMModel`.

    Parameters
    ----------
    model
        An instantiated SeZM / DPA-4 energy model (:class:`SeZMModel`). It is put
        into ``eval`` mode so the sparse-edge path is deterministic.
    atomic_number_to_type
        Optional explicit mapping ``{atomic_number: type_index}``. When ``None``
        the mapping is derived from the model ``type_map`` via the periodic
        table, which requires the type map to contain element symbols. Provide
        this override for non-element type maps.
    compute_stress
        If ``True``, ``"stress"`` is added to the active outputs so NPT / NPH
        barostats receive the Cauchy stress. Stress requires a periodic ``cell``.
    default_charge_spin
        Optional global ``[charge, spin]`` forwarded to SeZM models built with
        ``add_chg_spin_ebd=True``. It applies to the whole batch.

    Attributes
    ----------
    model
        The wrapped :class:`SeZMModel`, or ``None`` for a ``.pt2`` backend.
    model_config
        Mutable :class:`~nvalchemi.models.base.ModelConfig` controlling which
        outputs are produced and describing the required neighbour list.
    """

    model: SeZMModel | None

    def __init__(
        self,
        model: SeZMModel,
        *,
        atomic_number_to_type: dict[int, int] | None = None,
        compute_stress: bool = False,
        default_charge_spin: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self._aoti_runner: Any | None = None
        self._aoti_dim_fparam = 0
        self._aoti_dim_aparam = 0
        self._dtype = next(model.parameters()).dtype
        self._descriptor_dim: int | None = int(
            model.atomic_model.descriptor.get_dim_out()
        )
        self._configure(
            rcut=float(model.get_rcut()),
            type_map=list(model.get_type_map()),
            device=next(model.parameters()).device,
            atomic_number_to_type=atomic_number_to_type,
            compute_stress=compute_stress,
            default_charge_spin=default_charge_spin,
        )

    def _configure(
        self,
        *,
        rcut: float,
        type_map: list[str],
        device: torch.device,
        atomic_number_to_type: dict[int, int] | None,
        compute_stress: bool,
        default_charge_spin: list[float] | None,
    ) -> None:
        """Set up the shared configuration common to both backends."""
        self.default_charge_spin = default_charge_spin
        self.rcut = float(rcut)

        # Memoized species mapping, keyed on the atomic-number tensor identity so
        # a dynamics run validates types once instead of on every step.
        self._atype_cache: tuple[tuple[int, int, torch.device], torch.Tensor] | None = (
            None
        )
        # Pre-build the optional charge/spin condition once so the hot path does
        # not rebuild it from a Python list (a host-to-device copy) every step.
        charge_spin = (
            None
            if default_charge_spin is None
            else torch.tensor(
                default_charge_spin, dtype=self._dtype, device=device
            ).view(1, 2)
        )
        self.register_buffer("_charge_spin_buf", charge_spin, persistent=False)

        z_to_type = self._build_z_to_type(type_map, atomic_number_to_type, device)
        # persistent=False: derived from the type map, excluded from the state
        # dict but kept in sync with ``.to()`` device moves.
        self.register_buffer("_z_to_type", z_to_type, persistent=False)

        active: set[str] = {"energy", "forces"}
        if compute_stress:
            active.add("stress")
        # Forces and stress are produced by the model itself (SeZM's internal
        # ``edge_vec`` autograd), so they are returned directly rather than via
        # an ``nvalchemi`` autograd pass through ``positions``.
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            active_outputs=active,
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset(),
            optional_inputs=frozenset({"cell", "neighbor_list_shifts"}),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=self.rcut,
                format=NeighborListFormat.COO,
                half_list=False,
            ),
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_z_to_type(
        type_map: list[str],
        atomic_number_to_type: dict[int, int] | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a dense ``atomic_number -> type_index`` lookup tensor.

        Atomic numbers absent from the mapping map to ``-1`` so the forward pass
        can raise a clear error instead of silently mislabelling atoms.
        """
        if atomic_number_to_type is None:
            atomic_number_to_type = {}
            for type_index, symbol in enumerate(type_map):
                z = ELEMENT_TO_Z.get(symbol)
                if z is None:
                    raise ValueError(
                        f"Cannot map type map entry {symbol!r} to an atomic "
                        "number. Pass an explicit `atomic_number_to_type` "
                        "mapping for non-element type maps."
                    )
                atomic_number_to_type[z] = type_index
        if not atomic_number_to_type:
            raise ValueError("`atomic_number_to_type` resolved to an empty mapping.")
        max_z = max(atomic_number_to_type)
        table = torch.full((max_z + 1,), -1, dtype=torch.long, device=device)
        for z, type_index in atomic_number_to_type.items():
            table[int(z)] = int(type_index)
        return table

    # ------------------------------------------------------------------
    # BaseModelMixin required members
    # ------------------------------------------------------------------

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-atom and per-graph descriptor embedding widths."""
        if self._descriptor_dim is None:
            raise NotImplementedError(
                "Embeddings are only available for the `.pt` backend, not a "
                "frozen `.pt2` package."
            )
        return {
            "node_embeddings": (self._descriptor_dim,),
            "graph_embeddings": (self._descriptor_dim,),
        }

    # ------------------------------------------------------------------
    # Input / output adaptation
    # ------------------------------------------------------------------

    def _atype(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """Map atomic numbers to SeZM type indices via the lookup table.

        The result is memoized on the identity of *atomic_numbers* (storage
        pointer, length and device). A dynamics run reuses the same
        ``atomic_numbers`` tensor for every step while only mutating positions,
        so the species mapping -- and the two host-device synchronizations its
        validation needs -- run once on the first step and are skipped
        afterwards, keeping the MD hot path free of stream stalls.
        """
        key = (atomic_numbers.data_ptr(), atomic_numbers.numel(), atomic_numbers.device)
        cached = self._atype_cache
        if cached is not None and cached[0] == key:
            return cached[1]

        z = atomic_numbers.long()
        if z.numel() and int(z.max()) >= self._z_to_type.shape[0]:
            raise ValueError("Encountered an atomic number outside the model type map.")
        atype = self._z_to_type.index_select(0, z.clamp_min(0))
        if bool((atype < 0).any()):
            missing = sorted({int(v) for v in z[atype < 0].tolist()})
            raise ValueError(
                f"Atomic numbers {missing} are not present in the model type map."
            )
        self._atype_cache = (key, atype)
        return atype

    def _edge_schema(
        self, data: Batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Translate the COO neighbour list into SeZM's edge schema.

        Returns
        -------
        edge_index
            ``(2, E)`` with rows ``[src=neighbour, dst=centre]`` in flattened
            local-atom space.
        edge_vec
            ``(E, 3)`` displacement ``r_src - r_dst`` (PBC images folded in).
        edge_mask
            ``(E,)`` all-true validity mask for the real edges.
        """
        neighbor_list = getattr(data, "neighbor_list", None)
        if neighbor_list is None:
            raise KeyError(
                "Batch has no `neighbor_list`. Run `compute_neighbors(batch, "
                "config=model.model_config.neighbor_config)` or register a "
                "COO NeighborListHook before calling the model."
            )
        dtype = self._dtype
        device = data.positions.device
        positions = data.positions.to(dtype)

        neighbor_list = neighbor_list.long()
        source = neighbor_list[:, 0]
        target = neighbor_list[:, 1]
        n_edge = neighbor_list.shape[0]

        cell = getattr(data, "cell", None)
        neighbor_list_shifts = getattr(data, "neighbor_list_shifts", None)
        if cell is not None and neighbor_list_shifts is None:
            raise ValueError(
                "A periodic `cell` was provided without `neighbor_list_shifts`; "
                "PBC image shifts are required for correct edge vectors. Run "
                "`compute_neighbors` with the cell so the shifts are attached."
            )
        if neighbor_list_shifts is not None and cell is not None:
            cell = cell.to(dtype)
            shifts = neighbor_list_shifts.to(dtype)
            if cell.shape[0] == 1:
                # Single-frame fast path (the common MD case): every edge shares
                # the one cell, so a single (E,3)x(3,3) matmul replaces gathering
                # an (E,3,3) per-edge cell and the einsum over it.
                shift_vec = shifts @ cell[0]
            else:
                graph_per_edge = data.batch_idx.long().index_select(0, source)
                cell_per_edge = cell.index_select(0, graph_per_edge)
                shift_vec = torch.einsum("eb,ebc->ec", shifts, cell_per_edge)
        else:
            shift_vec = torch.zeros(n_edge, 3, dtype=dtype, device=device)

        edge_vec = (
            positions.index_select(0, target)
            - positions.index_select(0, source)
            + shift_vec
        )
        edge_index = torch.stack([target, source], dim=0)
        edge_mask = torch.ones(n_edge, dtype=torch.bool, device=device)
        return edge_index, edge_vec, edge_mask

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        """Build the lower-interface inputs for the wrapped model.

        The batch is presented as a single frame with ``nloc`` equal to the
        total number of atoms; the COO neighbour list already carries the global
        node offsets, so heterogeneous multi-graph batches need no special
        handling here.
        """
        del kwargs
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])
        dtype = self._dtype
        n_node = data.num_nodes

        coord = data.positions.to(dtype).view(1, n_node, 3)
        atype = self._atype(data.atomic_numbers).view(1, n_node)
        edge_index, edge_vec, edge_mask = self._edge_schema(data)
        return {
            "coord": coord,
            "atype": atype,
            "edge_index": edge_index,
            "edge_vec": edge_vec,
            "edge_scatter_index": edge_index,
            "edge_mask": edge_mask,
            "charge_spin": self._charge_spin_buf,
        }

    def adapt_output(
        self, model_output: dict[str, torch.Tensor], data: Batch
    ) -> ModelOutputs:
        """Map the lower-interface outputs to the ``nvalchemi`` output dict.

        Per-atom energy and virial are segment-summed with ``batch_idx`` so each
        graph gets its own total, even though the batch is run as a single frame.
        ``model_output`` carries the normalized keys ``atom_energy``,
        ``extended_force`` and ``extended_virial``.

        When ``stress`` is active the cell must be non-degenerate (positive
        volume): the stress is ``virial / |det(cell)|``, so a singular cell would
        yield non-finite values.
        """
        batch_idx = data.batch_idx.long()
        n_graph = data.num_graphs
        n_node = data.num_nodes
        out_dtype = data.positions.dtype

        atom_energy = model_output["atom_energy"].reshape(n_node)
        energy = torch.zeros(
            n_graph, dtype=atom_energy.dtype, device=atom_energy.device
        ).index_add_(0, batch_idx, atom_energy)

        output: ModelOutputs = OrderedDict()
        output["energy"] = energy.unsqueeze(-1).to(out_dtype)

        active = self.model_config.active_outputs
        if "forces" in active:
            output["forces"] = (
                model_output["extended_force"].reshape(n_node, 3).to(out_dtype)
            )
        if "stress" in active:
            cell = getattr(data, "cell", None)
            if cell is None:
                raise ValueError(
                    "stress output requires a periodic `cell` for the volume."
                )
            atom_virial = model_output["extended_virial"].reshape(n_node, 9)
            virial = torch.zeros(
                n_graph, 9, dtype=atom_virial.dtype, device=atom_virial.device
            ).index_add_(0, batch_idx, atom_virial)
            volume = torch.det(cell.to(virial.dtype)).abs().view(n_graph, 1, 1)
            output["stress"] = (virial.view(n_graph, 3, 3) / volume).to(out_dtype)
        return output

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run the model on a batch and return the ``nvalchemi`` output dict."""
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])
        model_inputs = self.adapt_input(data, **kwargs)
        if self._aoti_runner is not None:
            model_ret = self._run_pt2(model_inputs)
        else:
            need_virial = "stress" in self.model_config.active_outputs
            model_ret = self.model.forward_lower(
                model_inputs["coord"],
                model_inputs["atype"],
                model_inputs["edge_index"],
                model_inputs["edge_vec"],
                model_inputs["edge_scatter_index"],
                model_inputs["edge_mask"],
                do_atomic_virial=need_virial,
                charge_spin=model_inputs["charge_spin"],
            )
        return self.adapt_output(model_ret, data)

    def _run_pt2(self, model_inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run the frozen ``.pt2`` callable and normalize its output keys.

        The AOTInductor package traces ``forward_common_lower`` and returns a
        dict keyed by the raw DeePMD names; here they are renamed to the keys
        :meth:`adapt_output` expects. ``None`` trailing arguments are filtered
        by the loader, matching the export-time signature.
        """
        if self._aoti_dim_fparam or self._aoti_dim_aparam:
            raise NotImplementedError(
                "The `.pt2` backend does not support models requiring frame or "
                "atomic parameters (fparam / aparam) through nvalchemi."
            )
        ret = self._aoti_runner(
            model_inputs["coord"],
            model_inputs["atype"],
            model_inputs["edge_index"],
            model_inputs["edge_vec"],
            model_inputs["edge_scatter_index"],
            model_inputs["edge_mask"],
            None,
            None,
            model_inputs["charge_spin"],
        )
        return {
            "atom_energy": ret["energy"],
            "extended_force": ret["energy_derv_r"],
            "extended_virial": ret["energy_derv_c"],
        }

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Attach per-atom / per-graph descriptor embeddings to *data*.

        Writes ``node_embeddings`` (``[N, descriptor_dim]``) and
        ``graph_embeddings`` (``[B, descriptor_dim]``, sum-pooled over atoms) in
        place. Only supported for the ``.pt`` backend.
        """
        del kwargs
        if self.model is None:
            raise NotImplementedError(
                "Embeddings are only available for the `.pt` backend, not a "
                "frozen `.pt2` package."
            )
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])
        model_inputs = self.adapt_input(data)
        ret = self.model.forward_common_lower(
            model_inputs["coord"],
            model_inputs["atype"],
            model_inputs["edge_index"],
            model_inputs["edge_vec"],
            model_inputs["edge_scatter_index"],
            model_inputs["edge_mask"],
            charge_spin=model_inputs["charge_spin"],
            embedding_only=True,
        )
        n_node = data.num_nodes
        node_embeddings = ret["descriptor"].reshape(n_node, self._descriptor_dim)

        atoms_group = data._atoms_group
        if atoms_group is not None:
            atoms_group["node_embeddings"] = node_embeddings
        else:
            data.node_embeddings = node_embeddings

        graph_embeddings = torch.zeros(
            data.num_graphs,
            self._descriptor_dim,
            dtype=node_embeddings.dtype,
            device=node_embeddings.device,
        )
        graph_embeddings.index_add_(0, data.batch_idx.long(), node_embeddings)
        data.graph_embeddings = graph_embeddings
        return data

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        device: torch.device | str = "cpu",
        *,
        head: str | None = None,
        **wrapper_kwargs: Any,
    ) -> DPA4Wrapper:
        """Load a DeePMD-kit SeZM / DPA-4 model into a wrapper.

        Parameters
        ----------
        checkpoint_path
            Either a ``.pt`` training checkpoint or a frozen ``.pt2``
            AOTInductor package.
        device
            Target device. For ``.pt2`` the package is device-locked to its
            freeze host, so this must match.
        head
            Multi-task branch name; required for a multi-task ``.pt`` checkpoint.
        **wrapper_kwargs
            Forwarded to :class:`DPA4Wrapper` (e.g. ``compute_stress``,
            ``atomic_number_to_type``).
        """
        device = torch.device(device) if isinstance(device, str) else device
        if str(checkpoint_path).endswith(".pt2"):
            if head is not None:
                raise NotImplementedError(
                    "Head selection is not supported for a frozen `.pt2` package; "
                    "freeze the desired head instead."
                )
            return cls._from_pt2(checkpoint_path, device, **wrapper_kwargs)
        return cls._from_pt(checkpoint_path, device, head=head, **wrapper_kwargs)

    @classmethod
    def _from_pt(
        cls,
        checkpoint_path: Path | str,
        device: torch.device,
        *,
        head: str | None,
        **wrapper_kwargs: Any,
    ) -> DPA4Wrapper:
        """Load a ``.pt`` training checkpoint into a SeZM-backed wrapper."""
        from deepmd.pt.model.model import (
            get_model,
        )
        from deepmd.pt.train.wrapper import (
            ModelWrapper,
        )

        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model_params = state_dict["_extra_state"]["model_params"]

        if "model_dict" in model_params:
            if head is None:
                raise ValueError(
                    "`head` must be specified for a multi-task checkpoint. "
                    f"Available heads: {list(model_params['model_dict'])}."
                )
            if head not in model_params["model_dict"]:
                raise ValueError(
                    f"Unknown head {head!r} for this multi-task checkpoint. "
                    f"Available heads: {list(model_params['model_dict'])}."
                )
            model_params = model_params["model_dict"][head]
            head_state = {"_extra_state": state_dict["_extra_state"]}
            for key, value in state_dict.items():
                if f"model.{head}." in key:
                    head_state[key.replace(f"model.{head}.", "model.Default.")] = (
                        value.clone()
                    )
            state_dict = head_state

        model_params.pop("hessian_mode", None)
        model = get_model(model_params).to(device)
        model_wrapper = ModelWrapper(model)
        model_wrapper.load_state_dict(state_dict)
        sezm_model = model_wrapper.model["Default"]
        if not isinstance(sezm_model, SeZMModel):
            raise TypeError(
                "Checkpoint does not contain a SeZM / DPA-4 model; got "
                f"{type(sezm_model).__name__}."
            )
        return cls(sezm_model.to(device), **wrapper_kwargs)

    @classmethod
    def _from_pt2(
        cls,
        package_path: Path | str,
        device: torch.device,
        **wrapper_kwargs: Any,
    ) -> DPA4Wrapper:
        """Load a frozen ``.pt2`` AOTInductor package into a wrapper."""
        import json
        import zipfile

        from torch._inductor import (
            aoti_load_package,
        )

        with zipfile.ZipFile(package_path, "r") as archive:
            if _PT2_METADATA_ENTRY not in archive.namelist():
                raise ValueError(
                    f"{package_path!s} is missing {_PT2_METADATA_ENTRY!r}; it does "
                    "not look like a SeZM / DPA-4 `.pt2` archive."
                )
            metadata = json.loads(archive.read(_PT2_METADATA_ENTRY).decode("utf-8"))

        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.model = None
        self._aoti_runner = aoti_load_package(str(package_path))
        self._aoti_dim_fparam = int(metadata.get("dim_fparam", 0))
        self._aoti_dim_aparam = int(metadata.get("dim_aparam", 0))
        # `.pt2` packages are compiled with float64 I/O.
        self._dtype = torch.float64
        self._descriptor_dim = None
        self._configure(
            rcut=float(metadata["rcut"]),
            type_map=list(metadata["type_map"]),
            device=device,
            atomic_number_to_type=wrapper_kwargs.pop("atomic_number_to_type", None),
            compute_stress=wrapper_kwargs.pop("compute_stress", False),
            default_charge_spin=wrapper_kwargs.pop("default_charge_spin", None),
        )
        if wrapper_kwargs:
            raise TypeError(
                f"Unexpected keyword arguments for a `.pt2` wrapper: "
                f"{sorted(wrapper_kwargs)}."
            )
        return self


# The descriptor is registered as both SeZM and DPA4; this alias keeps the
# SeZM name available.
SeZMWrapper = DPA4Wrapper
