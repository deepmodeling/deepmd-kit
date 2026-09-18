# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

import numpy as np
import torch

from deepmd.dpmodel.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt.model.descriptor.descriptor import (
    DescriptorBlock,
)
from deepmd.pt.model.task.density import (
    DensityFittingNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    build_directional_neighbor_list,
    extend_input_and_build_neighbor_list,
)
from deepmd.pt.utils.region import (
    normalize_coord,
)
from deepmd.utils.path import (
    DPPath,
)

from .dp_atomic_model import (
    DPAtomicModel,
)

log = logging.getLogger(__name__)


class DPDensityAtomicModel(DPAtomicModel):
    def __init__(
        self,
        descriptor: BaseDescriptor,
        fitting: DensityFittingNet,
        type_map: list[str],
        **kwargs: Any,
    ) -> None:
        assert isinstance(fitting, DensityFittingNet)
        super().__init__(descriptor, fitting, type_map, **kwargs)
        self.rcut = self.descriptor.get_rcut()
        self.rcut_smth = self.descriptor.get_rcut_smth()
        if self.descriptor.get_env_protection() == 0.0:
            log.warning(
                "The descriptor env_protection is 0.0; grid points coincident "
                "with atoms would produce NaN densities. Setting it to 1e-6."
            )
            self._set_descriptor_env_protection(1e-6)
        self.sel = self.descriptor.get_sel()
        self.nnei = self.descriptor.get_nsel()

        wanted_shape = (1, self.nnei, 4)
        mean = torch.zeros(
            wanted_shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
        )
        stddev = torch.ones(
            wanted_shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
        )
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)

    def forward_atomic(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
        grid: torch.Tensor | None = None,
        grid_type: torch.Tensor | None = None,
        grid_nlist: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
        return_atomic_feature: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return atomic prediction.

        Parameters
        ----------
        extended_coord
            coodinates in extended region
        extended_atype
            atomic type in extended region
        nlist
            neighbor list. nf x nloc x nsel
        mapping
            mapps the extended indices to local indices
        fparam
            frame parameter. nf x ndf
        aparam
            atomic parameter. nf x nloc x nda

        Returns
        -------
        result_dict
            the result dict, defined by the `FittingOutputDef`.

        """
        del charge_spin, return_atomic_feature
        nframes, _, _ = nlist.shape
        if self.do_grad_r() or self.do_grad_c():
            extended_coord.requires_grad_(True)
        assert mapping is not None
        assert grid is not None
        assert grid_type is not None
        assert grid_nlist is not None
        _, ngrid, _ = grid_nlist.shape
        # nb x (ngrid+nall) x 3
        merged_coord = torch.cat([grid, extended_coord], dim=1)

        grid_atype = torch.ones(
            [nframes, ngrid], device=extended_atype.device, dtype=extended_atype.dtype
        ) * (self.descriptor.get_ntypes() - 1)
        # nb x (ngrid+nall)
        merged_atype = torch.cat([grid_atype, extended_atype], dim=1)

        # nb x ngrid
        grid_nlist_mask = grid_nlist >= 0
        shifted_grid_nlist = torch.where(grid_nlist_mask, grid_nlist + ngrid, -1)
        # nb x all
        nlist_mask = nlist >= 0
        shifted_nlist = torch.where(nlist_mask, nlist + ngrid, -1)
        # nb x (ngrid+nall)
        merged_nlist = torch.cat([shifted_grid_nlist, shifted_nlist], dim=1)

        # nb x ngrid, row i is the grid index i
        grid_mapping = torch.arange(
            ngrid, device=mapping.device, dtype=mapping.dtype
        ).expand(nframes, -1)
        # nb x (ngrid+nall)
        merged_mapping = torch.cat([grid_mapping, mapping + ngrid], dim=1)

        # the descriptor evaluates the environment of every merged point;
        # the atom rows are computed and discarded (only the grid rows feed
        # the fitting net), which is inherent to evaluating the merged
        # grid+atom system in one call
        descriptor, rot_mat, g2, h2, _sw = self.descriptor(
            merged_coord,
            merged_atype,
            merged_nlist,
            mapping=merged_mapping,
            comm_dict=comm_dict,
        )
        assert descriptor is not None

        # the fitting net must see the same grid type as the descriptor:
        # the reserved last entry of the type map
        grid_ftype = torch.ones(
            [nframes, ngrid], device=grid_type.device, dtype=grid_type.dtype
        ) * (self.descriptor.get_ntypes() - 1)
        ret = self.fitting_net(
            descriptor[:, :ngrid, :],
            grid_ftype,
            gr=rot_mat,
            g2=g2,
            h2=h2,
            fparam=fparam,
        )
        return ret

    def forward_common_atomic(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
        grid: torch.Tensor | None = None,
        grid_type: torch.Tensor | None = None,
        grid_nlist: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Common interface for atomic inference.

        This method accept extended coordinates, extended atom typs, neighbor list,
        and predict the atomic contribution of the fit property.

        Parameters
        ----------
        extended_coord
            extended coodinates, shape: nf x (nall x 3)
        extended_atype
            extended atom typs, shape: nf x nall
            for a type < 0 indicating the atomic is virtual.
        nlist
            neighbor list, shape: nf x nloc x nsel
        mapping
            extended to local index mapping, shape: nf x nall
        fparam
            frame parameters, shape: nf x dim_fparam
        aparam
            atomic parameter, shape: nf x nloc x dim_aparam
        comm_dict
            The data needed for communication for parallel inference.

        Returns
        -------
        ret_dict
            dict of output atomic properties.
            should implement the definition of `fitting_output_def`.
            ret_dict["mask"] of shape nf x nloc will be provided.
            ret_dict["mask"][ff,ii] == 1 indicating the ii-th atom of the ff-th frame is real.
            ret_dict["mask"][ff,ii] == 0 indicating the ii-th atom of the ff-th frame is virtual.

        """
        assert grid is not None
        assert grid_type is not None
        assert grid_nlist is not None
        del charge_spin
        nframes, _, _ = nlist.shape
        _, ngrid, _ = grid_nlist.shape

        if self.pair_excl is not None:
            pair_mask = self.pair_excl(nlist, extended_atype)
            # exclude neighbors in the nlist
            nlist = torch.where(pair_mask == 1, nlist, -1)
            # the directional grid-to-atom list: every center is the grid
            # type, so exclusions involving the grid type drop the neighbor
            reserved_type = self.descriptor.get_ntypes() - 1
            if any(
                reserved_type in pair for pair in self.pair_excl.get_exclude_types()
            ):
                nall = extended_atype.shape[1]
                nsel = grid_nlist.shape[-1]
                virtual_type = self.pair_excl.ntypes * torch.ones(
                    [nframes, 1],
                    dtype=extended_atype.dtype,
                    device=extended_atype.device,
                )
                ae = torch.cat([extended_atype, virtual_type], dim=-1)
                index = torch.where(grid_nlist == -1, nall, grid_nlist).view(
                    nframes, ngrid * nsel
                )
                type_j = torch.gather(ae, 1, index)
                type_ij = reserved_type * (self.pair_excl.ntypes + 1) + type_j
                grid_mask = self.pair_excl.type_mask[type_ij].view(nframes, ngrid, nsel)
                grid_nlist = torch.where(grid_mask == 1, grid_nlist, -1)

        ext_atom_mask = self.make_atom_mask(extended_atype)
        ret_dict = self.forward_atomic(
            extended_coord,
            torch.where(ext_atom_mask, extended_atype, 0),
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            comm_dict=comm_dict,
            grid=grid,
            grid_type=grid_type,
            grid_nlist=grid_nlist,
        )
        ret_dict = self.apply_out_stat(ret_dict, grid_type)

        # nf x ngrid
        grid_mask = torch.ones(
            [nframes, ngrid], dtype=torch.int32, device=ext_atom_mask.device
        )
        if self.atom_excl is not None:
            grid_mask *= self.atom_excl(grid_type)

        for kk in ret_dict.keys():
            out_shape = ret_dict[kk].shape
            out_shape2 = 1
            for ss in out_shape[2:]:
                out_shape2 *= ss
            ret_dict[kk] = (
                ret_dict[kk].reshape([out_shape[0], out_shape[1], out_shape2])
                * grid_mask[:, :, None]
            ).view(out_shape)
        ret_dict["mask"] = grid_mask

        return ret_dict

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "DPDensityAtomicModel.forward requires grid, grid_type and grid_nlist; "
            "use forward_common_atomic with explicit grid inputs instead."
        )

    def _get_forward_wrapper_func(
        self,
    ) -> Callable[..., dict[str, torch.Tensor]]:
        """Get a forward wrapper of the atomic model for output bias calculation."""

        def model_forward(
            coord: torch.Tensor,
            atype: torch.Tensor,
            box: torch.Tensor | None,
            fparam: torch.Tensor | None = None,
            aparam: torch.Tensor | None = None,
            charge_spin: torch.Tensor | None = None,
            spin: torch.Tensor | None = None,
            grid: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            del charge_spin, spin
            with torch.no_grad():
                (
                    extended_coord,
                    extended_atype,
                    mapping,
                    nlist,
                ) = extend_input_and_build_neighbor_list(
                    coord,
                    atype,
                    self.get_rcut(),
                    self.get_sel(),
                    mixed_types=self.mixed_types(),
                    box=box,
                )
                assert grid is not None
                if box is not None:
                    # same wrapping convention as forward_common: grid points
                    # outside the primary cell are periodically equivalent
                    grid = normalize_coord(
                        grid, box.to(grid.device).reshape(box.shape[0], 3, 3)
                    )
                grid_type = torch.full(
                    grid.shape[:-1],
                    self.descriptor.get_ntypes() - 1,
                    device=grid.device,
                    dtype=atype.dtype,
                )
                grid_nlist = build_directional_neighbor_list(
                    grid,
                    grid_type,
                    extended_coord,
                    extended_atype,
                    self.get_rcut(),
                    self.get_sel(),
                    distinguish_types=(not self.mixed_types()),
                )
                atomic_ret = self.forward_common_atomic(
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping=mapping,
                    fparam=fparam,
                    aparam=aparam,
                    grid=grid,
                    grid_type=grid_type,
                    grid_nlist=grid_nlist,
                )
                return {kk: vv.detach() for kk, vv in atomic_ret.items()}

        return model_forward

    def change_out_bias(
        self,
        sample_merged: Callable[[], list[dict]] | list[dict],
        stat_file_path: DPPath | None = None,
        bias_adjust_mode: str = "change-by-statistic",
    ) -> None:
        """Change the output bias according to the input data and the pretrained model.

        For density models, this operation is skipped because the output is
        grid-based rather than atomic-based, and the standard atomic bias
        adjustment (change-by-statistic / set-by-statistic) does not apply.
        The fitting net will adapt to the target dataset through normal
        gradient descent during training.
        """
        log.warning("change_out_bias is not supported for density models; skipping.")

    def compute_or_load_stat(
        self,
        sampled_func: Callable[[], list[dict]] | list[dict],
        stat_file_path: DPPath | None = None,
        compute_or_load_out_stat: bool = True,
        preset_observed_type: list[str] | None = None,
    ) -> None:
        """Compute or load statistics, with real statistics for the grid type.

        The reserved grid type (``ntypes - 1``) has no real atoms in any
        training system, so the standard pass yields zero samples for it
        and falls back to placeholder statistics (mean 0, stddev 0.1),
        which are then applied to exactly the rows that produce the
        density output. When the sampled data provides grids, a second
        pass is run with the grid points injected as pseudo-atoms of the
        grid type, and only the grid-type row of the statistics is taken
        from it; the real types keep their clean statistics from the
        standard pass.

        The injected pass only runs when the grid-type statistics are
        missing (no cache, or a cache whose grid-type row has zero
        samples), so the stat-file fast path still applies; when a stat
        file is used, the patched grid-type row is written back to disk so
        that the file and the in-memory statistics agree.
        """
        sampled = None
        grid_stats = None
        if self._grid_stat_missing(stat_file_path):
            sampled = sampled_func() if callable(sampled_func) else sampled_func
            if any("grid" in sample for sample in sampled):
                try:
                    # run the wrapped sampler first so the injected pass sees
                    # the same pair_exclude_types as the standard pass
                    wrapped = self._make_wrapped_sampler(lambda: sampled)
                    self.descriptor.compute_input_stats(
                        self._inject_grid_samples(wrapped())
                    )
                    grid_stats = [
                        (
                            block["davg"][-1].detach().clone(),
                            block["dstd"][-1].detach().clone(),
                        )
                        for block in self._descriptor_stat_blocks()
                    ]
                except (TypeError, KeyError) as err:
                    log.warning(
                        "Cannot compute input statistics for the grid type (%s); "
                        "falling back to the descriptor defaults.",
                        err,
                    )
            else:
                log.warning(
                    "No grid data in the sampled frames; the grid type gets the "
                    "descriptor's default input statistics."
                )
        super().compute_or_load_stat(
            (lambda: sampled) if sampled is not None else sampled_func,
            stat_file_path,
            compute_or_load_out_stat=compute_or_load_out_stat,
            preset_observed_type=preset_observed_type,
        )
        if grid_stats is not None:
            for block, (grid_davg, grid_dstd) in zip(
                self._descriptor_stat_blocks(), grid_stats, strict=True
            ):
                davg = block["davg"]
                dstd = block["dstd"]
                davg[-1] = grid_davg.to(device=davg.device, dtype=davg.dtype)
                dstd[-1] = grid_dstd.to(device=dstd.device, dtype=dstd.dtype)
            if stat_file_path is not None:
                self._write_back_grid_stat(stat_file_path, grid_stats)

    def _descriptor_stat_blocks(self) -> list:
        """Collect every descriptor block that carries davg/dstd statistics.

        Descriptors such as DPA-2 and hybrids carry several blocks, each with
        its own statistics; the grid-type row must be patched in all of them.
        The blocks are discovered by walking the module tree (via
        ``named_modules``) and probing the statistics item protocol, so no
        per-descriptor attribute list is needed.
        """
        blocks = []
        for _name, module in self.descriptor.named_modules():
            # filter networks etc. also implement __getitem__ with unrelated
            # semantics, so only descriptor-level modules are probed for the
            # statistics item protocol (some descriptors, e.g. se_r, carry
            # the statistics on the descriptor itself)
            if not isinstance(module, (BaseDescriptor, DescriptorBlock)):
                continue
            try:
                module["davg"]
                module["dstd"]
            except (TypeError, KeyError):
                continue
            blocks.append(module)
        if blocks:
            return blocks
        try:
            self.descriptor["davg"]
            self.descriptor["dstd"]
            return [self.descriptor]
        except (TypeError, KeyError):
            pass
        raise KeyError("davg/dstd not accessible on this descriptor")

    def _set_descriptor_env_protection(self, value: float) -> None:
        """Set env_protection on every descriptor block, hybrids included.

        Wrapper-level fields, if any, are derived from the blocks; the whole
        module tree is walked so no per-descriptor attribute list is needed.
        """
        for _name, module in self.descriptor.named_modules():
            if hasattr(module, "env_protection"):
                module.env_protection = value

    def _stat_cache_root(self, stat_file_path: DPPath) -> DPPath:
        """Apply the same type_map subdirectory as the parent's stat path."""
        if self.type_map is not None:
            stat_file_path = stat_file_path / " ".join(self.type_map)
        return stat_file_path

    def _grid_stat_missing(self, stat_file_path: DPPath | None) -> bool:
        """Tell whether the grid-type input statistics still need computing.

        True when there is no complete stat cache, or when the cache is
        complete but its grid-type row has zero samples (placeholder
        statistics saved by an older run).
        """
        if stat_file_path is None:
            return True
        try:
            blocks = self._descriptor_stat_blocks()
        except KeyError:
            # no accessible statistics: let the injected pass soft-fail below
            return True
        stat_file_path = self._stat_cache_root(stat_file_path)
        grid_type = self.descriptor.get_ntypes() - 1
        for block in blocks:
            env_stat = EnvMatStatSe(block)
            cache = stat_file_path / env_stat.get_hash()
            keys = env_stat.get_stat_keys()
            if not cache.is_dir() or any(not (cache / kk).is_file() for kk in keys):
                return True
            # number of samples of the grid-type row; zero means placeholder
            if float((cache / f"r_{grid_type}").load_numpy()[0]) == 0.0:
                return True
        return False

    def _write_back_grid_stat(
        self,
        stat_file_path: DPPath,
        grid_stats: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Write the patched grid-type row back to the stat cache files.

        The cache stores raw sums (number, sum, squared_sum) per stat key;
        a single sample reproduces the patched mean and stddev exactly.
        """
        grid_type = self.descriptor.get_ntypes() - 1
        stat_file_path = self._stat_cache_root(stat_file_path)
        for block, (grid_davg, grid_dstd) in zip(
            self._descriptor_stat_blocks(), grid_stats, strict=True
        ):
            env_stat = EnvMatStatSe(block)
            cache = stat_file_path / env_stat.get_hash()
            if not cache.is_dir():
                continue
            davg_np = grid_davg.detach().cpu().numpy()
            dstd_np = grid_dstd.detach().cpu().numpy()
            mean_r = float(davg_np[0, 0])
            std_r = float(dstd_np[0, 0])
            (cache / f"r_{grid_type}").save_numpy(
                np.array([1.0, mean_r, std_r**2 + mean_r**2])
            )
            if dstd_np.shape[-1] == 4:
                std_a = float(dstd_np[0, 1])
                (cache / f"a_{grid_type}").save_numpy(np.array([1.0, 0.0, std_a**2]))

    def _inject_grid_samples(self, sampled: list[dict]) -> list[dict]:
        """Append the grid points of each sample as pseudo-atoms of the
        reserved grid type, so the descriptor input statistics get real
        samples for it. Neighbor lists are rebuilt by the stat machinery,
        so only ``coord`` and ``atype`` need to be extended.

        Grid-grid pairs are excluded from the injected neighbor lists so
        that the grid-type statistics are computed over grid-to-atom
        pairs, matching the directional neighbor list of the forward pass
        (a no-op for per-type-sel descriptors with ``sel[-1] == 0``, but
        required for mixed-type descriptors with a scalar sel).
        """
        ntypes = self.descriptor.get_ntypes()
        grid_type = ntypes - 1
        injected = []
        for sample in sampled:
            grid = sample.get("grid")
            if grid is None:
                injected.append(sample)
                continue
            sample = dict(sample)
            coord = sample["coord"]
            atype = sample["atype"]
            nframes = atype.shape[0]
            gg = grid.reshape(nframes, -1, 3).to(coord.dtype)
            ngrid = gg.shape[1]
            sample["coord"] = torch.cat(
                [coord.reshape(nframes, -1, 3), gg], dim=1
            ).reshape(nframes, -1)
            sample["atype"] = torch.cat(
                [
                    atype,
                    torch.full(
                        (nframes, ngrid),
                        grid_type,
                        dtype=atype.dtype,
                        device=atype.device,
                    ),
                ],
                dim=1,
            )
            pair_exclude_types = [
                tuple(pair) for pair in sample.get("pair_exclude_types", [])
            ]
            if (grid_type, grid_type) not in pair_exclude_types:
                pair_exclude_types.append((grid_type, grid_type))
            sample["pair_exclude_types"] = pair_exclude_types
            injected.append(sample)
        return injected

    def compute_or_load_out_stat(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        stat_file_path: DPPath | None = None,
    ) -> None:
        """
        Compute the output statistics (e.g. energy bias) for the fitting net from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        stat_file_path : Optional[DPPath]
            The path to the stat file.

        """
        log.warning("Not implemented yet for density out stat!")
