# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Callable,
    Optional,
    Union,
)

import torch

from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat,
)
from deepmd.pt.model.descriptor.repformer_layer import (
    RepformerLayer,
    _make_nei_g1,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
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
from deepmd.utils.path import (
    DPPath,
)

from .dp_atomic_model import (
    DPAtomicModel,
)

log = logging.getLogger(__name__)


class DPDensityAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        assert isinstance(fitting, DensityFittingNet)
        super().__init__(descriptor, fitting, type_map, **kwargs)
        self.rcut = self.descriptor.get_rcut()
        self.rcut_smth = self.descriptor.get_rcut_smth()
        self.env_protection = self.descriptor.get_env_protection()
        if self.env_protection == 0.0:
            self.env_protection = 1e-6
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
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
        grid: Optional[torch.Tensor] = None,
        grid_type: Optional[torch.Tensor] = None,
        grid_nlist: Optional[torch.Tensor] = None,
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
        nframes, nloc, nnei = nlist.shape
        atype = extended_atype[:, :nloc]
        if self.do_grad_r() or self.do_grad_c():
            extended_coord.requires_grad_(True)
        assert mapping is not None
        assert grid is not None
        assert grid_type is not None
        assert grid_nlist is not None
        bsz, ngrid, nnei = grid_nlist.shape
        # nb x (ngrid+nall) x 3
        merged_coord = torch.cat([grid, extended_coord], dim=1)

        grid_atype = torch.ones([nframes, ngrid], device=extended_atype.device, dtype=extended_atype.dtype)*(self.descriptor.ntypes - 1)
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

        grid_mapping = torch.cat([torch.ones([nframes, 1], device=mapping.device, dtype=mapping.dtype)* i for i in range(ngrid)], dim=1)
        # nb x (ngrid+nall)
        merged_mapping = torch.cat([grid_mapping, mapping + ngrid], dim=1)

        descriptor, rot_mat, g2, h2, sw = self.descriptor(
            merged_coord,
            merged_atype,
            merged_nlist,
            mapping=merged_mapping,
            comm_dict=comm_dict,
        )
        assert descriptor is not None

        ret = self.fitting_net(
            descriptor[:, :ngrid, :],
            torch.zeros([nframes, ngrid], device=grid_type.device, dtype=grid_type.dtype),
            gr=rot_mat,
            g2=g2,
            h2=h2,
            fparam=fparam,
            aparam=aparam,
        )
        return ret

    def forward_common_atomic(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
        grid: Optional[torch.Tensor] = None,
        grid_type: Optional[torch.Tensor] = None,
        grid_nlist: Optional[torch.Tensor] = None,
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
        nframes, nloc, _ = nlist.shape
        _, ngrid, _ = grid_nlist.shape
        atype = extended_atype[:, :nloc]

        if self.pair_excl is not None:
            pair_mask = self.pair_excl(nlist, extended_atype)
            # exclude neighbors in the nlist
            nlist = torch.where(pair_mask == 1, nlist, -1)

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

        ext_grid_mask = self.make_atom_mask(grid_type)

        # nf x ngrid
        grid_mask = torch.ones([nframes, ngrid], dtype=torch.int32, device=ext_atom_mask.device)
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
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        return self.forward_common_atomic(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            comm_dict=comm_dict,
        )

    def _get_forward_wrapper_func(self) -> Callable[..., torch.Tensor]:
        """Get a forward wrapper of the atomic model for output bias calculation."""

        def model_forward(
            coord: torch.Tensor,
            atype: torch.Tensor,
            box: Optional[torch.Tensor],
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
            grid: Optional[torch.Tensor] = None,
        ) -> dict[str, torch.Tensor]:
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
                grid_type = torch.zeros(
                    grid.shape[:-1], device=grid.device, dtype=atype.dtype
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
        sample_merged: Union[Callable[[], list[dict]], list[dict]],
        stat_file_path: Optional[DPPath] = None,
        bias_adjust_mode: str = "change-by-statistic",
    ) -> None:
        """Change the output bias according to the input data and the pretrained model.

        For density models, this operation is skipped because the output is
        grid-based rather than atomic-based, and the standard atomic bias
        adjustment (change-by-statistic / set-by-statistic) does not apply.
        The fitting net will adapt to the target dataset through normal
        gradient descent during training.
        """
        log.warning(
            "change_out_bias is not supported for density models; skipping."
        )
        return

    def compute_or_load_out_stat(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        stat_file_path: Optional[DPPath] = None,
    ):
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
        pass
