# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
)

import torch

from deepmd.model_format import (
    FittingOutputDef,
)
from deepmd.pt.model.descriptor.descriptor import (
    Descriptor,
)
from deepmd.pt.model.task import (
    DenoiseNet,
    Fitting,
)

from .atomic_model import (
    AtomicModel,
)
from .model import (
    BaseModel,
)


class DPAtomicModel(BaseModel, AtomicModel):
    """Model give atomic prediction of some physical property.

    Parameters
    ----------
    descriptor
            Descriptor
    fitting_net
            Fitting net
    type_map
            Mapping atom type to the name (str) of the type.
            For example `type_map[1]` gives the name of the type 1.
    type_embedding
            Type embedding net
    resuming
            Whether to resume/fine-tune from checkpoint or not.
    stat_file_dir
            The directory to the state files.
    stat_file_path
            The path to the state files.
    sampled
            Sampled frames to compute the statistics.
    """

    def __init__(
        self,
        descriptor: dict,
        fitting_net: dict,
        type_map: Optional[List[str]],
        type_embedding: Optional[dict] = None,
        resuming: bool = False,
        stat_file_dir=None,
        stat_file_path=None,
        sampled=None,
        **kwargs,
    ):
        super().__init__()
        # Descriptor + Type Embedding Net (Optional)
        ntypes = len(type_map)
        self.type_map = type_map
        self.ntypes = ntypes
        descriptor["ntypes"] = ntypes
        self.combination = descriptor.get("combination", False)
        if self.combination:
            self.prefactor = descriptor.get("prefactor", [0.5, 0.5])
        self.descriptor_type = descriptor["type"]

        self.type_split = True
        if self.descriptor_type not in ["se_e2_a"]:
            self.type_split = False

        self.descriptor = Descriptor(**descriptor)
        self.rcut = self.descriptor.get_rcut()
        self.sel = self.descriptor.get_sel()
        self.split_nlist = False

        # Statistics
        self.compute_or_load_stat(
            fitting_net,
            ntypes,
            resuming=resuming,
            type_map=type_map,
            stat_file_dir=stat_file_dir,
            stat_file_path=stat_file_path,
            sampled=sampled,
        )

        # Fitting
        if fitting_net:
            fitting_net["type"] = fitting_net.get("type", "ener")
            if self.descriptor_type not in ["se_e2_a"]:
                fitting_net["ntypes"] = 1
            else:
                fitting_net["ntypes"] = self.descriptor.get_ntype()
                fitting_net["use_tebd"] = False
            fitting_net["embedding_width"] = self.descriptor.dim_out

            self.grad_force = "direct" not in fitting_net["type"]
            if not self.grad_force:
                fitting_net["out_dim"] = self.descriptor.dim_emb
                if "ener" in fitting_net["type"]:
                    fitting_net["return_energy"] = True
            self.fitting_net = Fitting(**fitting_net)
        else:
            self.fitting_net = None
            self.grad_force = False
            if not self.split_nlist:
                self.coord_denoise_net = DenoiseNet(
                    self.descriptor.dim_out, self.ntypes - 1, self.descriptor.dim_emb
                )
            elif self.combination:
                self.coord_denoise_net = DenoiseNet(
                    self.descriptor.dim_out,
                    self.ntypes - 1,
                    self.descriptor.dim_emb_list,
                    self.prefactor,
                )
            else:
                self.coord_denoise_net = DenoiseNet(
                    self.descriptor.dim_out, self.ntypes - 1, self.descriptor.dim_emb
                )

    def get_fitting_net(self) -> Fitting:
        """Get the fitting net."""
        return (
            self.fitting_net if self.fitting_net is not None else self.coord_denoise_net
        )

    def get_fitting_output_def(self) -> FittingOutputDef:
        """Get the output def of the fitting net."""
        return (
            self.fitting_net.output_def()
            if self.fitting_net is not None
            else self.coord_denoise_net.output_def()
        )

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.rcut

    def get_sel(self) -> List[int]:
        """Get the neighbor selection."""
        return self.sel

    def distinguish_types(self) -> bool:
        """If distinguish different types by sorting."""
        return self.type_split

    def forward_atomic(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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

        Returns
        -------
        result_dict
            the result dict, defined by the fitting net output def.

        """
        nframes, nloc, nnei = nlist.shape
        atype = extended_atype[:, :nloc]
        if self.do_grad():
            extended_coord.requires_grad_(True)
        descriptor, env_mat, diff, rot_mat, sw = self.descriptor(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
        )
        assert descriptor is not None
        # energy, force
        if self.fitting_net is not None:
            fit_ret = self.fitting_net(
                descriptor, atype, atype_tebd=None, rot_mat=rot_mat
            )
        # denoise
        else:
            nlist_list = [nlist]
            if not self.split_nlist:
                nnei_mask = nlist != -1
            elif self.combination:
                nnei_mask = []
                for item in nlist_list:
                    nnei_mask_item = item != -1
                    nnei_mask.append(nnei_mask_item)
            else:
                env_mat = env_mat[-1]
                diff = diff[-1]
                nnei_mask = nlist_list[-1] != -1
            fit_ret = self.coord_denoise_net(env_mat, diff, nnei_mask, descriptor, sw)
        return fit_ret
