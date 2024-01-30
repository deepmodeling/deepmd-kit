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
        descriptor, rot_mat, g2, h2, sw = self.descriptor(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
        )
        assert descriptor is not None
        # energy, force
        fit_ret = self.fitting_net(descriptor, atype, atype_tebd=None, rot_mat=rot_mat)
        return fit_ret
