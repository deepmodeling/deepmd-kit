# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    List,
    Optional,
    Union,
)

import torch

from deepmd.pt.model.descriptor import (
    DescriptorBlock,
)
from deepmd.pt.model.network.network import (
    Identity,
    Linear,
)
from deepmd.utils.path import (
    DPPath,
)


@DescriptorBlock.register("hybrid")
class DescrptBlockHybrid(DescriptorBlock):
    def __init__(
        self,
        list,
        ntypes: int,
        tebd_dim: int = 8,
        tebd_input_mode: str = "concat",
        hybrid_mode: str = "concat",
        **kwargs,
    ):
        """Construct a hybrid descriptor.

        Args:
        - descriptor_list: list of descriptors.
        - descriptor_param: descriptor configs.
        """
        super().__init__()
        supported_descrpt = ["se_atten", "se_uni"]
        descriptor_list = []
        for descriptor_param_item in list:
            descriptor_type_tmp = descriptor_param_item["type"]
            assert (
                descriptor_type_tmp in supported_descrpt
            ), f"Only descriptors in {supported_descrpt} are supported for `hybrid` descriptor!"
            descriptor_param_item["ntypes"] = ntypes
            if descriptor_type_tmp == "se_atten":
                descriptor_param_item["tebd_dim"] = tebd_dim
                descriptor_param_item["tebd_input_mode"] = tebd_input_mode
            descriptor_list.append(DescriptorBlock(**descriptor_param_item))
        self.descriptor_list = torch.nn.ModuleList(descriptor_list)
        self.descriptor_param = list
        self.rcut = [descrpt.rcut for descrpt in self.descriptor_list]
        self.sec = [descrpt.sec for descrpt in self.descriptor_list]
        self.sel = [descrpt.sel for descrpt in self.descriptor_list]
        self.split_sel = [sum(ii) for ii in self.sel]
        self.local_cluster_list = [
            descrpt.local_cluster for descrpt in self.descriptor_list
        ]
        self.local_cluster = True in self.local_cluster_list
        self.hybrid_mode = hybrid_mode
        self.tebd_dim = tebd_dim
        assert self.hybrid_mode in ["concat", "sequential"]
        sequential_transform = []
        if self.hybrid_mode == "sequential":
            for ii in range(len(descriptor_list) - 1):
                if descriptor_list[ii].dim_out == descriptor_list[ii + 1].dim_in:
                    sequential_transform.append(Identity())
                else:
                    sequential_transform.append(
                        Linear(
                            descriptor_list[ii].dim_out,
                            descriptor_list[ii + 1].dim_in,
                            bias=False,
                            init="glorot",
                        )
                    )
            sequential_transform.append(Identity())
        self.sequential_transform = torch.nn.ModuleList(sequential_transform)
        self.ntypes = ntypes

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.rcut

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return [sum(ii) for ii in self.get_sel()]

    def get_sel(self) -> List[int]:
        """Returns the number of selected atoms for each type."""
        return self.sel

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        return self.dim_out

    def get_dim_in(self) -> int:
        """Returns the input dimension."""
        return self.dim_in

    def get_dim_emb(self):
        return self.dim_emb

    def mixed_types(self) -> bool:
        """If true, the discriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the discriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return all(descriptor.mixed_types() for descriptor in self.descriptor_list)

    @property
    def dim_out(self):
        """Returns the output dimension of this descriptor."""
        if self.hybrid_mode == "concat":
            return sum([descrpt.dim_out for descrpt in self.descriptor_list])
        elif self.hybrid_mode == "sequential":
            return self.descriptor_list[-1].dim_out
        else:
            raise RuntimeError

    @property
    def dim_emb_list(self) -> List[int]:
        """Returns the output dimension list of embeddings."""
        return [descrpt.dim_emb for descrpt in self.descriptor_list]

    @property
    def dim_emb(self):
        """Returns the output dimension of embedding."""
        if self.hybrid_mode == "concat":
            return sum(self.dim_emb_list)
        elif self.hybrid_mode == "sequential":
            return self.descriptor_list[-1].dim_emb
        else:
            raise RuntimeError

    def share_params(self, base_class, shared_level, resume=False):
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some seperated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert (
            self.__class__ == base_class.__class__
        ), "Only descriptors of the same type can share params!"
        if shared_level == 0:
            for ii, des in enumerate(self.descriptor_list):
                self.descriptor_list[ii].share_params(
                    base_class.descriptor_list[ii], shared_level, resume=resume
                )
        else:
            raise NotImplementedError

    def compute_input_stats(
        self,
        merged: Union[Callable[[], List[dict]], List[dict]],
        path: Optional[DPPath] = None,
    ):
        """Update mean and stddev for descriptor elements."""
        for ii, descrpt in enumerate(self.descriptor_list):
            # need support for hybrid descriptors
            descrpt.compute_input_stats(merged, path)

    def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_atype_embd: Optional[torch.Tensor] = None,
        mapping: Optional[torch.Tensor] = None,
    ):
        """Calculate decoded embedding for each atom.

        Args:
        - extended_coord: Tell atom coordinates with shape [nframes, natoms[1]*3].
        - nlist: Tell atom types with shape [nframes, natoms[1]].
        - atype: Tell atom count and element count. Its shape is [2+self.ntypes].
        - nlist_type: Tell simulation box with shape [nframes, 9].
        - atype_tebd: Tell simulation box with shape [nframes, 9].
        - nlist_tebd: Tell simulation box with shape [nframes, 9].

        Returns
        -------
        - result: descriptor with shape [nframes, nloc, self.filter_neuron[-1] * self.axis_neuron].
        - ret: environment matrix with shape [nframes, nloc, self.neei, out_size]
        """
        nlist_list = list(torch.split(nlist, self.split_sel, -1))
        nframes, nloc, nnei = nlist.shape
        concat_rot_mat = True
        if self.hybrid_mode == "concat":
            out_descriptor = []
            # out_env_mat = []
            out_rot_mat_list = []
            # out_diff = []
            for ii, descrpt in enumerate(self.descriptor_list):
                descriptor, env_mat, diff, rot_mat, sw = descrpt(
                    nlist_list[ii],
                    extended_coord,
                    extended_atype,
                    extended_atype_embd,
                    mapping,
                )
                if descriptor.shape[0] == nframes * nloc:
                    # [nframes * nloc, 1 + nnei, emb_dim]
                    descriptor = descriptor[:, 0, :].reshape(nframes, nloc, -1)
                out_descriptor.append(descriptor)
                # out_env_mat.append(env_mat)
                # out_diff.append(diff)
                out_rot_mat_list.append(rot_mat)
                if rot_mat is None:
                    concat_rot_mat = False
            out_descriptor = torch.concat(out_descriptor, dim=-1)
            if concat_rot_mat:
                out_rot_mat = torch.concat(out_rot_mat_list, dim=-2)
            else:
                out_rot_mat = None
            return out_descriptor, None, None, out_rot_mat, sw
        elif self.hybrid_mode == "sequential":
            assert extended_atype_embd is not None
            assert mapping is not None
            nframes, nloc, nnei = nlist.shape
            nall = extended_coord.view(nframes, -1).shape[1] // 3
            seq_input_ext = extended_atype_embd
            seq_input = (
                seq_input_ext[:, :nloc, :] if len(self.descriptor_list) == 0 else None
            )
            env_mat, diff, rot_mat, sw = None, None, None, None
            env_mat_list, diff_list = [], []
            for ii, (descrpt, seq_transform) in enumerate(
                zip(self.descriptor_list, self.sequential_transform)
            ):
                seq_output, env_mat, diff, rot_mat, sw = descrpt(
                    nlist_list[ii],
                    extended_coord,
                    extended_atype,
                    seq_input_ext,
                    mapping,
                )
                seq_input = seq_transform(seq_output)
                mapping_ext = (
                    mapping.view(nframes, nall)
                    .unsqueeze(-1)
                    .expand(-1, -1, seq_input.shape[-1])
                )
                seq_input_ext = torch.gather(seq_input, 1, mapping_ext)
                env_mat_list.append(env_mat)
                diff_list.append(diff)
            return seq_input, env_mat_list, diff_list, rot_mat, sw
        else:
            raise RuntimeError
