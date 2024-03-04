# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np
import torch

from deepmd.pt.model.descriptor import (
    DescriptorBlock,
)
from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt.model.network.network import (
    Identity,
    Linear,
)
from deepmd.pt.utils.nlist import (
    nlist_distinguish_types,
)
from deepmd.pt.utils.utils import (
    to_torch_tensor,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


@BaseDescriptor.register("hybrid")
class DescrptHybrid(BaseDescriptor, torch.nn.Module):
    """Concate a list of descriptors to form a new descriptor.

    Parameters
    ----------
    list : list : List[Union[BaseDescriptor, Dict[str, Any]]]
        Build a descriptor from the concatenation of the list of descriptors.
        The descriptor can be either an object or a dictionary.
    """

    def __init__(
        self,
        list: List[Union[BaseDescriptor, Dict[str, Any]]],
        **kwargs,
    ) -> None:
        super().__init__()
        # warning: list is conflict with built-in list
        descrpt_list = list
        if descrpt_list == [] or descrpt_list is None:
            raise RuntimeError(
                "cannot build descriptor from an empty list of descriptors."
            )
        formatted_descript_list: List[BaseDescriptor] = []
        for ii in descrpt_list:
            if isinstance(ii, BaseDescriptor):
                formatted_descript_list.append(ii)
            elif isinstance(ii, dict):
                formatted_descript_list.append(
                    # pass other arguments (e.g. ntypes) to the descriptor
                    BaseDescriptor(**ii, **kwargs)
                )
            else:
                raise NotImplementedError
        self.descrpt_list = torch.nn.ModuleList(formatted_descript_list)
        self.numb_descrpt = len(self.descrpt_list)
        for ii in range(1, self.numb_descrpt):
            assert (
                self.descrpt_list[ii].get_ntypes() == self.descrpt_list[0].get_ntypes()
            ), f"number of atom types in {ii}th descrptor does not match others"
        # if hybrid sel is larger than sub sel, the nlist needs to be cut for each type
        self.nlist_cut_idx: List[torch.Tensor] = []
        if self.mixed_types() and not all(
            descrpt.mixed_types() for descrpt in self.descrpt_list
        ):
            self.sel_no_mixed_types = np.max(
                [
                    descrpt.get_sel()
                    for descrpt in self.descrpt_list
                    if not descrpt.mixed_types()
                ],
                axis=0,
            ).tolist()
        else:
            self.sel_no_mixed_types = None
        for ii in range(self.numb_descrpt):
            if self.mixed_types() == self.descrpt_list[ii].mixed_types():
                hybrid_sel = self.get_sel()
            else:
                assert self.sel_no_mixed_types is not None
                hybrid_sel = self.sel_no_mixed_types
            sub_sel = self.descrpt_list[ii].get_sel()
            start_idx = np.cumsum(np.pad(hybrid_sel, (1, 0), "constant"))[:-1]
            end_idx = start_idx + np.array(sub_sel)
            cut_idx = np.concatenate(
                [range(ss, ee) for ss, ee in zip(start_idx, end_idx)]
            ).astype(np.int64)
            self.nlist_cut_idx.append(to_torch_tensor(cut_idx))

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        # do not use numpy here - jit is not happy
        return max([descrpt.get_rcut() for descrpt in self.descrpt_list])

    def get_sel(self) -> List[int]:
        """Returns the number of selected atoms for each type."""
        if self.mixed_types():
            return [
                np.max(
                    [descrpt.get_nsel() for descrpt in self.descrpt_list], axis=0
                ).item()
            ]
        else:
            return np.max(
                [descrpt.get_sel() for descrpt in self.descrpt_list], axis=0
            ).tolist()

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.descrpt_list[0].get_ntypes()

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        return sum([descrpt.get_dim_out() for descrpt in self.descrpt_list])

    def get_dim_emb(self) -> int:
        """Returns the output dimension."""
        return sum([descrpt.get_dim_emb() for descrpt in self.descrpt_list])

    def mixed_types(self):
        """Returns if the descriptor requires a neighbor list that distinguish different
        atomic types or not.
        """
        return any(descrpt.mixed_types() for descrpt in self.descrpt_list)

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
            for ii, des in enumerate(self.descrpt_list):
                self.descrpt_list[ii].share_params(
                    base_class.descrpt_list[ii], shared_level, resume=resume
                )
        else:
            raise NotImplementedError

    def compute_input_stats(self, merged: List[dict], path: Optional[DPPath] = None):
        """Update mean and stddev for descriptor elements."""
        for descrpt in self.descrpt_list:
            descrpt.compute_input_stats(merged, path)

    def forward(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
    ):
        """Compute the descriptor.

        Parameters
        ----------
        coord_ext
            The extended coordinates of atoms. shape: nf x (nallx3)
        atype_ext
            The extended aotm types. shape: nf x nall
        nlist
            The neighbor list. shape: nf x nloc x nnei
        mapping
            The index mapping, not required by this descriptor.

        Returns
        -------
        descriptor
            The descriptor. shape: nf x nloc x (ng x axis_neuron)
        gr
            The rotationally equivariant and permutationally invariant single particle
            representation. shape: nf x nloc x ng x 3. This descriptor returns None
        g2
            The rotationally invariant pair-partical representation.
            this descriptor returns None
        h2
            The rotationally equivariant pair-partical representation.
            this descriptor returns None
        sw
            The smooth switch function. this descriptor returns None
        """
        out_descriptor = []
        out_gr = []
        out_g2: Optional[torch.Tensor] = None
        out_h2: Optional[torch.Tensor] = None
        out_sw: Optional[torch.Tensor] = None
        if self.sel_no_mixed_types is not None:
            nl_distinguish_types = nlist_distinguish_types(
                nlist,
                atype_ext,
                self.sel_no_mixed_types,
            )
        else:
            nl_distinguish_types = None
        # make jit happy
        # for descrpt, nci in zip(self.descrpt_list, self.nlist_cut_idx):
        for ii, descrpt in enumerate(self.descrpt_list):
            # cut the nlist to the correct length
            if self.mixed_types() == descrpt.mixed_types():
                nl = nlist[:, :, self.nlist_cut_idx[ii]]
            else:
                # mixed_types is True, but descrpt.mixed_types is False
                assert nl_distinguish_types is not None
                nl = nl_distinguish_types[:, :, self.nlist_cut_idx[ii]]
            odescriptor, gr, g2, h2, sw = descrpt(coord_ext, atype_ext, nl, mapping)
            out_descriptor.append(odescriptor)
            if gr is not None:
                out_gr.append(gr)
        out_descriptor = torch.cat(out_descriptor, dim=-1)
        out_gr = torch.cat(out_gr, dim=-2) if out_gr else None
        return out_descriptor, out_gr, out_g2, out_h2, out_sw

    @classmethod
    def update_sel(cls, global_jdata: dict, local_jdata: dict) -> dict:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        global_jdata : dict
            The global data, containing the training section
        local_jdata : dict
            The local data refer to the current class
        """
        local_jdata_cpy = local_jdata.copy()
        local_jdata_cpy["list"] = [
            BaseDescriptor.update_sel(global_jdata, sub_jdata)
            for sub_jdata in local_jdata["list"]
        ]
        return local_jdata_cpy

    def serialize(self) -> dict:
        return {
            "@class": "Descriptor",
            "type": "hybrid",
            "@version": 1,
            "list": [descrpt.serialize() for descrpt in self.descrpt_list],
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptHybrid":
        data = data.copy()
        class_name = data.pop("@class")
        assert class_name == "Descriptor"
        class_type = data.pop("type")
        assert class_type == "hybrid"
        check_version_compatibility(data.pop("@version"), 1, 1)
        obj = cls(
            list=[BaseDescriptor.deserialize(ii) for ii in data["list"]],
        )
        return obj


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
        """
        Compute the input statistics (e.g. mean and stddev) for the descriptors from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], List[dict]], List[dict]]
            - List[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], List[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        path : Optional[DPPath]
            The path to the stat file.

        """
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
