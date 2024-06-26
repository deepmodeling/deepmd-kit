# SPDX-License-Identifier: LGPL-3.0-or-later
import math
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch

from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt.utils.nlist import (
    nlist_distinguish_types,
)
from deepmd.pt.utils.utils import (
    to_torch_tensor,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
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

    nlist_cut_idx: List[torch.Tensor]

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

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        # may not be a good idea...
        # Note: Using the minimum rcut_smth might not be appropriate in all scenarios. Consider using a different approach or provide detailed documentation on why the minimum value is chosen.
        return min([descrpt.get_rcut_smth() for descrpt in self.descrpt_list])

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

    def get_type_map(self) -> List[str]:
        """Get the name to each type of atoms."""
        return self.descrpt_list[0].get_type_map()

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

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor has message passing."""
        return any(descrpt.has_message_passing() for descrpt in self.descrpt_list)

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix. All descriptors should be the same."""
        all_protection = [descrpt.get_env_protection() for descrpt in self.descrpt_list]
        same_as_0 = [math.isclose(ii, all_protection[0]) for ii in all_protection]
        if not all(same_as_0):
            raise ValueError(
                "Hybrid descriptor requires the same environment matrix protection for all descriptors. Found differing values."
            )
        return all_protection[0]

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

    def change_type_map(
        self, type_map: List[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        for ii, descrpt in enumerate(self.descrpt_list):
            descrpt.change_type_map(
                type_map=type_map,
                model_with_new_type_stat=model_with_new_type_stat.descrpt_list[ii]
                if model_with_new_type_stat is not None
                else None,
            )

    def compute_input_stats(self, merged: List[dict], path: Optional[DPPath] = None):
        """Update mean and stddev for descriptor elements."""
        for descrpt in self.descrpt_list:
            descrpt.compute_input_stats(merged, path)

    def set_stat_mean_and_stddev(
        self,
        mean: List[Union[torch.Tensor, List[torch.Tensor]]],
        stddev: List[Union[torch.Tensor, List[torch.Tensor]]],
    ) -> None:
        """Update mean and stddev for descriptor."""
        for ii, descrpt in enumerate(self.descrpt_list):
            descrpt.set_stat_mean_and_stddev(mean[ii], stddev[ii])

    def get_stat_mean_and_stddev(
        self,
    ) -> Tuple[
        List[Union[torch.Tensor, List[torch.Tensor]]],
        List[Union[torch.Tensor, List[torch.Tensor]]],
    ]:
        """Get mean and stddev for descriptor."""
        mean_list = []
        stddev_list = []
        for ii, descrpt in enumerate(self.descrpt_list):
            mean_item, stddev_item = descrpt.get_stat_mean_and_stddev()
            mean_list.append(mean_item)
            stddev_list.append(stddev_item)
        return mean_list, stddev_list

    def forward(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        comm_dict: Optional[Dict[str, torch.Tensor]] = None,
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
        comm_dict
            The data needed for communication for parallel inference.

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
                nl = nlist[:, :, self.nlist_cut_idx[ii].to(atype_ext.device)]
            else:
                # mixed_types is True, but descrpt.mixed_types is False
                assert nl_distinguish_types is not None
                nl = nl_distinguish_types[
                    :, :, self.nlist_cut_idx[ii].to(atype_ext.device)
                ]
            odescriptor, gr, g2, h2, sw = descrpt(coord_ext, atype_ext, nl, mapping)
            out_descriptor.append(odescriptor)
            if gr is not None:
                out_gr.append(gr)
        out_descriptor = torch.cat(out_descriptor, dim=-1)
        out_gr = torch.cat(out_gr, dim=-2) if out_gr else None
        return out_descriptor, out_gr, out_g2, out_h2, out_sw

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: Optional[List[str]],
        local_jdata: dict,
    ) -> Tuple[dict, Optional[float]]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statictics
        type_map : list[str], optional
            The name of each type of atoms
        local_jdata : dict
            The local data refer to the current class

        Returns
        -------
        dict
            The updated local data
        float
            The minimum distance between two atoms
        """
        local_jdata_cpy = local_jdata.copy()
        new_list = []
        min_nbor_dist = None
        for sub_jdata in local_jdata["list"]:
            new_sub_jdata, min_nbor_dist_ = BaseDescriptor.update_sel(
                train_data, type_map, sub_jdata
            )
            if min_nbor_dist_ is not None:
                min_nbor_dist = min_nbor_dist_
            new_list.append(new_sub_jdata)
        local_jdata_cpy["list"] = new_list
        return local_jdata_cpy, min_nbor_dist

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
