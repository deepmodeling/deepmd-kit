# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
)

import numpy as np

from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


@BaseDescriptor.register("hybrid")
class DescrptHybrid(BaseDescriptor, NativeOP):
    """Concate a list of descriptors to form a new descriptor.

    Parameters
    ----------
    list : list
            Build a descriptor from the concatenation of the list of descriptors.
    """

    def __init__(
        self,
        list: list,
    ) -> None:
        super().__init__()
        # warning: list is conflict with built-in list
        descrpt_list = list
        if descrpt_list == [] or descrpt_list is None:
            raise RuntimeError(
                "cannot build descriptor from an empty list of descriptors."
            )
        formatted_descript_list = []
        for ii in descrpt_list:
            if isinstance(ii, BaseDescriptor):
                formatted_descript_list.append(ii)
            elif isinstance(ii, dict):
                formatted_descript_list.append(BaseDescriptor(**ii))
            else:
                raise NotImplementedError
        self.descrpt_list = formatted_descript_list
        self.numb_descrpt = len(self.descrpt_list)
        for ii in range(1, self.numb_descrpt):
            assert (
                self.descrpt_list[ii].get_ntypes() == self.descrpt_list[0].get_ntypes()
            ), f"number of atom types in {ii}th descrptor {self.descrpt_list[0].__class__.__name__} does not match others"
        # if hybrid sel is larger than sub sel, the nlist needs to be cut for each type
        hybrid_sel = self.get_sel()
        self.nlist_cut_idx: List[np.ndarray] = []
        for ii in range(self.numb_descrpt):
            sub_sel = self.descrpt_list[ii].get_sel()
            start_idx = np.cumsum(np.pad(hybrid_sel, (1, 0), "constant"))[:-1]
            end_idx = start_idx + np.array(sub_sel)
            cut_idx = np.concatenate(
                [range(ss, ee) for ss, ee in zip(start_idx, end_idx)]
            )
            self.nlist_cut_idx.append(cut_idx)

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return np.max([descrpt.get_rcut() for descrpt in self.descrpt_list]).item()

    def get_sel(self) -> List[int]:
        """Returns the number of selected atoms for each type."""
        return np.max(
            [descrpt.get_sel() for descrpt in self.descrpt_list], axis=0
        ).tolist()

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.descrpt_list[0].get_ntypes()

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        return np.sum([descrpt.get_dim_out() for descrpt in self.descrpt_list]).item()

    def get_dim_emb(self) -> int:
        """Returns the output dimension."""
        return np.sum([descrpt.get_dim_emb() for descrpt in self.descrpt_list]).item()

    def mixed_types(self):
        """Returns if the descriptor requires a neighbor list that distinguish different
        atomic types or not.
        """
        return all(descrpt.mixed_types() for descrpt in self.descrpt_list)

    def compute_input_stats(self, merged: List[dict], path: Optional[DPPath] = None):
        """Update mean and stddev for descriptor elements."""
        for descrpt in self.descrpt_list:
            descrpt.compute_input_stats(merged, path)

    def call(
        self,
        coord_ext,
        atype_ext,
        nlist,
        mapping: Optional[np.ndarray] = None,
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
        for descrpt, nci in zip(self.descrpt_list, self.nlist_cut_idx):
            # cut the nlist to the correct length
            odescriptor, _, _, _, _ = descrpt(
                coord_ext, atype_ext, nlist[:, :, nci], mapping
            )
            out_descriptor.append(odescriptor)
        out_descriptor = np.concatenate(out_descriptor, axis=-1)
        return out_descriptor, None, None, None, None

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
