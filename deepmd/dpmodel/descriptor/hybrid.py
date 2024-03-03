# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np

from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.utils.nlist import (
    nlist_distinguish_types,
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
    list : list : List[Union[BaseDescriptor, Dict[str, Any]]]
        Build a descriptor from the concatenation of the list of descriptors.
        The descriptor can be either an object or a dictionary.
    """

    def __init__(
        self,
        list: List[Union[BaseDescriptor, Dict[str, Any]]],
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
            )
            self.nlist_cut_idx.append(cut_idx)

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return np.max([descrpt.get_rcut() for descrpt in self.descrpt_list]).item()

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
        return np.sum([descrpt.get_dim_out() for descrpt in self.descrpt_list]).item()

    def get_dim_emb(self) -> int:
        """Returns the output dimension."""
        return np.sum([descrpt.get_dim_emb() for descrpt in self.descrpt_list]).item()

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
        raise NotImplementedError

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
            representation. shape: nf x nloc x ng x 3.
        g2
            The rotationally invariant pair-partical representation.
        h2
            The rotationally equivariant pair-partical representation.
        sw
            The smooth switch function.
        """
        out_descriptor = []
        out_gr = []
        out_g2 = None
        out_h2 = None
        out_sw = None
        if self.sel_no_mixed_types is not None:
            nl_distinguish_types = nlist_distinguish_types(
                nlist,
                atype_ext,
                self.sel_no_mixed_types,
            )
        else:
            nl_distinguish_types = None
        for descrpt, nci in zip(self.descrpt_list, self.nlist_cut_idx):
            # cut the nlist to the correct length
            if self.mixed_types() == descrpt.mixed_types():
                nl = nlist[:, :, nci]
            else:
                # mixed_types is True, but descrpt.mixed_types is False
                assert nl_distinguish_types is not None
                nl = nl_distinguish_types[:, :, nci]
            odescriptor, gr, g2, h2, sw = descrpt(coord_ext, atype_ext, nl, mapping)
            out_descriptor.append(odescriptor)
            if gr is not None:
                out_gr.append(gr)

        out_descriptor = np.concatenate(out_descriptor, axis=-1)
        out_gr = np.concatenate(out_gr, axis=-2) if out_gr else None
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
