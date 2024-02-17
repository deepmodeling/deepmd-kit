# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Callable,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np
import torch

from deepmd.pt.model.network.network import (
    TypeEmbedNet,
)
from deepmd.pt.utils.plugin import (
    Plugin,
)
from deepmd.pt.utils.utils import (
    to_torch_tensor,
)
from deepmd.utils.path import (
    DPPath,
)

from .base_descriptor import (
    BaseDescriptor,
)

log = logging.getLogger(__name__)


class Descriptor(torch.nn.Module, BaseDescriptor):
    """The descriptor.
    Given the atomic coordinates, atomic types and neighbor list,
    calculate the descriptor.
    """

    __plugins = Plugin()
    local_cluster = False

    @staticmethod
    def register(key: str) -> Callable:
        """Register a descriptor plugin.

        Parameters
        ----------
        key : str
            the key of a descriptor

        Returns
        -------
        Descriptor
            the registered descriptor

        Examples
        --------
        >>> @Descriptor.register("some_descrpt")
            class SomeDescript(Descriptor):
                pass
        """
        return Descriptor.__plugins.register(key)

    @classmethod
    def get_data_process_key(cls, config):
        """
        Get the keys for the data preprocess.
        Usually need the information of rcut and sel.
        TODO Need to be deprecated when the dataloader has been cleaned up.
        """
        if cls is not Descriptor:
            raise NotImplementedError("get_data_process_key is not implemented!")
        descrpt_type = config["type"]
        return Descriptor.__plugins.plugins[descrpt_type].get_data_process_key(config)

    @property
    def data_stat_key(self):
        """
        Get the keys for the data statistic of the descriptor.
        Return a list of statistic names needed, such as "sumr", "suma" or "sumn".
        """
        raise NotImplementedError("data_stat_key is not implemented!")

    def __new__(cls, *args, **kwargs):
        if cls is Descriptor:
            try:
                descrpt_type = kwargs["type"]
            except KeyError:
                raise KeyError("the type of descriptor should be set by `type`")
            if descrpt_type in Descriptor.__plugins.plugins:
                cls = Descriptor.__plugins.plugins[descrpt_type]
            else:
                raise RuntimeError("Unknown descriptor type: " + descrpt_type)
        return super().__new__(cls)


class DescriptorBlock(torch.nn.Module, ABC):
    """The building block of descriptor.
    Given the input descriptor, provide with the atomic coordinates,
    atomic types and neighbor list, calculate the new descriptor.
    """

    __plugins = Plugin()
    local_cluster = False

    def __init__(
        self,
        ntypes: int,
        exclude_types: List[Tuple[int, int]] = [],
    ):
        super().__init__()
        _exclude_types: Set[Tuple[int, int]] = set()
        for tt in exclude_types:
            assert len(tt) == 2
            _exclude_types.add((tt[0], tt[1]))
            _exclude_types.add((tt[1], tt[0]))
        # ntypes + 1 for nlist masks
        self.type_mask = np.array(
            [
                [
                    1 if (tt_i, tt_j) not in _exclude_types else 0
                    for tt_i in range(ntypes + 1)
                ]
                for tt_j in range(ntypes + 1)
            ],
            dtype=np.int32,
        )
        # (ntypes+1 x ntypes+1)
        self.type_mask = to_torch_tensor(self.type_mask).view([-1])
        self.no_exclusion = len(_exclude_types) == 0

    @staticmethod
    def register(key: str) -> Callable:
        """Register a DescriptorBlock plugin.

        Parameters
        ----------
        key : str
            the key of a DescriptorBlock

        Returns
        -------
        DescriptorBlock
            the registered DescriptorBlock

        Examples
        --------
        >>> @DescriptorBlock.register("some_descrpt")
            class SomeDescript(DescriptorBlock):
                pass
        """
        return DescriptorBlock.__plugins.register(key)

    def __new__(cls, *args, **kwargs):
        if cls is DescriptorBlock:
            try:
                descrpt_type = kwargs["type"]
            except KeyError:
                raise KeyError("the type of DescriptorBlock should be set by `type`")
            if descrpt_type in DescriptorBlock.__plugins.plugins:
                cls = DescriptorBlock.__plugins.plugins[descrpt_type]
            else:
                raise RuntimeError("Unknown DescriptorBlock type: " + descrpt_type)
        return super().__new__(cls)

    @abstractmethod
    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        pass

    @abstractmethod
    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        pass

    @abstractmethod
    def get_sel(self) -> List[int]:
        """Returns the number of selected atoms for each type."""
        pass

    @abstractmethod
    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        pass

    @abstractmethod
    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        pass

    @abstractmethod
    def get_dim_in(self) -> int:
        """Returns the output dimension."""
        pass

    @abstractmethod
    def get_dim_emb(self) -> int:
        """Returns the embedding dimension."""
        pass

    def compute_input_stats(self, merged: List[dict], path: Optional[DPPath] = None):
        """Update mean and stddev for DescriptorBlock elements."""
        raise NotImplementedError

    def share_params(self, base_class, shared_level, resume=False):
        assert (
            self.__class__ == base_class.__class__
        ), "Only descriptors of the same type can share params!"
        if shared_level == 0:
            # link buffers
            if hasattr(self, "mean") and not resume:
                # in case of change params during resume
                sumr_base, suma_base, sumn_base, sumr2_base, suma2_base = (
                    base_class.sumr,
                    base_class.suma,
                    base_class.sumn,
                    base_class.sumr2,
                    base_class.suma2,
                )
                sumr, suma, sumn, sumr2, suma2 = (
                    self.sumr,
                    self.suma,
                    self.sumn,
                    self.sumr2,
                    self.suma2,
                )
                stat_dict = {
                    "sumr": sumr_base + sumr,
                    "suma": suma_base + suma,
                    "sumn": sumn_base + sumn,
                    "sumr2": sumr2_base + sumr2,
                    "suma2": suma2_base + suma2,
                }
                self.mean = base_class.mean
                self.stddev = base_class.stddev
            # self.load_state_dict(base_class.state_dict()) # this does not work, because it only inits the model
            # the following will successfully link all the params except buffers
            for item in self._modules:
                self._modules[item] = base_class._modules[item]
        else:
            raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_atype_embd: Optional[torch.Tensor] = None,
        mapping: Optional[torch.Tensor] = None,
    ):
        """Calculate DescriptorBlock."""
        pass

    # may have a better place for this method...
    def build_type_exclude_mask(
        self,
        nlist: torch.Tensor,
        atype_ext: torch.Tensor,
    ) -> torch.Tensor:
        """Compute type exclusion mask.

        Parameters
        ----------
        nlist
            The neighbor list. shape: nf x nloc x nnei
        atype_ext
            The extended aotm types. shape: nf x nall

        Returns
        -------
        mask
            The type exclusion mask of shape: nf x nloc x nnei.
            Element [ff,ii,jj] being 0 if type(ii), type(nlist[ff,ii,jj]) is excluded,
            otherwise being 1.

        """
        if self.no_exclusion:
            # safely return 1 if nothing is excluded.
            return torch.ones_like(nlist, dtype=torch.int32, device=nlist.device)
        nf, nloc, nnei = nlist.shape
        nall = atype_ext.shape[1]
        # add virtual atom of type ntypes. nf x nall+1
        ae = torch.cat(
            [
                atype_ext,
                self.get_ntypes()
                * torch.ones([nf, 1], dtype=atype_ext.dtype, device=atype_ext.device),
            ],
            dim=-1,
        )
        type_i = atype_ext[:, :nloc].view(nf, nloc) * (self.get_ntypes() + 1)
        # nf x nloc x nnei
        index = torch.where(nlist == -1, nall, nlist).view(nf, nloc * nnei)
        type_j = torch.gather(ae, 1, index).view(nf, nloc, nnei)
        type_ij = type_i[:, :, None] + type_j
        # nf x (nloc x nnei)
        type_ij = type_ij.view(nf, nloc * nnei)
        mask = self.type_mask[type_ij].view(nf, nloc, nnei)
        return mask


def compute_std(sumv2, sumv, sumn, rcut_r):
    """Compute standard deviation."""
    if sumn == 0:
        return 1.0 / rcut_r
    val = np.sqrt(sumv2 / sumn - np.multiply(sumv / sumn, sumv / sumn))
    if np.abs(val) < 1e-2:
        val = 1e-2
    return val


def make_default_type_embedding(
    ntypes,
):
    aux = {}
    aux["tebd_dim"] = 8
    return TypeEmbedNet(ntypes, aux["tebd_dim"]), aux
