# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Callable,
    List,
    Optional,
)

import numpy as np
import torch

from deepmd.pt.model.network.network import (
    TypeEmbedNet,
)
from deepmd.pt.utils.plugin import (
    Plugin,
)

from .base_descriptor import (
    BaseDescriptor,
)


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
    def get_stat_name(cls, config):
        descrpt_type = config["type"]
        return Descriptor.__plugins.plugins[descrpt_type].get_stat_name(config)

    @classmethod
    def get_data_process_key(cls, config):
        descrpt_type = config["type"]
        return Descriptor.__plugins.plugins[descrpt_type].get_data_process_key(config)

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

    @abstractmethod
    def compute_input_stats(self, merged):
        """Update mean and stddev for DescriptorBlock elements."""
        pass

    @abstractmethod
    def init_desc_stat(self, sumr, suma, sumn, sumr2, suma2):
        """Initialize the model bias by the statistics."""
        pass

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
                base_class.init_desc_stat(
                    sumr_base + sumr,
                    suma_base + suma,
                    sumn_base + sumn,
                    sumr2_base + sumr2,
                    suma2_base + suma2,
                )
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
        raise NotImplementedError


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
