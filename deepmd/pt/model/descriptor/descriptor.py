# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Type,
)

import torch

from deepmd.common import (
    j_get_type,
)
from deepmd.pt.model.network.network import (
    TypeEmbedNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env_mat_stat import (
    EnvMatStatSeA,
)
from deepmd.pt.utils.plugin import (
    Plugin,
)
from deepmd.utils.env_mat_stat import (
    StatItem,
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
            cls = cls.get_class_by_type(j_get_type(kwargs, cls.__name__))
        return super().__new__(cls)

    @classmethod
    def get_class_by_type(cls, descrpt_type: str) -> Type["Descriptor"]:
        if descrpt_type in Descriptor.__plugins.plugins:
            return Descriptor.__plugins.plugins[descrpt_type]
        else:
            raise RuntimeError("Unknown descriptor type: " + descrpt_type)

    @classmethod
    def deserialize(cls, data: dict) -> "Descriptor":
        """Deserialize the model.

        There is no suffix in a native DP model, but it is important
        for the TF backend.

        Parameters
        ----------
        data : dict
            The serialized data

        Returns
        -------
        Descriptor
            The deserialized descriptor
        """
        if cls is Descriptor:
            return Descriptor.get_class_by_type(data["type"]).deserialize(data)
        raise NotImplementedError("Not implemented in class %s" % cls.__name__)


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

    def compute_input_stats(self, merged: List[dict], path: Optional[DPPath] = None):
        """Update mean and stddev for DescriptorBlock elements."""
        raise NotImplementedError

    def get_stats(self) -> Dict[str, StatItem]:
        """Get the statistics of the descriptor."""
        raise NotImplementedError

    def share_params(self, base_class, shared_level, resume=False):
        assert (
            self.__class__ == base_class.__class__
        ), "Only descriptors of the same type can share params!"
        if shared_level == 0:
            # link buffers
            if hasattr(self, "mean") and not resume:
                # in case of change params during resume
                base_env = EnvMatStatSeA(base_class)
                base_env.stats = base_class.stats
                for kk in base_class.get_stats():
                    base_env.stats[kk] += self.get_stats()[kk]
                mean, stddev = base_env()
                if not base_class.set_davg_zero:
                    base_class.mean.copy_(torch.tensor(mean, device=env.DEVICE))
                base_class.stddev.copy_(torch.tensor(stddev, device=env.DEVICE))
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


def make_default_type_embedding(
    ntypes,
):
    aux = {}
    aux["tebd_dim"] = 8
    return TypeEmbedNet(ntypes, aux["tebd_dim"]), aux
