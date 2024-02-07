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
    Union,
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
    def get_stat_name(cls, config, ntypes):
        """
        Get the name for the statistic file of the descriptor.
        Usually use the combination of descriptor name, rcut, rcut_smth and sel as the statistic file name.
        """
        if cls is not Descriptor:
            raise NotImplementedError("get_stat_name is not implemented!")
        descrpt_type = config["type"]
        return Descriptor.__plugins.plugins[descrpt_type].get_stat_name(config, ntypes)

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

    def get_data_stat_key(self):
        """
        Get the keys for the data statistic of the descriptor.
        Return a list of statistic names needed, such as "sumr", "suma" or "sumn".
        """
        raise NotImplementedError("get_data_stat_key is not implemented!")

    def set_stats(
        self,
        type_map: List[str],
        sampled,
        stat_file_path: Optional[Union[str, List[str]]] = None,
    ):
        """
        Set the statistics parameters for the descriptor.
        Calculate and save the mean and standard deviation of the descriptor to `stat_file_path`
        if `sampled` is not None, otherwise load them from `stat_file_path`.

        Parameters
        ----------
        type_map
            Mapping atom type to the name (str) of the type.
            For example `type_map[1]` gives the name of the type 1.
        sampled
            The sampled data frames from different data systems.
        stat_file_path
            The path to the statistics files.
        """
        # TODO support hybrid descriptor
        descrpt_stat_key = self.get_data_stat_key()
        if sampled is not None:  # compute the statistics results
            tmp_dict = self.compute_input_stats(sampled)
            result_dict = {key: tmp_dict[key] for key in descrpt_stat_key}
            result_dict["type_map"] = type_map
            if stat_file_path is not None:
                self.save_stats(result_dict, stat_file_path)
        else:  # load the statistics results
            assert stat_file_path is not None, "No stat file to load!"
            result_dict = self.load_stats(type_map, stat_file_path)
        self.init_desc_stat(result_dict)

    def save_stats(self, result_dict, stat_file_path: Union[str, List[str]]):
        """
        Save the statistics results to `stat_file_path`.

        Parameters
        ----------
        result_dict
            The dictionary of statistics results.
        stat_file_path
            The path to the statistics file(s).
        """
        if not isinstance(stat_file_path, list):
            log.info(f"Saving stat file to {stat_file_path}")
            np.savez_compressed(stat_file_path, **result_dict)
        else:  # TODO hybrid descriptor not implemented
            raise NotImplementedError(
                "save_stats for hybrid descriptor is not implemented!"
            )

    def load_stats(self, type_map, stat_file_path: Union[str, List[str]]):
        """
        Load the statistics results to `stat_file_path`.

        Parameters
        ----------
        type_map
            Mapping atom type to the name (str) of the type.
            For example `type_map[1]` gives the name of the type 1.
        stat_file_path
            The path to the statistics file(s).

        Returns
        -------
        result_dict
            The dictionary of statistics results.
        """
        descrpt_stat_key = self.get_data_stat_key()
        target_type_map = type_map
        if not isinstance(stat_file_path, list):
            log.info(f"Loading stat file from {stat_file_path}")
            stats = np.load(stat_file_path)
            stat_type_map = list(stats["type_map"])
            missing_type = [i for i in target_type_map if i not in stat_type_map]
            assert not missing_type, (
                f"These type are not in stat file {stat_file_path}: {missing_type}! "
                f"Please change the stat file path!"
            )
            idx_map = [stat_type_map.index(i) for i in target_type_map]
            if stats[descrpt_stat_key[0]].size:  # not empty
                result_dict = {key: stats[key][idx_map] for key in descrpt_stat_key}
            else:
                result_dict = {key: [] for key in descrpt_stat_key}
        else:  # TODO hybrid descriptor not implemented
            raise NotImplementedError(
                "load_stats for hybrid descriptor is not implemented!"
            )
        return result_dict

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
    def init_desc_stat(self, stat_dict):
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
                stat_dict = {
                    "sumr": sumr_base + sumr,
                    "suma": suma_base + suma,
                    "sumn": sumn_base + sumn,
                    "sumr2": sumr2_base + sumr2,
                    "suma2": suma2_base + suma2,
                }
                base_class.init_desc_stat(stat_dict)
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
