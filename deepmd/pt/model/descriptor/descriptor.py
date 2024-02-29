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
    Union,
)

import torch

from deepmd.pt.model.network.network import (
    TypeEmbedNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.utils.env_mat_stat import (
    StatItem,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.plugin import (
    make_plugin_registry,
)

log = logging.getLogger(__name__)


class DescriptorBlock(torch.nn.Module, ABC, make_plugin_registry("DescriptorBlock")):
    """The building block of descriptor.
    Given the input descriptor, provide with the atomic coordinates,
    atomic types and neighbor list, calculate the new descriptor.
    """

    local_cluster = False

    def __new__(cls, *args, **kwargs):
        if cls is DescriptorBlock:
            try:
                descrpt_type = kwargs["type"]
            except KeyError:
                raise KeyError("the type of DescriptorBlock should be set by `type`")
            cls = cls.get_class_by_type(descrpt_type)
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
        raise NotImplementedError

    def get_stats(self) -> Dict[str, StatItem]:
        """Get the statistics of the descriptor."""
        raise NotImplementedError

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
            # link buffers
            if hasattr(self, "mean") and not resume:
                # in case of change params during resume
                base_env = EnvMatStatSe(base_class)
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
