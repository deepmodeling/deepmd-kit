# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Callable,
    NoReturn,
    Optional,
    Union,
)

import numpy as np

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


class DescriptorBlock(ABC, make_plugin_registry("DescriptorBlock")):
    """The building block of descriptor.
    Given the input descriptor, provide with the atomic coordinates,
    atomic types and neighbor list, calculate the new descriptor.
    """

    local_cluster = False

    def __new__(cls, *args, **kwargs):
        if cls is DescriptorBlock:
            try:
                descrpt_type = kwargs["type"]
            except KeyError as e:
                raise KeyError(
                    "the type of DescriptorBlock should be set by `type`"
                ) from e
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
    def get_sel(self) -> list[int]:
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
        """Returns the input dimension."""
        pass

    @abstractmethod
    def get_dim_emb(self) -> int:
        """Returns the embedding dimension."""
        pass

    def compute_input_stats(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        path: Optional[DPPath] = None,
    ) -> NoReturn:
        """
        Compute the input statistics (e.g. mean and stddev) for the descriptors from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        path : Optional[DPPath]
            The path to the stat file.

        """
        raise NotImplementedError

    def get_stats(self) -> dict[str, StatItem]:
        """Get the statistics of the descriptor."""
        raise NotImplementedError

    def share_params(self, base_class, shared_level, resume=False) -> NoReturn:
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        raise NotImplementedError

    @abstractmethod
    def call(
        self,
        nlist: np.ndarray,
        extended_coord: np.ndarray,
        extended_atype: np.ndarray,
        extended_atype_embd: Optional[np.ndarray] = None,
        mapping: Optional[np.ndarray] = None,
        type_embedding: Optional[np.ndarray] = None,
    ):
        """Calculate DescriptorBlock."""
        pass

    @abstractmethod
    def has_message_passing(self) -> bool:
        """Returns whether the descriptor block has message passing."""

    @abstractmethod
    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor block needs sorted nlist when using `forward_lower`."""


def extend_descrpt_stat(des, type_map, des_with_stat=None) -> None:
    r"""
    Extend the statistics of a descriptor block with types from newly provided `type_map`.

    After extending, the type related dimension of the extended statistics will have a length of
    `len(old_type_map) + len(type_map)`, where `old_type_map` represents the type map in `des`.
    The `get_index_between_two_maps()` function can then be used to correctly select statistics for types
    from `old_type_map` or `type_map`.
    Positive indices from 0 to `len(old_type_map) - 1` will select old statistics of types in `old_type_map`,
    while negative indices from `-len(type_map)` to -1 will select new statistics of types in `type_map`.

    Parameters
    ----------
    des : DescriptorBlock
        The descriptor block to be extended.
    type_map : list[str]
        The name of each type of atoms to be extended.
    des_with_stat : DescriptorBlock, Optional
        The descriptor block has additional statistics of types from newly provided `type_map`.
        If None, the default statistics will be used.
        Otherwise, the statistics provided in this DescriptorBlock will be used.

    """
    if des_with_stat is not None:
        extend_davg = des_with_stat["davg"]
        extend_dstd = des_with_stat["dstd"]
    else:
        extend_shape = [len(type_map), *list(des["davg"].shape[1:])]
        extend_davg = np.zeros(extend_shape, dtype=des["davg"].dtype)
        extend_dstd = np.ones(extend_shape, dtype=des["dstd"].dtype)
    des["davg"] = np.concatenate([des["davg"], extend_davg], axis=0)
    des["dstd"] = np.concatenate([des["dstd"], extend_dstd], axis=0)
