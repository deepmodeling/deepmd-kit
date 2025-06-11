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

import paddle

from deepmd.pd.model.network.network import (
    TypeEmbedNet,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.env_mat_stat import (
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


class DescriptorBlock(paddle.nn.Layer, ABC, make_plugin_registry("DescriptorBlock")):
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
    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
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

    @abstractmethod
    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
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
                Each element, `merged[i]`, is a data dictionary containing `keys`: `paddle.Tensor`
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

    def share_params(self, base_class, shared_level, resume=False) -> None:
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert self.__class__ == base_class.__class__, (
            "Only descriptors of the same type can share params!"
        )
        if shared_level == 0:
            # link buffers
            if hasattr(self, "mean"):
                if not resume and (
                    not getattr(self, "set_stddev_constant", False)
                    or not getattr(self, "set_davg_zero", False)
                ):
                    # in case of change params during resume
                    base_env = EnvMatStatSe(base_class)
                    base_env.stats = base_class.stats
                    for kk in base_class.get_stats():
                        base_env.stats[kk] += self.get_stats()[kk]
                    mean, stddev = base_env()
                    if not base_class.set_davg_zero:
                        paddle.assign(
                            paddle.to_tensor(mean, dtype=base_class.mean.dtype).to(
                                device=env.DEVICE
                            ),
                            base_class.mean,
                        )
                    paddle.assign(
                        paddle.to_tensor(stddev, dtype=base_class.stddev.dtype).to(
                            device=env.DEVICE
                        ),
                        base_class.stddev,
                    )
                # must share, even if not do stat
                self.mean = base_class.mean
                self.stddev = base_class.stddev
            # self.set_state_dict(base_class.state_dict()) # this does not work, because it only inits the model
            # the following will successfully link all the params except buffers
            for item in self._sub_layers:
                self._sub_layers[item] = base_class._sub_layers[item]
        else:
            raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        nlist: paddle.Tensor,
        extended_coord: paddle.Tensor,
        extended_atype: paddle.Tensor,
        extended_atype_embd: Optional[paddle.Tensor] = None,
        mapping: Optional[paddle.Tensor] = None,
        type_embedding: Optional[paddle.Tensor] = None,
    ):
        """Calculate DescriptorBlock."""
        pass

    @abstractmethod
    def has_message_passing(self) -> bool:
        """Returns whether the descriptor block has message passing."""

    @abstractmethod
    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor block needs sorted nlist when using `forward_lower`."""


def make_default_type_embedding(
    ntypes,
):
    aux = {}
    aux["tebd_dim"] = 8
    return TypeEmbedNet(ntypes, aux["tebd_dim"]), aux


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
        extend_davg = paddle.zeros(extend_shape, dtype=des["davg"].dtype).to(
            device=des["davg"].place
        )
        extend_dstd = paddle.ones(extend_shape, dtype=des["dstd"].dtype).to(
            device=des["dstd"].place
        )
    des["davg"] = paddle.concat([des["davg"], extend_davg], axis=0)
    des["dstd"] = paddle.concat([des["dstd"], extend_dstd], axis=0)
