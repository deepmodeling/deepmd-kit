# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
from typing import (
    Callable,
    ClassVar,
    Optional,
    Union,
)

import numpy as np
import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.descriptor import (
    DescriptorBlock,
    prod_env_mat,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
)
from deepmd.pt.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.env_mat_stat import (
    StatItem,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

try:
    from typing import (
        Final,
    )
except ImportError:
    from torch.jit import Final

from deepmd.dpmodel.utils import EnvMat as DPEnvMat
from deepmd.pt.model.network.mlp import (
    EmbeddingNet,
    NetworkCollection,
)
from deepmd.pt.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.pt.utils.tabulate import (
    DPTabulate,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
)

from .base_descriptor import (
    BaseDescriptor,
)

if not hasattr(torch.ops.deepmd, "tabulate_fusion_se_t"):

    def tabulate_fusion_se_t(
        argument0: torch.Tensor,
        argument1: torch.Tensor,
        argument2: torch.Tensor,
        argument3: torch.Tensor,
        argument4: int,
    ) -> list[torch.Tensor]:
        raise NotImplementedError(
            "tabulate_fusion_se_t is not available since customized PyTorch OP library is not built when freezing the model. "
            "See documentation for model compression for details."
        )

    # Note: this hack cannot actually save a model that can be runned using LAMMPS.
    torch.ops.deepmd.tabulate_fusion_se_t = tabulate_fusion_se_t


@BaseDescriptor.register("se_e3")
@BaseDescriptor.register("se_at")
@BaseDescriptor.register("se_a_3be")
class DescrptSeT(BaseDescriptor, torch.nn.Module):
    r"""DeepPot-SE constructed from all information (both angular and radial) of atomic
    configurations.

    The embedding takes angles between two neighboring atoms as input.

    Parameters
    ----------
    rcut : float
            The cut-off radius
    rcut_smth : float
            From where the environment matrix should be smoothed
    sel : list[int]
            sel[i] specifies the maxmum number of type i atoms in the cut-off radius
    neuron : list[int]
            Number of neurons in each hidden layers of the embedding net
    resnet_dt : bool
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
    set_davg_zero : bool
            Set the shift of embedding net input to zero.
    activation_function : str
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    env_protection : float
            Protection parameter to prevent division by zero errors during environment matrix calculations.
    exclude_types : list[list[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    precision : str
            The precision of the embedding net parameters. Supported options are |PRECISION|
    trainable : bool
            If the weights of embedding net are trainable.
    seed : int, Optional
            Random seed for initializing the network parameters.
    type_map: list[str], Optional
            A list of strings. Give the name to each type of atoms.
    """

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: list[int],
        neuron: list[int] = [24, 48, 96],
        resnet_dt: bool = False,
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        env_protection: float = 0.0,
        exclude_types: list[tuple[int, int]] = [],
        precision: str = "float64",
        trainable: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
        type_map: Optional[list[str]] = None,
        ntypes: Optional[int] = None,  # to be compat with input
        # not implemented
        spin=None,
    ) -> None:
        del ntypes
        if spin is not None:
            raise NotImplementedError("old implementation of spin is not supported.")
        super().__init__()
        self.type_map = type_map
        self.compress = False
        self.prec = PRECISION_DICT[precision]
        self.seat = DescrptBlockSeT(
            rcut,
            rcut_smth,
            sel,
            neuron=neuron,
            resnet_dt=resnet_dt,
            set_davg_zero=set_davg_zero,
            activation_function=activation_function,
            env_protection=env_protection,
            exclude_types=exclude_types,
            precision=precision,
            trainable=trainable,
            seed=seed,
        )

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.seat.get_rcut()

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.seat.get_rcut_smth()

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return self.seat.get_nsel()

    def get_sel(self) -> list[int]:
        """Returns the number of selected atoms for each type."""
        return self.seat.get_sel()

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.seat.get_ntypes()

    def get_type_map(self) -> list[str]:
        """Get the name to each type of atoms."""
        return self.type_map

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        return self.seat.get_dim_out()

    def get_dim_emb(self) -> int:
        """Returns the output dimension."""
        return self.seat.get_dim_emb()

    def mixed_types(self):
        """Returns if the descriptor requires a neighbor list that distinguish different
        atomic types or not.
        """
        return self.seat.mixed_types()

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor has message passing."""
        return self.seat.has_message_passing()

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor needs sorted nlist when using `forward_lower`."""
        return self.seat.need_sorted_nlist_for_lower()

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.seat.get_env_protection()

    def share_params(self, base_class, shared_level, resume=False) -> None:
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert self.__class__ == base_class.__class__, (
            "Only descriptors of the same type can share params!"
        )
        # For SeT descriptors, the user-defined share-level
        # shared_level: 0
        # share all parameters in sea
        if shared_level == 0:
            self.seat.share_params(base_class.seat, 0, resume=resume)
        # Other shared levels
        else:
            raise NotImplementedError

    @property
    def dim_out(self):
        """Returns the output dimension of this descriptor."""
        return self.seat.dim_out

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        raise NotImplementedError(
            "Descriptor se_e3 does not support changing for type related params!"
            "This feature is currently not implemented because it would require additional work to support the non-mixed-types case. "
            "We may consider adding this support in the future if there is a clear demand for it."
        )

    def compute_input_stats(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        path: Optional[DPPath] = None,
    ):
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
        return self.seat.compute_input_stats(merged, path)

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Receive the statisitcs (distance, max_nbor_size and env_mat_range) of the training data.

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between atoms
        table_extrapolate
            The scale of model extrapolation
        table_stride_1
            The uniform stride of the first table
        table_stride_2
            The uniform stride of the second table
        check_frequency
            The overflow check frequency
        """
        if self.compress:
            raise ValueError("Compression is already enabled.")
        data = self.serialize()
        self.table = DPTabulate(
            self,
            data["neuron"],
            exclude_types=data["exclude_types"],
            activation_fn=ActivationFn(data["activation_function"]),
        )
        stride_1_scaled = table_stride_1 * 10
        stride_2_scaled = table_stride_2 * 10
        self.table_config = [
            table_extrapolate,
            stride_1_scaled,
            stride_2_scaled,
            check_frequency,
        ]
        self.lower, self.upper = self.table.build(
            min_nbor_dist, table_extrapolate, stride_1_scaled, stride_2_scaled
        )
        self.seat.enable_compression(
            self.table.data, self.table_config, self.lower, self.upper
        )
        self.compress = True

    def reinit_exclude(
        self,
        exclude_types: list[tuple[int, int]] = [],
    ) -> None:
        """Update the type exclusions."""
        self.seat.reinit_exclude(exclude_types)

    def forward(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
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
            The descriptor. shape: nf x nloc x ng
        gr
            The rotationally equivariant and permutationally invariant single particle
            representation.
            This descriptor returns None.
        g2
            The rotationally invariant pair-partical representation.
            This descriptor returns None.
        h2
            The rotationally equivariant pair-partical representation.
            This descriptor returns None.
        sw
            The smooth switch function.

        """
        # cast the input to internal precsion
        coord_ext = coord_ext.to(dtype=self.prec)
        g1, rot_mat, g2, h2, sw = self.seat.forward(
            nlist, coord_ext, atype_ext, None, mapping
        )
        return (
            g1.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            None,
            None,
            None,
            sw.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
        )

    def set_stat_mean_and_stddev(
        self,
        mean: torch.Tensor,
        stddev: torch.Tensor,
    ) -> None:
        """Update mean and stddev for descriptor."""
        self.seat.mean = mean
        self.seat.stddev = stddev

    def get_stat_mean_and_stddev(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get mean and stddev for descriptor."""
        return self.seat.mean, self.seat.stddev

    def serialize(self) -> dict:
        obj = self.seat
        return {
            "@class": "Descriptor",
            "type": "se_e3",
            "@version": 2,
            "rcut": obj.rcut,
            "rcut_smth": obj.rcut_smth,
            "sel": obj.sel,
            "neuron": obj.neuron,
            "resnet_dt": obj.resnet_dt,
            "set_davg_zero": obj.set_davg_zero,
            "activation_function": obj.activation_function,
            "precision": RESERVED_PRECISION_DICT[obj.prec],
            "embeddings": obj.filter_layers.serialize(),
            "env_mat": DPEnvMat(
                obj.rcut, obj.rcut_smth, obj.env_protection
            ).serialize(),
            "exclude_types": obj.exclude_types,
            "env_protection": obj.env_protection,
            "type_map": self.type_map,
            "@variables": {
                "davg": obj["davg"].detach().cpu().numpy(),
                "dstd": obj["dstd"].detach().cpu().numpy(),
            },
            "trainable": obj.trainable,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptSeT":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 2, 1)
        data.pop("@class", None)
        data.pop("type", None)
        variables = data.pop("@variables")
        embeddings = data.pop("embeddings")
        env_mat = data.pop("env_mat")
        obj = cls(**data)

        def t_cvt(xx):
            return torch.tensor(xx, dtype=obj.seat.prec, device=env.DEVICE)

        obj.seat["davg"] = t_cvt(variables["davg"])
        obj.seat["dstd"] = t_cvt(variables["dstd"])
        obj.seat.filter_layers = NetworkCollection.deserialize(embeddings)
        return obj

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: Optional[list[str]],
        local_jdata: dict,
    ) -> tuple[dict, Optional[float]]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statistics
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
        min_nbor_dist, local_jdata_cpy["sel"] = UpdateSel().update_one_sel(
            train_data, type_map, local_jdata_cpy["rcut"], local_jdata_cpy["sel"], False
        )
        return local_jdata_cpy, min_nbor_dist


@DescriptorBlock.register("se_e3")
class DescrptBlockSeT(DescriptorBlock):
    ndescrpt: Final[int]
    __constants__: ClassVar[list] = ["ndescrpt"]

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: list[int],
        neuron: list[int] = [24, 48, 96],
        resnet_dt: bool = False,
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        env_protection: float = 0.0,
        exclude_types: list[tuple[int, int]] = [],
        precision: str = "float64",
        trainable: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        r"""Construct an embedding net of type `se_e3`.

        The embedding takes angles between two neighboring atoms as input.

        Parameters
        ----------
        rcut : float
            The cut-off radius
        rcut_smth : float
            From where the environment matrix should be smoothed
        sel : list[int]
            sel[i] specifies the maxmum number of type i atoms in the cut-off radius
        neuron : list[int]
            Number of neurons in each hidden layers of the embedding net
        resnet_dt : bool
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
        set_davg_zero : bool
            Set the shift of embedding net input to zero.
        activation_function : str
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
        env_protection : float
            Protection parameter to prevent division by zero errors during environment matrix calculations.
        exclude_types : list[list[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
        precision : str
            The precision of the embedding net parameters. Supported options are |PRECISION|
        trainable : bool
            If the weights of embedding net are trainable.
        seed : int, Optional
            Random seed for initializing the network parameters.
        """
        super().__init__()
        self.rcut = float(rcut)
        self.rcut_smth = float(rcut_smth)
        self.neuron = neuron
        self.filter_neuron = self.neuron
        self.set_davg_zero = set_davg_zero
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.resnet_dt = resnet_dt
        self.env_protection = env_protection
        self.ntypes = len(sel)
        self.seed = seed
        # order matters, placed after the assignment of self.ntypes
        self.reinit_exclude(exclude_types)

        self.sel = sel
        # should be on CPU to avoid D2H, as it is used as slice index
        self.sec = [0, *np.cumsum(self.sel).tolist()]
        self.split_sel = self.sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 4

        wanted_shape = (self.ntypes, self.nnei, 4)
        mean = torch.zeros(wanted_shape, dtype=self.prec, device=env.DEVICE)
        stddev = torch.ones(wanted_shape, dtype=self.prec, device=env.DEVICE)
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)

        ndim = 2
        filter_layers = NetworkCollection(
            ndim=ndim, ntypes=len(sel), network_type="embedding_network"
        )
        for ii, embedding_idx in enumerate(
            itertools.product(range(self.ntypes), repeat=ndim)
        ):
            filter_layers[embedding_idx] = EmbeddingNet(
                1,
                self.filter_neuron,
                activation_function=self.activation_function,
                precision=self.precision,
                resnet_dt=self.resnet_dt,
                seed=child_seed(self.seed, ii),
            )
        self.filter_layers = filter_layers
        self.stats = None
        # set trainable
        self.trainable = trainable
        for param in self.parameters():
            param.requires_grad = trainable

        # add for compression
        self.compress = False
        self.compress_info = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(0, dtype=self.prec, device="cpu"))
                for _ in range(len(self.filter_layers.networks))
            ]
        )
        self.compress_data = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(0, dtype=self.prec, device=env.DEVICE))
                for _ in range(len(self.filter_layers.networks))
            ]
        )

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.rcut

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.rcut_smth

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return sum(self.sel)

    def get_sel(self) -> list[int]:
        """Returns the number of selected atoms for each type."""
        return self.sel

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        return self.dim_out

    def get_dim_emb(self) -> int:
        """Returns the output dimension."""
        return self.neuron[-1]

    def get_dim_in(self) -> int:
        """Returns the input dimension."""
        return self.dim_in

    def mixed_types(self) -> bool:
        """If true, the descriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the descriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return False

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.env_protection

    @property
    def dim_out(self):
        """Returns the output dimension of this descriptor."""
        return self.filter_neuron[-1]

    @property
    def dim_in(self) -> int:
        """Returns the atomic input dimension of this descriptor."""
        return 0

    def __setitem__(self, key, value) -> None:
        if key in ("avg", "data_avg", "davg"):
            self.mean = value
        elif key in ("std", "data_std", "dstd"):
            self.stddev = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        if key in ("avg", "data_avg", "davg"):
            return self.mean
        elif key in ("std", "data_std", "dstd"):
            return self.stddev
        else:
            raise KeyError(key)

    def compute_input_stats(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        path: Optional[DPPath] = None,
    ) -> None:
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
        env_mat_stat = EnvMatStatSe(self)
        if path is not None:
            path = path / env_mat_stat.get_hash()
        if path is None or not path.is_dir():
            if callable(merged):
                # only get data for once
                sampled = merged()
            else:
                sampled = merged
        else:
            sampled = []
        env_mat_stat.load_or_compute_stats(sampled, path)
        self.stats = env_mat_stat.stats
        mean, stddev = env_mat_stat()
        if not self.set_davg_zero:
            self.mean.copy_(
                torch.tensor(mean, device=env.DEVICE, dtype=self.mean.dtype)
            )
        self.stddev.copy_(
            torch.tensor(stddev, device=env.DEVICE, dtype=self.stddev.dtype)
        )

    def get_stats(self) -> dict[str, StatItem]:
        """Get the statistics of the descriptor."""
        if self.stats is None:
            raise RuntimeError(
                "The statistics of the descriptor has not been computed."
            )
        return self.stats

    def reinit_exclude(
        self,
        exclude_types: list[tuple[int, int]] = [],
    ) -> None:
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    def enable_compression(
        self,
        table_data,
        table_config,
        lower,
        upper,
    ) -> None:
        for embedding_idx, ll in enumerate(self.filter_layers.networks):
            ti = embedding_idx % self.ntypes
            tj = embedding_idx // self.ntypes
            if ti <= tj:
                net = "filter_" + str(ti) + "_net_" + str(tj)
                info_ii = torch.as_tensor(
                    [
                        lower[net],
                        upper[net],
                        upper[net] * table_config[0],
                        table_config[1],
                        table_config[2],
                        table_config[3],
                    ],
                    dtype=self.prec,
                    device="cpu",
                )
                tensor_data_ii = table_data[net].to(device=env.DEVICE, dtype=self.prec)
                self.compress_data[embedding_idx] = tensor_data_ii
                self.compress_info[embedding_idx] = info_ii
        self.compress = True

    def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_atype_embd: Optional[torch.Tensor] = None,
        mapping: Optional[torch.Tensor] = None,
        type_embedding: Optional[torch.Tensor] = None,
    ):
        """Compute the descriptor.

        Parameters
        ----------
        nlist
            The neighbor list. shape: nf x nloc x nnei
        extended_coord
            The extended coordinates of atoms. shape: nf x (nallx3)
        extended_atype
            The extended aotm types. shape: nf x nall x nt
        extended_atype_embd
            The extended type embedding of atoms. shape: nf x nall
        mapping
            The index mapping, not required by this descriptor.
        type_embedding
            Full type embeddings. shape: (ntypes+1) x nt
            Required for stripped type embeddings.

        Returns
        -------
        result
            The descriptor. shape: nf x nloc x ng
        gr
            The rotationally equivariant and permutationally invariant single particle
            representation.
            This descriptor returns None.
        g2
            The rotationally invariant pair-partical representation.
            This descriptor returns None.
        h2
            The rotationally equivariant pair-partical representation.
            This descriptor returns None.
        sw
            The smooth switch function. shape: nf x nloc x nnei

        """
        del extended_atype_embd, mapping
        nf = nlist.shape[0]
        nloc = nlist.shape[1]
        atype = extended_atype[:, :nloc]
        dmatrix, diff, sw = prod_env_mat(
            extended_coord,
            nlist,
            atype,
            self.mean,
            self.stddev,
            self.rcut,
            self.rcut_smth,
            protection=self.env_protection,
        )
        dmatrix = dmatrix.view(-1, self.nnei, 4)
        nfnl = dmatrix.shape[0]
        # pre-allocate a shape to pass jit
        result = torch.zeros(
            [nfnl, self.filter_neuron[-1]],
            dtype=self.prec,
            device=extended_coord.device,
        )
        # nfnl x nnei
        exclude_mask = self.emask(nlist, extended_atype).view(nfnl, self.nnei)
        for embedding_idx, (ll, compress_data_ii, compress_info_ii) in enumerate(
            zip(self.filter_layers.networks, self.compress_data, self.compress_info)
        ):
            ti = embedding_idx % self.ntypes
            nei_type_j = self.sel[ti]
            tj = embedding_idx // self.ntypes
            nei_type_i = self.sel[tj]
            if ti <= tj:
                # avoid repeat calculation
                # nfnl x nt_i x 3
                rr_i = dmatrix[:, self.sec[ti] : self.sec[ti + 1], 1:]
                mm_i = exclude_mask[:, self.sec[ti] : self.sec[ti + 1]]
                rr_i = rr_i * mm_i[:, :, None]
                # nfnl x nt_j x 3
                rr_j = dmatrix[:, self.sec[tj] : self.sec[tj + 1], 1:]
                mm_j = exclude_mask[:, self.sec[tj] : self.sec[tj + 1]]
                rr_j = rr_j * mm_j[:, :, None]
                # nfnl x nt_i x nt_j
                env_ij = torch.einsum("ijm,ikm->ijk", rr_i, rr_j)
                if self.compress:
                    ebd_env_ij = env_ij.view(-1, 1)
                    res_ij = torch.ops.deepmd.tabulate_fusion_se_t(
                        compress_data_ii.contiguous(),
                        compress_info_ii.cpu().contiguous(),
                        ebd_env_ij.contiguous(),
                        env_ij.contiguous(),
                        self.filter_neuron[-1],
                    )[0]
                else:
                    # nfnl x nt_i x nt_j x 1
                    env_ij_reshape = env_ij.unsqueeze(-1)
                    # nfnl x nt_i x nt_j x ng
                    gg = ll.forward(env_ij_reshape)
                    # nfnl x nt_i x nt_j x ng
                    res_ij = torch.einsum("ijk,ijkm->im", env_ij, gg)
                res_ij = res_ij * (1.0 / float(nei_type_i) / float(nei_type_j))
                result += res_ij
        # xyz_scatter /= (self.nnei * self.nnei)
        result = result.view(nf, nloc, self.filter_neuron[-1])
        return (
            result,
            None,
            None,
            None,
            sw,
        )

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor block has message passing."""
        return False

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor block needs sorted nlist when using `forward_lower`."""
        return False
