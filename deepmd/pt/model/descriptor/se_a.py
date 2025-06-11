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

if not hasattr(torch.ops.deepmd, "tabulate_fusion_se_a"):

    def tabulate_fusion_se_a(
        argument0: torch.Tensor,
        argument1: torch.Tensor,
        argument2: torch.Tensor,
        argument3: torch.Tensor,
        argument4: int,
    ) -> list[torch.Tensor]:
        raise NotImplementedError(
            "tabulate_fusion_se_a is not available since customized PyTorch OP library is not built when freezing the model. "
            "See documentation for model compression for details."
        )

    # Note: this hack cannot actually save a model that can be runned using LAMMPS.
    torch.ops.deepmd.tabulate_fusion_se_a = tabulate_fusion_se_a


@BaseDescriptor.register("se_e2_a")
@BaseDescriptor.register("se_a")
class DescrptSeA(BaseDescriptor, torch.nn.Module):
    def __init__(
        self,
        rcut,
        rcut_smth,
        sel,
        neuron=[25, 50, 100],
        axis_neuron=16,
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = "float64",
        resnet_dt: bool = False,
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        type_one_side: bool = True,
        trainable: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
        ntypes: Optional[int] = None,  # to be compat with input
        type_map: Optional[list[str]] = None,
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
        self.sea = DescrptBlockSeA(
            rcut,
            rcut_smth,
            sel,
            neuron=neuron,
            axis_neuron=axis_neuron,
            set_davg_zero=set_davg_zero,
            activation_function=activation_function,
            precision=precision,
            resnet_dt=resnet_dt,
            exclude_types=exclude_types,
            env_protection=env_protection,
            type_one_side=type_one_side,
            trainable=trainable,
            seed=seed,
        )

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.sea.get_rcut()

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.sea.get_rcut_smth()

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return self.sea.get_nsel()

    def get_sel(self) -> list[int]:
        """Returns the number of selected atoms for each type."""
        return self.sea.get_sel()

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.sea.get_ntypes()

    def get_type_map(self) -> list[str]:
        """Get the name to each type of atoms."""
        return self.type_map

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        return self.sea.get_dim_out()

    def get_dim_emb(self) -> int:
        """Returns the output dimension."""
        return self.sea.get_dim_emb()

    def mixed_types(self):
        """Returns if the descriptor requires a neighbor list that distinguish different
        atomic types or not.
        """
        return self.sea.mixed_types()

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor has message passing."""
        return self.sea.has_message_passing()

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor needs sorted nlist when using `forward_lower`."""
        return self.sea.need_sorted_nlist_for_lower()

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.sea.get_env_protection()

    def share_params(self, base_class, shared_level, resume=False) -> None:
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert self.__class__ == base_class.__class__, (
            "Only descriptors of the same type can share params!"
        )
        # For SeA descriptors, the user-defined share-level
        # shared_level: 0
        # share all parameters in sea
        if shared_level == 0:
            self.sea.share_params(base_class.sea, 0, resume=resume)
        # Other shared levels
        else:
            raise NotImplementedError

    @property
    def dim_out(self):
        """Returns the output dimension of this descriptor."""
        return self.sea.dim_out

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        raise NotImplementedError(
            "Descriptor se_e2_a does not support changing for type related params!"
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
        return self.sea.compute_input_stats(merged, path)

    def reinit_exclude(
        self,
        exclude_types: list[tuple[int, int]] = [],
    ) -> None:
        """Update the type exclusions."""
        self.sea.reinit_exclude(exclude_types)

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
            data["type_one_side"],
            data["exclude_types"],
            ActivationFn(data["activation_function"]),
        )
        self.table_config = [
            table_extrapolate,
            table_stride_1,
            table_stride_2,
            check_frequency,
        ]
        self.lower, self.upper = self.table.build(
            min_nbor_dist, table_extrapolate, table_stride_1, table_stride_2
        )
        self.sea.enable_compression(
            self.table.data, self.table_config, self.lower, self.upper
        )
        self.compress = True

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
            The descriptor. shape: nf x nloc x (ng x axis_neuron)
        gr
            The rotationally equivariant and permutationally invariant single particle
            representation. shape: nf x nloc x ng x 3
        g2
            The rotationally invariant pair-partical representation.
            this descriptor returns None
        h2
            The rotationally equivariant pair-partical representation.
            this descriptor returns None
        sw
            The smooth switch function.

        """
        # cast the input to internal precsion
        coord_ext = coord_ext.to(dtype=self.prec)
        g1, rot_mat, g2, h2, sw = self.sea.forward(
            nlist, coord_ext, atype_ext, None, mapping
        )
        return (
            g1.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            rot_mat.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
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
        self.sea.mean = mean
        self.sea.stddev = stddev

    def get_stat_mean_and_stddev(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get mean and stddev for descriptor."""
        return self.sea.mean, self.sea.stddev

    def serialize(self) -> dict:
        obj = self.sea
        return {
            "@class": "Descriptor",
            "type": "se_e2_a",
            "@version": 2,
            "rcut": obj.rcut,
            "rcut_smth": obj.rcut_smth,
            "sel": obj.sel,
            "neuron": obj.neuron,
            "axis_neuron": obj.axis_neuron,
            "resnet_dt": obj.resnet_dt,
            "set_davg_zero": obj.set_davg_zero,
            "activation_function": obj.activation_function,
            # make deterministic
            "precision": RESERVED_PRECISION_DICT[obj.prec],
            "embeddings": obj.filter_layers.serialize(),
            "env_mat": DPEnvMat(
                obj.rcut, obj.rcut_smth, obj.env_protection
            ).serialize(),
            "exclude_types": obj.exclude_types,
            "env_protection": obj.env_protection,
            "@variables": {
                "davg": obj["davg"].detach().cpu().numpy(),
                "dstd": obj["dstd"].detach().cpu().numpy(),
            },
            "type_map": self.type_map,
            ## to be updated when the options are supported.
            "trainable": True,
            "type_one_side": obj.type_one_side,
            "spin": None,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptSeA":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 2, 1)
        data.pop("@class", None)
        data.pop("type", None)
        variables = data.pop("@variables")
        embeddings = data.pop("embeddings")
        env_mat = data.pop("env_mat")
        obj = cls(**data)

        def t_cvt(xx):
            return torch.tensor(xx, dtype=obj.sea.prec, device=env.DEVICE)

        obj.sea["davg"] = t_cvt(variables["davg"])
        obj.sea["dstd"] = t_cvt(variables["dstd"])
        obj.sea.filter_layers = NetworkCollection.deserialize(embeddings)
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


@DescriptorBlock.register("se_e2_a")
class DescrptBlockSeA(DescriptorBlock):
    ndescrpt: Final[int]
    __constants__: ClassVar[list] = ["ndescrpt"]

    def __init__(
        self,
        rcut,
        rcut_smth,
        sel,
        neuron=[25, 50, 100],
        axis_neuron=16,
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = "float64",
        resnet_dt: bool = False,
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        type_one_side: bool = True,
        trainable: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
        **kwargs,
    ) -> None:
        """Construct an embedding net of type `se_a`.

        Args:
        - rcut: Cut-off radius.
        - rcut_smth: Smooth hyper-parameter for pair force & energy.
        - sel: For each element type, how many atoms is selected as neighbors.
        - filter_neuron: Number of neurons in each hidden layers of the embedding net.
        - axis_neuron: Number of columns of the sub-matrix of the embedding matrix.
        """
        super().__init__()
        self.rcut = float(rcut)
        self.rcut_smth = float(rcut_smth)
        self.neuron = neuron
        self.filter_neuron = self.neuron
        self.axis_neuron = axis_neuron
        self.set_davg_zero = set_davg_zero
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.resnet_dt = resnet_dt
        self.env_protection = env_protection
        self.ntypes = len(sel)
        self.type_one_side = type_one_side
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

        ndim = 1 if self.type_one_side else 2
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

    def get_dim_rot_mat_1(self) -> int:
        """Returns the first dimension of the rotation matrix. The rotation is of shape dim_1 x 3."""
        return self.filter_neuron[-1]

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
        return self.filter_neuron[-1] * self.axis_neuron

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
        table_data: dict[str, torch.Tensor],
        table_config: list[Union[int, float]],
        lower: dict[str, int],
        upper: dict[str, int],
    ) -> None:
        for embedding_idx, ll in enumerate(self.filter_layers.networks):
            if self.type_one_side:
                ii = embedding_idx
                ti = -1
            else:
                # ti: center atom type, ii: neighbor type...
                ii = embedding_idx // self.ntypes
                ti = embedding_idx % self.ntypes
            if self.type_one_side:
                net = "filter_-1_net_" + str(ii)
            else:
                net = "filter_" + str(ti) + "_net_" + str(ii)
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
        """Calculate decoded embedding for each atom.

        Args:
        - coord: Tell atom coordinates with shape [nframes, natoms[1]*3].
        - atype: Tell atom types with shape [nframes, natoms[1]].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].
        - box: Tell simulation box with shape [nframes, 9].

        Returns
        -------
        - `torch.Tensor`: descriptor matrix with shape [nframes, natoms[0]*self.filter_neuron[-1]*self.axis_neuron].
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
        xyz_scatter = torch.zeros(
            [nfnl, 4, self.filter_neuron[-1]],
            dtype=self.prec,
            device=extended_coord.device,
        )
        # nfnl x nnei
        exclude_mask = self.emask(nlist, extended_atype).view(nfnl, self.nnei)
        for embedding_idx, (ll, compress_data_ii, compress_info_ii) in enumerate(
            zip(self.filter_layers.networks, self.compress_data, self.compress_info)
        ):
            if self.type_one_side:
                ii = embedding_idx
                ti = -1
                # torch.jit is not happy with slice(None)
                # ti_mask = torch.ones(nfnl, dtype=torch.bool, device=dmatrix.device)
                # applying a mask seems to cause performance degradation
                ti_mask = None
            else:
                # ti: center atom type, ii: neighbor type...
                ii = embedding_idx // self.ntypes
                ti = embedding_idx % self.ntypes
                ti_mask = atype.ravel().eq(ti)
            # nfnl x nt
            if ti_mask is not None:
                mm = exclude_mask[ti_mask, self.sec[ii] : self.sec[ii + 1]]
            else:
                mm = exclude_mask[:, self.sec[ii] : self.sec[ii + 1]]
            # nfnl x nt x 4
            if ti_mask is not None:
                rr = dmatrix[ti_mask, self.sec[ii] : self.sec[ii + 1], :]
            else:
                rr = dmatrix[:, self.sec[ii] : self.sec[ii + 1], :]
            rr = rr * mm[:, :, None]
            ss = rr[:, :, :1]

            if self.compress:
                ss = ss.reshape(-1, 1)  # xyz_scatter_tensor in tf

                gr = torch.ops.deepmd.tabulate_fusion_se_a(
                    compress_data_ii.contiguous(),
                    compress_info_ii.cpu().contiguous(),
                    ss.contiguous(),
                    rr.contiguous(),
                    self.filter_neuron[-1],
                )[0]
            else:
                # nfnl x nt x ng
                gg = ll.forward(ss)
                # nfnl x 4 x ng
                gr = torch.matmul(rr.permute(0, 2, 1), gg)

            if ti_mask is not None:
                xyz_scatter[ti_mask] += gr
            else:
                xyz_scatter += gr

        xyz_scatter /= self.nnei
        xyz_scatter_1 = xyz_scatter.permute(0, 2, 1)
        rot_mat = xyz_scatter_1[:, :, 1:4]
        xyz_scatter_2 = xyz_scatter[:, :, 0 : self.axis_neuron]
        result = torch.matmul(
            xyz_scatter_1, xyz_scatter_2
        )  # shape is [nframes*nall, self.filter_neuron[-1], self.axis_neuron]
        result = result.view(nf, nloc, self.filter_neuron[-1] * self.axis_neuron)
        rot_mat = rot_mat.view([nf, nloc] + list(rot_mat.shape[1:]))  # noqa:RUF005
        return (
            result,
            rot_mat,
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
