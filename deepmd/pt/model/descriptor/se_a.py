# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
from typing import (
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch

from deepmd.pt.model.descriptor import (
    DescriptorBlock,
    prod_env_mat,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
    RESERVED_PRECISON_DICT,
)
from deepmd.pt.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
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
from deepmd.pt.model.network.network import (
    TypeFilter,
)
from deepmd.pt.utils.exclude_mask import (
    PairExcludeMask,
)

from .base_descriptor import (
    BaseDescriptor,
)


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
        exclude_types: List[Tuple[int, int]] = [],
        old_impl: bool = False,
        type_one_side: bool = True,
        **kwargs,
    ):
        super().__init__()
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
            old_impl=old_impl,
            type_one_side=type_one_side,
            **kwargs,
        )

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.sea.get_rcut()

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return self.sea.get_nsel()

    def get_sel(self) -> List[int]:
        """Returns the number of selected atoms for each type."""
        return self.sea.get_sel()

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.sea.get_ntypes()

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

    def share_params(self, base_class, shared_level, resume=False):
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some seperated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert (
            self.__class__ == base_class.__class__
        ), "Only descriptors of the same type can share params!"
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
        return self.sea.compute_input_stats(merged, path)

    def reinit_exclude(
        self,
        exclude_types: List[Tuple[int, int]] = [],
    ):
        """Update the type exclusions."""
        self.sea.reinit_exclude(exclude_types)

    def forward(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
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
        return self.sea.forward(nlist, coord_ext, atype_ext, None, mapping)

    def set_stat_mean_and_stddev(
        self,
        mean: torch.Tensor,
        stddev: torch.Tensor,
    ) -> None:
        self.sea.mean = mean
        self.sea.stddev = stddev

    def serialize(self) -> dict:
        obj = self.sea
        return {
            "@class": "Descriptor",
            "type": "se_e2_a",
            "@version": 1,
            "rcut": obj.rcut,
            "rcut_smth": obj.rcut_smth,
            "sel": obj.sel,
            "neuron": obj.neuron,
            "axis_neuron": obj.axis_neuron,
            "resnet_dt": obj.resnet_dt,
            "set_davg_zero": obj.set_davg_zero,
            "activation_function": obj.activation_function,
            # make deterministic
            "precision": RESERVED_PRECISON_DICT[obj.prec],
            "embeddings": obj.filter_layers.serialize(),
            "env_mat": DPEnvMat(obj.rcut, obj.rcut_smth).serialize(),
            "exclude_types": obj.exclude_types,
            "@variables": {
                "davg": obj["davg"].detach().cpu().numpy(),
                "dstd": obj["dstd"].detach().cpu().numpy(),
            },
            ## to be updated when the options are supported.
            "trainable": True,
            "type_one_side": obj.type_one_side,
            "spin": None,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptSeA":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
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
    def update_sel(cls, global_jdata: dict, local_jdata: dict):
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        global_jdata : dict
            The global data, containing the training section
        local_jdata : dict
            The local data refer to the current class
        """
        local_jdata_cpy = local_jdata.copy()
        return UpdateSel().update_one_sel(global_jdata, local_jdata_cpy, False)


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
        exclude_types: List[Tuple[int, int]] = [],
        old_impl: bool = False,
        type_one_side: bool = True,
        trainable: bool = True,
        **kwargs,
    ):
        """Construct an embedding net of type `se_a`.

        Args:
        - rcut: Cut-off radius.
        - rcut_smth: Smooth hyper-parameter for pair force & energy.
        - sel: For each element type, how many atoms is selected as neighbors.
        - filter_neuron: Number of neurons in each hidden layers of the embedding net.
        - axis_neuron: Number of columns of the sub-matrix of the embedding matrix.
        """
        super().__init__()
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.neuron = neuron
        self.filter_neuron = self.neuron
        self.axis_neuron = axis_neuron
        self.set_davg_zero = set_davg_zero
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.resnet_dt = resnet_dt
        self.old_impl = old_impl
        self.ntypes = len(sel)
        self.type_one_side = type_one_side
        # order matters, placed after the assignment of self.ntypes
        self.reinit_exclude(exclude_types)

        self.sel = sel
        self.sec = torch.tensor(
            np.append([0], np.cumsum(self.sel)), dtype=int, device=env.DEVICE
        )
        self.split_sel = self.sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 4

        wanted_shape = (self.ntypes, self.nnei, 4)
        mean = torch.zeros(wanted_shape, dtype=self.prec, device=env.DEVICE)
        stddev = torch.ones(wanted_shape, dtype=self.prec, device=env.DEVICE)
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.filter_layers_old = None
        self.filter_layers = None

        if self.old_impl:
            if not self.type_one_side:
                raise ValueError(
                    "The old implementation does not support type_one_side=False."
                )
            filter_layers = []
            # TODO: remove
            start_index = 0
            for type_i in range(self.ntypes):
                one = TypeFilter(start_index, sel[type_i], self.filter_neuron)
                filter_layers.append(one)
                start_index += sel[type_i]
            self.filter_layers_old = torch.nn.ModuleList(filter_layers)
        else:
            ndim = 1 if self.type_one_side else 2
            filter_layers = NetworkCollection(
                ndim=ndim, ntypes=len(sel), network_type="embedding_network"
            )
            for embedding_idx in itertools.product(range(self.ntypes), repeat=ndim):
                filter_layers[embedding_idx] = EmbeddingNet(
                    1,
                    self.filter_neuron,
                    activation_function=self.activation_function,
                    precision=self.precision,
                    resnet_dt=self.resnet_dt,
                )
            self.filter_layers = filter_layers
        self.stats = None
        # set trainable
        for param in self.parameters():
            param.requires_grad = trainable

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.rcut

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return sum(self.sel)

    def get_sel(self) -> List[int]:
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
        """If true, the discriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the discriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return False

    @property
    def dim_out(self):
        """Returns the output dimension of this descriptor."""
        return self.filter_neuron[-1] * self.axis_neuron

    @property
    def dim_in(self):
        """Returns the atomic input dimension of this descriptor."""
        return 0

    def __setitem__(self, key, value):
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
            self.mean.copy_(torch.tensor(mean, device=env.DEVICE))
        self.stddev.copy_(torch.tensor(stddev, device=env.DEVICE))

    def get_stats(self) -> Dict[str, StatItem]:
        """Get the statistics of the descriptor."""
        if self.stats is None:
            raise RuntimeError(
                "The statistics of the descriptor has not been computed."
            )
        return self.stats

    def reinit_exclude(
        self,
        exclude_types: List[Tuple[int, int]] = [],
    ):
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_atype_embd: Optional[torch.Tensor] = None,
        mapping: Optional[torch.Tensor] = None,
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
        )

        if self.old_impl:
            assert self.filter_layers_old is not None
            dmatrix = dmatrix.view(
                -1, self.ndescrpt
            )  # shape is [nframes*nall, self.ndescrpt]
            xyz_scatter = torch.empty(
                1,
                device=env.DEVICE,
            )
            ret = self.filter_layers_old[0](dmatrix)
            xyz_scatter = ret
            for ii, transform in enumerate(self.filter_layers_old[1:]):
                # shape is [nframes*nall, 4, self.filter_neuron[-1]]
                ret = transform.forward(dmatrix)
                xyz_scatter = xyz_scatter + ret
        else:
            assert self.filter_layers is not None
            dmatrix = dmatrix.view(-1, self.nnei, 4)
            dmatrix = dmatrix.to(dtype=self.prec)
            nfnl = dmatrix.shape[0]
            # pre-allocate a shape to pass jit
            xyz_scatter = torch.zeros(
                [nfnl, 4, self.filter_neuron[-1]],
                dtype=self.prec,
                device=extended_coord.device,
            )
            # nfnl x nnei
            exclude_mask = self.emask(nlist, extended_atype).view(nfnl, -1)
            for embedding_idx, ll in enumerate(self.filter_layers.networks):
                if self.type_one_side:
                    ii = embedding_idx
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
        result = result.view(-1, nloc, self.filter_neuron[-1] * self.axis_neuron)
        rot_mat = rot_mat.view([-1, nloc] + list(rot_mat.shape[1:]))  # noqa:RUF005
        return (
            result.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            rot_mat.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            None,
            None,
            sw,
        )


def analyze_descrpt(matrix, ndescrpt, natoms):
    """Collect avg, square avg and count of descriptors in a batch."""
    ntypes = natoms.shape[1] - 2
    start_index = 0
    sysr = []
    sysa = []
    sysn = []
    sysr2 = []
    sysa2 = []
    for type_i in range(ntypes):
        end_index = start_index + natoms[0, 2 + type_i]
        dd = matrix[:, start_index:end_index]  # all descriptors for this element
        start_index = end_index
        dd = np.reshape(
            dd, [-1, 4]
        )  # Shape is [nframes*natoms[2+type_id]*self.nnei, 4]
        ddr = dd[:, :1]
        dda = dd[:, 1:]
        sumr = np.sum(ddr)
        suma = np.sum(dda) / 3.0
        sumn = dd.shape[0]  # Value is nframes*natoms[2+type_id]*self.nnei
        sumr2 = np.sum(np.multiply(ddr, ddr))
        suma2 = np.sum(np.multiply(dda, dda)) / 3.0
        sysr.append(sumr)
        sysa.append(suma)
        sysn.append(sumn)
        sysr2.append(sumr2)
        sysa2.append(suma2)
    return sysr, sysr2, sysa, sysa2, sysn
