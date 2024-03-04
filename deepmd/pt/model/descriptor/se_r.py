# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch

from deepmd.dpmodel.utils import EnvMat as DPEnvMat
from deepmd.pt.model.descriptor import (
    prod_env_mat,
)
from deepmd.pt.model.network.mlp import (
    EmbeddingNet,
    NetworkCollection,
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
from deepmd.pt.utils.exclude_mask import (
    PairExcludeMask,
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

from .base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("se_e2_r")
@BaseDescriptor.register("se_r")
class DescrptSeR(BaseDescriptor, torch.nn.Module):
    def __init__(
        self,
        rcut,
        rcut_smth,
        sel,
        neuron=[25, 50, 100],
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = "float64",
        resnet_dt: bool = False,
        exclude_types: List[Tuple[int, int]] = [],
        old_impl: bool = False,
        trainable: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.neuron = neuron
        self.filter_neuron = self.neuron
        self.set_davg_zero = set_davg_zero
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.resnet_dt = resnet_dt
        self.old_impl = False  # this does not support old implementation.
        self.exclude_types = exclude_types
        self.ntypes = len(sel)
        self.emask = PairExcludeMask(len(sel), exclude_types=exclude_types)

        self.sel = sel
        self.sec = torch.tensor(
            np.append([0], np.cumsum(self.sel)), dtype=int, device=env.DEVICE
        )
        self.split_sel = self.sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 1

        wanted_shape = (self.ntypes, self.nnei, 1)
        mean = torch.zeros(wanted_shape, dtype=self.prec, device=env.DEVICE)
        stddev = torch.ones(wanted_shape, dtype=self.prec, device=env.DEVICE)
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.filter_layers_old = None
        self.filter_layers = None

        filter_layers = NetworkCollection(
            ndim=1, ntypes=len(sel), network_type="embedding_network"
        )
        # TODO: ndim=2 if type_one_side=False
        for ii in range(self.ntypes):
            filter_layers[(ii,)] = EmbeddingNet(
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
        return self.neuron[-1]

    def get_dim_emb(self) -> int:
        """Returns the output dimension."""
        raise NotImplementedError

    def get_dim_in(self) -> int:
        """Returns the input dimension."""
        return 0

    def mixed_types(self) -> bool:
        """If true, the discriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the discriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return False

    def share_params(self, base_class, shared_level, resume=False):
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some seperated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert (
            self.__class__ == base_class.__class__
        ), "Only descriptors of the same type can share params!"
        # For SeR descriptors, the user-defined share-level
        # shared_level: 0
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
        # Other shared levels
        else:
            raise NotImplementedError

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
        del mapping
        nloc = nlist.shape[1]
        atype = atype_ext[:, :nloc]
        dmatrix, diff, sw = prod_env_mat(
            coord_ext,
            nlist,
            atype,
            self.mean,
            self.stddev,
            self.rcut,
            self.rcut_smth,
            True,
        )

        assert self.filter_layers is not None
        dmatrix = dmatrix.view(-1, self.nnei, 1)
        dmatrix = dmatrix.to(dtype=self.prec)
        nfnl = dmatrix.shape[0]
        # pre-allocate a shape to pass jit
        xyz_scatter = torch.zeros(
            [nfnl, 1, self.filter_neuron[-1]], dtype=self.prec, device=coord_ext.device
        )

        # nfnl x nnei
        exclude_mask = self.emask(nlist, atype_ext).view(nfnl, -1)
        for ii, ll in enumerate(self.filter_layers.networks):
            # nfnl x nt
            mm = exclude_mask[:, self.sec[ii] : self.sec[ii + 1]]
            # nfnl x nt x 1
            ss = dmatrix[:, self.sec[ii] : self.sec[ii + 1], :]
            ss = ss * mm[:, :, None]
            # nfnl x nt x ng
            gg = ll.forward(ss)
            gg = torch.mean(gg, dim=1).unsqueeze(1)
            xyz_scatter += gg * (self.sel[ii] / self.nnei)

        res_rescale = 1.0 / 5.0
        result = xyz_scatter * res_rescale
        result = result.view(-1, nloc, self.filter_neuron[-1])
        return (
            result.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            None,
            None,
            None,
            sw,
        )

    def set_stat_mean_and_stddev(
        self,
        mean: torch.Tensor,
        stddev: torch.Tensor,
    ) -> None:
        self.mean = mean
        self.stddev = stddev

    def serialize(self) -> dict:
        return {
            "@class": "Descriptor",
            "type": "se_r",
            "@version": 1,
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel,
            "neuron": self.neuron,
            "resnet_dt": self.resnet_dt,
            "set_davg_zero": self.set_davg_zero,
            "activation_function": self.activation_function,
            # make deterministic
            "precision": RESERVED_PRECISON_DICT[self.prec],
            "embeddings": self.filter_layers.serialize(),
            "env_mat": DPEnvMat(self.rcut, self.rcut_smth).serialize(),
            "exclude_types": self.exclude_types,
            "@variables": {
                "davg": self["davg"].detach().cpu().numpy(),
                "dstd": self["dstd"].detach().cpu().numpy(),
            },
            ## to be updated when the options are supported.
            "trainable": True,
            "type_one_side": True,
            "spin": None,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptSeR":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        variables = data.pop("@variables")
        embeddings = data.pop("embeddings")
        env_mat = data.pop("env_mat")
        obj = cls(**data)

        def t_cvt(xx):
            return torch.tensor(xx, dtype=obj.prec, device=env.DEVICE)

        obj["davg"] = t_cvt(variables["davg"])
        obj["dstd"] = t_cvt(variables["dstd"])
        obj.filter_layers = NetworkCollection.deserialize(embeddings)
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
