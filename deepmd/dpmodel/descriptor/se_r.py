# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from deepmd.dpmodel.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

try:
    from deepmd._version import version as __version__
except ImportError:
    __version__ = "unknown"

import copy
from typing import (
    Any,
    List,
    Optional,
)

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.utils import (
    EmbeddingNet,
    EnvMat,
    NetworkCollection,
    PairExcludeMask,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from .base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("se_e2_r")
@BaseDescriptor.register("se_r")
class DescrptSeR(NativeOP, BaseDescriptor):
    r"""DeepPot-SE_R constructed from only the radial imformation of atomic configurations.


    Parameters
    ----------
    rcut
            The cut-off radius :math:`r_c`
    rcut_smth
            From where the environment matrix should be smoothed :math:`r_s`
    sel : list[str]
            sel[i] specifies the maxmum number of type i atoms in the cut-off radius
    neuron : list[int]
            Number of neurons in each hidden layers of the embedding net :math:`\mathcal{N}`
    resnet_dt
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
    trainable
            If the weights of embedding net are trainable.
    type_one_side
            Try to build N_types embedding nets. Otherwise, building N_types^2 embedding nets
    exclude_types : List[List[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    set_davg_zero
            Set the shift of embedding net input to zero.
    activation_function
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision
            The precision of the embedding net parameters. Supported options are |PRECISION|
    multi_task
            If the model has multi fitting nets to train.
    spin
            The deepspin object.

    Limitations
    -----------
    The currently implementation does not support the following features

    1. type_one_side == False
    2. exclude_types != []
    3. spin is not None

    References
    ----------
    .. [1] Linfeng Zhang, Jiequn Han, Han Wang, Wissam A. Saidi, Roberto Car, and E. Weinan. 2018.
       End-to-end symmetry preserving inter-atomic potential energy model for finite and extended
       systems. In Proceedings of the 32nd International Conference on Neural Information Processing
       Systems (NIPS'18). Curran Associates Inc., Red Hook, NY, USA, 4441-4451.
    """

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: List[str],
        neuron: List[int] = [24, 48, 96],
        resnet_dt: bool = False,
        trainable: bool = True,
        type_one_side: bool = True,
        exclude_types: List[List[int]] = [],
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        spin: Optional[Any] = None,
        # consistent with argcheck, not used though
        seed: Optional[int] = None,
    ) -> None:
        ## seed, uniform_seed, multi_task, not included.
        if not type_one_side:
            raise NotImplementedError("type_one_side == False not implemented")
        if spin is not None:
            raise NotImplementedError("spin is not implemented")

        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.sel = sel
        self.ntypes = len(self.sel)
        self.neuron = neuron
        self.resnet_dt = resnet_dt
        self.trainable = trainable
        self.type_one_side = type_one_side
        self.exclude_types = exclude_types
        self.set_davg_zero = set_davg_zero
        self.activation_function = activation_function
        self.precision = precision
        self.spin = spin
        self.emask = PairExcludeMask(self.ntypes, self.exclude_types)

        in_dim = 1  # not considiering type embedding
        self.embeddings = NetworkCollection(
            ntypes=self.ntypes,
            ndim=(1 if self.type_one_side else 2),
            network_type="embedding_network",
        )
        if not self.type_one_side:
            raise NotImplementedError("type_one_side == False not implemented")
        for ii in range(self.ntypes):
            self.embeddings[(ii,)] = EmbeddingNet(
                in_dim,
                self.neuron,
                self.activation_function,
                self.resnet_dt,
                self.precision,
            )
        self.env_mat = EnvMat(self.rcut, self.rcut_smth)
        self.nnei = np.sum(self.sel)
        self.davg = np.zeros(
            [self.ntypes, self.nnei, 1], dtype=PRECISION_DICT[self.precision]
        )
        self.dstd = np.ones(
            [self.ntypes, self.nnei, 1], dtype=PRECISION_DICT[self.precision]
        )
        self.orig_sel = self.sel

    def __setitem__(self, key, value):
        if key in ("avg", "data_avg", "davg"):
            self.davg = value
        elif key in ("std", "data_std", "dstd"):
            self.dstd = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        if key in ("avg", "data_avg", "davg"):
            return self.davg
        elif key in ("std", "data_std", "dstd"):
            return self.dstd
        else:
            raise KeyError(key)

    @property
    def dim_out(self):
        """Returns the output dimension of this descriptor."""
        return self.get_dim_out()

    def get_dim_out(self):
        """Returns the output dimension of this descriptor."""
        return self.neuron[-1]

    def get_dim_emb(self):
        """Returns the embedding (g2) dimension of this descriptor."""
        raise NotImplementedError

    def get_rcut(self):
        """Returns cutoff radius."""
        return self.rcut

    def get_sel(self):
        """Returns cutoff radius."""
        return self.sel

    def mixed_types(self):
        """Returns if the descriptor requires a neighbor list that distinguish different
        atomic types or not.
        """
        return False

    def share_params(self, base_class, shared_level, resume=False):
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some seperated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        raise NotImplementedError

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def compute_input_stats(self, merged: List[dict], path: Optional[DPPath] = None):
        """Update mean and stddev for descriptor elements."""
        raise NotImplementedError

    def cal_g(
        self,
        ss,
        ll,
    ):
        nf, nloc, nnei = ss.shape[0:3]
        ss = ss.reshape(nf, nloc, nnei, 1)
        # nf x nloc x nnei x ng
        gg = self.embeddings[(ll,)].call(ss)
        return gg

    def call(
        self,
        coord_ext,
        atype_ext,
        nlist,
        mapping: Optional[np.ndarray] = None,
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
            The index mapping from extended to lcoal region. not used by this descriptor.

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
        # nf x nloc x nnei x 1
        rr, ww = self.env_mat.call(
            coord_ext, atype_ext, nlist, self.davg, self.dstd, True
        )
        nf, nloc, nnei, _ = rr.shape
        sec = np.append([0], np.cumsum(self.sel))

        ng = self.neuron[-1]
        xyz_scatter = np.zeros([nf, nloc, ng], dtype=PRECISION_DICT[self.precision])
        exclude_mask = self.emask.build_type_exclude_mask(nlist, atype_ext)
        for tt in range(self.ntypes):
            mm = exclude_mask[:, :, sec[tt] : sec[tt + 1]]
            tr = rr[:, :, sec[tt] : sec[tt + 1], :]
            tr = tr * mm[:, :, :, None]
            gg = self.cal_g(tr, tt)
            gg = np.mean(gg, axis=2)
            # nf x nloc x ng x 1
            xyz_scatter += gg * (self.sel[tt] / self.nnei)

        res_rescale = 1.0 / 5.0
        res = xyz_scatter * res_rescale
        res = res.reshape(nf, nloc, -1).astype(GLOBAL_NP_FLOAT_PRECISION)
        return res, None, None, None, ww

    def serialize(self) -> dict:
        """Serialize the descriptor to dict."""
        return {
            "@class": "Descriptor",
            "type": "se_r",
            "@version": 1,
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel,
            "neuron": self.neuron,
            "resnet_dt": self.resnet_dt,
            "trainable": self.trainable,
            "type_one_side": self.type_one_side,
            "exclude_types": self.exclude_types,
            "set_davg_zero": self.set_davg_zero,
            "activation_function": self.activation_function,
            # make deterministic
            "precision": np.dtype(PRECISION_DICT[self.precision]).name,
            "spin": self.spin,
            "env_mat": self.env_mat.serialize(),
            "embeddings": self.embeddings.serialize(),
            "@variables": {
                "davg": self.davg,
                "dstd": self.dstd,
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptSeR":
        """Deserialize from dict."""
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        data.pop("type", None)
        variables = data.pop("@variables")
        embeddings = data.pop("embeddings")
        env_mat = data.pop("env_mat")
        obj = cls(**data)

        obj["davg"] = variables["davg"]
        obj["dstd"] = variables["dstd"]
        obj.embeddings = NetworkCollection.deserialize(embeddings)
        obj.env_mat = EnvMat.deserialize(env_mat)
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
