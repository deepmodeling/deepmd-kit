# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

try:
    from deepmd_utils._version import version as __version__
except ImportError:
    __version__ = "unknown"

from typing import (
    Any,
    List,
    Optional,
)

from .common import (
    DEFAULT_PRECISION,
    NativeOP,
)
from .env_mat import (
    EnvMat,
)
from .network import (
    EmbeddingNet,
)


class DescrptSeA(NativeOP):
    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: List[str],
        neuron: List[int] = [24, 48, 96],
        axis_neuron: int = 8,
        resnet_dt: bool = False,
        trainable: bool = True,
        type_one_side: bool = True,
        exclude_types: List[List[int]] = [],
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        spin: Optional[Any] = None,
        stripped_type_embedding: bool = False,
    ) -> None:
        ## seed, uniform_seed, multi_task, not included.
        if not type_one_side:
            raise NotImplementedError("type_one_side == False not implemented")
        if stripped_type_embedding:
            raise NotImplementedError("stripped_type_embedding is not implemented")
        if exclude_types != []:
            raise NotImplementedError("exclude_types is not implemented")
        if spin is not None:
            raise NotImplementedError("spin is not implemented")

        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.sel = sel
        self.ntypes = len(self.sel)
        self.neuron = neuron
        self.axis_neuron = axis_neuron
        self.resnet_dt = resnet_dt
        self.trainable = trainable
        self.type_one_side = type_one_side
        self.exclude_types = exclude_types
        self.set_davg_zero = set_davg_zero
        self.activation_function = activation_function
        self.precision = precision
        self.spin = spin
        self.stripped_type_embedding = stripped_type_embedding

        in_dim = 1  # not considiering type embedding
        self.embeddings = []
        for ii in range(self.ntypes):
            self.embeddings.append(
                EmbeddingNet(
                    in_dim,
                    self.neuron,
                    self.activation_function,
                    self.resnet_dt,
                    self.precision,
                )
            )
        self.env_mat = EnvMat(self.rcut, self.rcut_smth)
        self.nnei = np.sum(self.sel)
        self.nneix4 = self.nnei * 4
        self.davg = np.zeros([self.ntypes, self.nneix4])
        self.dstd = np.ones([self.ntypes, self.nneix4])
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

    def cal_g(
        self,
        ss,
        ll,
    ):
        nf, nloc, nnei = ss.shape[0:3]
        ss = ss.reshape(nf, nloc, nnei, 1)
        # nf x nloc x nnei x ng
        gg = self.embeddings[ll].call(ss)
        return gg

    def call(
        self,
        coord_ext,
        atype_ext,
        nlist,
    ):
        """Compute the environment matrix.

        Parameters
        ----------
        coord_ext
            The extended coordinates of atoms. shape: nf x (nallx3)
        atype_ext
            The extended aotm types. shape: nf x nall
        nlist
            The neighbor list. shape: nf x nloc x nnei

        Returns
        -------
        descriptor
            The descriptor. shape: nf x nloc x ng x axis_neuron
        """
        # nf x nloc x nnei x 4
        rr, ww = self.env_mat.call(nlist, coord_ext, atype_ext, self.davg, self.dstd)
        nf, nloc, nnei, _ = rr.shape
        sec = np.append([0], np.cumsum(self.sel))

        ng = self.neuron[-1]
        gr = np.zeros([nf, nloc, ng, 4])
        for tt in range(self.ntypes):
            tr = rr[:, :, sec[tt] : sec[tt + 1], :]
            ss = tr[..., 0:1]
            gg = self.cal_g(ss, tt)
            # nf x nloc x ng x 4
            gr += np.einsum("flni,flnj->flij", gg, tr)
        gr /= self.nnei
        gr1 = gr[:, :, : self.axis_neuron, :]
        # nf x nloc x ng x ng1
        grrg = np.einsum("flid,fljd->flij", gr, gr1)
        # nf x nloc x (ng x ng1)
        grrg = grrg.reshape(nf, nloc, ng * self.axis_neuron)
        return grrg

    def serialize(self) -> dict:
        return {
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel,
            "neuron": self.neuron,
            "axis_neuron": self.axis_neuron,
            "resnet_dt": self.resnet_dt,
            "trainable": self.trainable,
            "type_one_side": self.type_one_side,
            "exclude_types": self.exclude_types,
            "set_davg_zero": self.set_davg_zero,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "spin": self.spin,
            "stripped_type_embedding": self.stripped_type_embedding,
            "env_mat": self.env_mat.serialize(),
            "embeddings": [ii.serialize() for ii in self.embeddings],
            "@variables": {
                "davg": self.davg,
                "dstd": self.dstd,
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptSeA":
        variables = data.pop("@variables")
        embeddings = data.pop("embeddings")
        env_mat = data.pop("env_mat")
        obj = cls(**data)

        obj["davg"] = variables["davg"]
        obj["dstd"] = variables["dstd"]
        obj.embeddings = [EmbeddingNet.deserialize(dd) for dd in embeddings]
        obj.env_mat = EnvMat.deserialize(env_mat)
        return obj
