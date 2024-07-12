# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import itertools
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

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
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.dpmodel.utils.update_sel import (
    UpdateSel,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
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


@BaseDescriptor.register("se_e3")
@BaseDescriptor.register("se_at")
@BaseDescriptor.register("se_a_3be")
class DescrptSeT(NativeOP, BaseDescriptor):
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
    exclude_types : List[List[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    precision : str
            The precision of the embedding net parameters. Supported options are |PRECISION|
    trainable : bool
            If the weights of embedding net are trainable.
    seed : int, Optional
            Random seed for initializing the network parameters.
    type_map: List[str], Optional
            A list of strings. Give the name to each type of atoms.
    ntypes : int
            Number of element types.
            Not used in this descriptor, only to be compat with input.
    """

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: List[int],
        neuron: List[int] = [24, 48, 96],
        resnet_dt: bool = False,
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        env_protection: float = 0.0,
        exclude_types: List[Tuple[int, int]] = [],
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: Optional[Union[int, List[int]]] = None,
        type_map: Optional[List[str]] = None,
        ntypes: Optional[int] = None,  # to be compat with input
    ) -> None:
        del ntypes
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.sel = sel
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
        self.type_map = type_map
        # order matters, placed after the assignment of self.ntypes
        self.reinit_exclude(exclude_types)
        self.trainable = trainable

        in_dim = 1  # not considiering type embedding
        self.embeddings = NetworkCollection(
            ntypes=self.ntypes,
            ndim=2,
            network_type="embedding_network",
        )
        for ii, embedding_idx in enumerate(
            itertools.product(range(self.ntypes), repeat=self.embeddings.ndim)
        ):
            self.embeddings[embedding_idx] = EmbeddingNet(
                in_dim,
                self.neuron,
                self.activation_function,
                self.resnet_dt,
                self.precision,
                seed=child_seed(self.seed, ii),
            )
        self.env_mat = EnvMat(self.rcut, self.rcut_smth, protection=self.env_protection)
        self.nnei = np.sum(self.sel)
        self.davg = np.zeros(
            [self.ntypes, self.nnei, 4], dtype=PRECISION_DICT[self.precision]
        )
        self.dstd = np.ones(
            [self.ntypes, self.nnei, 4], dtype=PRECISION_DICT[self.precision]
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

    def change_type_map(
        self, type_map: List[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        raise NotImplementedError(
            "Descriptor se_e3 does not support changing for type related params!"
            "This feature is currently not implemented because it would require additional work to support the non-mixed-types case. "
            "We may consider adding this support in the future if there is a clear demand for it."
        )

    def get_dim_out(self):
        """Returns the output dimension of this descriptor."""
        return self.neuron[-1]

    def get_dim_emb(self):
        """Returns the embedding (g2) dimension of this descriptor."""
        return self.neuron[-1]

    def get_rcut(self):
        """Returns cutoff radius."""
        return self.rcut

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.rcut_smth

    def get_sel(self):
        """Returns cutoff radius."""
        return self.sel

    def mixed_types(self):
        """Returns if the descriptor requires a neighbor list that distinguish different
        atomic types or not.
        """
        return False

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor has message passing."""
        return False

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.env_protection

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

    def get_type_map(self) -> List[str]:
        """Get the name to each type of atoms."""
        return self.type_map

    def compute_input_stats(self, merged: List[dict], path: Optional[DPPath] = None):
        """Update mean and stddev for descriptor elements."""
        raise NotImplementedError

    def set_stat_mean_and_stddev(
        self,
        mean: np.ndarray,
        stddev: np.ndarray,
    ) -> None:
        """Update mean and stddev for descriptor."""
        self.davg = mean
        self.dstd = stddev

    def get_stat_mean_and_stddev(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get mean and stddev for descriptor."""
        return self.davg, self.dstd

    def reinit_exclude(
        self,
        exclude_types: List[Tuple[int, int]] = [],
    ):
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

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
        del mapping
        # nf x nloc x nnei x 4
        rr, diff, ww = self.env_mat.call(
            coord_ext, atype_ext, nlist, self.davg, self.dstd
        )
        nf, nloc, nnei, _ = rr.shape
        sec = np.append([0], np.cumsum(self.sel))

        ng = self.neuron[-1]
        result = np.zeros([nf * nloc, ng], dtype=PRECISION_DICT[self.precision])
        exclude_mask = self.emask.build_type_exclude_mask(nlist, atype_ext)
        # merge nf and nloc axis, so for type_one_side == False,
        # we don't require atype is the same in all frames
        exclude_mask = exclude_mask.reshape(nf * nloc, nnei)
        rr = rr.reshape(nf * nloc, nnei, 4)

        for embedding_idx in itertools.product(
            range(self.ntypes), repeat=self.embeddings.ndim
        ):
            ti, tj = embedding_idx
            nei_type_i = self.sel[ti]
            nei_type_j = self.sel[tj]
            if ti <= tj:
                # avoid repeat calculation
                # nfnl x nt_i x 3
                rr_i = rr[:, sec[ti] : sec[ti + 1], 1:]
                mm_i = exclude_mask[:, sec[ti] : sec[ti + 1]]
                rr_i = rr_i * mm_i[:, :, None]
                # nfnl x nt_j x 3
                rr_j = rr[:, sec[tj] : sec[tj + 1], 1:]
                mm_j = exclude_mask[:, sec[tj] : sec[tj + 1]]
                rr_j = rr_j * mm_j[:, :, None]
                # nfnl x nt_i x nt_j
                env_ij = np.einsum("ijm,ikm->ijk", rr_i, rr_j)
                # nfnl x nt_i x nt_j x 1
                env_ij_reshape = env_ij[:, :, :, None]
                # nfnl x nt_i x nt_j x ng
                gg = self.embeddings[embedding_idx].call(env_ij_reshape)
                # nfnl x nt_i x nt_j x ng
                res_ij = np.einsum("ijk,ijkm->im", env_ij, gg)
                res_ij = res_ij * (1.0 / float(nei_type_i) / float(nei_type_j))
                result += res_ij
        # nf x nloc x ng
        result = result.reshape(nf, nloc, ng).astype(GLOBAL_NP_FLOAT_PRECISION)
        return result, None, None, None, ww

    def serialize(self) -> dict:
        """Serialize the descriptor to dict."""
        for embedding_idx in itertools.product(range(self.ntypes), repeat=2):
            # not actually used; to match serilization data from TF to pass the test
            ti, tj = embedding_idx
            if (self.exclude_types and embedding_idx in self.emask) or tj < ti:
                self.embeddings[embedding_idx].clear()

        return {
            "@class": "Descriptor",
            "type": "se_e3",
            "@version": 2,
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel,
            "neuron": self.neuron,
            "resnet_dt": self.resnet_dt,
            "set_davg_zero": self.set_davg_zero,
            "activation_function": self.activation_function,
            "precision": np.dtype(PRECISION_DICT[self.precision]).name,
            "embeddings": self.embeddings.serialize(),
            "env_mat": self.env_mat.serialize(),
            "exclude_types": self.exclude_types,
            "env_protection": self.env_protection,
            "@variables": {
                "davg": self.davg,
                "dstd": self.dstd,
            },
            "type_map": self.type_map,
            "trainable": self.trainable,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptSeT":
        """Deserialize from dict."""
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 2, 1)
        data.pop("@class", None)
        data.pop("type", None)
        variables = data.pop("@variables")
        embeddings = data.pop("embeddings")
        env_mat = data.pop("env_mat")
        obj = cls(**data)

        obj["davg"] = variables["davg"]
        obj["dstd"] = variables["dstd"]
        obj.embeddings = NetworkCollection.deserialize(embeddings)
        return obj

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: Optional[List[str]],
        local_jdata: dict,
    ) -> Tuple[dict, Optional[float]]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statictics
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
