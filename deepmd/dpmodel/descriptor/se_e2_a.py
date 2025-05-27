# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
from typing import (
    Any,
    Callable,
    NoReturn,
    Optional,
    Union,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.common import (
    cast_precision,
    to_numpy_array,
)
from deepmd.dpmodel.utils import (
    EmbeddingNet,
    EnvMat,
    NetworkCollection,
    PairExcludeMask,
)
from deepmd.dpmodel.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.dpmodel.utils.update_sel import (
    UpdateSel,
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


@BaseDescriptor.register("se_e2_a")
@BaseDescriptor.register("se_a")
class DescrptSeA(NativeOP, BaseDescriptor):
    r"""DeepPot-SE constructed from all information (both angular and radial) of
    atomic configurations. The embedding takes the distance between atoms as input.

    The descriptor :math:`\mathcal{D}^i \in \mathcal{R}^{M_1 \times M_2}` is given by [1]_

    .. math::
        \mathcal{D}^i = (\mathcal{G}^i)^T \mathcal{R}^i (\mathcal{R}^i)^T \mathcal{G}^i_<

    where :math:`\mathcal{R}^i \in \mathbb{R}^{N \times 4}` is the coordinate
    matrix, and each row of :math:`\mathcal{R}^i` can be constructed as follows

    .. math::
        (\mathcal{R}^i)_j = [
        \begin{array}{c}
            s(r_{ji}) & \frac{s(r_{ji})x_{ji}}{r_{ji}} & \frac{s(r_{ji})y_{ji}}{r_{ji}} & \frac{s(r_{ji})z_{ji}}{r_{ji}}
        \end{array}
        ]

    where :math:`\mathbf{R}_{ji}=\mathbf{R}_j-\mathbf{R}_i = (x_{ji}, y_{ji}, z_{ji})` is
    the relative coordinate and :math:`r_{ji}=\lVert \mathbf{R}_{ji} \lVert` is its norm.
    The switching function :math:`s(r)` is defined as:

    .. math::
        s(r)=
        \begin{cases}
        \frac{1}{r}, & r<r_s \\
        \frac{1}{r} \{ {(\frac{r - r_s}{ r_c - r_s})}^3 (-6 {(\frac{r - r_s}{ r_c - r_s})}^2 +15 \frac{r - r_s}{ r_c - r_s} -10) +1 \}, & r_s \leq r<r_c \\
        0, & r \geq r_c
        \end{cases}

    Each row of the embedding matrix  :math:`\mathcal{G}^i \in \mathbb{R}^{N \times M_1}` consists of outputs
    of a embedding network :math:`\mathcal{N}` of :math:`s(r_{ji})`:

    .. math::
        (\mathcal{G}^i)_j = \mathcal{N}(s(r_{ji}))

    :math:`\mathcal{G}^i_< \in \mathbb{R}^{N \times M_2}` takes first :math:`M_2` columns of
    :math:`\mathcal{G}^i`. The equation of embedding network :math:`\mathcal{N}` can be found at
    :meth:`deepmd.tf.utils.network.embedding_net`.

    Parameters
    ----------
    rcut
            The cut-off radius :math:`r_c`
    rcut_smth
            From where the environment matrix should be smoothed :math:`r_s`
    sel : list[int]
            sel[i] specifies the maxmum number of type i atoms in the cut-off radius
    neuron : list[int]
            Number of neurons in each hidden layers of the embedding net :math:`\mathcal{N}`
    axis_neuron
            Number of the axis neuron :math:`M_2` (number of columns of the sub-matrix of the embedding matrix)
    resnet_dt
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
    trainable
            If the weights of embedding net are trainable.
    type_one_side
            Try to build N_types embedding nets. Otherwise, building N_types^2 embedding nets
    exclude_types : list[list[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    env_protection: float
            Protection parameter to prevent division by zero errors during environment matrix calculations.
    set_davg_zero
            Set the shift of embedding net input to zero.
    activation_function
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision
            The precision of the embedding net parameters. Supported options are |PRECISION|
    spin
            The deepspin object.
    type_map: list[str], Optional
            A list of strings. Give the name to each type of atoms.
    ntypes : int
            Number of element types.
            Not used in this descriptor, only to be compat with input.

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
        sel: list[int],
        neuron: list[int] = [24, 48, 96],
        axis_neuron: int = 8,
        resnet_dt: bool = False,
        trainable: bool = True,
        type_one_side: bool = True,
        exclude_types: list[list[int]] = [],
        env_protection: float = 0.0,
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        spin: Optional[Any] = None,
        type_map: Optional[list[str]] = None,
        ntypes: Optional[int] = None,  # to be compat with input
        # consistent with argcheck, not used though
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        del ntypes
        ## seed, uniform_seed, not included.
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
        self.env_protection = env_protection
        self.set_davg_zero = set_davg_zero
        self.activation_function = activation_function
        self.precision = precision
        self.spin = spin
        self.type_map = type_map
        # order matters, placed after the assignment of self.ntypes
        self.reinit_exclude(exclude_types)

        in_dim = 1  # not considiering type embedding
        embeddings = NetworkCollection(
            ntypes=self.ntypes,
            ndim=(1 if self.type_one_side else 2),
            network_type="embedding_network",
        )
        for ii, embedding_idx in enumerate(
            itertools.product(range(self.ntypes), repeat=embeddings.ndim)
        ):
            embeddings[embedding_idx] = EmbeddingNet(
                in_dim,
                self.neuron,
                self.activation_function,
                self.resnet_dt,
                self.precision,
                seed=child_seed(seed, ii),
            )
        self.embeddings = embeddings
        self.env_mat = EnvMat(self.rcut, self.rcut_smth, protection=self.env_protection)
        self.nnei = np.sum(self.sel).item()
        self.davg = np.zeros(
            [self.ntypes, self.nnei, 4], dtype=PRECISION_DICT[self.precision]
        )
        self.dstd = np.ones(
            [self.ntypes, self.nnei, 4], dtype=PRECISION_DICT[self.precision]
        )
        self.orig_sel = self.sel
        self.sel_cumsum = [0, *np.cumsum(self.sel).tolist()]
        self.ndescrpt = self.nnei * 4

    def __setitem__(self, key, value) -> None:
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
        return self.neuron[-1] * self.axis_neuron

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

    def mixed_types(self) -> bool:
        """Returns if the descriptor requires a neighbor list that distinguish different
        atomic types or not.
        """
        return False

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor has message passing."""
        return False

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor needs sorted nlist when using `forward_lower`."""
        return False

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.env_protection

    def share_params(self, base_class, shared_level, resume=False) -> NoReturn:
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        raise NotImplementedError

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

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def get_type_map(self) -> list[str]:
        """Get the name to each type of atoms."""
        return self.type_map

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
                Each element, `merged[i]`, is a data dictionary containing `keys`: `paddle.Tensor`
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
        xp = array_api_compat.array_namespace(self.dstd)
        if not self.set_davg_zero:
            self.davg = xp.asarray(mean, dtype=self.davg.dtype, copy=True)
        self.dstd = xp.asarray(stddev, dtype=self.dstd.dtype, copy=True)

    def set_stat_mean_and_stddev(
        self,
        mean: np.ndarray,
        stddev: np.ndarray,
    ) -> None:
        """Update mean and stddev for descriptor."""
        self.davg = mean
        self.dstd = stddev

    def get_stat_mean_and_stddev(self) -> tuple[np.ndarray, np.ndarray]:
        """Get mean and stddev for descriptor."""
        return self.davg, self.dstd

    def cal_g(
        self,
        ss,
        embedding_idx,
    ):
        xp = array_api_compat.array_namespace(ss)
        nf_times_nloc, nnei = ss.shape[0:2]
        ss = xp.reshape(ss, (nf_times_nloc, nnei, 1))
        # (nf x nloc) x nnei x ng
        gg = self.embeddings[embedding_idx].call(ss)
        return gg

    def reinit_exclude(
        self,
        exclude_types: list[tuple[int, int]] = [],
    ) -> None:
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    @cast_precision
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
            The index mapping from extended to local region. not used by this descriptor.

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
        # nf x nloc x nnei x 4
        rr, diff, ww = self.env_mat.call(
            coord_ext, atype_ext, nlist, self.davg, self.dstd
        )
        nf, nloc, nnei, _ = rr.shape
        sec = np.append([0], np.cumsum(self.sel))

        ng = self.neuron[-1]
        gr = np.zeros([nf * nloc, ng, 4], dtype=PRECISION_DICT[self.precision])
        exclude_mask = self.emask.build_type_exclude_mask(nlist, atype_ext)
        # merge nf and nloc axis, so for type_one_side == False,
        # we don't require atype is the same in all frames
        exclude_mask = exclude_mask.reshape(nf * nloc, nnei)
        rr = rr.reshape(nf * nloc, nnei, 4)

        for embedding_idx in itertools.product(
            range(self.ntypes), repeat=self.embeddings.ndim
        ):
            if self.type_one_side:
                (tt,) = embedding_idx
                ti_mask = np.s_[:]
            else:
                ti, tt = embedding_idx
                ti_mask = atype_ext[:, :nloc].ravel() == ti
            mm = exclude_mask[ti_mask, sec[tt] : sec[tt + 1]]
            tr = rr[ti_mask, sec[tt] : sec[tt + 1], :]
            tr = tr * mm[:, :, None]
            ss = tr[..., 0:1]
            gg = self.cal_g(ss, embedding_idx)
            gr_tmp = np.einsum("lni,lnj->lij", gg, tr)
            gr[ti_mask] += gr_tmp
        gr = gr.reshape(nf, nloc, ng, 4)
        # nf x nloc x ng x 4
        gr /= self.nnei
        gr1 = gr[:, :, : self.axis_neuron, :]
        # nf x nloc x ng x ng1
        grrg = np.einsum("flid,fljd->flij", gr, gr1)
        # nf x nloc x (ng x ng1)
        grrg = grrg.reshape(nf, nloc, ng * self.axis_neuron)
        return grrg, gr[..., 1:], None, None, ww

    def serialize(self) -> dict:
        """Serialize the descriptor to dict."""
        if not self.type_one_side and self.exclude_types:
            for embedding_idx in itertools.product(range(self.ntypes), repeat=2):
                # not actually used; to match serilization data from TF to pass the test
                if embedding_idx in self.emask:
                    self.embeddings[embedding_idx].clear()

        return {
            "@class": "Descriptor",
            "type": "se_e2_a",
            "@version": 2,
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel,
            "neuron": self.neuron,
            "axis_neuron": self.axis_neuron,
            "resnet_dt": self.resnet_dt,
            "trainable": self.trainable,
            "type_one_side": self.type_one_side,
            "exclude_types": self.exclude_types,
            "env_protection": self.env_protection,
            "set_davg_zero": self.set_davg_zero,
            "activation_function": self.activation_function,
            # make deterministic
            "precision": np.dtype(PRECISION_DICT[self.precision]).name,
            "spin": self.spin,
            "env_mat": self.env_mat.serialize(),
            "embeddings": self.embeddings.serialize(),
            "@variables": {
                "davg": to_numpy_array(self.davg),
                "dstd": to_numpy_array(self.dstd),
            },
            "type_map": self.type_map,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptSeA":
        """Deserialize from dict."""
        data = data.copy()
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


class DescrptSeAArrayAPI(DescrptSeA):
    @cast_precision
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
            The index mapping from extended to local region. not used by this descriptor.

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
        if not self.type_one_side:
            raise NotImplementedError(
                "type_one_side == False is not supported in DescrptSeAArrayAPI"
            )
        del mapping
        xp = array_api_compat.array_namespace(coord_ext, atype_ext, nlist)
        input_dtype = coord_ext.dtype
        # nf x nloc x nnei x 4
        rr, diff, ww = self.env_mat.call(
            coord_ext,
            atype_ext,
            nlist,
            self.davg[...],
            self.dstd[...],
        )
        nf, nloc, nnei, _ = rr.shape
        sec = self.sel_cumsum

        ng = self.neuron[-1]
        gr = xp.zeros([nf * nloc, ng, 4], dtype=self.dstd.dtype)
        exclude_mask = self.emask.build_type_exclude_mask(nlist, atype_ext)
        # merge nf and nloc axis, so for type_one_side == False,
        # we don't require atype is the same in all frames
        exclude_mask = xp.reshape(exclude_mask, (nf * nloc, nnei))
        rr = xp.reshape(rr, (nf * nloc, nnei, 4))
        rr = xp.astype(rr, self.dstd.dtype)

        for embedding_idx in itertools.product(
            range(self.ntypes), repeat=self.embeddings.ndim
        ):
            (tt,) = embedding_idx
            mm = exclude_mask[:, sec[tt] : sec[tt + 1]]
            tr = rr[:, sec[tt] : sec[tt + 1], :]
            tr = tr * xp.astype(mm[:, :, None], tr.dtype)
            ss = tr[..., 0:1]
            gg = self.cal_g(ss, embedding_idx)
            # gr_tmp = xp.einsum("lni,lnj->lij", gg, tr)
            gr_tmp = xp.sum(gg[:, :, :, None] * tr[:, :, None, :], axis=1)
            gr += gr_tmp
        gr = xp.reshape(gr, (nf, nloc, ng, 4))
        # nf x nloc x ng x 4
        gr /= self.nnei
        gr1 = gr[:, :, : self.axis_neuron, :]
        # nf x nloc x ng x ng1
        # grrg = xp.einsum("flid,fljd->flij", gr, gr1)
        grrg = xp.sum(gr[:, :, :, None, :] * gr1[:, :, None, :, :], axis=4)
        # nf x nloc x (ng x ng1)
        grrg = xp.reshape(grrg, (nf, nloc, ng * self.axis_neuron))
        return grrg, gr[..., 1:], None, None, ww
