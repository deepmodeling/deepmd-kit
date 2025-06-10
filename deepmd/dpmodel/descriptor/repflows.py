# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Optional,
    Union,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    xp_take_along_axis,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils import (
    EnvMat,
    PairExcludeMask,
    aggregate,
    get_graph_index,
)
from deepmd.dpmodel.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.dpmodel.utils.network import (
    NativeLayer,
    get_activation_fn,
)
from deepmd.dpmodel.utils.safe_gradient import (
    safe_for_vector_norm,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
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

from .descriptor import (
    DescriptorBlock,
)
from .repformers import (
    _cal_grrg,
    _cal_hg,
    _make_nei_g1,
    get_residual,
    symmetrization_op,
)


@DescriptorBlock.register("se_repflow")
class DescrptBlockRepflows(NativeOP, DescriptorBlock):
    r"""
    The repflow descriptor block.

    Parameters
    ----------
    n_dim : int, optional
        The dimension of node representation.
    e_dim : int, optional
        The dimension of edge representation.
    a_dim : int, optional
        The dimension of angle representation.
    nlayers : int, optional
        Number of repflow layers.
    e_rcut : float, optional
        The edge cut-off radius.
    e_rcut_smth : float, optional
        Where to start smoothing for edge. For example the 1/r term is smoothed from rcut to rcut_smth.
    e_sel : int, optional
        Maximally possible number of selected edge neighbors.
    a_rcut : float, optional
        The angle cut-off radius.
    a_rcut_smth : float, optional
        Where to start smoothing for angle. For example the 1/r term is smoothed from rcut to rcut_smth.
    a_sel : int, optional
        Maximally possible number of selected angle neighbors.
    a_compress_rate : int, optional
        The compression rate for angular messages. The default value is 0, indicating no compression.
        If a non-zero integer c is provided, the node and edge dimensions will be compressed
        to a_dim/c and a_dim/2c, respectively, within the angular message.
    a_compress_e_rate : int, optional
        The extra compression rate for edge in angular message compression. The default value is 1.
        When using angular message compression with a_compress_rate c and a_compress_e_rate c_e,
        the edge dimension will be compressed to (c_e * a_dim / 2c) within the angular message.
    a_compress_use_split : bool, optional
        Whether to split first sub-vectors instead of linear mapping during angular message compression.
        The default value is False.
    n_multi_edge_message : int, optional
        The head number of multiple edge messages to update node feature.
        Default is 1, indicating one head edge message.
    axis_neuron : int, optional
        The number of dimension of submatrix in the symmetrization ops.
    update_angle : bool, optional
        Where to update the angle rep. If not, only node and edge rep will be used.
    update_style : str, optional
        Style to update a representation.
        Supported options are:
        -'res_avg': Updates a rep `u` with: u = 1/\\sqrt{n+1} (u + u_1 + u_2 + ... + u_n)
        -'res_incr': Updates a rep `u` with: u = u + 1/\\sqrt{n} (u_1 + u_2 + ... + u_n)
        -'res_residual': Updates a rep `u` with: u = u + (r1*u_1 + r2*u_2 + ... + r3*u_n)
        where `r1`, `r2` ... `r3` are residual weights defined by `update_residual`
        and `update_residual_init`.
    update_residual : float, optional
        When update using residual mode, the initial std of residual vector weights.
    update_residual_init : str, optional
        When update using residual mode, the initialization mode of residual vector weights.
    fix_stat_std : float, optional
        If non-zero (default is 0.3), use this constant as the normalization standard deviation
        instead of computing it from data statistics.
    optim_update : bool, optional
        Whether to enable the optimized update method.
        Uses a more efficient process when enabled. Defaults to True
    smooth_edge_update : bool, optional
        Whether to make edge update smooth.
        If True, the edge update from angle message will not use self as padding.
    edge_init_use_dist : bool, optional
        Whether to use direct distance r to initialize the edge features instead of 1/r.
        Note that when using this option, the activation function will not be used when initializing edge features.
    use_exp_switch : bool, optional
        Whether to use an exponential switch function instead of a polynomial one in the neighbor update.
        The exponential switch function ensures neighbor contributions smoothly diminish as the interatomic distance
        `r` approaches the cutoff radius `rcut`. Specifically, the function is defined as:
        s(r) = \\exp(-\\exp(20 * (r - rcut_smth) / rcut_smth)) for 0 < r \\leq rcut, and s(r) = 0 for r > rcut.
        Here, `rcut_smth` is an adjustable smoothing factor and `rcut_smth` should be chosen carefully
        according to `rcut`, ensuring s(r) approaches zero smoothly at the cutoff.
        Typical recommended values are `rcut_smth` = 5.3 for `rcut` = 6.0, and 3.5 for `rcut` = 4.0.
    use_dynamic_sel : bool, optional
        Whether to dynamically select neighbors within the cutoff radius.
        If True, the exact number of neighbors within the cutoff radius is used
        without padding to a fixed selection numbers.
        When enabled, users can safely set larger values for `e_sel` or `a_sel` (e.g., 1200 or 300, respectively)
        to guarantee capturing all neighbors within the cutoff radius.
        Note that when using dynamic selection, the `smooth_edge_update` must be True.
    sel_reduce_factor : float, optional
        Reduction factor applied to neighbor-scale normalization when `use_dynamic_sel` is True.
        In the dynamic selection case, neighbor-scale normalization will use `e_sel / sel_reduce_factor`
        or `a_sel / sel_reduce_factor` instead of the raw `e_sel` or `a_sel` values,
        accommodating larger selection numbers.
    use_loc_mapping : bool, optional
        Whether to use local atom index mapping in training or non-parallel inference.
        When True, local indexing and mapping are applied to neighbor lists and embeddings during descriptor computation.
    ntypes : int
        Number of element types
    activation_function : str, optional
        The activation function in the embedding net.
    set_davg_zero : bool, optional
        Set the normalization average to zero.
    precision : str, optional
        The precision of the embedding net parameters.
    exclude_types : list[list[int]], optional
        The excluded pairs of types which have no interaction with each other.
        For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    env_protection : float, optional
        Protection parameter to prevent division by zero errors during environment matrix calculations.
        For example, when using paddings, there may be zero distances of neighbors, which may make division by zero error during environment matrix calculations without protection.
    seed : int, optional
        Random seed for parameter initialization.
    """

    def __init__(
        self,
        e_rcut,
        e_rcut_smth,
        e_sel: int,
        a_rcut,
        a_rcut_smth,
        a_sel: int,
        ntypes: int,
        nlayers: int = 6,
        n_dim: int = 128,
        e_dim: int = 64,
        a_dim: int = 64,
        a_compress_rate: int = 0,
        a_compress_e_rate: int = 1,
        a_compress_use_split: bool = False,
        n_multi_edge_message: int = 1,
        axis_neuron: int = 4,
        update_angle: bool = True,
        activation_function: str = "silu",
        update_style: str = "res_residual",
        update_residual: float = 0.1,
        update_residual_init: str = "const",
        set_davg_zero: bool = True,
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        precision: str = "float64",
        fix_stat_std: float = 0.3,
        optim_update: bool = True,
        smooth_edge_update: bool = False,
        edge_init_use_dist: bool = False,
        use_exp_switch: bool = False,
        use_dynamic_sel: bool = False,
        sel_reduce_factor: float = 10.0,
        use_loc_mapping: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        super().__init__()
        self.e_rcut = float(e_rcut)
        self.e_rcut_smth = float(e_rcut_smth)
        self.e_sel = e_sel
        self.a_rcut = float(a_rcut)
        self.a_rcut_smth = float(a_rcut_smth)
        self.a_sel = a_sel
        self.ntypes = ntypes
        self.nlayers = nlayers
        # for other common desciptor method
        sel = [e_sel] if isinstance(e_sel, int) else e_sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 4  # use full descriptor.
        assert len(sel) == 1
        self.sel = sel
        self.rcut = e_rcut
        self.rcut_smth = e_rcut_smth
        self.sec = self.sel
        self.split_sel = self.sel
        self.a_compress_rate = a_compress_rate
        self.a_compress_e_rate = a_compress_e_rate
        self.n_multi_edge_message = n_multi_edge_message
        self.axis_neuron = axis_neuron
        self.set_davg_zero = set_davg_zero
        self.fix_stat_std = fix_stat_std
        self.set_stddev_constant = fix_stat_std != 0.0
        self.a_compress_use_split = a_compress_use_split
        self.optim_update = optim_update
        self.smooth_edge_update = smooth_edge_update
        self.edge_init_use_dist = edge_init_use_dist
        self.use_exp_switch = use_exp_switch
        self.use_dynamic_sel = use_dynamic_sel
        self.use_loc_mapping = use_loc_mapping
        self.sel_reduce_factor = sel_reduce_factor
        if self.use_dynamic_sel and not self.smooth_edge_update:
            raise NotImplementedError(
                "smooth_edge_update must be True when use_dynamic_sel is True!"
            )
        if self.sel_reduce_factor <= 0:
            raise ValueError(
                f"`sel_reduce_factor` must be > 0, got {self.sel_reduce_factor}"
            )

        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.update_angle = update_angle

        self.activation_function = activation_function
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.act = get_activation_fn(self.activation_function)
        self.prec = PRECISION_DICT[precision]

        # order matters, placed after the assignment of self.ntypes
        self.reinit_exclude(exclude_types)
        self.env_protection = env_protection
        self.precision = precision
        self.epsilon = 1e-4
        self.seed = seed

        self.edge_embd = NativeLayer(
            1, self.e_dim, precision=precision, seed=child_seed(seed, 0)
        )
        self.angle_embd = NativeLayer(
            1, self.a_dim, precision=precision, bias=False, seed=child_seed(seed, 1)
        )
        layers = []
        for ii in range(nlayers):
            layers.append(
                RepFlowLayer(
                    e_rcut=self.e_rcut,
                    e_rcut_smth=self.e_rcut_smth,
                    e_sel=self.sel,
                    a_rcut=self.a_rcut,
                    a_rcut_smth=self.a_rcut_smth,
                    a_sel=self.a_sel,
                    ntypes=self.ntypes,
                    n_dim=self.n_dim,
                    e_dim=self.e_dim,
                    a_dim=self.a_dim,
                    a_compress_rate=self.a_compress_rate,
                    a_compress_use_split=self.a_compress_use_split,
                    a_compress_e_rate=self.a_compress_e_rate,
                    n_multi_edge_message=self.n_multi_edge_message,
                    axis_neuron=self.axis_neuron,
                    update_angle=self.update_angle,
                    activation_function=self.activation_function,
                    update_style=self.update_style,
                    update_residual=self.update_residual,
                    update_residual_init=self.update_residual_init,
                    precision=precision,
                    optim_update=self.optim_update,
                    use_dynamic_sel=self.use_dynamic_sel,
                    sel_reduce_factor=self.sel_reduce_factor,
                    smooth_edge_update=self.smooth_edge_update,
                    seed=child_seed(child_seed(seed, 1), ii),
                )
            )
        self.layers = layers

        wanted_shape = (self.ntypes, self.nnei, 4)
        self.env_mat_edge = EnvMat(
            self.e_rcut,
            self.e_rcut_smth,
            protection=self.env_protection,
            use_exp_switch=self.use_exp_switch,
        )
        self.env_mat_angle = EnvMat(
            self.a_rcut,
            self.a_rcut_smth,
            protection=self.env_protection,
            use_exp_switch=self.use_exp_switch,
        )
        self.mean = np.zeros(wanted_shape, dtype=PRECISION_DICT[self.precision])
        self.stddev = np.ones(wanted_shape, dtype=PRECISION_DICT[self.precision])
        if self.set_stddev_constant:
            self.stddev = self.stddev * self.fix_stat_std

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.e_rcut

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.e_rcut_smth

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

    def get_dim_in(self) -> int:
        """Returns the input dimension."""
        return self.dim_in

    def get_dim_emb(self) -> int:
        """Returns the embedding dimension e_dim."""
        return self.e_dim

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

    def mixed_types(self) -> bool:
        """If true, the descriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the descriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return True

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.env_protection

    @property
    def dim_out(self):
        """Returns the output dimension of this descriptor."""
        return self.n_dim

    @property
    def dim_in(self):
        """Returns the atomic input dimension of this descriptor."""
        return self.n_dim

    @property
    def dim_emb(self):
        """Returns the embedding dimension e_dim."""
        return self.get_dim_emb()

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
        xp = array_api_compat.array_namespace(self.stddev)
        if not self.set_davg_zero:
            self.mean = xp.asarray(mean, dtype=self.mean.dtype, copy=True)
        self.stddev = xp.asarray(stddev, dtype=self.stddev.dtype, copy=True)

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

    def call(
        self,
        nlist: np.ndarray,
        coord_ext: np.ndarray,
        atype_ext: np.ndarray,
        atype_embd_ext: Optional[np.ndarray] = None,
        mapping: Optional[np.ndarray] = None,
    ):
        xp = array_api_compat.array_namespace(nlist, coord_ext, atype_ext)
        nframes, nloc, nnei = nlist.shape
        nall = xp.reshape(coord_ext, (nframes, -1)).shape[1] // 3
        exclude_mask = self.emask.build_type_exclude_mask(nlist, atype_ext)
        exclude_mask = xp.astype(exclude_mask, xp.bool)
        # nb x nloc x nnei
        nlist = xp.where(exclude_mask, nlist, xp.full_like(nlist, -1))
        # nb x nloc x nnei x 4, nb x nloc x nnei x 3, nb x nloc x nnei x 1
        dmatrix, diff, sw = self.env_mat_edge.call(
            coord_ext,
            atype_ext,
            nlist,
            self.mean[...],
            self.stddev[...],
        )
        # nb x nloc x nnei
        nlist_mask = nlist != -1
        sw = xp.reshape(sw, (nframes, nloc, nnei))
        # beyond the cutoff sw should be 0.0
        sw = xp.where(nlist_mask, sw, xp.zeros_like(sw))

        # get angle nlist (maybe smaller)
        a_dist_mask = (xp.linalg.vector_norm(diff, axis=-1) < self.a_rcut)[
            :, :, : self.a_sel
        ]
        a_nlist = nlist[:, :, : self.a_sel]
        a_nlist = xp.where(a_dist_mask, a_nlist, xp.full_like(a_nlist, -1))

        _, a_diff, a_sw = self.env_mat_angle.call(
            coord_ext,
            atype_ext,
            a_nlist,
            self.mean[:, : self.a_sel, :],
            self.stddev[:, : self.a_sel, :],
        )

        # nb x nloc x a_nnei
        a_nlist_mask = a_nlist != -1
        a_sw = xp.reshape(a_sw, (nframes, nloc, self.a_sel))
        # beyond the cutoff sw should be 0.0
        a_sw = xp.where(a_nlist_mask, a_sw, xp.zeros_like(a_sw))
        # set all padding positions to index of 0
        # if a neighbor is real or not is indicated by nlist_mask
        nlist = xp.where(nlist == -1, xp.zeros_like(nlist), nlist)
        a_nlist = xp.where(a_nlist == -1, xp.zeros_like(a_nlist), a_nlist)

        # get node embedding
        # nb x nloc x tebd_dim
        atype_embd = atype_embd_ext[:, :nloc, :]
        assert list(atype_embd.shape) == [nframes, nloc, self.n_dim]

        node_ebd = self.act(atype_embd)

        # get edge and angle embedding input
        # nb x nloc x nnei x 1,  nb x nloc x nnei x 3
        # edge_input, h2 = xp.split(dmatrix, [1], axis=-1)
        # nb x nloc x nnei x 1
        if self.edge_init_use_dist:
            edge_input = safe_for_vector_norm(diff, axis=-1, keepdims=True)
        else:
            edge_input = dmatrix[:, :, :, :1]
        h2 = dmatrix[:, :, :, 1:]

        # nf x nloc x a_nnei x 3
        normalized_diff_i = a_diff / (
            safe_for_vector_norm(a_diff, axis=-1, keepdims=True) + 1e-6
        )
        # nf x nloc x 3 x a_nnei
        normalized_diff_j = xp.matrix_transpose(normalized_diff_i)
        # nf x nloc x a_nnei x a_nnei
        # 1 - 1e-6 for torch.acos stability
        cosine_ij = xp.matmul(normalized_diff_i, normalized_diff_j) * (1 - 1e-6)
        # nf x nloc x a_nnei x a_nnei x 1
        angle_input = xp.reshape(
            cosine_ij, (nframes, nloc, self.a_sel, self.a_sel, 1)
        ) / (xp.pi**0.5)

        if self.use_loc_mapping:
            assert mapping is not None
            flat_map = xp.reshape(mapping, (nframes, -1))
            nlist = xp.reshape(
                xp_take_along_axis(flat_map, xp.reshape(nlist, (nframes, -1)), axis=1),
                nlist.shape,
            )

        if self.use_dynamic_sel:
            # get graph index
            edge_index, angle_index = get_graph_index(
                nlist,
                nlist_mask,
                a_nlist_mask,
                nall,
                use_loc_mapping=self.use_loc_mapping,
            )
            # flat all the tensors
            # n_edge x 1
            edge_input = edge_input[nlist_mask]
            # n_edge x 3
            h2 = h2[nlist_mask]
            # n_edge x 1
            sw = sw[nlist_mask]
            # nb x nloc x a_nnei x a_nnei
            a_nlist_mask = xp.logical_and(
                a_nlist_mask[:, :, :, None], a_nlist_mask[:, :, None, :]
            )
            # n_angle x 1
            angle_input = angle_input[a_nlist_mask]
            # n_angle x 1
            a_sw = (a_sw[:, :, :, None] * a_sw[:, :, None, :])[a_nlist_mask]
        else:
            edge_index = angle_index = xp.zeros([1, 3], dtype=nlist.dtype)

        # get edge and angle embedding
        # nb x nloc x nnei x e_dim [OR] n_edge x e_dim
        if not self.edge_init_use_dist:
            edge_ebd = self.act(self.edge_embd(edge_input))
        else:
            edge_ebd = self.edge_embd(edge_input)
        # nf x nloc x a_nnei x a_nnei x a_dim [OR] n_angle x a_dim
        angle_ebd = self.angle_embd(angle_input)

        # nb x nall x n_dim
        mapping = xp.tile(xp.reshape(mapping, (nframes, -1, 1)), (1, 1, self.n_dim))
        for idx, ll in enumerate(self.layers):
            # node_ebd:     nb x nloc x n_dim
            # node_ebd_ext: nb x nall x n_dim
            node_ebd_ext = (
                node_ebd
                if self.use_loc_mapping
                else xp_take_along_axis(node_ebd, mapping, axis=1)
            )
            node_ebd, edge_ebd, angle_ebd = ll.call(
                node_ebd_ext,
                edge_ebd,
                h2,
                angle_ebd,
                nlist,
                nlist_mask,
                sw,
                a_nlist,
                a_nlist_mask,
                a_sw,
                edge_index=edge_index,
                angle_index=angle_index,
            )

        # nb x nloc x 3 x e_dim
        h2g2 = (
            _cal_hg(edge_ebd, h2, nlist_mask, sw)
            if not self.use_dynamic_sel
            else _cal_hg_dynamic(
                edge_ebd,
                h2,
                sw,
                owner=edge_index[:, 0],
                num_owner=nframes * nloc,
                nb=nframes,
                nloc=nloc,
                scale_factor=(self.nnei / self.sel_reduce_factor) ** (-0.5),
            )
        )
        # nb x nloc x e_dim x 3
        rot_mat = xp.matrix_transpose(h2g2)

        return (
            node_ebd,
            edge_ebd,
            h2,
            xp.reshape(rot_mat, (nframes, nloc, self.dim_emb, 3)),
            sw,
        )

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor block has message passing."""
        return True

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor block needs sorted nlist when using `forward_lower`."""
        return True

    @classmethod
    def deserialize(cls, data):
        """Deserialize the descriptor block."""
        data = data.copy()
        edge_embd = NativeLayer.deserialize(data.pop("edge_embd"))
        angle_embd = NativeLayer.deserialize(data.pop("angle_embd"))
        layers = [RepFlowLayer.deserialize(dd) for dd in data.pop("repflow_layers")]
        env_mat_edge = EnvMat.deserialize(data.pop("env_mat_edge"))
        env_mat_angle = EnvMat.deserialize(data.pop("env_mat_angle"))
        variables = data.pop("@variables")
        davg = variables["davg"]
        dstd = variables["dstd"]
        obj = cls(**data)
        obj.edge_embd = edge_embd
        obj.angle_embd = angle_embd
        obj.layers = layers
        obj.env_mat_edge = env_mat_edge
        obj.env_mat_angle = env_mat_angle
        obj.mean = davg
        obj.stddev = dstd
        return obj

    def serialize(self):
        """Serialize the descriptor block."""
        return {
            "e_rcut": self.e_rcut,
            "e_rcut_smth": self.e_rcut_smth,
            "e_sel": self.e_sel,
            "a_rcut": self.a_rcut,
            "a_rcut_smth": self.a_rcut_smth,
            "a_sel": self.a_sel,
            "ntypes": self.ntypes,
            "nlayers": self.nlayers,
            "n_dim": self.n_dim,
            "e_dim": self.e_dim,
            "a_dim": self.a_dim,
            "a_compress_rate": self.a_compress_rate,
            "a_compress_e_rate": self.a_compress_e_rate,
            "a_compress_use_split": self.a_compress_use_split,
            "n_multi_edge_message": self.n_multi_edge_message,
            "axis_neuron": self.axis_neuron,
            "update_angle": self.update_angle,
            "activation_function": self.activation_function,
            "update_style": self.update_style,
            "update_residual": self.update_residual,
            "update_residual_init": self.update_residual_init,
            "set_davg_zero": self.set_davg_zero,
            "exclude_types": self.exclude_types,
            "env_protection": self.env_protection,
            "precision": self.precision,
            "fix_stat_std": self.fix_stat_std,
            "optim_update": self.optim_update,
            "edge_init_use_dist": self.edge_init_use_dist,
            "use_exp_switch": self.use_exp_switch,
            "smooth_edge_update": self.smooth_edge_update,
            "use_dynamic_sel": self.use_dynamic_sel,
            "sel_reduce_factor": self.sel_reduce_factor,
            "use_loc_mapping": self.use_loc_mapping,
            # variables
            "edge_embd": self.edge_embd.serialize(),
            "angle_embd": self.angle_embd.serialize(),
            "repflow_layers": [layer.serialize() for layer in self.layers],
            "env_mat_edge": self.env_mat_edge.serialize(),
            "env_mat_angle": self.env_mat_angle.serialize(),
            "@variables": {
                "davg": to_numpy_array(self["davg"]),
                "dstd": to_numpy_array(self["dstd"]),
            },
        }


def _cal_hg_dynamic(
    flat_edge_ebd: np.ndarray,
    flat_h2: np.ndarray,
    flat_sw: np.ndarray,
    owner: np.ndarray,
    num_owner: int,
    nb: int,
    nloc: int,
    scale_factor: float,
) -> np.ndarray:
    """
    Calculate the transposed rotation matrix.

    Parameters
    ----------
    flat_edge_ebd
        Flatted neighbor-wise/pair-wise invariant rep tensors, with shape n_edge x e_dim.
    flat_h2
        Flatted neighbor-wise/pair-wise equivariant rep tensors, with shape n_edge x 3.
    flat_sw
        Flatted switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
        and remains 0 beyond rcut, with shape n_edge.
    owner
        The owner index of the neighbor to reduce on.
    num_owner : int
        The total number of the owner.
    nb : int
        The number of batches.
    nloc : int
        The number of local atoms.
    scale_factor : float
        The scale factor to apply after reduce.

    Returns
    -------
    hg
        The transposed rotation matrix, with shape nf x nloc x 3 x e_dim.
    """
    xp = array_api_compat.array_namespace(flat_edge_ebd, flat_h2, flat_sw, owner)
    n_edge, e_dim = flat_edge_ebd.shape
    # n_edge x e_dim
    flat_edge_ebd = flat_edge_ebd * xp.expand_dims(flat_sw, axis=-1)
    # n_edge x 3 x e_dim
    flat_h2g2 = xp.reshape(
        (xp.expand_dims(flat_h2, axis=-1) * xp.expand_dims(flat_edge_ebd, axis=1)),
        (-1, 3 * e_dim),
    )

    # nf x nloc x 3 x e_dim
    h2g2 = aggregate(flat_h2g2, owner, average=False, num_owner=num_owner)
    h2g2 = xp.reshape(h2g2, (nb, nloc, 3, e_dim)) * scale_factor

    return h2g2


def symmetrization_op_dynamic(
    flat_edge_ebd: np.ndarray,
    flat_h2: np.ndarray,
    flat_sw: np.ndarray,
    owner: np.ndarray,
    num_owner: int,
    nb: int,
    nloc: int,
    scale_factor: float,
    axis_neuron: int,
) -> np.ndarray:
    """
    Symmetrization operator to obtain atomic invariant rep.

    Parameters
    ----------
    flat_edge_ebd
        Flatted neighbor-wise/pair-wise invariant rep tensors, with shape n_edge x e_dim.
    flat_h2
        Flatted neighbor-wise/pair-wise equivariant rep tensors, with shape n_edge x 3.
    flat_sw
        Flatted switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
        and remains 0 beyond rcut, with shape n_edge.
    owner
        The owner index of the neighbor to reduce on.
    num_owner : int
        The total number of the owner.
    nb : int
        The number of batches.
    nloc : int
        The number of local atoms.
    scale_factor : float
        The scale factor to apply after reduce.
    axis_neuron
        Size of the submatrix.

    Returns
    -------
    grrg
        Atomic invariant rep, with shape nb x nloc x (axis_neuron x e_dim)
    """
    # nb x nloc x 3 x e_dim
    h2g2 = _cal_hg_dynamic(
        flat_edge_ebd,
        flat_h2,
        flat_sw,
        owner,
        num_owner,
        nb,
        nloc,
        scale_factor,
    )
    # nb x nloc x (axis x e_dim)
    grrg = _cal_grrg(h2g2, axis_neuron)
    return grrg


class RepFlowLayer(NativeOP):
    def __init__(
        self,
        e_rcut: float,
        e_rcut_smth: float,
        e_sel: int,
        a_rcut: float,
        a_rcut_smth: float,
        a_sel: int,
        ntypes: int,
        n_dim: int = 128,
        e_dim: int = 16,
        a_dim: int = 64,
        a_compress_rate: int = 0,
        a_compress_use_split: bool = False,
        a_compress_e_rate: int = 1,
        n_multi_edge_message: int = 1,
        axis_neuron: int = 4,
        update_angle: bool = True,
        optim_update: bool = True,
        use_dynamic_sel: bool = False,
        sel_reduce_factor: float = 10.0,
        smooth_edge_update: bool = False,
        activation_function: str = "silu",
        update_style: str = "res_residual",
        update_residual: float = 0.1,
        update_residual_init: str = "const",
        precision: str = "float64",
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        super().__init__()
        self.epsilon = 1e-4  # protection of 1./nnei
        self.e_rcut = float(e_rcut)
        self.e_rcut_smth = float(e_rcut_smth)
        self.ntypes = ntypes
        e_sel = [e_sel] if isinstance(e_sel, int) else e_sel
        self.nnei = sum(e_sel)
        assert len(e_sel) == 1
        self.e_sel = e_sel
        self.sec = self.e_sel
        self.a_rcut = a_rcut
        self.a_rcut_smth = a_rcut_smth
        self.a_sel = a_sel
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.a_compress_rate = a_compress_rate
        if a_compress_rate != 0:
            assert (a_dim * a_compress_e_rate) % (2 * a_compress_rate) == 0, (
                f"For a_compress_rate of {a_compress_rate}, a_dim*a_compress_e_rate must be divisible by {2 * a_compress_rate}. "
                f"Currently, a_dim={a_dim} and a_compress_e_rate={a_compress_e_rate} is not valid."
            )
        self.n_multi_edge_message = n_multi_edge_message
        assert self.n_multi_edge_message >= 1, "n_multi_edge_message must >= 1!"
        self.axis_neuron = axis_neuron
        self.update_angle = update_angle
        self.activation_function = activation_function
        self.act = get_activation_fn(self.activation_function)
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.a_compress_e_rate = a_compress_e_rate
        self.a_compress_use_split = a_compress_use_split
        self.precision = precision
        self.seed = seed
        self.prec = PRECISION_DICT[precision]
        self.optim_update = optim_update
        self.smooth_edge_update = smooth_edge_update
        self.use_dynamic_sel = use_dynamic_sel
        self.sel_reduce_factor = sel_reduce_factor
        self.dynamic_e_sel = self.nnei / self.sel_reduce_factor
        self.dynamic_a_sel = self.a_sel / self.sel_reduce_factor

        assert update_residual_init in [
            "norm",
            "const",
        ], "'update_residual_init' only support 'norm' or 'const'!"

        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.n_residual = []
        self.e_residual = []
        self.a_residual = []
        self.edge_info_dim = self.n_dim * 2 + self.e_dim

        # node self mlp
        self.node_self_mlp = NativeLayer(
            n_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 0),
        )
        if self.update_style == "res_residual":
            self.n_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 1),
                )
            )

        # node sym (grrg + drrd)
        self.n_sym_dim = n_dim * self.axis_neuron + e_dim * self.axis_neuron
        self.node_sym_linear = NativeLayer(
            self.n_sym_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 2),
        )
        if self.update_style == "res_residual":
            self.n_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 3),
                )
            )

        # node edge message
        self.node_edge_linear = NativeLayer(
            self.edge_info_dim,
            self.n_multi_edge_message * n_dim,
            precision=precision,
            seed=child_seed(seed, 4),
        )
        if self.update_style == "res_residual":
            for head_index in range(self.n_multi_edge_message):
                self.n_residual.append(
                    get_residual(
                        n_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(child_seed(seed, 5), head_index),
                    )
                )

        # edge self message
        self.edge_self_linear = NativeLayer(
            self.edge_info_dim,
            e_dim,
            precision=precision,
            seed=child_seed(seed, 6),
        )
        if self.update_style == "res_residual":
            self.e_residual.append(
                get_residual(
                    e_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 7),
                )
            )

        if self.update_angle:
            self.angle_dim = self.a_dim
            if self.a_compress_rate == 0:
                # angle + node + edge * 2
                self.angle_dim += self.n_dim + 2 * self.e_dim
                self.a_compress_n_linear = None
                self.a_compress_e_linear = None
                self.e_a_compress_dim = e_dim
                self.n_a_compress_dim = n_dim
            else:
                # angle + a_dim/c + a_dim/2c * 2 * e_rate
                self.angle_dim += (1 + self.a_compress_e_rate) * (
                    self.a_dim // self.a_compress_rate
                )
                self.e_a_compress_dim = (
                    self.a_dim // (2 * self.a_compress_rate) * self.a_compress_e_rate
                )
                self.n_a_compress_dim = self.a_dim // self.a_compress_rate
                if not self.a_compress_use_split:
                    self.a_compress_n_linear = NativeLayer(
                        self.n_dim,
                        self.n_a_compress_dim,
                        precision=precision,
                        bias=False,
                        seed=child_seed(seed, 8),
                    )
                    self.a_compress_e_linear = NativeLayer(
                        self.e_dim,
                        self.e_a_compress_dim,
                        precision=precision,
                        bias=False,
                        seed=child_seed(seed, 9),
                    )
                else:
                    self.a_compress_n_linear = None
                    self.a_compress_e_linear = None

            # edge angle message
            self.edge_angle_linear1 = NativeLayer(
                self.angle_dim,
                self.e_dim,
                precision=precision,
                seed=child_seed(seed, 10),
            )
            self.edge_angle_linear2 = NativeLayer(
                self.e_dim,
                self.e_dim,
                precision=precision,
                seed=child_seed(seed, 11),
            )
            if self.update_style == "res_residual":
                self.e_residual.append(
                    get_residual(
                        self.e_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 12),
                    )
                )

            # angle self message
            self.angle_self_linear = NativeLayer(
                self.angle_dim,
                self.a_dim,
                precision=precision,
                seed=child_seed(seed, 13),
            )
            if self.update_style == "res_residual":
                self.a_residual.append(
                    get_residual(
                        self.a_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 14),
                    )
                )
        else:
            self.angle_self_linear = None
            self.edge_angle_linear1 = None
            self.edge_angle_linear2 = None
            self.a_compress_n_linear = None
            self.a_compress_e_linear = None
            self.angle_dim = 0

    def optim_angle_update(
        self,
        angle_ebd: np.ndarray,
        node_ebd: np.ndarray,
        edge_ebd: np.ndarray,
        feat: str = "edge",
    ) -> np.ndarray:
        xp = array_api_compat.array_namespace(angle_ebd, node_ebd, edge_ebd)

        if feat == "edge":
            assert self.edge_angle_linear1 is not None
            matrix, bias = self.edge_angle_linear1.w, self.edge_angle_linear1.b
        elif feat == "angle":
            assert self.angle_self_linear is not None
            matrix, bias = self.angle_self_linear.w, self.angle_self_linear.b
        else:
            raise NotImplementedError
        assert bias is not None

        angle_dim = angle_ebd.shape[-1]
        node_dim = node_ebd.shape[-1]
        edge_dim = edge_ebd.shape[-1]
        assert angle_dim + node_dim + 2 * edge_dim == matrix.shape[0]
        # Array API does not provide a way to split the array
        sub_angle = matrix[:angle_dim, ...]  # angle_dim
        sub_node = matrix[angle_dim : angle_dim + node_dim, ...]  # node_dim
        sub_edge_ik = matrix[
            angle_dim + node_dim : angle_dim + node_dim + edge_dim, ...
        ]  # edge_dim
        sub_edge_ij = matrix[angle_dim + node_dim + edge_dim :, ...]  # edge_dim

        # nf * nloc * a_sel * a_sel * angle_dim
        sub_angle_update = xp.matmul(angle_ebd, sub_angle)
        # nf * nloc * angle_dim
        sub_node_update = xp.matmul(node_ebd, sub_node)
        # nf * nloc * a_nnei * angle_dim
        sub_edge_update_ik = xp.matmul(edge_ebd, sub_edge_ik)
        sub_edge_update_ij = xp.matmul(edge_ebd, sub_edge_ij)

        result_update = (
            bias
            + sub_node_update[:, :, xp.newaxis, xp.newaxis, :]
            + sub_edge_update_ik[:, :, xp.newaxis, :, :]
            + sub_edge_update_ij[:, :, :, xp.newaxis, :]
            + sub_angle_update
        )
        return result_update

    def optim_angle_update_dynamic(
        self,
        flat_angle_ebd: np.ndarray,
        node_ebd: np.ndarray,
        flat_edge_ebd: np.ndarray,
        n2a_index: np.ndarray,
        eij2a_index: np.ndarray,
        eik2a_index: np.ndarray,
        feat="edge",
    ):
        xp = array_api_compat.array_namespace(
            flat_angle_ebd, node_ebd, flat_edge_ebd, n2a_index, eij2a_index, eik2a_index
        )

        if feat == "edge":
            matrix, bias = self.edge_angle_linear1.w, self.edge_angle_linear1.b
        elif feat == "angle":
            matrix, bias = self.angle_self_linear.w, self.angle_self_linear.b
        else:
            raise NotImplementedError
        assert bias is not None

        nf, nloc, node_dim = node_ebd.shape
        edge_dim = flat_edge_ebd.shape[-1]
        angle_dim = flat_angle_ebd.shape[-1]
        assert angle_dim + node_dim + 2 * edge_dim == matrix.shape[0]

        # Array API does not provide a way to split the array
        sub_angle = matrix[:angle_dim, ...]  # angle_dim
        sub_node = matrix[angle_dim : angle_dim + node_dim, ...]  # node_dim
        sub_edge_ik = matrix[
            angle_dim + node_dim : angle_dim + node_dim + edge_dim, ...
        ]  # edge_dim
        sub_edge_ij = matrix[angle_dim + node_dim + edge_dim :, ...]  # edge_dim

        sub_angle_update = xp.matmul(flat_angle_ebd, sub_angle)

        sub_node_update = xp.matmul(node_ebd, sub_node)
        sub_node_update = xp.take(
            xp.reshape(sub_node_update, (nf * nloc, sub_node_update.shape[-1])),
            n2a_index,
            axis=0,
        )

        sub_edge_update_ik = xp.matmul(flat_edge_ebd, sub_edge_ik)
        sub_edge_update_ij = xp.matmul(flat_edge_ebd, sub_edge_ij)

        sub_edge_update_ik = xp.take(sub_edge_update_ik, eik2a_index, axis=0)
        sub_edge_update_ij = xp.take(sub_edge_update_ij, eij2a_index, axis=0)

        result_update = (
            bias
            + sub_node_update
            + sub_edge_update_ik
            + sub_edge_update_ij
            + sub_angle_update
        )
        return result_update

    def optim_edge_update(
        self,
        node_ebd: np.ndarray,
        node_ebd_ext: np.ndarray,
        edge_ebd: np.ndarray,
        nlist: np.ndarray,
        feat: str = "node",
    ) -> np.ndarray:
        xp = array_api_compat.array_namespace(node_ebd, node_ebd_ext, edge_ebd, nlist)

        if feat == "node":
            matrix, bias = self.node_edge_linear.w, self.node_edge_linear.b
        elif feat == "edge":
            matrix, bias = self.edge_self_linear.w, self.edge_self_linear.b
        else:
            raise NotImplementedError
        node_dim = node_ebd.shape[-1]
        edge_dim = edge_ebd.shape[-1]
        assert node_dim * 2 + edge_dim == matrix.shape[0]
        # Array API does not provide a way to split the array
        node = matrix[:node_dim, ...]  # node_dim
        node_ext = matrix[node_dim : 2 * node_dim, ...]  # node_dim
        edge = matrix[2 * node_dim : 2 * node_dim + edge_dim, ...]  # edge_dim

        # nf * nloc * node/edge_dim
        sub_node_update = xp.matmul(node_ebd, node)

        # nf * nall * node/edge_dim
        sub_node_ext_update = xp.matmul(node_ebd_ext, node_ext)
        # nf * nloc * nnei * node/edge_dim
        sub_node_ext_update = _make_nei_g1(sub_node_ext_update, nlist)

        # nf * nloc * nnei * node/edge_dim
        sub_edge_update = xp.matmul(edge_ebd, edge)

        result_update = (
            bias
            + sub_node_update[:, :, xp.newaxis, :]
            + sub_edge_update
            + sub_node_ext_update
        )
        return result_update

    def optim_edge_update_dynamic(
        self,
        node_ebd: np.ndarray,
        node_ebd_ext: np.ndarray,
        flat_edge_ebd: np.ndarray,
        n2e_index: np.ndarray,
        n_ext2e_index: np.ndarray,
        feat: str = "node",
    ):
        xp = array_api_compat.array_namespace(
            node_ebd, node_ebd_ext, flat_edge_ebd, n2e_index, n_ext2e_index
        )

        if feat == "node":
            matrix, bias = self.node_edge_linear.w, self.node_edge_linear.b
        elif feat == "edge":
            matrix, bias = self.edge_self_linear.w, self.edge_self_linear.b
        else:
            raise NotImplementedError
        nf, nall, node_dim = node_ebd_ext.shape
        _, nloc, _ = node_ebd.shape
        edge_dim = flat_edge_ebd.shape[-1]
        assert node_dim * 2 + edge_dim == matrix.shape[0]
        # Array API does not provide a way to split the array
        node = matrix[:node_dim, ...]  # node_dim
        node_ext = matrix[node_dim : 2 * node_dim, ...]  # node_dim
        edge = matrix[2 * node_dim : 2 * node_dim + edge_dim, ...]  # edge_dim

        # nf * nloc * node/edge_dim
        sub_node_update = xp.matmul(node_ebd, node)
        sub_node_update = xp.take(
            xp.reshape(sub_node_update, (nf * nloc, sub_node_update.shape[-1])),
            n2e_index,
            axis=0,
        )

        sub_node_ext_update = xp.matmul(node_ebd_ext, node_ext)
        sub_node_ext_update = xp.take(
            xp.reshape(sub_node_ext_update, (nf * nall, sub_node_ext_update.shape[-1])),
            n_ext2e_index,
            axis=0,
        )

        sub_edge_update = xp.matmul(flat_edge_ebd, edge)

        result_update = bias + sub_node_update + sub_edge_update + sub_node_ext_update
        return result_update

    def call(
        self,
        node_ebd_ext: np.ndarray,  # nf x nall x n_dim
        edge_ebd: np.ndarray,  # nf x nloc x nnei x e_dim
        h2: np.ndarray,  # nf x nloc x nnei x 3
        angle_ebd: np.ndarray,  # nf x nloc x a_nnei x a_nnei x a_dim
        nlist: np.ndarray,  # nf x nloc x nnei
        nlist_mask: np.ndarray,  # nf x nloc x nnei
        sw: np.ndarray,  # switch func, nf x nloc x nnei
        a_nlist: np.ndarray,  # nf x nloc x a_nnei
        a_nlist_mask: np.ndarray,  # nf x nloc x a_nnei
        a_sw: np.ndarray,  # switch func, nf x nloc x a_nnei
        edge_index: np.ndarray,  # n_edge x 2
        angle_index: np.ndarray,  # n_angle x 3
    ):
        """
        Parameters
        ----------
        node_ebd_ext : nf x nall x n_dim
            Extended node embedding.
        edge_ebd : nf x nloc x nnei x e_dim
            Edge embedding.
        h2 : nf x nloc x nnei x 3
            Pair-atom channel, equivariant.
        angle_ebd : nf x nloc x a_nnei x a_nnei x a_dim
            Angle embedding.
        nlist : nf x nloc x nnei
            Neighbor list. (padded neis are set to 0)
        nlist_mask : nf x nloc x nnei
            Masks of the neighbor list. real nei 1 otherwise 0
        sw : nf x nloc x nnei
            Switch function.
        a_nlist : nf x nloc x a_nnei
            Neighbor list for angle. (padded neis are set to 0)
        a_nlist_mask : nf x nloc x a_nnei
            Masks of the neighbor list for angle. real nei 1 otherwise 0
        a_sw : nf x nloc x a_nnei
            Switch function for angle.
        edge_index : Optional for dynamic sel, n_edge x 2
            n2e_index : n_edge
                Broadcast indices from node(i) to edge(ij), or reduction indices from edge(ij) to node(i).
            n_ext2e_index : n_edge
                Broadcast indices from extended node(j) to edge(ij).
        angle_index : Optional for dynamic sel, n_angle x 3
            n2a_index : n_angle
                Broadcast indices from extended node(j) to angle(ijk).
            eij2a_index : n_angle
                Broadcast indices from extended edge(ij) to angle(ijk), or reduction indices from angle(ijk) to edge(ij).
            eik2a_index : n_angle
                Broadcast indices from extended edge(ik) to angle(ijk).

        Returns
        -------
        n_updated:     nf x nloc x n_dim
            Updated node embedding.
        e_updated:     nf x nloc x nnei x e_dim
            Updated edge embedding.
        a_updated : nf x nloc x a_nnei x a_nnei x a_dim
            Updated angle embedding.
        """
        xp = array_api_compat.array_namespace(
            node_ebd_ext,
            edge_ebd,
            h2,
            angle_ebd,
            nlist,
            nlist_mask,
            sw,
            a_nlist,
            a_nlist_mask,
            a_sw,
            edge_index,
            angle_index,
        )
        nb, nloc, nnei = nlist.shape
        nall = node_ebd_ext.shape[1]
        # int cannot jit; do not run it when self.use_dynamic_sel == False
        n_edge = (
            int(xp.sum(xp.astype(nlist_mask, xp.int32))) if self.use_dynamic_sel else 0
        )
        node_ebd = node_ebd_ext[:, :nloc, :]
        assert (nb, nloc) == node_ebd.shape[:2]
        if not self.use_dynamic_sel:
            assert (nb, nloc, nnei) == h2.shape[:3]
        else:
            assert (n_edge, 3) == h2.shape
        del a_nlist  # may be used in the future

        n2e_index, n_ext2e_index = edge_index[:, 0], edge_index[:, 1]
        n2a_index, eij2a_index, eik2a_index = (
            angle_index[:, 0],
            angle_index[:, 1],
            angle_index[:, 2],
        )

        # nb x nloc x nnei x n_dim [OR] n_edge x n_dim
        nei_node_ebd = (
            _make_nei_g1(node_ebd_ext, nlist)
            if not self.use_dynamic_sel
            else xp.take(
                xp.reshape(node_ebd_ext, (-1, self.n_dim)), n_ext2e_index, axis=0
            )
        )

        n_update_list: list[np.ndarray] = [node_ebd]
        e_update_list: list[np.ndarray] = [edge_ebd]
        a_update_list: list[np.ndarray] = [angle_ebd]

        # node self mlp
        node_self_mlp = self.act(self.node_self_mlp(node_ebd))
        n_update_list.append(node_self_mlp)

        # node sym (grrg + drrd)
        node_sym_list: list[np.ndarray] = []
        node_sym_list.append(
            symmetrization_op(
                edge_ebd,
                h2,
                nlist_mask,
                sw,
                self.axis_neuron,
            )
            if not self.use_dynamic_sel
            else symmetrization_op_dynamic(
                edge_ebd,
                h2,
                sw,
                owner=n2e_index,
                num_owner=nb * nloc,
                nb=nb,
                nloc=nloc,
                scale_factor=self.dynamic_e_sel ** (-0.5),
                axis_neuron=self.axis_neuron,
            )
        )
        node_sym_list.append(
            symmetrization_op(
                nei_node_ebd,
                h2,
                nlist_mask,
                sw,
                self.axis_neuron,
            )
            if not self.use_dynamic_sel
            else symmetrization_op_dynamic(
                nei_node_ebd,
                h2,
                sw,
                owner=n2e_index,
                num_owner=nb * nloc,
                nb=nb,
                nloc=nloc,
                scale_factor=self.dynamic_e_sel ** (-0.5),
                axis_neuron=self.axis_neuron,
            )
        )
        node_sym = self.act(self.node_sym_linear(xp.concat(node_sym_list, axis=-1)))
        n_update_list.append(node_sym)

        if not self.optim_update:
            if not self.use_dynamic_sel:
                # nb x nloc x nnei x (n_dim * 2 + e_dim)
                edge_info = xp.concat(
                    [
                        xp.tile(
                            xp.reshape(node_ebd, (nb, nloc, 1, self.n_dim)),
                            (1, 1, self.nnei, 1),
                        ),
                        nei_node_ebd,
                        edge_ebd,
                    ],
                    axis=-1,
                )
            else:
                # n_edge x (n_dim * 2 + e_dim)
                edge_info = xp.concat(
                    [
                        xp.take(
                            xp.reshape(node_ebd, (-1, self.n_dim)),
                            n2e_index,
                            axis=0,
                        ),
                        nei_node_ebd,
                        edge_ebd,
                    ],
                    axis=-1,
                )
        else:
            edge_info = None

        # node edge message
        # nb x nloc x nnei x (h * n_dim)
        if not self.optim_update:
            assert edge_info is not None
            node_edge_update = self.act(
                self.node_edge_linear(edge_info)
            ) * xp.expand_dims(sw, axis=-1)
        else:
            node_edge_update = self.act(
                self.optim_edge_update(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    nlist,
                    "node",
                )
                if not self.use_dynamic_sel
                else self.optim_edge_update_dynamic(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    n2e_index,
                    n_ext2e_index,
                    "node",
                )
            ) * xp.expand_dims(sw, axis=-1)

        node_edge_update = (
            (xp.sum(node_edge_update, axis=-2) / self.nnei)
            if not self.use_dynamic_sel
            else (
                xp.reshape(
                    aggregate(
                        node_edge_update,
                        n2e_index,
                        average=False,
                        num_owner=nb * nloc,
                    ),
                    (nb, nloc, node_edge_update.shape[-1]),
                )
                / self.dynamic_e_sel
            )
        )
        if self.n_multi_edge_message > 1:
            # nb x nloc x h x n_dim
            node_edge_update_mul_head = xp.reshape(
                node_edge_update, (nb, nloc, self.n_multi_edge_message, self.n_dim)
            )
            for head_index in range(self.n_multi_edge_message):
                n_update_list.append(node_edge_update_mul_head[:, :, head_index, :])
        else:
            n_update_list.append(node_edge_update)
        # update node_ebd
        n_updated = self.list_update(n_update_list, "node")

        # edge self message
        if not self.optim_update:
            assert edge_info is not None
            edge_self_update = self.act(self.edge_self_linear(edge_info))
        else:
            edge_self_update = self.act(
                self.optim_edge_update(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    nlist,
                    "edge",
                )
                if not self.use_dynamic_sel
                else self.optim_edge_update_dynamic(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    n2e_index,
                    n_ext2e_index,
                    "edge",
                )
            )
        e_update_list.append(edge_self_update)

        if self.update_angle:
            assert self.angle_self_linear is not None
            assert self.edge_angle_linear1 is not None
            assert self.edge_angle_linear2 is not None
            # get angle info
            if self.a_compress_rate != 0:
                if not self.a_compress_use_split:
                    assert self.a_compress_n_linear is not None
                    assert self.a_compress_e_linear is not None
                    node_ebd_for_angle = self.a_compress_n_linear(node_ebd)
                    edge_ebd_for_angle = self.a_compress_e_linear(edge_ebd)
                else:
                    # use the first a_compress_dim dim for node and edge
                    node_ebd_for_angle = node_ebd[..., : self.n_a_compress_dim]
                    edge_ebd_for_angle = edge_ebd[..., : self.e_a_compress_dim]
            else:
                node_ebd_for_angle = node_ebd
                edge_ebd_for_angle = edge_ebd

            if not self.use_dynamic_sel:
                # nb x nloc x a_nnei x e_dim
                edge_ebd_for_angle = edge_ebd_for_angle[..., : self.a_sel, :]
                # nb x nloc x a_nnei x e_dim
                edge_ebd_for_angle = xp.where(
                    xp.expand_dims(a_nlist_mask, axis=-1),
                    edge_ebd_for_angle,
                    xp.zeros_like(edge_ebd_for_angle),
                )
            if not self.optim_update:
                # nb x nloc x a_nnei x a_nnei x n_dim [OR] n_angle x n_dim
                node_for_angle_info = (
                    xp.tile(
                        xp.reshape(
                            node_ebd_for_angle, (nb, nloc, 1, 1, self.n_a_compress_dim)
                        ),
                        (1, 1, self.a_sel, self.a_sel, 1),
                    )
                    if not self.use_dynamic_sel
                    else xp.take(
                        xp.reshape(node_ebd_for_angle, (-1, self.n_a_compress_dim)),
                        n2a_index,
                        axis=0,
                    )
                )

                # nb x nloc x (a_nnei) x a_nnei x e_dim [OR] n_angle x e_dim
                edge_for_angle_k = (
                    xp.tile(
                        xp.reshape(
                            edge_ebd_for_angle,
                            (nb, nloc, 1, self.a_sel, self.e_a_compress_dim),
                        ),
                        (1, 1, self.a_sel, 1, 1),
                    )
                    if not self.use_dynamic_sel
                    else xp.take(
                        edge_ebd_for_angle,
                        eik2a_index,
                        axis=0,
                    )
                )
                # nb x nloc x a_nnei x (a_nnei) x e_dim
                edge_for_angle_j = (
                    xp.tile(
                        xp.reshape(
                            edge_ebd_for_angle,
                            (nb, nloc, self.a_sel, 1, self.e_a_compress_dim),
                        ),
                        (1, 1, 1, self.a_sel, 1),
                    )
                    if not self.use_dynamic_sel
                    else xp.take(
                        edge_ebd_for_angle,
                        eij2a_index,
                        axis=0,
                    )
                )
                # nb x nloc x a_nnei x a_nnei x (e_dim + e_dim)
                edge_for_angle_info = xp.concat(
                    [edge_for_angle_k, edge_for_angle_j], axis=-1
                )
                angle_info_list = [angle_ebd]
                angle_info_list.append(node_for_angle_info)
                angle_info_list.append(edge_for_angle_info)
                # nb x nloc x a_nnei x a_nnei x (a + n_dim + e_dim*2) or (a + a/c + a/c)
                # [OR]
                # n_angle x (a + n_dim + e_dim*2) or (a + a/c + a/c)
                angle_info = xp.concat(angle_info_list, axis=-1)
            else:
                angle_info = None

            # edge angle message
            # nb x nloc x a_nnei x a_nnei x e_dim [OR] n_angle x e_dim
            if not self.optim_update:
                assert angle_info is not None
                edge_angle_update = self.act(self.edge_angle_linear1(angle_info))
            else:
                edge_angle_update = self.act(
                    self.optim_angle_update(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_ebd_for_angle,
                        "edge",
                    )
                    if not self.use_dynamic_sel
                    else self.optim_angle_update_dynamic(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_ebd_for_angle,
                        n2a_index,
                        eij2a_index,
                        eik2a_index,
                        "edge",
                    )
                )
            if not self.use_dynamic_sel:
                # nb x nloc x a_nnei x a_nnei x e_dim
                weighted_edge_angle_update = (
                    a_sw[:, :, :, xp.newaxis, xp.newaxis]
                    * a_sw[:, :, xp.newaxis, :, xp.newaxis]
                    * edge_angle_update
                )
                # nb x nloc x a_nnei x e_dim
                reduced_edge_angle_update = xp.sum(
                    weighted_edge_angle_update, axis=-2
                ) / (self.a_sel**0.5)
                # nb x nloc x nnei x e_dim
                padding_edge_angle_update = xp.concat(
                    [
                        reduced_edge_angle_update,
                        xp.zeros(
                            (nb, nloc, self.nnei - self.a_sel, self.e_dim),
                            dtype=edge_ebd.dtype,
                        ),
                    ],
                    axis=2,
                )
            else:
                # n_angle x e_dim
                weighted_edge_angle_update = edge_angle_update * xp.expand_dims(
                    a_sw, axis=-1
                )
                # n_edge x e_dim
                padding_edge_angle_update = aggregate(
                    weighted_edge_angle_update,
                    eij2a_index,
                    average=False,
                    num_owner=n_edge,
                ) / (self.dynamic_a_sel**0.5)

            if not self.smooth_edge_update:
                # will be deprecated in the future
                # not support dynamic index, will pass anyway
                if self.use_dynamic_sel:
                    raise NotImplementedError(
                        "smooth_edge_update must be True when use_dynamic_sel is True!"
                    )
                full_mask = xp.concat(
                    [
                        a_nlist_mask,
                        xp.zeros(
                            (nb, nloc, self.nnei - self.a_sel),
                            dtype=a_nlist_mask.dtype,
                        ),
                    ],
                    axis=-1,
                )
                padding_edge_angle_update = xp.where(
                    xp.expand_dims(full_mask, axis=-1),
                    padding_edge_angle_update,
                    edge_ebd,
                )
            e_update_list.append(
                self.act(self.edge_angle_linear2(padding_edge_angle_update))
            )
            # update edge_ebd
            e_updated = self.list_update(e_update_list, "edge")

            # angle self message
            # nb x nloc x a_nnei x a_nnei x dim_a
            if not self.optim_update:
                assert angle_info is not None
                angle_self_update = self.act(self.angle_self_linear(angle_info))
            else:
                angle_self_update = self.act(
                    self.optim_angle_update(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_ebd_for_angle,
                        "angle",
                    )
                    if not self.use_dynamic_sel
                    else self.optim_angle_update_dynamic(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_ebd_for_angle,
                        n2a_index,
                        eij2a_index,
                        eik2a_index,
                        "angle",
                    )
                )
            a_update_list.append(angle_self_update)
        else:
            # update edge_ebd
            e_updated = self.list_update(e_update_list, "edge")

        # update angle_ebd
        a_updated = self.list_update(a_update_list, "angle")
        return n_updated, e_updated, a_updated

    def list_update_res_avg(
        self,
        update_list: list[np.ndarray],
    ) -> np.ndarray:
        nitem = len(update_list)
        uu = update_list[0]
        for ii in range(1, nitem):
            uu = uu + update_list[ii]
        return uu / (float(nitem) ** 0.5)

    def list_update_res_incr(self, update_list: list[np.ndarray]) -> np.ndarray:
        nitem = len(update_list)
        uu = update_list[0]
        scale = 1.0 / (float(nitem - 1) ** 0.5) if nitem > 1 else 0.0
        for ii in range(1, nitem):
            uu = uu + scale * update_list[ii]
        return uu

    def list_update_res_residual(
        self, update_list: list[np.ndarray], update_name: str = "node"
    ) -> np.ndarray:
        nitem = len(update_list)
        uu = update_list[0]
        if update_name == "node":
            for ii, vv in enumerate(self.n_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "edge":
            for ii, vv in enumerate(self.e_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "angle":
            for ii, vv in enumerate(self.a_residual):
                uu = uu + vv * update_list[ii + 1]
        else:
            raise NotImplementedError
        return uu

    def list_update(
        self, update_list: list[np.ndarray], update_name: str = "node"
    ) -> np.ndarray:
        if self.update_style == "res_avg":
            return self.list_update_res_avg(update_list)
        elif self.update_style == "res_incr":
            return self.list_update_res_incr(update_list)
        elif self.update_style == "res_residual":
            return self.list_update_res_residual(update_list, update_name=update_name)
        else:
            raise RuntimeError(f"unknown update style {self.update_style}")

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        data = {
            "@class": "RepFlowLayer",
            "@version": 2,
            "e_rcut": self.e_rcut,
            "e_rcut_smth": self.e_rcut_smth,
            "e_sel": self.e_sel,
            "a_rcut": self.a_rcut,
            "a_rcut_smth": self.a_rcut_smth,
            "a_sel": self.a_sel,
            "ntypes": self.ntypes,
            "n_dim": self.n_dim,
            "e_dim": self.e_dim,
            "a_dim": self.a_dim,
            "a_compress_rate": self.a_compress_rate,
            "a_compress_e_rate": self.a_compress_e_rate,
            "a_compress_use_split": self.a_compress_use_split,
            "n_multi_edge_message": self.n_multi_edge_message,
            "axis_neuron": self.axis_neuron,
            "activation_function": self.activation_function,
            "update_angle": self.update_angle,
            "update_style": self.update_style,
            "update_residual": self.update_residual,
            "update_residual_init": self.update_residual_init,
            "precision": self.precision,
            "optim_update": self.optim_update,
            "smooth_edge_update": self.smooth_edge_update,
            "use_dynamic_sel": self.use_dynamic_sel,
            "sel_reduce_factor": self.sel_reduce_factor,
            "node_self_mlp": self.node_self_mlp.serialize(),
            "node_sym_linear": self.node_sym_linear.serialize(),
            "node_edge_linear": self.node_edge_linear.serialize(),
            "edge_self_linear": self.edge_self_linear.serialize(),
        }
        if self.update_angle:
            data.update(
                {
                    "edge_angle_linear1": self.edge_angle_linear1.serialize(),
                    "edge_angle_linear2": self.edge_angle_linear2.serialize(),
                    "angle_self_linear": self.angle_self_linear.serialize(),
                }
            )
            if self.a_compress_rate != 0 and not self.a_compress_use_split:
                data.update(
                    {
                        "a_compress_n_linear": self.a_compress_n_linear.serialize(),
                        "a_compress_e_linear": self.a_compress_e_linear.serialize(),
                    }
                )
        if self.update_style == "res_residual":
            data.update(
                {
                    "@variables": {
                        "n_residual": [to_numpy_array(t) for t in self.n_residual],
                        "e_residual": [to_numpy_array(t) for t in self.e_residual],
                        "a_residual": [to_numpy_array(t) for t in self.a_residual],
                    }
                }
            )
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "RepFlowLayer":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 2, 1)
        data.pop("@class")
        update_angle = data["update_angle"]
        a_compress_rate = data["a_compress_rate"]
        a_compress_use_split = data["a_compress_use_split"]
        node_self_mlp = data.pop("node_self_mlp")
        node_sym_linear = data.pop("node_sym_linear")
        node_edge_linear = data.pop("node_edge_linear")
        edge_self_linear = data.pop("edge_self_linear")
        edge_angle_linear1 = data.pop("edge_angle_linear1", None)
        edge_angle_linear2 = data.pop("edge_angle_linear2", None)
        angle_self_linear = data.pop("angle_self_linear", None)
        a_compress_n_linear = data.pop("a_compress_n_linear", None)
        a_compress_e_linear = data.pop("a_compress_e_linear", None)
        update_style = data["update_style"]
        variables = data.pop("@variables", {})
        n_residual = variables.get("n_residual", data.pop("n_residual", []))
        e_residual = variables.get("e_residual", data.pop("e_residual", []))
        a_residual = variables.get("a_residual", data.pop("a_residual", []))

        obj = cls(**data)
        obj.node_self_mlp = NativeLayer.deserialize(node_self_mlp)
        obj.node_sym_linear = NativeLayer.deserialize(node_sym_linear)
        obj.node_edge_linear = NativeLayer.deserialize(node_edge_linear)
        obj.edge_self_linear = NativeLayer.deserialize(edge_self_linear)

        if update_angle:
            assert isinstance(edge_angle_linear1, dict)
            assert isinstance(edge_angle_linear2, dict)
            assert isinstance(angle_self_linear, dict)
            obj.edge_angle_linear1 = NativeLayer.deserialize(edge_angle_linear1)
            obj.edge_angle_linear2 = NativeLayer.deserialize(edge_angle_linear2)
            obj.angle_self_linear = NativeLayer.deserialize(angle_self_linear)
            if a_compress_rate != 0 and not a_compress_use_split:
                assert isinstance(a_compress_n_linear, dict)
                assert isinstance(a_compress_e_linear, dict)
                obj.a_compress_n_linear = NativeLayer.deserialize(a_compress_n_linear)
                obj.a_compress_e_linear = NativeLayer.deserialize(a_compress_e_linear)

        if update_style == "res_residual":
            obj.n_residual = n_residual
            obj.e_residual = e_residual
            obj.a_residual = a_residual
        return obj
