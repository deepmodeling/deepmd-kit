# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Optional,
    Union,
)

import torch

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.descriptor.descriptor import (
    DescriptorBlock,
)
from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.model.network.utils import (
    get_graph_index,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)
from deepmd.pt.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.pt.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.pt.utils.spin import (
    concat_switch_virtual,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
)
from deepmd.utils.env_mat_stat import (
    StatItem,
)
from deepmd.utils.path import (
    DPPath,
)

from .repflow_layer import (
    RepFlowLayer,
)

if not hasattr(torch.ops.deepmd, "border_op"):

    def border_op(
        argument0,
        argument1,
        argument2,
        argument3,
        argument4,
        argument5,
        argument6,
        argument7,
        argument8,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "border_op is not available since customized PyTorch OP library is not built when freezing the model. "
            "See documentation for DPA3 for details."
        )

    # Note: this hack cannot actually save a model that can be run using LAMMPS.
    torch.ops.deepmd.border_op = border_op


@DescriptorBlock.register("se_repflow")
class DescrptBlockRepflows(DescriptorBlock):
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
    use_loc_mapping : bool, Optional
        Whether to use local atom index mapping in training or non-parallel inference.
        When True, local indexing and mapping are applied to neighbor lists and embeddings during descriptor computation.
    optim_update : bool, optional
        Whether to enable the optimized update method.
        Uses a more efficient process when enabled. Defaults to True
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
        smooth_edge_update: bool = False,
        edge_init_use_dist: bool = False,
        use_exp_switch: bool = False,
        use_dynamic_sel: bool = False,
        sel_reduce_factor: float = 10.0,
        use_loc_mapping: bool = True,
        optim_update: bool = True,
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
        self.use_loc_mapping = use_loc_mapping
        self.optim_update = optim_update
        self.smooth_edge_update = smooth_edge_update
        self.edge_init_use_dist = edge_init_use_dist
        self.use_exp_switch = use_exp_switch
        self.use_dynamic_sel = use_dynamic_sel
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
        self.act = ActivationFn(activation_function)
        self.prec = PRECISION_DICT[precision]

        # order matters, placed after the assignment of self.ntypes
        self.reinit_exclude(exclude_types)
        self.env_protection = env_protection
        self.precision = precision
        self.epsilon = 1e-4
        self.seed = seed

        self.edge_embd = MLPLayer(
            1, self.e_dim, precision=precision, seed=child_seed(seed, 0)
        )
        self.angle_embd = MLPLayer(
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
        self.layers = torch.nn.ModuleList(layers)

        wanted_shape = (self.ntypes, self.nnei, 4)
        mean = torch.zeros(wanted_shape, dtype=self.prec, device=env.DEVICE)
        stddev = torch.ones(wanted_shape, dtype=self.prec, device=env.DEVICE)
        if self.set_stddev_constant:
            stddev = stddev * self.fix_stat_std
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.stats = None

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

    def reinit_exclude(
        self,
        exclude_types: list[tuple[int, int]] = [],
    ) -> None:
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_atype_embd: Optional[torch.Tensor] = None,
        mapping: Optional[torch.Tensor] = None,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ):
        parallel_mode = comm_dict is not None
        if not parallel_mode:
            assert mapping is not None
        nframes, nloc, nnei = nlist.shape
        nall = extended_coord.view(nframes, -1).shape[1] // 3
        atype = extended_atype[:, :nloc]
        # nb x nloc x nnei
        exclude_mask = self.emask(nlist, extended_atype)
        nlist = torch.where(exclude_mask != 0, nlist, -1)
        # nb x nloc x nnei x 4, nb x nloc x nnei x 3, nb x nloc x nnei x 1
        dmatrix, diff, sw = prod_env_mat(
            extended_coord,
            nlist,
            atype,
            self.mean,
            self.stddev,
            self.e_rcut,
            self.e_rcut_smth,
            protection=self.env_protection,
            use_exp_switch=self.use_exp_switch,
        )
        nlist_mask = nlist != -1
        sw = torch.squeeze(sw, -1)
        # beyond the cutoff sw should be 0.0
        sw = sw.masked_fill(~nlist_mask, 0.0)

        # get angle nlist (maybe smaller)
        a_dist_mask = (torch.linalg.norm(diff, dim=-1) < self.a_rcut)[
            :, :, : self.a_sel
        ]
        a_nlist = nlist[:, :, : self.a_sel]
        a_nlist = torch.where(a_dist_mask, a_nlist, -1)
        _, a_diff, a_sw = prod_env_mat(
            extended_coord,
            a_nlist,
            atype,
            self.mean[:, : self.a_sel],
            self.stddev[:, : self.a_sel],
            self.a_rcut,
            self.a_rcut_smth,
            protection=self.env_protection,
            use_exp_switch=self.use_exp_switch,
        )
        a_nlist_mask = a_nlist != -1
        a_sw = torch.squeeze(a_sw, -1)
        # beyond the cutoff sw should be 0.0
        a_sw = a_sw.masked_fill(~a_nlist_mask, 0.0)
        # set all padding positions to index of 0
        # if the a neighbor is real or not is indicated by nlist_mask
        nlist[nlist == -1] = 0
        a_nlist[a_nlist == -1] = 0

        # get node embedding
        # [nframes, nloc, tebd_dim]
        assert extended_atype_embd is not None
        atype_embd = extended_atype_embd[:, :nloc, :]
        assert list(atype_embd.shape) == [nframes, nloc, self.n_dim]
        assert isinstance(atype_embd, torch.Tensor)  # for jit
        node_ebd = self.act(atype_embd)
        n_dim = node_ebd.shape[-1]

        # get edge and angle embedding input
        # nb x nloc x nnei x 1,  nb x nloc x nnei x 3
        edge_input, h2 = torch.split(dmatrix, [1, 3], dim=-1)
        if self.edge_init_use_dist:
            # nb x nloc x nnei x 1
            edge_input = torch.linalg.norm(diff, dim=-1, keepdim=True)

        # nf x nloc x a_nnei x 3
        normalized_diff_i = a_diff / (
            torch.linalg.norm(a_diff, dim=-1, keepdim=True) + 1e-6
        )
        # nf x nloc x 3 x a_nnei
        normalized_diff_j = torch.transpose(normalized_diff_i, 2, 3)
        # nf x nloc x a_nnei x a_nnei
        # 1 - 1e-6 for torch.acos stability
        cosine_ij = torch.matmul(normalized_diff_i, normalized_diff_j) * (1 - 1e-6)
        angle_input = cosine_ij.unsqueeze(-1) / (torch.pi**0.5)

        if not parallel_mode and self.use_loc_mapping:
            assert mapping is not None
            # convert nlist from nall to nloc index
            nlist = torch.gather(
                mapping,
                1,
                index=nlist.reshape(nframes, -1),
            ).reshape(nlist.shape)
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
            a_nlist_mask = a_nlist_mask[:, :, :, None] & a_nlist_mask[:, :, None, :]
            # n_angle x 1
            angle_input = angle_input[a_nlist_mask]
            # n_angle x 1
            a_sw = (a_sw[:, :, :, None] * a_sw[:, :, None, :])[a_nlist_mask]
        else:
            # avoid jit assertion
            edge_index = angle_index = torch.zeros(
                [1, 3], device=nlist.device, dtype=nlist.dtype
            )
        # get edge and angle embedding
        # nb x nloc x nnei x e_dim [OR] n_edge x e_dim
        if not self.edge_init_use_dist:
            edge_ebd = self.act(self.edge_embd(edge_input))
        else:
            edge_ebd = self.edge_embd(edge_input)
        # nf x nloc x a_nnei x a_nnei x a_dim [OR] n_angle x a_dim
        angle_ebd = self.angle_embd(angle_input)

        # nb x nall x n_dim
        if not parallel_mode:
            assert mapping is not None
            mapping = (
                mapping.view(nframes, nall).unsqueeze(-1).expand(-1, -1, self.n_dim)
            )
        for idx, ll in enumerate(self.layers):
            # node_ebd:     nb x nloc x n_dim
            # node_ebd_ext: nb x nall x n_dim [OR] nb x nloc x n_dim when not parallel_mode
            if not parallel_mode:
                assert mapping is not None
                node_ebd_ext = (
                    torch.gather(node_ebd, 1, mapping)
                    if not self.use_loc_mapping
                    else node_ebd
                )
            else:
                assert comm_dict is not None
                has_spin = "has_spin" in comm_dict
                if not has_spin:
                    n_padding = nall - nloc
                    node_ebd = torch.nn.functional.pad(
                        node_ebd.squeeze(0), (0, 0, 0, n_padding), value=0.0
                    )
                    real_nloc = nloc
                    real_nall = nall
                else:
                    # for spin
                    real_nloc = nloc // 2
                    real_nall = nall // 2
                    real_n_padding = real_nall - real_nloc
                    node_ebd_real, node_ebd_virtual = torch.split(
                        node_ebd, [real_nloc, real_nloc], dim=1
                    )
                    # mix_node_ebd: nb x real_nloc x (n_dim * 2)
                    mix_node_ebd = torch.cat([node_ebd_real, node_ebd_virtual], dim=2)
                    # nb x real_nall x (n_dim * 2)
                    node_ebd = torch.nn.functional.pad(
                        mix_node_ebd.squeeze(0), (0, 0, 0, real_n_padding), value=0.0
                    )

                assert "send_list" in comm_dict
                assert "send_proc" in comm_dict
                assert "recv_proc" in comm_dict
                assert "send_num" in comm_dict
                assert "recv_num" in comm_dict
                assert "communicator" in comm_dict
                ret = torch.ops.deepmd.border_op(
                    comm_dict["send_list"],
                    comm_dict["send_proc"],
                    comm_dict["recv_proc"],
                    comm_dict["send_num"],
                    comm_dict["recv_num"],
                    node_ebd,
                    comm_dict["communicator"],
                    torch.tensor(
                        real_nloc,
                        dtype=torch.int32,
                        device=torch.device("cpu"),
                    ),  # should be int of c++, placed on cpu
                    torch.tensor(
                        real_nall - real_nloc,
                        dtype=torch.int32,
                        device=torch.device("cpu"),
                    ),  # should be int of c++, placed on cpu
                )
                node_ebd_ext = ret[0].unsqueeze(0)
                if has_spin:
                    node_ebd_real_ext, node_ebd_virtual_ext = torch.split(
                        node_ebd_ext, [n_dim, n_dim], dim=2
                    )
                    node_ebd_ext = concat_switch_virtual(
                        node_ebd_real_ext, node_ebd_virtual_ext, real_nloc
                    )
            node_ebd, edge_ebd, angle_ebd = ll.forward(
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
            RepFlowLayer._cal_hg(edge_ebd, h2, nlist_mask, sw)
            if not self.use_dynamic_sel
            else RepFlowLayer._cal_hg_dynamic(
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
        # (nb x nloc) x e_dim x 3
        rot_mat = torch.permute(h2g2, (0, 1, 3, 2))

        return node_ebd, edge_ebd, h2, rot_mat.view(nframes, nloc, self.dim_emb, 3), sw

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
        if self.set_stddev_constant and self.set_davg_zero:
            return
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
        if not self.set_stddev_constant:
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

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor block has message passing."""
        return True

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor block needs sorted nlist when using `forward_lower`."""
        return True
