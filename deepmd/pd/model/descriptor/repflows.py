# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Optional,
    Union,
)

import paddle

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pd.model.descriptor.descriptor import (
    DescriptorBlock,
)
from deepmd.pd.model.descriptor.env_mat import (
    prod_env_mat,
)
from deepmd.pd.model.network.mlp import (
    MLPLayer,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.env import (
    PRECISION_DICT,
)
from deepmd.pd.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.pd.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.pd.utils.utils import (
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

# if not hasattr(paddle.ops.deepmd, "border_op"):

#     def border_op(
#         argument0,
#         argument1,
#         argument2,
#         argument3,
#         argument4,
#         argument5,
#         argument6,
#         argument7,
#         argument8,
#     ) -> paddle.Tensor:
#         raise NotImplementedError(
#             "border_op is not available since customized PyTorch OP library is not built when freezing the model. "
#             "See documentation for DPA-3 for details."
#         )

#     # Note: this hack cannot actually save a model that can be run using LAMMPS.
#     paddle.ops.deepmd.border_op = border_op


@DescriptorBlock.register("se_repflow")
class DescrptBlockRepflows(DescriptorBlock):
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
        skip_stat: bool = True,
        optim_update: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
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
        self.skip_stat = skip_stat
        self.a_compress_use_split = a_compress_use_split
        self.optim_update = optim_update

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
                    seed=child_seed(child_seed(seed, 1), ii),
                )
            )
        self.layers = paddle.nn.LayerList(layers)

        wanted_shape = (self.ntypes, self.nnei, 4)
        mean = paddle.zeros(wanted_shape, dtype=self.prec).to(device=env.DEVICE)
        stddev = paddle.ones(wanted_shape, dtype=self.prec).to(device=env.DEVICE)
        if self.skip_stat:
            stddev = stddev * 0.3
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
        nlist: paddle.Tensor,
        extended_coord: paddle.Tensor,
        extended_atype: paddle.Tensor,
        extended_atype_embd: Optional[paddle.Tensor] = None,
        mapping: Optional[paddle.Tensor] = None,
        comm_dict: Optional[dict[str, paddle.Tensor]] = None,
    ):
        if comm_dict is None:
            assert mapping is not None
            assert extended_atype_embd is not None
        nframes, nloc, nnei = nlist.shape
        nall = extended_coord.reshape([nframes, -1]).shape[1] // 3
        atype = extended_atype[:, :nloc]
        # nb x nloc x nnei
        exclude_mask = self.emask(nlist, extended_atype)
        nlist = paddle.where(exclude_mask != 0, nlist, -1)
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
        )
        nlist_mask = nlist != -1
        sw = paddle.squeeze(sw, -1)
        # beyond the cutoff sw should be 0.0
        sw = sw.masked_fill(~nlist_mask, 0.0)

        # [nframes, nloc, tebd_dim]
        if comm_dict is None:
            if paddle.in_dynamic_mode():
                assert isinstance(extended_atype_embd, paddle.Tensor)  # for jit
            atype_embd = extended_atype_embd[:, :nloc, :]
            if paddle.in_dynamic_mode():
                assert atype_embd.shape == [nframes, nloc, self.n_dim]
        else:
            atype_embd = extended_atype_embd
            if paddle.in_dynamic_mode():
                assert isinstance(atype_embd, paddle.Tensor)  # for jit
        node_ebd = self.act(atype_embd)
        n_dim = node_ebd.shape[-1]
        # nb x nloc x nnei x 1,  nb x nloc x nnei x 3
        edge_input, h2 = paddle.split(dmatrix, [1, 3], axis=-1)
        # nb x nloc x nnei x e_dim
        edge_ebd = self.act(self.edge_embd(edge_input))

        # get angle nlist (maybe smaller)
        a_dist_mask = (paddle.linalg.norm(diff, axis=-1) < self.a_rcut)[
            :, :, : self.a_sel
        ]
        a_nlist = nlist[:, :, : self.a_sel]
        a_nlist = paddle.where(a_dist_mask, a_nlist, -1)
        _, a_diff, a_sw = prod_env_mat(
            extended_coord,
            a_nlist,
            atype,
            self.mean[:, : self.a_sel],
            self.stddev[:, : self.a_sel],
            self.a_rcut,
            self.a_rcut_smth,
            protection=self.env_protection,
        )
        a_nlist_mask = a_nlist != -1
        a_sw = paddle.squeeze(a_sw, -1)
        # beyond the cutoff sw should be 0.0
        a_sw = a_sw.masked_fill(~a_nlist_mask, 0.0)
        a_nlist[a_nlist == -1] = 0

        # nf x nloc x a_nnei x 3
        normalized_diff_i = a_diff / (
            paddle.linalg.norm(a_diff, axis=-1, keepdim=True) + 1e-6
        )
        # nf x nloc x 3 x a_nnei
        normalized_diff_j = paddle.transpose(normalized_diff_i, [0, 1, 3, 2])
        # nf x nloc x a_nnei x a_nnei
        # 1 - 1e-6 for paddle.acos stability
        cosine_ij = paddle.matmul(normalized_diff_i, normalized_diff_j) * (1 - 1e-6)
        # nf x nloc x a_nnei x a_nnei x 1
        cosine_ij = cosine_ij.unsqueeze(-1) / (paddle.pi**0.5)
        # nf x nloc x a_nnei x a_nnei x a_dim
        angle_ebd = self.angle_embd(cosine_ij).reshape(
            [nframes, nloc, self.a_sel, self.a_sel, self.a_dim]
        )

        # set all padding positions to index of 0
        # if the a neighbor is real or not is indicated by nlist_mask
        nlist[nlist == -1] = 0
        # nb x nall x n_dim
        if comm_dict is None:
            assert mapping is not None
            mapping = (
                mapping.reshape([nframes, nall])
                .unsqueeze(-1)
                .expand([-1, -1, self.n_dim])
            )
        for idx, ll in enumerate(self.layers):
            # node_ebd:     nb x nloc x n_dim
            # node_ebd_ext: nb x nall x n_dim
            if comm_dict is None:
                assert mapping is not None
                node_ebd_ext = paddle.take_along_axis(node_ebd, mapping, 1)
            else:
                raise NotImplementedError("border_op is not supported in paddle yet")
                # has_spin = "has_spin" in comm_dict
                # if not has_spin:
                #     n_padding = nall - nloc
                #     node_ebd = paddle.nn.functional.pad(
                #         node_ebd.squeeze(0), (0, 0, 0, n_padding), value=0.0
                #     )
                #     real_nloc = nloc
                #     real_nall = nall
                # else:
                #     # for spin
                #     real_nloc = nloc // 2
                #     real_nall = nall // 2
                #     real_n_padding = real_nall - real_nloc
                #     node_ebd_real, node_ebd_virtual = paddle.split(
                #         node_ebd, [real_nloc, real_nloc], axis=1
                #     )
                #     # mix_node_ebd: nb x real_nloc x (n_dim * 2)
                #     mix_node_ebd = paddle.concat([node_ebd_real, node_ebd_virtual], axis=2)
                #     # nb x real_nall x (n_dim * 2)
                #     node_ebd = paddle.nn.functional.pad(
                #         mix_node_ebd.squeeze(0), (0, 0, 0, real_n_padding), value=0.0
                #     )

                # assert "send_list" in comm_dict
                # assert "send_proc" in comm_dict
                # assert "recv_proc" in comm_dict
                # assert "send_num" in comm_dict
                # assert "recv_num" in comm_dict
                # assert "communicator" in comm_dict
                # ret = paddle.ops.deepmd.border_op(
                #     comm_dict["send_list"],
                #     comm_dict["send_proc"],
                #     comm_dict["recv_proc"],
                #     comm_dict["send_num"],
                #     comm_dict["recv_num"],
                #     node_ebd,
                #     comm_dict["communicator"],
                #     paddle.to_tensor(
                #         real_nloc,
                #         dtype=paddle.int32,
                #         place=env.DEVICE,
                #     ),  # should be int of c++
                #     paddle.to_tensor(
                #         real_nall - real_nloc,
                #         dtype=paddle.int32,
                #         place=env.DEVICE,
                #     ),  # should be int of c++
                # )
                # node_ebd_ext = ret[0].unsqueeze(0)
                # if has_spin:
                #     node_ebd_real_ext, node_ebd_virtual_ext = paddle.split(
                #         node_ebd_ext, [n_dim, n_dim], axis=2
                #     )
                #     node_ebd_ext = concat_switch_virtual(
                #         node_ebd_real_ext, node_ebd_virtual_ext, real_nloc
                #     )
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
            )

        # nb x nloc x 3 x e_dim
        h2g2 = RepFlowLayer._cal_hg(edge_ebd, h2, nlist_mask, sw)
        # (nb x nloc) x e_dim x 3
        rot_mat = paddle.transpose(h2g2, (0, 1, 3, 2))

        return (
            node_ebd,
            edge_ebd,
            h2,
            rot_mat.reshape([nframes, nloc, self.dim_emb, 3]),
            sw,
        )

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
        if self.skip_stat and self.set_davg_zero:
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
            paddle.assign(
                paddle.to_tensor(mean, dtype=self.mean.dtype, place=env.DEVICE),
                self.mean,
            )
        paddle.assign(
            paddle.to_tensor(stddev, dtype=self.stddev.dtype, place=env.DEVICE),
            self.stddev,
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
