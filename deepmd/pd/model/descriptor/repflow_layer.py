# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Union,
)

import paddle
import paddle.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pd.model.descriptor.repformer_layer import (
    _apply_nlist_mask,
    _apply_switch,
    _make_nei_g1,
    get_residual,
)
from deepmd.pd.model.network.mlp import (
    MLPLayer,
)
from deepmd.pd.utils.env import (
    PRECISION_DICT,
)
from deepmd.pd.utils.utils import (
    ActivationFn,
    to_numpy_array,
    to_paddle_tensor,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class RepFlowLayer(paddle.nn.Layer):
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
        self.act = ActivationFn(activation_function)
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
        self.node_self_mlp = MLPLayer(
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
        self.node_sym_linear = MLPLayer(
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
        self.node_edge_linear = MLPLayer(
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
        self.edge_self_linear = MLPLayer(
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
                    self.a_compress_n_linear = MLPLayer(
                        self.n_dim,
                        self.n_a_compress_dim,
                        precision=precision,
                        bias=False,
                        seed=child_seed(seed, 8),
                    )
                    self.a_compress_e_linear = MLPLayer(
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
            self.edge_angle_linear1 = MLPLayer(
                self.angle_dim,
                self.e_dim,
                precision=precision,
                seed=child_seed(seed, 10),
            )
            self.edge_angle_linear2 = MLPLayer(
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
            self.angle_self_linear = MLPLayer(
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

        self.n_residual = nn.ParameterList(self.n_residual)
        self.e_residual = nn.ParameterList(self.e_residual)
        self.a_residual = nn.ParameterList(self.a_residual)

    @staticmethod
    def _cal_hg(
        edge_ebd: paddle.Tensor,
        h2: paddle.Tensor,
        nlist_mask: paddle.Tensor,
        sw: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Calculate the transposed rotation matrix.

        Parameters
        ----------
        edge_ebd
            Neighbor-wise/Pair-wise edge embeddings, with shape nb x nloc x nnei x e_dim.
        h2
            Neighbor-wise/Pair-wise equivariant rep tensors, with shape nb x nloc x nnei x 3.
        nlist_mask
            Neighbor list mask, where zero means no neighbor, with shape nb x nloc x nnei.
        sw
            The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape nb x nloc x nnei.

        Returns
        -------
        hg
            The transposed rotation matrix, with shape nb x nloc x 3 x e_dim.
        """
        # edge_ebd:  nb x nloc x nnei x e_dim
        # h2:  nb x nloc x nnei x 3
        # msk: nb x nloc x nnei
        nb, nloc, nnei, _ = edge_ebd.shape
        e_dim = edge_ebd.shape[-1]
        # nb x nloc x nnei x e_dim
        edge_ebd = _apply_nlist_mask(edge_ebd, nlist_mask)
        edge_ebd = _apply_switch(edge_ebd, sw)
        invnnei = paddle.rsqrt(
            float(nnei)
            * paddle.ones((nb, nloc, 1, 1), dtype=edge_ebd.dtype).to(
                device=edge_ebd.place
            )
        )
        # nb x nloc x 3 x e_dim
        h2g2 = paddle.matmul(paddle.matrix_transpose(h2), edge_ebd) * invnnei
        return h2g2

    @staticmethod
    def _cal_grrg(h2g2: paddle.Tensor, axis_neuron: int) -> paddle.Tensor:
        """
        Calculate the atomic invariant rep.

        Parameters
        ----------
        h2g2
            The transposed rotation matrix, with shape nb x nloc x 3 x e_dim.
        axis_neuron
            Size of the submatrix.

        Returns
        -------
        grrg
            Atomic invariant rep, with shape nb x nloc x (axis_neuron x e_dim)
        """
        # nb x nloc x 3 x e_dim
        nb, nloc, _, e_dim = h2g2.shape
        # nb x nloc x 3 x axis
        h2g2m = h2g2[..., :axis_neuron]
        # nb x nloc x axis x e_dim
        g1_13 = paddle.matmul(paddle.matrix_transpose(h2g2m), h2g2) / (3.0**1)
        # nb x nloc x (axisxng2)
        g1_13 = g1_13.reshape([nb, nloc, axis_neuron * e_dim])
        return g1_13

    def symmetrization_op(
        self,
        edge_ebd: paddle.Tensor,
        h2: paddle.Tensor,
        nlist_mask: paddle.Tensor,
        sw: paddle.Tensor,
        axis_neuron: int,
    ) -> paddle.Tensor:
        """
        Symmetrization operator to obtain atomic invariant rep.

        Parameters
        ----------
        edge_ebd
            Neighbor-wise/Pair-wise invariant rep tensors, with shape nb x nloc x nnei x e_dim.
        h2
            Neighbor-wise/Pair-wise equivariant rep tensors, with shape nb x nloc x nnei x 3.
        nlist_mask
            Neighbor list mask, where zero means no neighbor, with shape nb x nloc x nnei.
        sw
            The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape nb x nloc x nnei.
        axis_neuron
            Size of the submatrix.

        Returns
        -------
        grrg
            Atomic invariant rep, with shape nb x nloc x (axis_neuron x e_dim)
        """
        # edge_ebd:  nb x nloc x nnei x e_dim
        # h2:  nb x nloc x nnei x 3
        # msk: nb x nloc x nnei
        nb, nloc, nnei, _ = edge_ebd.shape
        # nb x nloc x 3 x e_dim
        h2g2 = self._cal_hg(
            edge_ebd,
            h2,
            nlist_mask,
            sw,
        )
        # nb x nloc x (axisxng2)
        g1_13 = self._cal_grrg(h2g2, axis_neuron)
        return g1_13

    def optim_angle_update(
        self,
        angle_ebd: paddle.Tensor,
        node_ebd: paddle.Tensor,
        edge_ebd: paddle.Tensor,
        feat: str = "edge",
    ) -> paddle.Tensor:
        if feat == "edge":
            assert self.edge_angle_linear1 is not None
            matrix, bias = self.edge_angle_linear1.matrix, self.edge_angle_linear1.bias
        elif feat == "angle":
            assert self.angle_self_linear is not None
            matrix, bias = self.angle_self_linear.matrix, self.angle_self_linear.bias
        else:
            raise NotImplementedError
        assert bias is not None

        angle_dim = angle_ebd.shape[-1]
        node_dim = node_ebd.shape[-1]
        edge_dim = edge_ebd.shape[-1]
        # angle_dim, node_dim, edge_dim, edge_dim
        sub_angle, sub_node, sub_edge_ij, sub_edge_ik = paddle.split(
            matrix, [angle_dim, node_dim, edge_dim, edge_dim]
        )

        # nf * nloc * a_sel * a_sel * angle_dim
        sub_angle_update = paddle.matmul(angle_ebd, sub_angle)
        # nf * nloc * angle_dim
        sub_node_update = paddle.matmul(node_ebd, sub_node)
        # nf * nloc * a_nnei * angle_dim
        sub_edge_update_ij = paddle.matmul(edge_ebd, sub_edge_ij)
        sub_edge_update_ik = paddle.matmul(edge_ebd, sub_edge_ik)

        result_update = (
            bias
            + sub_node_update.unsqueeze(2).unsqueeze(3)
            + sub_edge_update_ij.unsqueeze(2)
            + sub_edge_update_ik.unsqueeze(3)
            + sub_angle_update
        )
        return result_update

    def optim_edge_update(
        self,
        node_ebd: paddle.Tensor,
        node_ebd_ext: paddle.Tensor,
        edge_ebd: paddle.Tensor,
        nlist: paddle.Tensor,
        feat: str = "node",
    ) -> paddle.Tensor:
        if feat == "node":
            matrix, bias = self.node_edge_linear.matrix, self.node_edge_linear.bias
        elif feat == "edge":
            matrix, bias = self.edge_self_linear.matrix, self.edge_self_linear.bias
        else:
            raise NotImplementedError
        assert bias is not None

        node_dim = node_ebd.shape[-1]
        edge_dim = edge_ebd.shape[-1]
        # node_dim, node_dim, edge_dim
        node, node_ext, edge = paddle.split(matrix, [node_dim, node_dim, edge_dim])

        # nf * nloc * node/edge_dim
        sub_node_update = paddle.matmul(node_ebd, node)
        # nf * nall * node/edge_dim
        sub_node_ext_update = paddle.matmul(node_ebd_ext, node_ext)
        # nf * nloc * nnei * node/edge_dim
        sub_node_ext_update = _make_nei_g1(sub_node_ext_update, nlist)
        # nf * nloc * nnei * node/edge_dim
        sub_edge_update = paddle.matmul(edge_ebd, edge)

        result_update = (
            bias + sub_node_update.unsqueeze(2) + sub_edge_update + sub_node_ext_update
        )
        return result_update

    def forward(
        self,
        node_ebd_ext: paddle.Tensor,  # nf x nall x n_dim
        edge_ebd: paddle.Tensor,  # nf x nloc x nnei x e_dim
        h2: paddle.Tensor,  # nf x nloc x nnei x 3
        angle_ebd: paddle.Tensor,  # nf x nloc x a_nnei x a_nnei x a_dim
        nlist: paddle.Tensor,  # nf x nloc x nnei
        nlist_mask: paddle.Tensor,  # nf x nloc x nnei
        sw: paddle.Tensor,  # switch func, nf x nloc x nnei
        a_nlist: paddle.Tensor,  # nf x nloc x a_nnei
        a_nlist_mask: paddle.Tensor,  # nf x nloc x a_nnei
        a_sw: paddle.Tensor,  # switch func, nf x nloc x a_nnei
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

        Returns
        -------
        n_updated:     nf x nloc x n_dim
            Updated node embedding.
        e_updated:     nf x nloc x nnei x e_dim
            Updated edge embedding.
        a_updated : nf x nloc x a_nnei x a_nnei x a_dim
            Updated angle embedding.
        """
        nb, nloc, nnei, _ = edge_ebd.shape
        nall = node_ebd_ext.shape[1]
        node_ebd = node_ebd_ext[:, :nloc, :]
        if paddle.in_dynamic_mode():
            assert [nb, nloc] == node_ebd.shape[:2]
        if paddle.in_dynamic_mode():
            assert [nb, nloc, nnei] == h2.shape[:3]
        del a_nlist  # may be used in the future

        n_update_list: list[paddle.Tensor] = [node_ebd]
        e_update_list: list[paddle.Tensor] = [edge_ebd]
        a_update_list: list[paddle.Tensor] = [angle_ebd]

        # node self mlp
        node_self_mlp = self.act(self.node_self_mlp(node_ebd))
        n_update_list.append(node_self_mlp)

        nei_node_ebd = _make_nei_g1(node_ebd_ext, nlist)

        # node sym (grrg + drrd)
        node_sym_list: list[paddle.Tensor] = []
        node_sym_list.append(
            self.symmetrization_op(
                edge_ebd,
                h2,
                nlist_mask,
                sw,
                self.axis_neuron,
            )
        )
        node_sym_list.append(
            self.symmetrization_op(
                nei_node_ebd,
                h2,
                nlist_mask,
                sw,
                self.axis_neuron,
            )
        )
        node_sym = self.act(self.node_sym_linear(paddle.concat(node_sym_list, axis=-1)))
        n_update_list.append(node_sym)

        if not self.optim_update:
            # nb x nloc x nnei x (n_dim * 2 + e_dim)
            edge_info = paddle.concat(
                [
                    paddle.tile(node_ebd.unsqueeze(-2), [1, 1, self.nnei, 1]),
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
            ) * sw.unsqueeze(-1)
        else:
            node_edge_update = self.act(
                self.optim_edge_update(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    nlist,
                    "node",
                )
            ) * sw.unsqueeze(-1)

        node_edge_update = paddle.sum(node_edge_update, axis=-2) / self.nnei
        if self.n_multi_edge_message > 1:
            # nb x nloc x nnei x h x n_dim
            node_edge_update_mul_head = node_edge_update.reshape(
                [nb, nloc, self.n_multi_edge_message, self.n_dim]
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
            )
        e_update_list.append(edge_self_update)

        if self.update_angle:
            if paddle.in_dynamic_mode():
                assert self.angle_self_linear is not None
            if paddle.in_dynamic_mode():
                assert self.edge_angle_linear1 is not None
            if paddle.in_dynamic_mode():
                assert self.edge_angle_linear2 is not None
            # get angle info
            if self.a_compress_rate != 0:
                if not self.a_compress_use_split:
                    if paddle.in_dynamic_mode():
                        assert self.a_compress_n_linear is not None
                    if paddle.in_dynamic_mode():
                        assert self.a_compress_e_linear is not None
                    node_ebd_for_angle = self.a_compress_n_linear(node_ebd)
                    edge_ebd_for_angle = self.a_compress_e_linear(edge_ebd)
                else:
                    # use the first a_compress_dim dim for node and edge
                    node_ebd_for_angle = node_ebd[:, :, : self.n_a_compress_dim]
                    edge_ebd_for_angle = edge_ebd[:, :, :, : self.e_a_compress_dim]
            else:
                node_ebd_for_angle = node_ebd
                edge_ebd_for_angle = edge_ebd

            # nb x nloc x a_nnei x e_dim
            edge_for_angle = edge_ebd_for_angle[:, :, : self.a_sel, :]
            # nb x nloc x a_nnei x e_dim
            edge_for_angle = paddle.where(
                a_nlist_mask.unsqueeze(-1),
                edge_for_angle,
                paddle.zeros_like(edge_for_angle),
            ).astype(edge_for_angle.dtype)
            if not self.optim_update:
                # nb x nloc x a_nnei x a_nnei x n_dim
                node_for_angle_info = paddle.tile(
                    node_ebd_for_angle.unsqueeze(2).unsqueeze(2),
                    [1, 1, self.a_sel, self.a_sel, 1],
                )
                # nb x nloc x (a_nnei) x a_nnei x edge_ebd
                edge_for_angle_i = paddle.tile(
                    edge_for_angle.unsqueeze(2), (1, 1, self.a_sel, 1, 1)
                )
                # nb x nloc x a_nnei x (a_nnei) x e_dim
                edge_for_angle_j = paddle.tile(
                    edge_for_angle.unsqueeze(3), (1, 1, 1, self.a_sel, 1)
                )
                # nb x nloc x a_nnei x a_nnei x (e_dim + e_dim)
                edge_for_angle_info = paddle.concat(
                    [edge_for_angle_i, edge_for_angle_j], axis=-1
                )
                angle_info_list = [angle_ebd]
                angle_info_list.append(node_for_angle_info)
                angle_info_list.append(edge_for_angle_info)
                # nb x nloc x a_nnei x a_nnei x (a + n_dim + e_dim*2) or (a + a/c + a/c)
                angle_info = paddle.concat(angle_info_list, axis=-1)
            else:
                angle_info = None

            # edge angle message
            # nb x nloc x a_nnei x a_nnei x e_dim
            if not self.optim_update:
                assert angle_info is not None
                edge_angle_update = self.act(self.edge_angle_linear1(angle_info))
            else:
                edge_angle_update = self.act(
                    self.optim_angle_update(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_for_angle,
                        "edge",
                    )
                )

            # nb x nloc x a_nnei x a_nnei x e_dim
            weighted_edge_angle_update = (
                a_sw[..., None, None] * a_sw[..., None, :, None] * edge_angle_update
            )
            # nb x nloc x a_nnei x e_dim
            reduced_edge_angle_update = paddle.sum(
                weighted_edge_angle_update, axis=-2
            ) / (self.a_sel**0.5)
            # nb x nloc x nnei x e_dim
            padding_edge_angle_update = paddle.concat(
                [
                    reduced_edge_angle_update,
                    paddle.zeros(
                        [nb, nloc, self.nnei - self.a_sel, self.e_dim],
                        dtype=edge_ebd.dtype,
                    ).to(device=edge_ebd.place),
                ],
                axis=2,
            )
            if not self.smooth_edge_update:
                # will be deprecated in the future
                full_mask = paddle.concat(
                    [
                        a_nlist_mask,
                        paddle.zeros(
                            [nb, nloc, self.nnei - self.a_sel],
                            dtype=a_nlist_mask.dtype,
                        ).to(a_nlist_mask.place),
                    ],
                    axis=-1,
                )
                padding_edge_angle_update = paddle.where(
                    full_mask.unsqueeze(-1), padding_edge_angle_update, edge_ebd
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
                        edge_for_angle,
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
        update_list: list[paddle.Tensor],
    ) -> paddle.Tensor:
        nitem = len(update_list)
        uu = update_list[0]
        for ii in range(1, nitem):
            uu = uu + update_list[ii]
        return uu / (float(nitem) ** 0.5)

    def list_update_res_incr(self, update_list: list[paddle.Tensor]) -> paddle.Tensor:
        nitem = len(update_list)
        uu = update_list[0]
        scale = 1.0 / (float(nitem - 1) ** 0.5) if nitem > 1 else 0.0
        for ii in range(1, nitem):
            uu = uu + scale * update_list[ii]
        return uu

    def list_update_res_residual(
        self, update_list: list[paddle.Tensor], update_name: str = "node"
    ) -> paddle.Tensor:
        nitem = len(update_list)
        uu = update_list[0]
        # make jit happy
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
        self, update_list: list[paddle.Tensor], update_name: str = "node"
    ) -> paddle.Tensor:
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
        obj.node_self_mlp = MLPLayer.deserialize(node_self_mlp)
        obj.node_sym_linear = MLPLayer.deserialize(node_sym_linear)
        obj.node_edge_linear = MLPLayer.deserialize(node_edge_linear)
        obj.edge_self_linear = MLPLayer.deserialize(edge_self_linear)

        if update_angle:
            if paddle.in_dynamic_mode():
                assert isinstance(edge_angle_linear1, dict)
            if paddle.in_dynamic_mode():
                assert isinstance(edge_angle_linear2, dict)
            if paddle.in_dynamic_mode():
                assert isinstance(angle_self_linear, dict)
            obj.edge_angle_linear1 = MLPLayer.deserialize(edge_angle_linear1)
            obj.edge_angle_linear2 = MLPLayer.deserialize(edge_angle_linear2)
            obj.angle_self_linear = MLPLayer.deserialize(angle_self_linear)
            if a_compress_rate != 0 and not a_compress_use_split:
                if paddle.in_dynamic_mode():
                    assert isinstance(a_compress_n_linear, dict)
                if paddle.in_dynamic_mode():
                    assert isinstance(a_compress_e_linear, dict)
                obj.a_compress_n_linear = MLPLayer.deserialize(a_compress_n_linear)
                obj.a_compress_e_linear = MLPLayer.deserialize(a_compress_e_linear)

        if update_style == "res_residual":
            for ii, t in enumerate(obj.n_residual):
                t.data = to_paddle_tensor(n_residual[ii])
            for ii, t in enumerate(obj.e_residual):
                t.data = to_paddle_tensor(e_residual[ii])
            for ii, t in enumerate(obj.a_residual):
                t.data = to_paddle_tensor(a_residual[ii])
        return obj
