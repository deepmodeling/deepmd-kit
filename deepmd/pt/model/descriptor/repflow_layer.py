# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Union,
)

import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.network.init import (
    constant_,
    normal_,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
    get_generator,
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


def get_residual(
    _dim: int,
    _scale: float,
    _mode: str = "norm",
    trainable: bool = True,
    precision: str = "float64",
    seed: Optional[Union[int, list[int]]] = None,
) -> torch.Tensor:
    r"""
    Get residual tensor for one update vector.

    Parameters
    ----------
    _dim : int
        The dimension of the update vector.
    _scale
        The initial scale of the residual tensor. See `_mode` for details.
    _mode
        The mode of residual initialization for the residual tensor.
        - "norm" (default): init residual using normal with `_scale` std.
        - "const": init residual using element-wise constants of `_scale`.
    trainable
        Whether the residual tensor is trainable.
    precision
        The precision of the residual tensor.
    seed : int, optional
        Random seed for parameter initialization.
    """
    random_generator = get_generator(seed)
    residual = nn.Parameter(
        data=torch.zeros(_dim, dtype=PRECISION_DICT[precision], device=env.DEVICE),
        requires_grad=trainable,
    )
    if _mode == "norm":
        normal_(residual.data, std=_scale, generator=random_generator)
    elif _mode == "const":
        constant_(residual.data, val=_scale)
    else:
        raise RuntimeError(f"Unsupported initialization mode '{_mode}'!")
    return residual


# common ops
def _make_nei_g1(
    g1_ext: torch.Tensor,
    nlist: torch.Tensor,
) -> torch.Tensor:
    """
    Make neighbor-wise atomic invariant rep.

    Parameters
    ----------
    g1_ext
        Extended atomic invariant rep, with shape nb x nall x ng1.
    nlist
        Neighbor list, with shape nb x nloc x nnei.

    Returns
    -------
    gg1: torch.Tensor
        Neighbor-wise atomic invariant rep, with shape nb x nloc x nnei x ng1.

    """
    # nlist: nb x nloc x nnei
    nb, nloc, nnei = nlist.shape
    # g1_ext: nb x nall x ng1
    ng1 = g1_ext.shape[-1]
    # index: nb x (nloc x nnei) x ng1
    index = nlist.reshape(nb, nloc * nnei).unsqueeze(-1).expand(-1, -1, ng1)
    # gg1  : nb x (nloc x nnei) x ng1
    gg1 = torch.gather(g1_ext, dim=1, index=index)
    # gg1  : nb x nloc x nnei x ng1
    gg1 = gg1.view(nb, nloc, nnei, ng1)
    return gg1


def _apply_nlist_mask(
    gg: torch.Tensor,
    nlist_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply nlist mask to neighbor-wise rep tensors.

    Parameters
    ----------
    gg
        Neighbor-wise rep tensors, with shape nf x nloc x nnei x d.
    nlist_mask
        Neighbor list mask, where zero means no neighbor, with shape nf x nloc x nnei.
    """
    # gg:  nf x nloc x nnei x d
    # msk: nf x nloc x nnei
    return gg.masked_fill(~nlist_mask.unsqueeze(-1), 0.0)


def _apply_switch(gg: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    """
    Apply switch function to neighbor-wise rep tensors.

    Parameters
    ----------
    gg
        Neighbor-wise rep tensors, with shape nf x nloc x nnei x d.
    sw
        The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
        and remains 0 beyond rcut, with shape nf x nloc x nnei.
    """
    # gg:  nf x nloc x nnei x d
    # sw:  nf x nloc x nnei
    return gg * sw.unsqueeze(-1)


class RepFlowLayer(torch.nn.Module):
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
        axis_neuron: int = 4,
        update_angle: bool = True,  # angle
        update_g1_has_conv: bool = True,
        activation_function: str = "silu",
        update_style: str = "res_avg",
        update_residual: float = 0.001,
        update_residual_init: str = "norm",
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
        self.axis_neuron = axis_neuron
        self.update_angle = update_angle
        self.activation_function = activation_function
        self.act = ActivationFn(activation_function)
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.precision = precision
        self.seed = seed
        self.prec = PRECISION_DICT[precision]

        self.update_g1_has_conv = update_g1_has_conv

        assert update_residual_init in [
            "norm",
            "const",
        ], "'update_residual_init' only support 'norm' or 'const'!"

        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.g1_residual = []
        self.g2_residual = []
        self.h2_residual = []
        self.a_residual = []
        self.proj_g1g2 = None
        self.edge_info_dim = self.n_dim * 2 + self.e_dim

        # g1 self mlp
        self.node_self_mlp = MLPLayer(
            n_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 15),
        )
        if self.update_style == "res_residual":
            self.g1_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 16),
                )
            )

        # g1 conv # tmp
        if self.update_g1_has_conv:
            self.proj_g1g2 = MLPLayer(
                e_dim,
                n_dim,
                bias=False,
                precision=precision,
                seed=child_seed(seed, 4),
            )
            if self.update_style == "res_residual":
                self.g1_residual.append(
                    get_residual(
                        n_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 17),
                    )
                )

        # g1 sym
        self.g1_sym_dim = self.cal_1_dim(n_dim, e_dim, self.axis_neuron)
        self.linear1 = MLPLayer(
            self.g1_sym_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 1),
        )
        if self.update_style == "res_residual":
            self.g1_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 0),
                )
            )

        # g1 edge
        self.g1_edge_linear1 = MLPLayer(
            self.edge_info_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 11),
        )  # need act
        if self.update_style == "res_residual":
            self.g1_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 13),
                )
            )

        # g2 edge
        self.linear2 = MLPLayer(
            self.edge_info_dim,
            e_dim,
            precision=precision,
            seed=child_seed(seed, 2),
        )
        if self.update_style == "res_residual":
            self.g2_residual.append(
                get_residual(
                    e_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 3),
                )
            )

        if self.update_angle:
            angle_seed = 20
            self.angle_dim = self.a_dim + self.n_dim + 2 * self.e_dim
            self.angle_linear = MLPLayer(
                self.angle_dim,
                self.a_dim,
                precision=precision,
                seed=child_seed(seed, angle_seed + 1),
            )  # need act
            if self.update_style == "res_residual":
                self.a_residual.append(
                    get_residual(
                        self.a_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, angle_seed + 2),
                    )
                )

            self.g2_angle_linear1 = MLPLayer(
                self.angle_dim,
                self.e_dim,
                precision=precision,
                seed=child_seed(seed, angle_seed + 3),
            )  # need act
            self.g2_angle_linear2 = MLPLayer(
                self.e_dim,
                self.e_dim,
                precision=precision,
                seed=child_seed(seed, angle_seed + 4),
            )
            if self.update_style == "res_residual":
                self.g2_residual.append(
                    get_residual(
                        self.e_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, angle_seed + 5),
                    )
                )

        else:
            self.angle_linear = None
            self.g2_angle_linear1 = None
            self.g2_angle_linear2 = None
            self.angle_dim = 0
            self.angle_dim = 0

        self.g1_residual = nn.ParameterList(self.g1_residual)
        self.g2_residual = nn.ParameterList(self.g2_residual)
        self.h2_residual = nn.ParameterList(self.h2_residual)
        self.a_residual = nn.ParameterList(self.a_residual)

    def cal_1_dim(self, g1d: int, g2d: int, ax: int) -> int:
        ret = g2d * ax + g1d * ax
        return ret

    def _update_g1_conv(
        self,
        gg1: torch.Tensor,
        g2: torch.Tensor,
        nlist_mask: torch.Tensor,
        sw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the convolution update for atomic invariant rep.

        Parameters
        ----------
        gg1
            Neighbor-wise atomic invariant rep, with shape nb x nloc x nnei x ng1.
        g2
            Pair invariant rep, with shape nb x nloc x nnei x ng2.
        nlist_mask
            Neighbor list mask, where zero means no neighbor, with shape nb x nloc x nnei.
        sw
            The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape nb x nloc x nnei.
        """
        assert self.proj_g1g2 is not None
        nb, nloc, nnei, _ = g2.shape
        ng1 = gg1.shape[-1]
        ng2 = g2.shape[-1]
        gg1 = gg1.view(nb, nloc, nnei, ng1)
        # nb x nloc x nnei x ng2/ng1
        gg1 = _apply_nlist_mask(gg1, nlist_mask)
        gg1 = _apply_switch(gg1, sw)
        invnnei = (1.0 / float(nnei)) * torch.ones(
            (nb, nloc, 1), dtype=gg1.dtype, device=gg1.device
        )
        g2 = self.proj_g1g2(g2).view(nb, nloc, nnei, ng1)
        # nb x nloc x ng1
        g1_11 = torch.sum(g2 * gg1, dim=2) * invnnei
        return g1_11

    @staticmethod
    def _cal_hg(
        g2: torch.Tensor,
        h2: torch.Tensor,
        nlist_mask: torch.Tensor,
        sw: torch.Tensor,
        smooth: bool = True,
        epsilon: float = 1e-4,
        use_sqrt_nnei: bool = True,
    ) -> torch.Tensor:
        """
        Calculate the transposed rotation matrix.

        Parameters
        ----------
        g2
            Neighbor-wise/Pair-wise invariant rep tensors, with shape nb x nloc x nnei x ng2.
        h2
            Neighbor-wise/Pair-wise equivariant rep tensors, with shape nb x nloc x nnei x 3.
        nlist_mask
            Neighbor list mask, where zero means no neighbor, with shape nb x nloc x nnei.
        sw
            The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape nb x nloc x nnei.
        smooth
            Whether to use smoothness in processes such as attention weights calculation.
        epsilon
            Protection of 1./nnei.

        Returns
        -------
        hg
            The transposed rotation matrix, with shape nb x nloc x 3 x ng2.
        """
        # g2:  nb x nloc x nnei x ng2
        # h2:  nb x nloc x nnei x 3
        # msk: nb x nloc x nnei
        nb, nloc, nnei, _ = g2.shape
        ng2 = g2.shape[-1]
        # nb x nloc x nnei x ng2
        g2 = _apply_nlist_mask(g2, nlist_mask)
        if not smooth:
            # nb x nloc
            # must use type_as here to convert bool to float, otherwise there will be numerical difference from numpy
            if not use_sqrt_nnei:
                invnnei = 1.0 / (epsilon + torch.sum(nlist_mask.type_as(g2), dim=-1))
            else:
                invnnei = 1.0 / (
                    epsilon + torch.sqrt(torch.sum(nlist_mask.type_as(g2), dim=-1))
                )
            # nb x nloc x 1 x 1
            invnnei = invnnei.unsqueeze(-1).unsqueeze(-1)
        else:
            g2 = _apply_switch(g2, sw)
            if not use_sqrt_nnei:
                invnnei = (1.0 / float(nnei)) * torch.ones(
                    (nb, nloc, 1, 1), dtype=g2.dtype, device=g2.device
                )
            else:
                invnnei = torch.rsqrt(
                    float(nnei)
                    * torch.ones((nb, nloc, 1, 1), dtype=g2.dtype, device=g2.device)
                )
        # nb x nloc x 3 x ng2
        h2g2 = torch.matmul(torch.transpose(h2, -1, -2), g2) * invnnei
        return h2g2

    @staticmethod
    def _cal_grrg(h2g2: torch.Tensor, axis_neuron: int) -> torch.Tensor:
        """
        Calculate the atomic invariant rep.

        Parameters
        ----------
        h2g2
            The transposed rotation matrix, with shape nb x nloc x 3 x ng2.
        axis_neuron
            Size of the submatrix.

        Returns
        -------
        grrg
            Atomic invariant rep, with shape nb x nloc x (axis_neuron x ng2)
        """
        # nb x nloc x 3 x ng2
        nb, nloc, _, ng2 = h2g2.shape
        # nb x nloc x 3 x axis
        h2g2m = torch.split(h2g2, axis_neuron, dim=-1)[0]
        # nb x nloc x axis x ng2
        g1_13 = torch.matmul(torch.transpose(h2g2m, -1, -2), h2g2) / (3.0**1)
        # nb x nloc x (axisxng2)
        g1_13 = g1_13.view(nb, nloc, axis_neuron * ng2)
        return g1_13

    def symmetrization_op(
        self,
        g2: torch.Tensor,
        h2: torch.Tensor,
        nlist_mask: torch.Tensor,
        sw: torch.Tensor,
        axis_neuron: int,
        smooth: bool = True,
        epsilon: float = 1e-4,
    ) -> torch.Tensor:
        """
        Symmetrization operator to obtain atomic invariant rep.

        Parameters
        ----------
        g2
            Neighbor-wise/Pair-wise invariant rep tensors, with shape nb x nloc x nnei x ng2.
        h2
            Neighbor-wise/Pair-wise equivariant rep tensors, with shape nb x nloc x nnei x 3.
        nlist_mask
            Neighbor list mask, where zero means no neighbor, with shape nb x nloc x nnei.
        sw
            The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape nb x nloc x nnei.
        axis_neuron
            Size of the submatrix.
        smooth
            Whether to use smoothness in processes such as attention weights calculation.
        epsilon
            Protection of 1./nnei.

        Returns
        -------
        grrg
            Atomic invariant rep, with shape nb x nloc x (axis_neuron x ng2)
        """
        # g2:  nb x nloc x nnei x ng2
        # h2:  nb x nloc x nnei x 3
        # msk: nb x nloc x nnei
        nb, nloc, nnei, _ = g2.shape
        # nb x nloc x 3 x ng2
        h2g2 = self._cal_hg(
            g2,
            h2,
            nlist_mask,
            sw,
            smooth=smooth,
            epsilon=epsilon,
            use_sqrt_nnei=True,
        )
        # nb x nloc x (axisxng2)
        g1_13 = self._cal_grrg(h2g2, axis_neuron)
        return g1_13

    def forward(
        self,
        g1_ext: torch.Tensor,  # nf x nall x ng1
        g2: torch.Tensor,  # nf x nloc x nnei x ng2
        h2: torch.Tensor,  # nf x nloc x nnei x 3
        angle_embed: torch.Tensor,  # nf x nloc x a_nnei x a_nnei x a_dim
        nlist: torch.Tensor,  # nf x nloc x nnei
        nlist_mask: torch.Tensor,  # nf x nloc x nnei
        sw: torch.Tensor,  # switch func, nf x nloc x nnei
        angle_nlist: torch.Tensor,  # nf x nloc x a_nnei
        angle_nlist_mask: torch.Tensor,  # nf x nloc x a_nnei
        angle_sw: torch.Tensor,  # switch func, nf x nloc x a_nnei
    ):
        """
        Parameters
        ----------
        g1_ext : nf x nall x ng1         extended single-atom channel
        g2 : nf x nloc x nnei x ng2  pair-atom channel, invariant
        h2 : nf x nloc x nnei x 3    pair-atom channel, equivariant
        nlist : nf x nloc x nnei        neighbor list (padded neis are set to 0)
        nlist_mask : nf x nloc x nnei   masks of the neighbor list. real nei 1 otherwise 0
        sw : nf x nloc x nnei        switch function

        Returns
        -------
        g1:     nf x nloc x ng1         updated single-atom channel
        g2:     nf x nloc x nnei x ng2  updated pair-atom channel, invariant
        h2:     nf x nloc x nnei x 3    updated pair-atom channel, equivariant
        """
        nb, nloc, nnei, _ = g2.shape
        nall = g1_ext.shape[1]
        g1, _ = torch.split(g1_ext, [nloc, nall - nloc], dim=1)
        assert (nb, nloc) == g1.shape[:2]
        assert (nb, nloc, nnei) == h2.shape[:3]

        g1_update: list[torch.Tensor] = [g1]
        g2_update: list[torch.Tensor] = [g2]
        a_update: list[torch.Tensor] = [angle_embed]
        h2_update: list[torch.Tensor] = [h2]

        g1_sym: list[torch.Tensor] = []

        # g1 self mlp
        node_self_mlp = self.act(self.node_self_mlp(g1))
        g1_update.append(node_self_mlp)

        gg1 = _make_nei_g1(g1_ext, nlist)
        # g1 conv # tmp
        if self.update_g1_has_conv:
            assert gg1 is not None
            g1_conv = self._update_g1_conv(gg1, g2, nlist_mask, sw)
            g1_update.append(g1_conv)

        # g1 sym mlp
        g1_sym.append(
            self.symmetrization_op(
                g2,
                h2,
                nlist_mask,
                sw,
                self.axis_neuron,
                smooth=True,
                epsilon=self.epsilon,
            )
        )
        g1_sym.append(
            self.symmetrization_op(
                gg1,
                h2,
                nlist_mask,
                sw,
                self.axis_neuron,
                smooth=True,
                epsilon=self.epsilon,
            )
        )
        g1_1 = self.act(self.linear1(torch.cat(g1_sym, dim=-1)))
        g1_update.append(g1_1)

        edge_info = torch.cat(
            [torch.tile(g1.unsqueeze(-2), [1, 1, self.nnei, 1]), gg1, g2], dim=-1
        )

        # g1 edge update
        # nb x nloc x nnei x ng1
        g1_edge_info = self.act(self.g1_edge_linear1(edge_info)) * sw.unsqueeze(-1)
        g1_edge_update = torch.sum(g1_edge_info, dim=-2) / self.nnei
        g1_update.append(g1_edge_update)
        # update g1
        g1_new = self.list_update(g1_update, "g1")

        # g2 edge update
        g2_edge_info = self.act(self.linear2(edge_info))
        g2_update.append(g2_edge_info)

        if self.update_angle:
            assert self.angle_linear is not None
            assert self.g2_angle_linear1 is not None
            assert self.g2_angle_linear2 is not None
            # nb x nloc x a_nnei x a_nnei x g1
            g1_angle_embed = torch.tile(
                g1.unsqueeze(2).unsqueeze(2), (1, 1, self.a_sel, self.a_sel, 1)
            )
            # nb x nloc x a_nnei x g2
            g2_angle = g2[:, :, : self.a_sel, :]
            # nb x nloc x a_nnei x g2
            g2_angle = torch.where(angle_nlist_mask.unsqueeze(-1), g2_angle, 0.0)
            # nb x nloc x (a_nnei) x a_nnei x g2
            g2_angle_i = torch.tile(g2_angle.unsqueeze(2), (1, 1, self.a_sel, 1, 1))
            # nb x nloc x a_nnei x (a_nnei) x g2
            g2_angle_j = torch.tile(g2_angle.unsqueeze(3), (1, 1, 1, self.a_sel, 1))
            # nb x nloc x a_nnei x a_nnei x (g2 + g2)
            g2_angle_embed = torch.cat([g2_angle_i, g2_angle_j], dim=-1)

            # angle for g2:
            updated_g2_angle_list = [angle_embed]
            # nb x nloc x a_nnei x a_nnei x (a + g1 + g2*2)
            updated_g2_angle_list += [g1_angle_embed, g2_angle_embed]
            updated_g2_angle = torch.cat(updated_g2_angle_list, dim=-1)
            # nb x nloc x a_nnei x a_nnei x g2
            updated_angle_g2 = self.act(self.g2_angle_linear1(updated_g2_angle))
            # nb x nloc x a_nnei x a_nnei x g2
            weighted_updated_angle_g2 = (
                updated_angle_g2
                * angle_sw[:, :, :, None, None]
                * angle_sw[:, :, None, :, None]
            )
            # nb x nloc x a_nnei x g2
            reduced_updated_angle_g2 = torch.sum(weighted_updated_angle_g2, dim=-2) / (
                self.a_sel**0.5
            )
            # nb x nloc x nnei x g2
            padding_updated_angle_g2 = torch.concat(
                [
                    reduced_updated_angle_g2,
                    torch.zeros(
                        [nb, nloc, self.nnei - self.a_sel, self.e_dim],
                        dtype=g2.dtype,
                        device=g2.device,
                    ),
                ],
                dim=2,
            )
            full_mask = torch.concat(
                [
                    angle_nlist_mask,
                    torch.zeros(
                        [nb, nloc, self.nnei - self.a_sel],
                        dtype=angle_nlist_mask.dtype,
                        device=angle_nlist_mask.device,
                    ),
                ],
                dim=-1,
            )
            padding_updated_angle_g2 = torch.where(
                full_mask.unsqueeze(-1), padding_updated_angle_g2, g2
            )
            g2_update.append(self.act(self.g2_angle_linear2(padding_updated_angle_g2)))

            # update g2
            g2_new = self.list_update(g2_update, "g2")
            # angle for angle
            updated_angle = updated_g2_angle
            # nb x nloc x a_nnei x a_nnei x dim_a
            angle_message = self.act(self.angle_linear(updated_angle))
            # angle update
            a_update.append(angle_message)
        else:
            # update g2
            g2_new = self.list_update(g2_update, "g2")

        # update
        h2_new = self.list_update(h2_update, "h2")
        a_new = self.list_update(a_update, "a")
        return g1_new, g2_new, h2_new, a_new

    @torch.jit.export
    def list_update_res_avg(
        self,
        update_list: list[torch.Tensor],
    ) -> torch.Tensor:
        nitem = len(update_list)
        uu = update_list[0]
        for ii in range(1, nitem):
            uu = uu + update_list[ii]
        return uu / (float(nitem) ** 0.5)

    @torch.jit.export
    def list_update_res_incr(self, update_list: list[torch.Tensor]) -> torch.Tensor:
        nitem = len(update_list)
        uu = update_list[0]
        scale = 1.0 / (float(nitem - 1) ** 0.5) if nitem > 1 else 0.0
        for ii in range(1, nitem):
            uu = uu + scale * update_list[ii]
        return uu

    @torch.jit.export
    def list_update_res_residual(
        self, update_list: list[torch.Tensor], update_name: str = "g1"
    ) -> torch.Tensor:
        nitem = len(update_list)
        uu = update_list[0]
        # make jit happy
        if update_name == "g1":
            for ii, vv in enumerate(self.g1_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "g2":
            for ii, vv in enumerate(self.g2_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "h2":
            for ii, vv in enumerate(self.h2_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "a":
            for ii, vv in enumerate(self.a_residual):
                uu = uu + vv * update_list[ii + 1]
        else:
            raise NotImplementedError
        return uu

    @torch.jit.export
    def list_update(
        self, update_list: list[torch.Tensor], update_name: str = "g1"
    ) -> torch.Tensor:
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
            "@class": "RepformerLayer",
            "@version": 1,
            "e_rcut": self.e_rcut,
            "e_rcut_smth": self.e_rcut_smth,
            "e_sel": self.e_sel,
            "ntypes": self.ntypes,
            "n_dim": self.n_dim,
            "e_dim": self.e_dim,
            "axis_neuron": self.axis_neuron,
            "activation_function": self.activation_function,
            "update_style": self.update_style,
            "precision": self.precision,
            "linear1": self.linear1.serialize(),
        }
        if self.update_g1_has_conv:
            data.update(
                {
                    "proj_g1g2": self.proj_g1g2.serialize(),
                }
            )

        if self.update_g2_has_attn or self.update_h2:
            data.update(
                {
                    "attn2g_map": self.attn2g_map.serialize(),
                }
            )
            if self.update_g2_has_attn:
                data.update(
                    {
                        "attn2_mh_apply": self.attn2_mh_apply.serialize(),
                        "attn2_lm": self.attn2_lm.serialize(),
                    }
                )

            if self.update_h2:
                data.update(
                    {
                        "attn2_ev_apply": self.attn2_ev_apply.serialize(),
                    }
                )
        if self.update_g1_has_attn:
            data.update(
                {
                    "loc_attn": self.loc_attn.serialize(),
                }
            )
        if self.g1_out_mlp:
            data.update(
                {
                    "node_self_mlp": self.node_self_mlp.serialize(),
                }
            )
        if self.update_style == "res_residual":
            data.update(
                {
                    "@variables": {
                        "g1_residual": [to_numpy_array(t) for t in self.g1_residual],
                        "g2_residual": [to_numpy_array(t) for t in self.g2_residual],
                        "h2_residual": [to_numpy_array(t) for t in self.h2_residual],
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
        linear1 = data.pop("linear1")
        update_chnnl_2 = data["update_chnnl_2"]
        update_g1_has_conv = data["update_g1_has_conv"]
        update_g2_has_g1g1 = data["update_g2_has_g1g1"]
        update_g2_has_attn = data["update_g2_has_attn"]
        update_h2 = data["update_h2"]
        update_g1_has_attn = data["update_g1_has_attn"]
        update_style = data["update_style"]
        g1_out_mlp = data["g1_out_mlp"]

        linear2 = data.pop("linear2", None)
        proj_g1g2 = data.pop("proj_g1g2", None)
        attn2g_map = data.pop("attn2g_map", None)
        attn2_mh_apply = data.pop("attn2_mh_apply", None)
        attn2_lm = data.pop("attn2_lm", None)
        attn2_ev_apply = data.pop("attn2_ev_apply", None)
        loc_attn = data.pop("loc_attn", None)
        node_self_mlp = data.pop("node_self_mlp", None)
        variables = data.pop("@variables", {})
        g1_residual = variables.get("g1_residual", data.pop("g1_residual", []))
        g2_residual = variables.get("g2_residual", data.pop("g2_residual", []))
        h2_residual = variables.get("h2_residual", data.pop("h2_residual", []))

        obj = cls(**data)
        obj.linear1 = MLPLayer.deserialize(linear1)
        if update_chnnl_2:
            assert isinstance(linear2, dict)
            obj.linear2 = MLPLayer.deserialize(linear2)
        if update_g1_has_conv:
            assert isinstance(proj_g1g2, dict)
            obj.proj_g1g2 = MLPLayer.deserialize(proj_g1g2)

        if g1_out_mlp:
            assert isinstance(node_self_mlp, dict)
            obj.node_self_mlp = MLPLayer.deserialize(node_self_mlp)
        if update_style == "res_residual":
            for ii, t in enumerate(obj.g1_residual):
                t.data = to_torch_tensor(g1_residual[ii])
            for ii, t in enumerate(obj.g2_residual):
                t.data = to_torch_tensor(g2_residual[ii])
            for ii, t in enumerate(obj.h2_residual):
                t.data = to_torch_tensor(h2_residual[ii])
        return obj
