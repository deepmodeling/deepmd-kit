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
from deepmd.pt.model.network.layernorm import (
    LayerNorm,
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


class Atten2Map(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        head_num: int,
        has_gate: bool = False,  # apply gate to attn map
        smooth: bool = True,
        attnw_shift: float = 20.0,
        precision: str = "float64",
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        """Return neighbor-wise multi-head self-attention maps, with gate mechanism."""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.mapqk = MLPLayer(
            input_dim,
            hidden_dim * 2 * head_num,
            bias=False,
            precision=precision,
            seed=seed,
        )
        self.has_gate = has_gate
        self.smooth = smooth
        self.attnw_shift = attnw_shift
        self.precision = precision

    def forward(
        self,
        g2: torch.Tensor,  # nb x nloc x nnei x ng2
        h2: torch.Tensor,  # nb x nloc x nnei x 3
        nlist_mask: torch.Tensor,  # nb x nloc x nnei
        sw: torch.Tensor,  # nb x nloc x nnei
    ) -> torch.Tensor:
        (
            nb,
            nloc,
            nnei,
            _,
        ) = g2.shape
        nd, nh = self.hidden_dim, self.head_num
        # nb x nloc x nnei x nd x (nh x 2)
        g2qk = self.mapqk(g2).view(nb, nloc, nnei, nd, nh * 2)
        # nb x nloc x (nh x 2) x nnei x nd
        g2qk = torch.permute(g2qk, (0, 1, 4, 2, 3))
        # nb x nloc x nh x nnei x nd
        g2q, g2k = torch.split(g2qk, nh, dim=2)
        # g2q = torch.nn.functional.normalize(g2q, dim=-1)
        # g2k = torch.nn.functional.normalize(g2k, dim=-1)
        # nb x nloc x nh x nnei x nnei
        attnw = torch.matmul(g2q, torch.transpose(g2k, -1, -2)) / nd**0.5
        if self.has_gate:
            gate = torch.matmul(h2, torch.transpose(h2, -1, -2)).unsqueeze(-3)
            attnw = attnw * gate
        # mask the attenmap, nb x nloc x 1 x 1 x nnei
        attnw_mask = ~nlist_mask.unsqueeze(2).unsqueeze(2)
        # mask the attenmap, nb x nloc x 1 x nnei x 1
        attnw_mask_c = ~nlist_mask.unsqueeze(2).unsqueeze(-1)
        if self.smooth:
            attnw = (attnw + self.attnw_shift) * sw[:, :, None, :, None] * sw[
                :, :, None, None, :
            ] - self.attnw_shift
        else:
            attnw = attnw.masked_fill(
                attnw_mask,
                float("-inf"),
            )
        attnw = torch.softmax(attnw, dim=-1)
        attnw = attnw.masked_fill(
            attnw_mask,
            0.0,
        )
        # nb x nloc x nh x nnei x nnei
        attnw = attnw.masked_fill(
            attnw_mask_c,
            0.0,
        )
        if self.smooth:
            attnw = attnw * sw[:, :, None, :, None] * sw[:, :, None, None, :]
        # nb x nloc x nnei x nnei
        h2h2t = torch.matmul(h2, torch.transpose(h2, -1, -2)) / 3.0**0.5
        # nb x nloc x nh x nnei x nnei
        ret = attnw * h2h2t[:, :, None, :, :]
        # ret = torch.softmax(g2qk, dim=-1)
        # nb x nloc x nnei x nnei x nh
        ret = torch.permute(ret, (0, 1, 3, 4, 2))
        return ret

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        return {
            "@class": "Atten2Map",
            "@version": 1,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "head_num": self.head_num,
            "has_gate": self.has_gate,
            "smooth": self.smooth,
            "attnw_shift": self.attnw_shift,
            "precision": self.precision,
            "mapqk": self.mapqk.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Atten2Map":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        mapqk = data.pop("mapqk")
        obj = cls(**data)
        obj.mapqk = MLPLayer.deserialize(mapqk)
        return obj


class Atten2MultiHeadApply(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_num: int,
        precision: str = "float64",
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.head_num = head_num
        self.mapv = MLPLayer(
            input_dim,
            input_dim * head_num,
            bias=False,
            precision=precision,
            seed=child_seed(seed, 0),
        )
        self.head_map = MLPLayer(
            input_dim * head_num,
            input_dim,
            precision=precision,
            seed=child_seed(seed, 1),
        )
        self.precision = precision

    def forward(
        self,
        AA: torch.Tensor,  # nf x nloc x nnei x nnei x nh
        g2: torch.Tensor,  # nf x nloc x nnei x ng2
    ) -> torch.Tensor:
        nf, nloc, nnei, ng2 = g2.shape
        nh = self.head_num
        # nf x nloc x nnei x ng2 x nh
        g2v = self.mapv(g2).view(nf, nloc, nnei, ng2, nh)
        # nf x nloc x nh x nnei x ng2
        g2v = torch.permute(g2v, (0, 1, 4, 2, 3))
        # g2v = torch.nn.functional.normalize(g2v, dim=-1)
        # nf x nloc x nh x nnei x nnei
        AA = torch.permute(AA, (0, 1, 4, 2, 3))
        # nf x nloc x nh x nnei x ng2
        ret = torch.matmul(AA, g2v)
        # nf x nloc x nnei x ng2 x nh
        ret = torch.permute(ret, (0, 1, 3, 4, 2)).reshape(nf, nloc, nnei, (ng2 * nh))
        # nf x nloc x nnei x ng2
        return self.head_map(ret)

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        return {
            "@class": "Atten2MultiHeadApply",
            "@version": 1,
            "input_dim": self.input_dim,
            "head_num": self.head_num,
            "precision": self.precision,
            "mapv": self.mapv.serialize(),
            "head_map": self.head_map.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Atten2MultiHeadApply":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        mapv = data.pop("mapv")
        head_map = data.pop("head_map")
        obj = cls(**data)
        obj.mapv = MLPLayer.deserialize(mapv)
        obj.head_map = MLPLayer.deserialize(head_map)
        return obj


class Atten2EquiVarApply(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_num: int,
        precision: str = "float64",
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.head_num = head_num
        self.head_map = MLPLayer(
            head_num, 1, bias=False, precision=precision, seed=seed
        )
        self.precision = precision

    def forward(
        self,
        AA: torch.Tensor,  # nf x nloc x nnei x nnei x nh
        h2: torch.Tensor,  # nf x nloc x nnei x 3
    ) -> torch.Tensor:
        nf, nloc, nnei, _ = h2.shape
        nh = self.head_num
        # nf x nloc x nh x nnei x nnei
        AA = torch.permute(AA, (0, 1, 4, 2, 3))
        h2m = torch.unsqueeze(h2, dim=2)
        # nf x nloc x nh x nnei x 3
        h2m = torch.tile(h2m, [1, 1, nh, 1, 1])
        # nf x nloc x nh x nnei x 3
        ret = torch.matmul(AA, h2m)
        # nf x nloc x nnei x 3 x nh
        ret = torch.permute(ret, (0, 1, 3, 4, 2)).view(nf, nloc, nnei, 3, nh)
        # nf x nloc x nnei x 3
        return torch.squeeze(self.head_map(ret), dim=-1)

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        return {
            "@class": "Atten2EquiVarApply",
            "@version": 1,
            "input_dim": self.input_dim,
            "head_num": self.head_num,
            "precision": self.precision,
            "head_map": self.head_map.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Atten2EquiVarApply":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        head_map = data.pop("head_map")
        obj = cls(**data)
        obj.head_map = MLPLayer.deserialize(head_map)
        return obj


class LocalAtten(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        head_num: int,
        smooth: bool = True,
        attnw_shift: float = 20.0,
        precision: str = "float64",
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.mapq = MLPLayer(
            input_dim,
            hidden_dim * 1 * head_num,
            bias=False,
            precision=precision,
            seed=child_seed(seed, 0),
        )
        self.mapkv = MLPLayer(
            input_dim,
            (hidden_dim + input_dim) * head_num,
            bias=False,
            precision=precision,
            seed=child_seed(seed, 1),
        )
        self.head_map = MLPLayer(
            input_dim * head_num,
            input_dim,
            precision=precision,
            seed=child_seed(seed, 2),
        )
        self.smooth = smooth
        self.attnw_shift = attnw_shift
        self.precision = precision

    def forward(
        self,
        g1: torch.Tensor,  # nb x nloc x ng1
        gg1: torch.Tensor,  # nb x nloc x nnei x ng1
        nlist_mask: torch.Tensor,  # nb x nloc x nnei
        sw: torch.Tensor,  # nb x nloc x nnei
    ) -> torch.Tensor:
        nb, nloc, nnei = nlist_mask.shape
        ni, nd, nh = self.input_dim, self.hidden_dim, self.head_num
        assert ni == g1.shape[-1]
        assert ni == gg1.shape[-1]
        # nb x nloc x nd x nh
        g1q = self.mapq(g1).view(nb, nloc, nd, nh)
        # nb x nloc x nh x nd
        g1q = torch.permute(g1q, (0, 1, 3, 2))
        # nb x nloc x nnei x (nd+ni) x nh
        gg1kv = self.mapkv(gg1).view(nb, nloc, nnei, nd + ni, nh)
        gg1kv = torch.permute(gg1kv, (0, 1, 4, 2, 3))
        # nb x nloc x nh x nnei x nd, nb x nloc x nh x nnei x ng1
        gg1k, gg1v = torch.split(gg1kv, [nd, ni], dim=-1)

        # nb x nloc x nh x 1 x nnei
        attnw = torch.matmul(g1q.unsqueeze(-2), torch.transpose(gg1k, -1, -2)) / nd**0.5
        # nb x nloc x nh x nnei
        attnw = attnw.squeeze(-2)
        # mask the attenmap, nb x nloc x 1 x nnei
        attnw_mask = ~nlist_mask.unsqueeze(-2)
        # nb x nloc x nh x nnei
        if self.smooth:
            attnw = (attnw + self.attnw_shift) * sw.unsqueeze(-2) - self.attnw_shift
        else:
            attnw = attnw.masked_fill(
                attnw_mask,
                float("-inf"),
            )
        attnw = torch.softmax(attnw, dim=-1)
        attnw = attnw.masked_fill(
            attnw_mask,
            0.0,
        )
        if self.smooth:
            attnw = attnw * sw.unsqueeze(-2)

        # nb x nloc x nh x ng1
        ret = (
            torch.matmul(attnw.unsqueeze(-2), gg1v).squeeze(-2).view(nb, nloc, nh * ni)
        )
        # nb x nloc x ng1
        ret = self.head_map(ret)
        return ret

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        return {
            "@class": "LocalAtten",
            "@version": 1,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "head_num": self.head_num,
            "smooth": self.smooth,
            "attnw_shift": self.attnw_shift,
            "precision": self.precision,
            "mapq": self.mapq.serialize(),
            "mapkv": self.mapkv.serialize(),
            "head_map": self.head_map.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "LocalAtten":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        mapq = data.pop("mapq")
        mapkv = data.pop("mapkv")
        head_map = data.pop("head_map")
        obj = cls(**data)
        obj.mapq = MLPLayer.deserialize(mapq)
        obj.mapkv = MLPLayer.deserialize(mapkv)
        obj.head_map = MLPLayer.deserialize(head_map)
        return obj


class RepformerLayer(torch.nn.Module):
    def __init__(
        self,
        rcut,
        rcut_smth,
        sel: int,
        ntypes: int,
        g1_dim=128,
        g2_dim=16,
        axis_neuron: int = 4,
        update_chnnl_2: bool = True,
        update_g1_has_conv: bool = True,
        update_g1_has_drrd: bool = True,
        update_g1_has_grrg: bool = True,
        update_g1_has_attn: bool = True,
        update_g2_has_g1g1: bool = True,
        update_g2_has_attn: bool = True,
        update_h2: bool = False,
        attn1_hidden: int = 64,
        attn1_nhead: int = 4,
        attn2_hidden: int = 16,
        attn2_nhead: int = 4,
        attn2_has_gate: bool = False,
        activation_function: str = "tanh",
        update_style: str = "res_avg",
        update_residual: float = 0.001,
        update_residual_init: str = "norm",
        smooth: bool = True,
        precision: str = "float64",
        trainable_ln: bool = True,
        ln_eps: Optional[float] = 1e-5,
        use_sqrt_nnei: bool = True,
        g1_out_conv: bool = True,
        g1_out_mlp: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        super().__init__()
        self.epsilon = 1e-4  # protection of 1./nnei
        self.rcut = float(rcut)
        self.rcut_smth = float(rcut_smth)
        self.ntypes = ntypes
        sel = [sel] if isinstance(sel, int) else sel
        self.nnei = sum(sel)
        assert len(sel) == 1
        self.sel = sel
        self.sec = self.sel
        self.axis_neuron = axis_neuron
        self.activation_function = activation_function
        self.act = ActivationFn(activation_function)
        self.update_g1_has_grrg = update_g1_has_grrg
        self.update_g1_has_drrd = update_g1_has_drrd
        self.update_g1_has_conv = update_g1_has_conv
        self.update_g1_has_attn = update_g1_has_attn
        self.update_chnnl_2 = update_chnnl_2
        self.update_g2_has_g1g1 = update_g2_has_g1g1 if self.update_chnnl_2 else False
        self.update_g2_has_attn = update_g2_has_attn if self.update_chnnl_2 else False
        self.update_h2 = update_h2 if self.update_chnnl_2 else False
        del update_g2_has_g1g1, update_g2_has_attn, update_h2
        self.attn1_hidden = attn1_hidden
        self.attn1_nhead = attn1_nhead
        self.attn2_hidden = attn2_hidden
        self.attn2_nhead = attn2_nhead
        self.attn2_has_gate = attn2_has_gate
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.smooth = smooth
        self.g1_dim = g1_dim
        self.g2_dim = g2_dim
        self.trainable_ln = trainable_ln
        self.ln_eps = ln_eps
        self.precision = precision
        self.seed = seed
        self.use_sqrt_nnei = use_sqrt_nnei
        self.g1_out_conv = g1_out_conv
        self.g1_out_mlp = g1_out_mlp

        assert update_residual_init in [
            "norm",
            "const",
        ], "'update_residual_init' only support 'norm' or 'const'!"
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.g1_residual = []
        self.g2_residual = []
        self.h2_residual = []

        if self.update_style == "res_residual":
            self.g1_residual.append(
                get_residual(
                    g1_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 0),
                )
            )

        g1_in_dim = self.cal_1_dim(g1_dim, g2_dim, self.axis_neuron)
        self.linear1 = MLPLayer(
            g1_in_dim,
            g1_dim,
            precision=precision,
            seed=child_seed(seed, 1),
        )
        self.linear2 = None
        self.proj_g1g2 = None
        self.proj_g1g1g2 = None
        self.attn2g_map = None
        self.attn2_mh_apply = None
        self.attn2_lm = None
        self.attn2_ev_apply = None
        self.loc_attn = None

        if self.update_chnnl_2:
            self.linear2 = MLPLayer(
                g2_dim,
                g2_dim,
                precision=precision,
                seed=child_seed(seed, 2),
            )
            if self.update_style == "res_residual":
                self.g2_residual.append(
                    get_residual(
                        g2_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 3),
                    )
                )
        if self.g1_out_mlp:
            self.g1_self_mlp = MLPLayer(
                g1_dim,
                g1_dim,
                precision=precision,
                seed=child_seed(seed, 15),
            )
            if self.update_style == "res_residual":
                self.g1_residual.append(
                    get_residual(
                        g1_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 16),
                    )
                )
        else:
            self.g1_self_mlp = None
        if self.update_g1_has_conv:
            if not self.g1_out_conv:
                self.proj_g1g2 = MLPLayer(
                    g1_dim,
                    g2_dim,
                    bias=False,
                    precision=precision,
                    seed=child_seed(seed, 4),
                )
            else:
                self.proj_g1g2 = MLPLayer(
                    g2_dim,
                    g1_dim,
                    bias=False,
                    precision=precision,
                    seed=child_seed(seed, 4),
                )
                if self.update_style == "res_residual":
                    self.g1_residual.append(
                        get_residual(
                            g1_dim,
                            self.update_residual,
                            self.update_residual_init,
                            precision=precision,
                            seed=child_seed(seed, 17),
                        )
                    )
        if self.update_g2_has_g1g1:
            self.proj_g1g1g2 = MLPLayer(
                g1_dim,
                g2_dim,
                bias=False,
                precision=precision,
                seed=child_seed(seed, 5),
            )
            if self.update_style == "res_residual":
                self.g2_residual.append(
                    get_residual(
                        g2_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 6),
                    )
                )
        if self.update_g2_has_attn or self.update_h2:
            self.attn2g_map = Atten2Map(
                g2_dim,
                attn2_hidden,
                attn2_nhead,
                attn2_has_gate,
                self.smooth,
                precision=precision,
                seed=child_seed(seed, 7),
            )
            if self.update_g2_has_attn:
                self.attn2_mh_apply = Atten2MultiHeadApply(
                    g2_dim, attn2_nhead, precision=precision, seed=child_seed(seed, 8)
                )
                self.attn2_lm = LayerNorm(
                    g2_dim,
                    eps=ln_eps,
                    trainable=trainable_ln,
                    precision=precision,
                    seed=child_seed(seed, 9),
                )
                if self.update_style == "res_residual":
                    self.g2_residual.append(
                        get_residual(
                            g2_dim,
                            self.update_residual,
                            self.update_residual_init,
                            precision=precision,
                            seed=child_seed(seed, 10),
                        )
                    )

            if self.update_h2:
                self.attn2_ev_apply = Atten2EquiVarApply(
                    g2_dim, attn2_nhead, precision=precision, seed=child_seed(seed, 11)
                )
                if self.update_style == "res_residual":
                    self.h2_residual.append(
                        get_residual(
                            1,
                            self.update_residual,
                            self.update_residual_init,
                            precision=precision,
                            seed=child_seed(seed, 12),
                        )
                    )
        if self.update_g1_has_attn:
            self.loc_attn = LocalAtten(
                g1_dim,
                attn1_hidden,
                attn1_nhead,
                self.smooth,
                precision=precision,
                seed=child_seed(seed, 13),
            )
            if self.update_style == "res_residual":
                self.g1_residual.append(
                    get_residual(
                        g1_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 14),
                    )
                )

        self.g1_residual = nn.ParameterList(self.g1_residual)
        self.g2_residual = nn.ParameterList(self.g2_residual)
        self.h2_residual = nn.ParameterList(self.h2_residual)

    def cal_1_dim(self, g1d: int, g2d: int, ax: int) -> int:
        ret = g1d if not self.g1_out_mlp else 0
        if self.update_g1_has_grrg:
            ret += g2d * ax
        if self.update_g1_has_drrd:
            ret += g1d * ax
        if self.update_g1_has_conv and not self.g1_out_conv:
            ret += g2d
        return ret

    def _update_h2(
        self,
        h2: torch.Tensor,
        attn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the attention weights update for pair-wise equivariant rep.

        Parameters
        ----------
        h2
            Pair-wise equivariant rep tensors, with shape nf x nloc x nnei x 3.
        attn
            Attention weights from g2 attention, with shape nf x nloc x nnei x nnei x nh2.
        """
        assert self.attn2_ev_apply is not None
        # nf x nloc x nnei x nh2
        h2_1 = self.attn2_ev_apply(attn, h2)
        return h2_1

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
        if not self.g1_out_conv:
            # gg1  : nb x nloc x nnei x ng2
            gg1 = self.proj_g1g2(gg1).view(nb, nloc, nnei, ng2)
        else:
            gg1 = gg1.view(nb, nloc, nnei, ng1)
        # nb x nloc x nnei x ng2/ng1
        gg1 = _apply_nlist_mask(gg1, nlist_mask)
        if not self.smooth:
            # normalized by number of neighbors, not smooth
            # nb x nloc x 1
            # must use type_as here to convert bool to float, otherwise there will be numerical difference from numpy
            invnnei = 1.0 / (
                self.epsilon + torch.sum(nlist_mask.type_as(gg1), dim=-1)
            ).unsqueeze(-1)
        else:
            gg1 = _apply_switch(gg1, sw)
            invnnei = (1.0 / float(nnei)) * torch.ones(
                (nb, nloc, 1), dtype=gg1.dtype, device=gg1.device
            )
        if not self.g1_out_conv:
            # nb x nloc x ng2
            g1_11 = torch.sum(g2 * gg1, dim=2) * invnnei
        else:
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
        h2g2m = h2g2[..., :axis_neuron]
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
            use_sqrt_nnei=self.use_sqrt_nnei,
        )
        # nb x nloc x (axisxng2)
        g1_13 = self._cal_grrg(h2g2, axis_neuron)
        return g1_13

    def _update_g2_g1g1(
        self,
        g1: torch.Tensor,  # nb x nloc x ng1
        gg1: torch.Tensor,  # nb x nloc x nnei x ng1
        nlist_mask: torch.Tensor,  # nb x nloc x nnei
        sw: torch.Tensor,  # nb x nloc x nnei
    ) -> torch.Tensor:
        """
        Update the g2 using element-wise dot g1_i * g1_j.

        Parameters
        ----------
        g1
            Atomic invariant rep, with shape nb x nloc x ng1.
        gg1
            Neighbor-wise atomic invariant rep, with shape nb x nloc x nnei x ng1.
        nlist_mask
            Neighbor list mask, where zero means no neighbor, with shape nb x nloc x nnei.
        sw
            The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape nb x nloc x nnei.
        """
        ret = g1.unsqueeze(-2) * gg1
        # nb x nloc x nnei x ng1
        ret = _apply_nlist_mask(ret, nlist_mask)
        if self.smooth:
            ret = _apply_switch(ret, sw)
        return ret

    def forward(
        self,
        g1_ext: torch.Tensor,  # nf x nall x ng1
        g2: torch.Tensor,  # nf x nloc x nnei x ng2
        h2: torch.Tensor,  # nf x nloc x nnei x 3
        nlist: torch.Tensor,  # nf x nloc x nnei
        nlist_mask: torch.Tensor,  # nf x nloc x nnei
        sw: torch.Tensor,  # switch func, nf x nloc x nnei
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
        cal_gg1 = (
            self.update_g1_has_drrd
            or self.update_g1_has_conv
            or self.update_g1_has_attn
            or self.update_g2_has_g1g1
        )

        nb, nloc, nnei, _ = g2.shape
        nall = g1_ext.shape[1]
        g1, _ = torch.split(g1_ext, [nloc, nall - nloc], dim=1)
        assert (nb, nloc) == g1.shape[:2]
        assert (nb, nloc, nnei) == h2.shape[:3]

        g2_update: list[torch.Tensor] = [g2]
        h2_update: list[torch.Tensor] = [h2]
        g1_update: list[torch.Tensor] = [g1]
        g1_mlp: list[torch.Tensor] = [g1] if not self.g1_out_mlp else []
        if self.g1_out_mlp:
            assert self.g1_self_mlp is not None
            g1_self_mlp = self.act(self.g1_self_mlp(g1))
            g1_update.append(g1_self_mlp)

        if cal_gg1:
            gg1 = _make_nei_g1(g1_ext, nlist)
        else:
            gg1 = None

        if self.update_chnnl_2:
            # mlp(g2)
            assert self.linear2 is not None
            # nb x nloc x nnei x ng2
            g2_1 = self.act(self.linear2(g2))
            g2_update.append(g2_1)

            if self.update_g2_has_g1g1:
                # linear(g1_i * g1_j)
                assert gg1 is not None
                assert self.proj_g1g1g2 is not None
                g2_update.append(
                    self.proj_g1g1g2(self._update_g2_g1g1(g1, gg1, nlist_mask, sw))
                )

            if self.update_g2_has_attn or self.update_h2:
                # gated_attention(g2, h2)
                assert self.attn2g_map is not None
                # nb x nloc x nnei x nnei x nh
                AAg = self.attn2g_map(g2, h2, nlist_mask, sw)

                if self.update_g2_has_attn:
                    assert self.attn2_mh_apply is not None
                    assert self.attn2_lm is not None
                    # nb x nloc x nnei x ng2
                    g2_2 = self.attn2_mh_apply(AAg, g2)
                    g2_2 = self.attn2_lm(g2_2)
                    g2_update.append(g2_2)

                if self.update_h2:
                    # linear_head(attention_weights * h2)
                    h2_update.append(self._update_h2(h2, AAg))

        if self.update_g1_has_conv:
            assert gg1 is not None
            g1_conv = self._update_g1_conv(gg1, g2, nlist_mask, sw)
            if not self.g1_out_conv:
                g1_mlp.append(g1_conv)
            else:
                g1_update.append(g1_conv)

        if self.update_g1_has_grrg:
            g1_mlp.append(
                self.symmetrization_op(
                    g2,
                    h2,
                    nlist_mask,
                    sw,
                    self.axis_neuron,
                    smooth=self.smooth,
                    epsilon=self.epsilon,
                )
            )

        if self.update_g1_has_drrd:
            assert gg1 is not None
            g1_mlp.append(
                self.symmetrization_op(
                    gg1,
                    h2,
                    nlist_mask,
                    sw,
                    self.axis_neuron,
                    smooth=self.smooth,
                    epsilon=self.epsilon,
                )
            )

        # nb x nloc x [ng1+ng2+(axisxng2)+(axisxng1)]
        #                  conv   grrg      drrd
        g1_1 = self.act(self.linear1(torch.cat(g1_mlp, dim=-1)))
        g1_update.append(g1_1)

        if self.update_g1_has_attn:
            assert gg1 is not None
            assert self.loc_attn is not None
            g1_update.append(self.loc_attn(g1, gg1, nlist_mask, sw))

        # update
        if self.update_chnnl_2:
            g2_new = self.list_update(g2_update, "g2")
            h2_new = self.list_update(h2_update, "h2")
        else:
            g2_new, h2_new = g2, h2
        g1_new = self.list_update(g1_update, "g1")
        return g1_new, g2_new, h2_new

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
            "@version": 2,
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel,
            "ntypes": self.ntypes,
            "g1_dim": self.g1_dim,
            "g2_dim": self.g2_dim,
            "axis_neuron": self.axis_neuron,
            "update_chnnl_2": self.update_chnnl_2,
            "update_g1_has_conv": self.update_g1_has_conv,
            "update_g1_has_drrd": self.update_g1_has_drrd,
            "update_g1_has_grrg": self.update_g1_has_grrg,
            "update_g1_has_attn": self.update_g1_has_attn,
            "update_g2_has_g1g1": self.update_g2_has_g1g1,
            "update_g2_has_attn": self.update_g2_has_attn,
            "update_h2": self.update_h2,
            "attn1_hidden": self.attn1_hidden,
            "attn1_nhead": self.attn1_nhead,
            "attn2_hidden": self.attn2_hidden,
            "attn2_nhead": self.attn2_nhead,
            "attn2_has_gate": self.attn2_has_gate,
            "activation_function": self.activation_function,
            "update_style": self.update_style,
            "smooth": self.smooth,
            "precision": self.precision,
            "trainable_ln": self.trainable_ln,
            "use_sqrt_nnei": self.use_sqrt_nnei,
            "g1_out_conv": self.g1_out_conv,
            "g1_out_mlp": self.g1_out_mlp,
            "ln_eps": self.ln_eps,
            "linear1": self.linear1.serialize(),
        }
        if self.update_chnnl_2:
            data.update(
                {
                    "linear2": self.linear2.serialize(),
                }
            )
        if self.update_g1_has_conv:
            data.update(
                {
                    "proj_g1g2": self.proj_g1g2.serialize(),
                }
            )
        if self.update_g2_has_g1g1:
            data.update(
                {
                    "proj_g1g1g2": self.proj_g1g1g2.serialize(),
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
                    "g1_self_mlp": self.g1_self_mlp.serialize(),
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
    def deserialize(cls, data: dict) -> "RepformerLayer":
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
        proj_g1g1g2 = data.pop("proj_g1g1g2", None)
        attn2g_map = data.pop("attn2g_map", None)
        attn2_mh_apply = data.pop("attn2_mh_apply", None)
        attn2_lm = data.pop("attn2_lm", None)
        attn2_ev_apply = data.pop("attn2_ev_apply", None)
        loc_attn = data.pop("loc_attn", None)
        g1_self_mlp = data.pop("g1_self_mlp", None)
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
        if update_g2_has_g1g1:
            assert isinstance(proj_g1g1g2, dict)
            obj.proj_g1g1g2 = MLPLayer.deserialize(proj_g1g1g2)
        if update_g2_has_attn or update_h2:
            assert isinstance(attn2g_map, dict)
            obj.attn2g_map = Atten2Map.deserialize(attn2g_map)
            if update_g2_has_attn:
                assert isinstance(attn2_mh_apply, dict)
                assert isinstance(attn2_lm, dict)
                obj.attn2_mh_apply = Atten2MultiHeadApply.deserialize(attn2_mh_apply)
                obj.attn2_lm = LayerNorm.deserialize(attn2_lm)
            if update_h2:
                assert isinstance(attn2_ev_apply, dict)
                obj.attn2_ev_apply = Atten2EquiVarApply.deserialize(attn2_ev_apply)
        if update_g1_has_attn:
            assert isinstance(loc_attn, dict)
            obj.loc_attn = LocalAtten.deserialize(loc_attn)
        if g1_out_mlp:
            assert isinstance(g1_self_mlp, dict)
            obj.g1_self_mlp = MLPLayer.deserialize(g1_self_mlp)
        if update_style == "res_residual":
            for ii, t in enumerate(obj.g1_residual):
                t.data = to_torch_tensor(g1_residual[ii])
            for ii, t in enumerate(obj.g2_residual):
                t.data = to_torch_tensor(g2_residual[ii])
            for ii, t in enumerate(obj.h2_residual):
                t.data = to_torch_tensor(h2_residual[ii])
        return obj
