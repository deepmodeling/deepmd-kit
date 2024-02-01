# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_func

from deepmd.pt.model.descriptor.descriptor import (
    DescriptorBlock,
    compute_std,
)
from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat_se_a,
)
from deepmd.pt.model.network.mlp import (
    EmbeddingNet,
    LayerNorm,
    MLPLayer,
    NetworkCollection,
)
from deepmd.pt.model.network.network import (
    NeighborWiseAttention,
    TypeFilter,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)


@DescriptorBlock.register("se_atten")
class DescrptBlockSeAtten(DescriptorBlock):
    def __init__(
        self,
        rcut,
        rcut_smth,
        sel,
        ntypes: int,
        neuron: list = [25, 50, 100],
        axis_neuron: int = 16,
        tebd_dim: int = 8,
        tebd_input_mode: str = "concat",
        # set_davg_zero: bool = False,
        set_davg_zero: bool = True,  # TODO
        attn: int = 128,
        attn_layer: int = 2,
        attn_dotr: bool = True,
        attn_mask: bool = False,
        activation_function="tanh",
        precision: str = "float64",
        resnet_dt: bool = False,
        scaling_factor=1.0,
        normalize=True,
        temperature=None,
        type: Optional[str] = None,
        old_impl: bool = False,
        **kwargs,
    ):
        """Construct an embedding net of type `se_atten`.

        Args:
        - rcut: Cut-off radius.
        - rcut_smth: Smooth hyper-parameter for pair force & energy.
        - sel: For each element type, how many atoms is selected as neighbors.
        - filter_neuron: Number of neurons in each hidden layers of the embedding net.
        - axis_neuron: Number of columns of the sub-matrix of the embedding matrix.
        """
        super().__init__()
        del type
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.neuron = neuron
        self.filter_neuron = self.neuron
        self.filter_neuron = neuron
        self.axis_neuron = axis_neuron
        self.tebd_dim = tebd_dim
        self.tebd_input_mode = tebd_input_mode
        self.set_davg_zero = set_davg_zero
        self.attn_dim = attn
        self.attn_layer = attn_layer
        self.attn_dotr = attn_dotr
        self.attn_mask = attn_mask
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.resnet_dt = resnet_dt
        self.scaling_factor = scaling_factor
        self.normalize = normalize
        self.temperature = temperature
        self.old_impl = old_impl

        if isinstance(sel, int):
            sel = [sel]

        self.ntypes = ntypes
        self.sel = sel
        self.sec = self.sel
        self.split_sel = self.sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 4
        if self.old_impl:
            self.dpa1_attention = NeighborWiseAttention(
                self.attn_layer,
                self.nnei,
                self.filter_neuron[-1],
                self.attn_dim,
                dotr=self.attn_dotr,
                do_mask=self.attn_mask,
                activation=self.activation_function,
                scaling_factor=self.scaling_factor,
                normalize=self.normalize,
                temperature=self.temperature,
            )
        else:
            self.dpa1_attention = NeighborGatedAttention(
                self.attn_layer,
                self.nnei,
                self.filter_neuron[-1],
                self.attn_dim,
                dotr=self.attn_dotr,
                do_mask=self.attn_mask,
                scaling_factor=self.scaling_factor,
                normalize=self.normalize,
                temperature=self.temperature,
            )

        wanted_shape = (self.ntypes, self.nnei, 4)
        mean = torch.zeros(
            wanted_shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
        )
        stddev = torch.ones(
            wanted_shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
        )
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.embd_input_dim = (
            1 + self.tebd_dim * 2 if self.tebd_input_mode in ["concat"] else 1
        )
        self.filter_layers_old = None
        self.filter_layers = None

        if self.old_impl:
            filter_layers = []
            one = TypeFilter(
                0,
                self.nnei,
                self.filter_neuron,
                return_G=True,
                tebd_dim=self.tebd_dim,
                use_tebd=True,
                tebd_mode=self.tebd_input_mode,
            )
            filter_layers.append(one)
            self.filter_layers_old = torch.nn.ModuleList(filter_layers)
        else:
            filter_layers = NetworkCollection(
                ndim=0, ntypes=len(sel), network_type="embedding_network"
            )
            filter_layers[0] = EmbeddingNet(
                self.embd_input_dim,
                self.filter_neuron,
                activation_function=self.activation_function,
                precision=self.precision,
                resnet_dt=self.resnet_dt,
            )
            self.filter_layers = filter_layers

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.rcut

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return sum(self.sel)

    def get_sel(self) -> List[int]:
        """Returns the number of selected atoms for each type."""
        return self.sel

    def get_ntype(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def get_dim_in(self) -> int:
        """Returns the output dimension."""
        return self.dim_in

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        return self.dim_out

    @property
    def dim_out(self):
        """Returns the output dimension of this descriptor."""
        return self.filter_neuron[-1] * self.axis_neuron

    @property
    def dim_in(self):
        """Returns the atomic input dimension of this descriptor."""
        return self.tebd_dim

    @property
    def dim_emb(self):
        """Returns the output dimension of embedding."""
        return self.filter_neuron[-1]

    def __setitem__(self, key, value):
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

    def compute_input_stats(self, merged):
        """Update mean and stddev for descriptor elements."""
        sumr = []
        suma = []
        sumn = []
        sumr2 = []
        suma2 = []
        mixed_type = "real_natoms_vec" in merged[0]
        for system in merged:
            index = system["mapping"].unsqueeze(-1).expand(-1, -1, 3)
            extended_coord = torch.gather(system["coord"], dim=1, index=index)
            extended_coord = extended_coord - system["shift"]
            env_mat, _, _ = prod_env_mat_se_a(
                extended_coord,
                system["nlist"],
                system["atype"],
                self.mean,
                self.stddev,
                self.rcut,
                self.rcut_smth,
            )
            if not mixed_type:
                sysr, sysr2, sysa, sysa2, sysn = analyze_descrpt(
                    env_mat.detach().cpu().numpy(), self.ndescrpt, system["natoms"]
                )
            else:
                sysr, sysr2, sysa, sysa2, sysn = analyze_descrpt(
                    env_mat.detach().cpu().numpy(),
                    self.ndescrpt,
                    system["real_natoms_vec"],
                    mixed_type=mixed_type,
                    real_atype=system["atype"].detach().cpu().numpy(),
                )
            sumr.append(sysr)
            suma.append(sysa)
            sumn.append(sysn)
            sumr2.append(sysr2)
            suma2.append(sysa2)
        sumr = np.sum(sumr, axis=0)
        suma = np.sum(suma, axis=0)
        sumn = np.sum(sumn, axis=0)
        sumr2 = np.sum(sumr2, axis=0)
        suma2 = np.sum(suma2, axis=0)
        return sumr, suma, sumn, sumr2, suma2

    def init_desc_stat(self, sumr, suma, sumn, sumr2, suma2):
        all_davg = []
        all_dstd = []
        for type_i in range(self.ntypes):
            davgunit = [[sumr[type_i] / (sumn[type_i] + 1e-15), 0, 0, 0]]
            dstdunit = [
                [
                    compute_std(sumr2[type_i], sumr[type_i], sumn[type_i], self.rcut),
                    compute_std(suma2[type_i], suma[type_i], sumn[type_i], self.rcut),
                    compute_std(suma2[type_i], suma[type_i], sumn[type_i], self.rcut),
                    compute_std(suma2[type_i], suma[type_i], sumn[type_i], self.rcut),
                ]
            ]
            davg = np.tile(davgunit, [self.nnei, 1])
            dstd = np.tile(dstdunit, [self.nnei, 1])
            all_davg.append(davg)
            all_dstd.append(dstd)
        self.sumr = sumr
        self.suma = suma
        self.sumn = sumn
        self.sumr2 = sumr2
        self.suma2 = suma2
        if not self.set_davg_zero:
            mean = np.stack(all_davg)
            self.mean.copy_(torch.tensor(mean, device=env.DEVICE))
        stddev = np.stack(all_dstd)
        self.stddev.copy_(torch.tensor(stddev, device=env.DEVICE))

    def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_atype_embd: Optional[torch.Tensor] = None,
        mapping: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Calculate decoded embedding for each atom.

        Args:
        - coord: Tell atom coordinates with shape [nframes, natoms[1]*3].
        - atype: Tell atom types with shape [nframes, natoms[1]].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].
        - box: Tell simulation box with shape [nframes, 9].

        Returns
        -------
        - result: descriptor with shape [nframes, nloc, self.filter_neuron[-1] * self.axis_neuron].
        - ret: environment matrix with shape [nframes, nloc, self.neei, out_size]
        """
        del mapping
        assert extended_atype_embd is not None
        nframes, nloc, nnei = nlist.shape
        atype = extended_atype[:, :nloc]
        nb = nframes
        nall = extended_coord.view(nb, -1, 3).shape[1]
        dmatrix, diff, sw = prod_env_mat_se_a(
            extended_coord,
            nlist,
            atype,
            self.mean,
            self.stddev,
            self.rcut,
            self.rcut_smth,
        )
        # [nfxnlocxnnei, self.ndescrpt]
        nlist_mask = nlist != -1
        nlist[nlist == -1] = 0
        sw = torch.squeeze(sw, -1)
        # beyond the cutoff sw should be 0.0
        sw = sw.masked_fill(~nlist_mask, 0.0)
        # nf x nloc x nt -> nf x nloc x nnei x nt
        atype_tebd = extended_atype_embd[:, :nloc, :]
        atype_tebd_nnei = atype_tebd.unsqueeze(2).expand(-1, -1, self.nnei, -1)
        # nf x nall x nt
        nt = extended_atype_embd.shape[-1]
        atype_tebd_ext = extended_atype_embd
        # nb x (nloc x nnei) x nt
        index = nlist.reshape(nb, nloc * nnei).unsqueeze(-1).expand(-1, -1, nt)
        # nb x (nloc x nnei) x nt
        atype_tebd_nlist = torch.gather(atype_tebd_ext, dim=1, index=index)
        # nb x nloc x nnei x nt
        atype_tebd_nlist = atype_tebd_nlist.view(nb, nloc, nnei, nt)
        if self.old_impl:
            assert self.filter_layers_old is not None
            dmatrix = dmatrix.view(
                -1, self.ndescrpt
            )  # shape is [nframes*nall, self.ndescrpt]
            gg = self.filter_layers_old[0](
                dmatrix,
                atype_tebd=atype_tebd_nnei,
                nlist_tebd=atype_tebd_nlist,
            )  # shape is [nframes*nall, self.neei, out_size]
            input_r = torch.nn.functional.normalize(
                dmatrix.reshape(-1, self.nnei, 4)[:, :, 1:4], dim=-1
            )
            gg = self.dpa1_attention(
                gg, nlist_mask, input_r=input_r, sw=sw
            )  # shape is [nframes*nloc, self.neei, out_size]
            inputs_reshape = dmatrix.view(-1, self.nnei, 4).permute(
                0, 2, 1
            )  # shape is [nframes*natoms[0], 4, self.neei]
            xyz_scatter = torch.matmul(
                inputs_reshape, gg
            )  # shape is [nframes*natoms[0], 4, out_size]
        else:
            assert self.filter_layers is not None
            dmatrix = dmatrix.view(-1, self.nnei, 4)
            nfnl = dmatrix.shape[0]
            # nfnl x nnei x 4
            rr = dmatrix
            ss = rr[:, :, :1]
            if self.tebd_input_mode in ["concat"]:
                nlist_tebd = atype_tebd_nlist.reshape(nfnl, nnei, self.tebd_dim)
                atype_tebd = atype_tebd_nnei.reshape(nfnl, nnei, self.tebd_dim)
                # nfnl x nnei x (1 + tebd_dim * 2)
                ss = torch.concat([ss, nlist_tebd, atype_tebd], dim=2)
            # nfnl x nnei x ng
            gg = self.filter_layers._networks[0](ss)
            input_r = torch.nn.functional.normalize(
                dmatrix.reshape(-1, self.nnei, 4)[:, :, 1:4], dim=-1
            )
            gg = self.dpa1_attention(
                gg, nlist_mask, input_r=input_r, sw=sw
            )  # shape is [nframes*nloc, self.neei, out_size]
            # nfnl x 4 x ng
            xyz_scatter = torch.matmul(rr.permute(0, 2, 1), gg)
        xyz_scatter = xyz_scatter / self.nnei
        xyz_scatter_1 = xyz_scatter.permute(0, 2, 1)
        rot_mat = xyz_scatter_1[:, :, 1:4]
        xyz_scatter_2 = xyz_scatter[:, :, 0 : self.axis_neuron]
        result = torch.matmul(
            xyz_scatter_1, xyz_scatter_2
        )  # shape is [nframes*nloc, self.filter_neuron[-1], self.axis_neuron]
        return (
            result.view(-1, nloc, self.filter_neuron[-1] * self.axis_neuron),
            gg.view(-1, nloc, self.nnei, self.filter_neuron[-1]),
            dmatrix.view(-1, nloc, self.nnei, 4)[..., 1:],
            rot_mat.view(-1, self.filter_neuron[-1], 3),
            sw,
        )


class NeighborGatedAttention(nn.Module):
    def __init__(
        self,
        layer_num: int,
        nnei: int,
        embed_dim: int,
        hidden_dim: int,
        dotr: bool = False,
        do_mask: bool = False,
        scaling_factor: float = 1.0,
        normalize: bool = True,
        temperature: Optional[float] = None,
        precision: str = DEFAULT_PRECISION,
    ):
        """Construct a neighbor-wise attention net."""
        super().__init__()
        self.layer_num = layer_num
        self.nnei = nnei
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dotr = dotr
        self.do_mask = do_mask
        self.scaling_factor = scaling_factor
        self.normalize = normalize
        self.temperature = temperature
        self.precision = precision
        self.network_type = NeighborGatedAttentionLayer
        attention_layers = []
        for i in range(self.layer_num):
            attention_layers.append(
                NeighborGatedAttentionLayer(
                    nnei,
                    embed_dim,
                    hidden_dim,
                    dotr=dotr,
                    do_mask=do_mask,
                    scaling_factor=scaling_factor,
                    normalize=normalize,
                    temperature=temperature,
                    precision=precision,
                )
            )
        self.attention_layers = nn.ModuleList(attention_layers)

    def forward(
        self,
        input_G,
        nei_mask,
        input_r: Optional[torch.Tensor] = None,
        sw: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_G: Input G, [nframes * nloc, nnei, embed_dim].
            nei_mask: neighbor mask, [nframes * nloc, nnei].
            input_r: normalized radial, [nframes, nloc, nei, 3].

        Returns
        -------
        out: Output G, [nframes * nloc, nnei, embed_dim]
        """
        out = input_G
        # https://github.com/pytorch/pytorch/issues/39165#issuecomment-635472592
        for layer in self.attention_layers:
            out = layer(out, nei_mask, input_r=input_r, sw=sw)
        return out

    def _convert_key(self, key):
        if isinstance(key, int):
            idx = key
        else:
            if isinstance(key, tuple):
                pass
            elif isinstance(key, str):
                key = tuple([int(tt) for tt in key.split("_")[1:]])
            else:
                raise TypeError(key)
            assert isinstance(key, tuple)
            assert len(key) == self.ndim
            idx = sum([tt * self.ntypes**ii for ii, tt in enumerate(key)])
        return idx

    def __getitem__(self, key):
        return self.attention_layers[self._convert_key(key)]

    def __setitem__(self, key, value):
        if isinstance(value, self.network_type):
            pass
        elif isinstance(value, dict):
            value = self.network_type.deserialize(value)
        else:
            raise TypeError(value)
        self.attention_layers[self._convert_key(key)] = value

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        # network_type_map_inv = {v: k for k, v in self.NETWORK_TYPE_MAP.items()}
        # network_type_name = network_type_map_inv[self.network_type]
        return {
            "layer_num": self.layer_num,
            "nnei": self.nnei,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "dotr": self.dotr,
            "do_mask": self.do_mask,
            "scaling_factor": self.scaling_factor,
            "normalize": self.normalize,
            "temperature": self.temperature,
            "precision": self.precision,
            "attention_layers": [layer.serialize() for layer in self.attention_layers],
        }

    @classmethod
    def deserialize(cls, data: dict) -> "NeighborGatedAttention":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        attention_layers = data.pop("attention_layers")
        obj = cls(**data)
        for ii, network in enumerate(attention_layers):
            obj[ii] = network
        return obj


class NeighborGatedAttentionLayer(nn.Module):
    def __init__(
        self,
        nnei: int,
        embed_dim: int,
        hidden_dim: int,
        dotr: bool = False,
        do_mask: bool = False,
        scaling_factor: float = 1.0,
        normalize: bool = True,
        temperature: Optional[float] = None,
        precision: str = DEFAULT_PRECISION,
    ):
        """Construct a neighbor-wise attention layer."""
        super().__init__()
        self.nnei = nnei
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dotr = dotr
        self.do_mask = do_mask
        self.scaling_factor = scaling_factor
        self.normalize = normalize
        self.temperature = temperature
        self.precision = precision
        self.attention_layer = GatedAttentionLayer(
            nnei,
            embed_dim,
            hidden_dim,
            dotr=dotr,
            do_mask=do_mask,
            scaling_factor=scaling_factor,
            normalize=normalize,
            temperature=temperature,
            precision=precision,
        )
        self.attn_layer_norm = LayerNorm(self.embed_dim, precision=precision)

    def forward(
        self,
        x,
        nei_mask,
        input_r: Optional[torch.Tensor] = None,
        sw: Optional[torch.Tensor] = None,
    ):
        residual = x
        x = self.attention_layer(x, nei_mask, input_r=input_r, sw=sw)
        x = residual + x
        x = self.attn_layer_norm(x)
        return x

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        return {
            "nnei": self.nnei,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "dotr": self.dotr,
            "do_mask": self.do_mask,
            "scaling_factor": self.scaling_factor,
            "normalize": self.normalize,
            "temperature": self.temperature,
            "precision": self.precision,
            "attention_layer": self.attention_layer.serialize(),
            "attn_layer_norm": self.attn_layer_norm.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "NeighborGatedAttentionLayer":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        attention_layer = data.pop("attention_layer")
        attn_layer_norm = data.pop("attn_layer_norm")
        obj = cls(**data)
        obj.attention_layer = GatedAttentionLayer.deserialize(attention_layer)
        obj.attn_layer_norm = LayerNorm.deserialize(attn_layer_norm)
        return obj


class GatedAttentionLayer(nn.Module):
    def __init__(
        self,
        nnei: int,
        embed_dim: int,
        hidden_dim: int,
        dotr: bool = False,
        do_mask: bool = False,
        scaling_factor: float = 1.0,
        normalize: bool = True,
        temperature: Optional[float] = None,
        bias: bool = True,
        smooth: bool = True,
        precision: str = DEFAULT_PRECISION,
    ):
        """Construct a neighbor-wise attention net."""
        super().__init__()
        self.nnei = nnei
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dotr = dotr
        self.do_mask = do_mask
        self.bias = bias
        self.smooth = smooth
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        self.precision = precision
        if temperature is None:
            self.scaling = (self.hidden_dim * scaling_factor) ** -0.5
        else:
            self.scaling = temperature
        self.normalize = normalize
        self.in_proj = MLPLayer(
            embed_dim,
            hidden_dim * 3,
            bias=bias,
            use_timestep=False,
            bavg=0.0,
            stddev=1.0,
            precision=precision,
        )
        self.out_proj = MLPLayer(
            hidden_dim,
            embed_dim,
            bias=bias,
            use_timestep=False,
            bavg=0.0,
            stddev=1.0,
            precision=precision,
        )

    def forward(
        self,
        query,
        nei_mask,
        input_r: Optional[torch.Tensor] = None,
        sw: Optional[torch.Tensor] = None,
        attnw_shift: float = 20.0,
    ):
        """
        Args:
            query: input G, [nframes * nloc, nnei, embed_dim].
            nei_mask: neighbor mask, [nframes * nloc, nnei].
            input_r: normalized radial, [nframes, nloc, nei, 3].

        Returns
        -------
        type_embedding:
        """
        q, k, v = self.in_proj(query).chunk(3, dim=-1)
        #  [nframes * nloc, nnei, hidden_dim]
        q = q.view(-1, self.nnei, self.hidden_dim)
        k = k.view(-1, self.nnei, self.hidden_dim)
        v = v.view(-1, self.nnei, self.hidden_dim)
        if self.normalize:
            q = torch_func.normalize(q, dim=-1)
            k = torch_func.normalize(k, dim=-1)
            v = torch_func.normalize(v, dim=-1)
        q = q * self.scaling
        k = k.transpose(1, 2)
        #  [nframes * nloc, nnei, nnei]
        attn_weights = torch.bmm(q, k)
        #  [nframes * nloc, nnei]
        nei_mask = nei_mask.view(-1, self.nnei)
        if self.smooth:
            # [nframes * nloc, nnei]
            assert sw is not None
            sw = sw.view([-1, self.nnei])
            attn_weights = (attn_weights + attnw_shift) * sw[:, :, None] * sw[
                :, None, :
            ] - attnw_shift
        else:
            attn_weights = attn_weights.masked_fill(
                ~nei_mask.unsqueeze(1), float("-inf")
            )
        attn_weights = torch_func.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.masked_fill(~nei_mask.unsqueeze(-1), 0.0)
        if self.smooth:
            assert sw is not None
            attn_weights = attn_weights * sw[:, :, None] * sw[:, None, :]
        if self.dotr:
            assert input_r is not None, "input_r must be provided when dotr is True!"
            angular_weight = torch.bmm(input_r, input_r.transpose(1, 2))
            attn_weights = attn_weights * angular_weight
        o = torch.bmm(attn_weights, v)
        output = self.out_proj(o)
        return output

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        # network_type_map_inv = {v: k for k, v in self.NETWORK_TYPE_MAP.items()}
        # network_type_name = network_type_map_inv[self.network_type]
        return {
            "nnei": self.nnei,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "dotr": self.dotr,
            "do_mask": self.do_mask,
            "scaling_factor": self.scaling_factor,
            "normalize": self.normalize,
            "temperature": self.temperature,
            "bias": self.bias,
            "smooth": self.smooth,
            "precision": self.precision,
            "in_proj": self.in_proj.serialize(),
            "out_proj": self.out_proj.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "GatedAttentionLayer":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        in_proj = data.pop("in_proj")
        out_proj = data.pop("out_proj")
        obj = cls(**data)
        obj.in_proj = MLPLayer.deserialize(in_proj)
        obj.out_proj = MLPLayer.deserialize(out_proj)
        return obj


def analyze_descrpt(matrix, ndescrpt, natoms, mixed_type=False, real_atype=None):
    """Collect avg, square avg and count of descriptors in a batch."""
    ntypes = natoms.shape[1] - 2
    if not mixed_type:
        sysr = []
        sysa = []
        sysn = []
        sysr2 = []
        sysa2 = []
        start_index = 0
        for type_i in range(ntypes):
            end_index = start_index + natoms[0, 2 + type_i]
            dd = matrix[:, start_index:end_index]
            start_index = end_index
            dd = np.reshape(
                dd, [-1, 4]
            )  # Shape is [nframes*natoms[2+type_id]*self.nnei, 4]
            ddr = dd[:, :1]
            dda = dd[:, 1:]
            sumr = np.sum(ddr)
            suma = np.sum(dda) / 3.0
            sumn = dd.shape[0]  # Value is nframes*natoms[2+type_id]*self.nnei
            sumr2 = np.sum(np.multiply(ddr, ddr))
            suma2 = np.sum(np.multiply(dda, dda)) / 3.0
            sysr.append(sumr)
            sysa.append(suma)
            sysn.append(sumn)
            sysr2.append(sumr2)
            sysa2.append(suma2)
    else:
        sysr = [0.0 for i in range(ntypes)]
        sysa = [0.0 for i in range(ntypes)]
        sysn = [0 for i in range(ntypes)]
        sysr2 = [0.0 for i in range(ntypes)]
        sysa2 = [0.0 for i in range(ntypes)]
        for frame_item in range(matrix.shape[0]):
            dd_ff = matrix[frame_item]
            atype_frame = real_atype[frame_item]
            for type_i in range(ntypes):
                type_idx = atype_frame == type_i
                dd = dd_ff[type_idx]
                dd = np.reshape(dd, [-1, 4])  # typen_atoms * nnei, 4
                ddr = dd[:, :1]
                dda = dd[:, 1:]
                sumr = np.sum(ddr)
                suma = np.sum(dda) / 3.0
                sumn = dd.shape[0]
                sumr2 = np.sum(np.multiply(ddr, ddr))
                suma2 = np.sum(np.multiply(dda, dda)) / 3.0
                sysr[type_i] += sumr
                sysa[type_i] += suma
                sysn[type_i] += sumn
                sysr2[type_i] += sumr2
                sysa2[type_i] += suma2

    return sysr, sysr2, sysa, sysa2, sysn
