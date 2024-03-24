# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepmd.pt.utils import (
    env,
)

try:
    from typing import (
        Final,
    )
except ImportError:
    from torch.jit import Final

from functools import (
    partial,
)

import torch.utils.checkpoint

from deepmd.pt.utils.utils import (
    ActivationFn,
    get_activation_fn,
)


def Tensor(*shape):
    return torch.empty(shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)


class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and self.training:
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, prob=None):
        super().__init__()
        self.drop_prob = prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return f"prob={self.drop_prob}"


def softmax_dropout(
    input_x, dropout_prob, is_training=True, mask=None, bias=None, inplace=True
):
    input_x = input_x.contiguous()
    if not inplace:
        input_x = input_x.clone()
    if mask is not None:
        input_x += mask
    if bias is not None:
        input_x += bias
    return F.dropout(F.softmax(input_x, dim=-1), p=dropout_prob, training=is_training)


def checkpoint_sequential(
    functions,
    input_x,
    enabled=True,
):
    def wrap_tuple(a):
        return (a,) if type(a) is not tuple else a

    def exec(func, a):
        return wrap_tuple(func(*a))

    def get_wrap_exec(func):
        def wrap_exec(*a):
            return exec(func, a)

        return wrap_exec

    input_x = wrap_tuple(input_x)

    is_grad_enabled = torch.is_grad_enabled()

    if enabled and is_grad_enabled:
        for func in functions:
            input_x = torch.utils.checkpoint.checkpoint(get_wrap_exec(func), *input_x)
    else:
        for func in functions:
            input_x = exec(func, input_x)
    return input_x


class ResidualLinear(nn.Module):
    resnet: Final[int]

    def __init__(self, num_in, num_out, bavg=0.0, stddev=1.0, resnet_dt=False):
        """Construct a residual linear layer.

        Args:
        - num_in: Width of input tensor.
        - num_out: Width of output tensor.
        - resnet_dt: Using time-step in the ResNet construction.
        """
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.resnet = resnet_dt

        self.matrix = nn.Parameter(data=Tensor(num_in, num_out))
        nn.init.normal_(self.matrix.data, std=stddev / np.sqrt(num_out + num_in))
        self.bias = nn.Parameter(data=Tensor(1, num_out))
        nn.init.normal_(self.bias.data, mean=bavg, std=stddev)
        if self.resnet:
            self.idt = nn.Parameter(data=Tensor(1, num_out))
            nn.init.normal_(self.idt.data, mean=1.0, std=0.001)

    def forward(self, inputs):
        """Return X ?+ X*W+b."""
        xw_plus_b = torch.matmul(inputs, self.matrix) + self.bias
        hidden = torch.tanh(xw_plus_b)
        if self.resnet:
            hidden = hidden * self.idt
        if self.num_in == self.num_out:
            return inputs + hidden
        elif self.num_in * 2 == self.num_out:
            return torch.cat([inputs, inputs], dim=1) + hidden
        else:
            return hidden


class TypeFilter(nn.Module):
    use_tebd: Final[bool]
    tebd_mode: Final[str]

    def __init__(
        self,
        offset,
        length,
        neuron,
        return_G=False,
        tebd_dim=0,
        use_tebd=False,
        tebd_mode="concat",
    ):
        """Construct a filter on the given element as neighbor.

        Args:
        - offset: Element offset in the descriptor matrix.
        - length: Atom count of this element.
        - neuron: Number of neurons in each hidden layers of the embedding net.
        """
        super().__init__()
        self.offset = offset
        self.length = length
        self.tebd_dim = tebd_dim
        self.use_tebd = use_tebd
        self.tebd_mode = tebd_mode
        supported_tebd_mode = ["concat", "dot", "dot_residual_s", "dot_residual_t"]
        assert (
            tebd_mode in supported_tebd_mode
        ), f"Unknown tebd_mode {tebd_mode}! Supported are {supported_tebd_mode}."
        if use_tebd and tebd_mode == "concat":
            self.neuron = [1 + tebd_dim * 2, *neuron]
        else:
            self.neuron = [1, *neuron]

        deep_layers = []
        for ii in range(1, len(self.neuron)):
            one = ResidualLinear(self.neuron[ii - 1], self.neuron[ii])
            deep_layers.append(one)
        self.deep_layers = nn.ModuleList(deep_layers)

        deep_layers_t = []
        if use_tebd and tebd_mode in ["dot", "dot_residual_s", "dot_residual_t"]:
            self.neuron_t = [tebd_dim * 2, *neuron]
            for ii in range(1, len(self.neuron_t)):
                one = ResidualLinear(self.neuron_t[ii - 1], self.neuron_t[ii])
                deep_layers_t.append(one)
        self.deep_layers_t = nn.ModuleList(deep_layers_t)

        self.return_G = return_G

    def forward(
        self,
        inputs,
        atype_tebd: Optional[torch.Tensor] = None,
        nlist_tebd: Optional[torch.Tensor] = None,
    ):
        """Calculate decoded embedding for each atom.

        Args:
        - inputs: Descriptor matrix. Its shape is [nframes*natoms[0], len_descriptor].

        Returns
        -------
        - `torch.Tensor`: Embedding contributed by me. Its shape is [nframes*natoms[0], 4, self.neuron[-1]].
        """
        inputs_i = inputs[:, self.offset * 4 : (self.offset + self.length) * 4]
        inputs_reshape = inputs_i.reshape(
            -1, 4
        )  # shape is [nframes*natoms[0]*self.length, 4]
        xyz_scatter = inputs_reshape[:, 0:1]

        # concat the tebd as input
        if self.use_tebd and self.tebd_mode == "concat":
            assert nlist_tebd is not None and atype_tebd is not None
            nlist_tebd = nlist_tebd.reshape(-1, self.tebd_dim)
            atype_tebd = atype_tebd.reshape(-1, self.tebd_dim)
            # [nframes * nloc * nnei, 1 + tebd_dim * 2]
            xyz_scatter = torch.concat([xyz_scatter, nlist_tebd, atype_tebd], dim=1)

        for linear in self.deep_layers:
            xyz_scatter = linear(xyz_scatter)
            # [nframes * nloc * nnei, out_size]

        # dot the tebd output
        if self.use_tebd and self.tebd_mode in [
            "dot",
            "dot_residual_s",
            "dot_residual_t",
        ]:
            assert nlist_tebd is not None and atype_tebd is not None
            nlist_tebd = nlist_tebd.reshape(-1, self.tebd_dim)
            atype_tebd = atype_tebd.reshape(-1, self.tebd_dim)
            # [nframes * nloc * nnei, tebd_dim * 2]
            two_side_tebd = torch.concat([nlist_tebd, atype_tebd], dim=1)
            for linear in self.deep_layers_t:
                two_side_tebd = linear(two_side_tebd)
                # [nframes * nloc * nnei, out_size]
            if self.tebd_mode == "dot":
                xyz_scatter = xyz_scatter * two_side_tebd
            elif self.tebd_mode == "dot_residual_s":
                xyz_scatter = xyz_scatter * two_side_tebd + xyz_scatter
            elif self.tebd_mode == "dot_residual_t":
                xyz_scatter = xyz_scatter * two_side_tebd + two_side_tebd

        xyz_scatter = xyz_scatter.view(
            -1, self.length, self.neuron[-1]
        )  # shape is [nframes*natoms[0], self.length, self.neuron[-1]]
        if self.return_G:
            return xyz_scatter
        else:
            # shape is [nframes*natoms[0], 4, self.length]
            inputs_reshape = inputs_i.view(-1, self.length, 4).permute(0, 2, 1)
            return torch.matmul(inputs_reshape, xyz_scatter)


class SimpleLinear(nn.Module):
    use_timestep: Final[bool]

    def __init__(
        self,
        num_in,
        num_out,
        bavg=0.0,
        stddev=1.0,
        use_timestep=False,
        activate=None,
        bias: bool = True,
    ):
        """Construct a linear layer.

        Args:
        - num_in: Width of input tensor.
        - num_out: Width of output tensor.
        - use_timestep: Apply time-step to weight.
        - activate: type of activate func.
        """
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.use_timestep = use_timestep
        self.activate = ActivationFn(activate)

        self.matrix = nn.Parameter(data=Tensor(num_in, num_out))
        nn.init.normal_(self.matrix.data, std=stddev / np.sqrt(num_out + num_in))
        if bias:
            self.bias = nn.Parameter(data=Tensor(1, num_out))
            nn.init.normal_(self.bias.data, mean=bavg, std=stddev)
        else:
            self.bias = None
        if self.use_timestep:
            self.idt = nn.Parameter(data=Tensor(1, num_out))
            nn.init.normal_(self.idt.data, mean=0.1, std=0.001)

    def forward(self, inputs):
        """Return X*W+b."""
        xw = torch.matmul(inputs, self.matrix)
        hidden = xw + self.bias if self.bias is not None else xw
        hidden = self.activate(hidden)
        if self.use_timestep:
            hidden = hidden * self.idt
        return hidden


class Linear(nn.Linear):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        init: str = "default",
    ):
        super().__init__(
            d_in,
            d_out,
            bias=bias,
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )

        self.use_bias = bias

        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "relu":
            self._trunc_normal_init(2.0)
        elif init == "glorot":
            self._glorot_uniform_init()
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "normal":
            self._normal_init()
        elif init == "final":
            self._zero_init(False)
        else:
            raise ValueError("Invalid init method.")

    def _trunc_normal_init(self, scale=1.0):
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _glorot_uniform_init(self):
        nn.init.xavier_uniform_(self.weight, gain=1)

    def _zero_init(self, use_bias=True):
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self):
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")


class Transition(nn.Module):
    def __init__(self, d_in, n, dropout=0.0):
        super().__init__()

        self.d_in = d_in
        self.n = n

        self.linear_1 = Linear(self.d_in, self.n * self.d_in, init="relu")
        self.act = nn.GELU()
        self.linear_2 = Linear(self.n * self.d_in, d_in, init="final")
        self.dropout = dropout

    def _transition(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_2(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self._transition(x=x)
        return x


class Embedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        dtype=torch.float64,
    ):
        super().__init__(
            num_embeddings, embedding_dim, padding_idx=padding_idx, dtype=dtype
        )
        self._normal_init()

        if padding_idx is not None:
            self.weight.data[self.padding_idx].zero_()

    def _normal_init(self, std=0.02):
        nn.init.normal_(self.weight, mean=0.0, std=std)


class NonLinearHead(nn.Module):
    def __init__(self, input_dim, out_dim, activation_fn, hidden=None):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = SimpleLinear(input_dim, hidden, activate=activation_fn)
        self.linear2 = SimpleLinear(hidden, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super().__init__()

        if hidden is None:
            hidden = input
        self.layer1 = Linear(input, hidden, init="relu")
        self.layer2 = Linear(hidden, output_size, init="final")

    def forward(self, x):
        x = F.linear(x, self.layer1.weight)
        # x = fused_ops.bias_torch_gelu(x, self.layer1.bias)
        x = nn.GELU()(x) + self.layer1.bias
        x = self.layer2(x)
        return x

    def zero_init(self):
        nn.init.zeros_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = SimpleLinear(embed_dim, embed_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.layer_norm = nn.LayerNorm(embed_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)

        if weight is None:
            weight = nn.Linear(
                embed_dim, output_dim, bias=False, dtype=env.GLOBAL_PT_FLOAT_PRECISION
            ).weight
        self.weight = weight
        self.bias = nn.Parameter(
            torch.zeros(output_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
        )

    def forward(self, features, masked_tokens: Optional[torch.Tensor] = None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class ResidualDeep(nn.Module):
    def __init__(
        self, type_id, embedding_width, neuron, bias_atom_e, out_dim=1, resnet_dt=False
    ):
        """Construct a filter on the given element as neighbor.

        Args:
        - typei: Element ID.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neurons in each hidden layers of the embedding net.
        - resnet_dt: Using time-step in the ResNet construction.
        """
        super().__init__()
        self.type_id = type_id
        self.neuron = [embedding_width, *neuron]
        self.out_dim = out_dim

        deep_layers = []
        for ii in range(1, len(self.neuron)):
            one = SimpleLinear(
                num_in=self.neuron[ii - 1],
                num_out=self.neuron[ii],
                use_timestep=(
                    resnet_dt and ii > 1 and self.neuron[ii - 1] == self.neuron[ii]
                ),
                activate="tanh",
            )
            deep_layers.append(one)
        self.deep_layers = nn.ModuleList(deep_layers)
        if not env.ENERGY_BIAS_TRAINABLE:
            bias_atom_e = 0
        self.final_layer = SimpleLinear(self.neuron[-1], self.out_dim, bias_atom_e)

    def forward(self, inputs):
        """Calculate decoded embedding for each atom.

        Args:
        - inputs: Embedding net output per atom. Its shape is [nframes*nloc, self.embedding_width].

        Returns
        -------
        - `torch.Tensor`: Output layer with shape [nframes*nloc, self.neuron[-1]].
        """
        outputs = inputs
        for idx, linear in enumerate(self.deep_layers):
            if idx > 0 and linear.num_in == linear.num_out:
                outputs = outputs + linear(outputs)
            else:
                outputs = linear(outputs)
        outputs = self.final_layer(outputs)
        return outputs


class TypeEmbedNet(nn.Module):
    def __init__(self, type_nums, embed_dim, bavg=0.0, stddev=1.0):
        """Construct a type embedding net."""
        super().__init__()
        self.embedding = nn.Embedding(
            type_nums + 1,
            embed_dim,
            padding_idx=type_nums,
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )
        # nn.init.normal_(self.embedding.weight[:-1], mean=bavg, std=stddev)

    def forward(self, atype):
        """
        Args:
            atype: Type of each input, [nframes, nloc] or [nframes, nloc, nnei].

        Returns
        -------
        type_embedding:

        """
        return self.embedding(atype)

    def share_params(self, base_class, shared_level, resume=False):
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some seperated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert (
            self.__class__ == base_class.__class__
        ), "Only TypeEmbedNet of the same type can share params!"
        if shared_level == 0:
            # the following will successfully link all the params except buffers, which need manually link.
            for item in self._modules:
                self._modules[item] = base_class._modules[item]
        else:
            raise NotImplementedError


@torch.jit.script
def gaussian(x, mean, std: float):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianKernel(nn.Module):
    def __init__(self, K=128, num_pair=512, std_width=1.0, start=0.0, stop=9.0):
        super().__init__()
        self.K = K
        std_width = std_width
        start = start
        stop = stop
        mean = torch.linspace(start, stop, K, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
        self.std = (std_width * (mean[1] - mean[0])).item()
        self.register_buffer("mean", mean)
        self.mul = Embedding(
            num_pair + 1, 1, padding_idx=num_pair, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )
        self.bias = Embedding(
            num_pair + 1, 1, padding_idx=num_pair, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1.0)

    def forward(self, x, atom_pair):
        mul = self.mul(atom_pair).abs().sum(dim=-2)
        bias = self.bias(atom_pair).sum(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        # [nframes, nloc, nnei, K]
        x = x.expand(-1, -1, -1, self.K)
        mean = self.mean.view(-1)
        return gaussian(x, mean, self.std)


class GaussianEmbedding(nn.Module):
    def __init__(
        self,
        rcut,
        kernel_num,
        num_pair,
        embed_dim,
        pair_embed_dim,
        sel,
        ntypes,
        atomic_sum_gbf,
    ):
        """Construct a gaussian kernel based embedding of pair representation.

        Args:
            rcut: Radial cutoff.
            kernel_num: Number of gaussian kernels.
            num_pair: Number of different pairs.
            embed_dim: Dimension of atomic representation.
            pair_embed_dim: Dimension of pair representation.
            sel: Number of neighbors.
            ntypes: Number of atom types.
        """
        super().__init__()
        self.gbf = GaussianKernel(K=kernel_num, num_pair=num_pair, stop=rcut)
        self.gbf_proj = NonLinear(kernel_num, pair_embed_dim)
        self.embed_dim = embed_dim
        self.pair_embed_dim = pair_embed_dim
        self.atomic_sum_gbf = atomic_sum_gbf
        if self.atomic_sum_gbf:
            if kernel_num != self.embed_dim:
                self.edge_proj = torch.nn.Linear(
                    kernel_num, self.embed_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION
                )
            else:
                self.edge_proj = None
        self.ntypes = ntypes
        self.nnei = sel

    def forward(self, coord_selected, atom_feature, edge_type_2dim, edge_feature):
        ## local cluster forward
        """Calculate decoded embedding for each atom.
        Args:
            coord_selected: Clustered atom coordinates with shape [nframes*nloc, natoms, 3].
            atom_feature: Previous calculated atomic features with shape [nframes*nloc, natoms, embed_dim].
            edge_type_2dim: Edge index for gbf calculation with shape [nframes*nloc, natoms, natoms, 2].
            edge_feature: Previous calculated edge features with shape [nframes*nloc, natoms, natoms, pair_dim].

        Returns
        -------
        atom_feature: Updated atomic features with shape [nframes*nloc, natoms, embed_dim].
        attn_bias: Updated edge features as attention bias with shape [nframes*nloc, natoms, natoms, pair_dim].
        delta_pos: Delta position for force/vector prediction with shape [nframes*nloc, natoms, natoms, 3].
        """
        ncluster, natoms, _ = coord_selected.shape
        # ncluster x natoms x natoms x 3
        delta_pos = coord_selected.unsqueeze(1) - coord_selected.unsqueeze(2)
        # (ncluster x natoms x natoms
        dist = delta_pos.norm(dim=-1).view(-1, natoms, natoms)
        # [ncluster, natoms, natoms, K]
        gbf_feature = self.gbf(dist, edge_type_2dim)
        if self.atomic_sum_gbf:
            edge_features = gbf_feature
            # [ncluster, natoms, K]
            sum_edge_features = edge_features.sum(dim=-2)
            if self.edge_proj is not None:
                sum_edge_features = self.edge_proj(sum_edge_features)
            # [ncluster, natoms, embed_dim]
            atom_feature = atom_feature + sum_edge_features

        # [ncluster, natoms, natoms, pair_dim]
        gbf_result = self.gbf_proj(gbf_feature)

        attn_bias = gbf_result + edge_feature
        return atom_feature, attn_bias, delta_pos


class NeighborWiseAttention(nn.Module):
    def __init__(
        self,
        layer_num,
        nnei,
        embed_dim,
        hidden_dim,
        dotr=False,
        do_mask=False,
        post_ln=True,
        ffn=False,
        ffn_embed_dim=1024,
        activation="tanh",
        scaling_factor=1.0,
        head_num=1,
        normalize=True,
        temperature=None,
    ):
        """Construct a neighbor-wise attention net."""
        super().__init__()
        self.layer_num = layer_num
        attention_layers = []
        for i in range(self.layer_num):
            attention_layers.append(
                NeighborWiseAttentionLayer(
                    nnei,
                    embed_dim,
                    hidden_dim,
                    dotr=dotr,
                    do_mask=do_mask,
                    post_ln=post_ln,
                    ffn=ffn,
                    ffn_embed_dim=ffn_embed_dim,
                    activation=activation,
                    scaling_factor=scaling_factor,
                    head_num=head_num,
                    normalize=normalize,
                    temperature=temperature,
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


class NeighborWiseAttentionLayer(nn.Module):
    ffn: Final[bool]

    def __init__(
        self,
        nnei,
        embed_dim,
        hidden_dim,
        dotr=False,
        do_mask=False,
        post_ln=True,
        ffn=False,
        ffn_embed_dim=1024,
        activation="tanh",
        scaling_factor=1.0,
        head_num=1,
        normalize=True,
        temperature=None,
    ):
        """Construct a neighbor-wise attention layer."""
        super().__init__()
        self.nnei = nnei
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dotr = dotr
        self.do_mask = do_mask
        self.post_ln = post_ln
        self.ffn = ffn
        self.attention_layer = GatedSelfAttetion(
            nnei,
            embed_dim,
            hidden_dim,
            dotr=dotr,
            do_mask=do_mask,
            scaling_factor=scaling_factor,
            head_num=head_num,
            normalize=normalize,
            temperature=temperature,
        )
        self.attn_layer_norm = nn.LayerNorm(
            self.embed_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
        )
        if self.ffn:
            self.ffn_embed_dim = ffn_embed_dim
            self.fc1 = nn.Linear(
                self.embed_dim, self.ffn_embed_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION
            )
            self.activation_fn = get_activation_fn(activation)
            self.fc2 = nn.Linear(
                self.ffn_embed_dim, self.embed_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION
            )
            self.final_layer_norm = nn.LayerNorm(
                self.embed_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION
            )

    def forward(
        self,
        x,
        nei_mask,
        input_r: Optional[torch.Tensor] = None,
        sw: Optional[torch.Tensor] = None,
    ):
        residual = x
        if not self.post_ln:
            x = self.attn_layer_norm(x)
        x = self.attention_layer(x, nei_mask, input_r=input_r, sw=sw)
        x = residual + x
        if self.post_ln:
            x = self.attn_layer_norm(x)
        if self.ffn:
            residual = x
            if not self.post_ln:
                x = self.final_layer_norm(x)
            x = self.fc1(x)
            x = self.activation_fn(x)
            x = self.fc2(x)
            x = residual + x
            if self.post_ln:
                x = self.final_layer_norm(x)
        return x


class GatedSelfAttetion(nn.Module):
    def __init__(
        self,
        nnei,
        embed_dim,
        hidden_dim,
        dotr=False,
        do_mask=False,
        scaling_factor=1.0,
        head_num=1,
        normalize=True,
        temperature=None,
        bias=True,
        smooth=True,
    ):
        """Construct a neighbor-wise attention net."""
        super().__init__()
        self.nnei = nnei
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.dotr = dotr
        self.do_mask = do_mask
        if temperature is None:
            self.scaling = (self.hidden_dim * scaling_factor) ** -0.5
        else:
            self.scaling = temperature
        self.normalize = normalize
        self.in_proj = SimpleLinear(
            embed_dim,
            hidden_dim * 3,
            bavg=0.0,
            stddev=1.0,
            use_timestep=False,
            bias=bias,
        )
        self.out_proj = SimpleLinear(
            hidden_dim, embed_dim, bavg=0.0, stddev=1.0, use_timestep=False, bias=bias
        )
        self.smooth = smooth

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
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            v = F.normalize(v, dim=-1)
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
        attn_weights = F.softmax(attn_weights, dim=-1)
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


class LocalSelfMultiheadAttention(nn.Module):
    def __init__(self, feature_dim, attn_head, scaling_factor=1.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.attn_head = attn_head
        self.head_dim = feature_dim // attn_head
        assert (
            feature_dim % attn_head == 0
        ), f"feature_dim {feature_dim} must be divided by attn_head {attn_head}!"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5
        self.in_proj = SimpleLinear(self.feature_dim, self.feature_dim * 3)
        # TODO debug
        # self.out_proj = SimpleLinear(self.feature_dim, self.feature_dim)

    def forward(
        self,
        query,
        attn_bias: Optional[torch.Tensor] = None,
        nlist_mask: Optional[torch.Tensor] = None,
        nlist: Optional[torch.Tensor] = None,
        return_attn=True,
    ):
        nframes, nloc, feature_dim = query.size()
        _, _, nnei = nlist.size()
        assert feature_dim == self.feature_dim
        # [nframes, nloc, feature_dim]
        q, k, v = self.in_proj(query).chunk(3, dim=-1)
        # [nframes * attn_head * nloc, 1, head_dim]
        q = (
            q.view(nframes, nloc, self.attn_head, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(nframes * self.attn_head * nloc, 1, self.head_dim)
            * self.scaling
        )
        # [nframes, nloc, feature_dim] --> [nframes, nloc + 1, feature_dim]
        # with nlist [nframes, nloc, nnei] --> [nframes, nloc, nnei, feature_dim]
        # padding = torch.zeros(feature_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION).to(k.device)
        # k = torch.concat([k, padding.unsqueeze(0).unsqueeze(1)], dim=1)
        # v = torch.concat([v, padding.unsqueeze(0).unsqueeze(1)], dim=1)

        # [nframes, nloc * nnei, feature_dim]
        index = nlist.view(nframes, -1).unsqueeze(-1).expand(-1, -1, feature_dim)
        k = torch.gather(k, dim=1, index=index)
        # [nframes, nloc * nnei, feature_dim]
        v = torch.gather(v, dim=1, index=index)
        # [nframes * attn_head * nloc, nnei, head_dim]
        k = (
            k.view(nframes, nloc, nnei, self.attn_head, self.head_dim)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(nframes * self.attn_head * nloc, nnei, self.head_dim)
        )
        v = (
            v.view(nframes, nloc, nnei, self.attn_head, self.head_dim)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(nframes * self.attn_head * nloc, nnei, self.head_dim)
        )
        # [nframes * attn_head * nloc, 1, nnei]
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # maskfill
        # [nframes, attn_head, nloc, nnei]
        attn_weights = attn_weights.view(
            nframes, self.attn_head, nloc, nnei
        ).masked_fill(~nlist_mask.unsqueeze(1), float("-inf"))
        # add bias
        if return_attn:
            attn_weights = attn_weights + attn_bias
        # softmax
        # [nframes * attn_head * nloc, 1, nnei]
        attn = F.softmax(attn_weights, dim=-1).view(
            nframes * self.attn_head * nloc, 1, nnei
        )
        # bmm
        # [nframes * attn_head * nloc, 1, head_dim]
        o = torch.bmm(attn, v)
        assert list(o.size()) == [nframes * self.attn_head * nloc, 1, self.head_dim]
        # [nframes, nloc, feature_dim]
        o = (
            o.view(nframes, self.attn_head, nloc, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(nframes, nloc, self.feature_dim)
        )
        # out
        ## TODO debug:
        # o = self.out_proj(o)
        if not return_attn:
            return o
        else:
            return o, attn_weights, attn


class NodeTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        pair_dim: int,
        num_head: int,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
        self.pair_norm = nn.LayerNorm(pair_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
        self.embed_dim = embed_dim
        self.q_proj = Linear(embed_dim, embed_dim, bias=False, init="glorot")
        self.k_proj = Linear(embed_dim, embed_dim, bias=False, init="glorot")
        self.v_proj = Linear(embed_dim, embed_dim, bias=False, init="glorot")
        self.num_heads = num_head
        self.head_dim = embed_dim // num_head
        self.scaling = self.head_dim**-0.5
        self.force_proj = Linear(embed_dim, 1, init="final", bias=False)
        self.linear_bias = Linear(pair_dim, num_head)
        self.dropout = 0.1

    def zero_init(self):
        nn.init.zeros_(self.force_proj.weight)

    def forward(
        self,
        query: Tensor,
        pair: Tensor,
        delta_pos: Tensor,
        attn_mask: Tensor = None,
    ) -> Tensor:
        ncluster, natoms, _ = query.size()
        query = self.layer_norm(query)
        # [ncluster, natoms, natoms, pair_dim]
        pair = self.pair_norm(pair)

        # [ncluster, attn_head, natoms, head_dim]
        q = (
            self.q_proj(query)
            .view(ncluster, natoms, self.num_heads, -1)
            .transpose(1, 2)
            * self.scaling
        )
        # [ncluster, attn_head, natoms, head_dim]
        k = (
            self.k_proj(query)
            .view(ncluster, natoms, self.num_heads, -1)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(query)
            .view(ncluster, natoms, self.num_heads, -1)
            .transpose(1, 2)
        )
        # [ncluster, attn_head, natoms, natoms]
        attn = q @ k.transpose(-1, -2)
        del q, k
        # [ncluster, attn_head, natoms, natoms]
        bias = self.linear_bias(pair).permute(0, 3, 1, 2).contiguous()

        # [ncluster, attn_head, natoms, natoms]
        attn_probs = softmax_dropout(
            attn,
            self.dropout,
            self.training,
            mask=attn_mask,
            bias=bias.contiguous(),
        ).view(ncluster, self.num_heads, natoms, natoms)

        # delta_pos: [ncluster, natoms, natoms, 3]
        # [ncluster, attn_head, natoms, natoms, 3]
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )
        # [ncluster, attn_head, 3, natoms, natoms]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        # [ncluster, attn_head, 3, natoms, head_dim]
        x = rot_attn_probs @ v.unsqueeze(2)
        # [ncluster, natoms, 3, embed_dim]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(ncluster, natoms, 3, -1)
        cur_force = self.force_proj(x).view(ncluster, natoms, 3)
        return cur_force


class EnergyHead(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
        self.linear_in = Linear(input_dim, input_dim, init="relu")

        self.linear_out = Linear(input_dim, output_dim, bias=True, init="final")

    def forward(self, x):
        x = x.type(self.linear_in.weight.dtype)
        x = F.gelu(self.layer_norm(self.linear_in(x)))
        x = self.linear_out(x)
        return x


class OuterProduct(nn.Module):
    def __init__(self, d_atom, d_pair, d_hid=32):
        super().__init__()

        self.d_atom = d_atom
        self.d_pair = d_pair
        self.d_hid = d_hid

        self.linear_in = nn.Linear(
            d_atom, d_hid * 2, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )
        self.linear_out = nn.Linear(
            d_hid**2, d_pair, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )
        self.act = nn.GELU()

    def _opm(self, a, b):
        # [nframes, nloc, d]
        nframes, nloc, d = a.shape
        a = a.view(nframes, nloc, 1, d, 1)
        b = b.view(nframes, 1, nloc, 1, d)
        # [nframes, nloc, nloc, d, d]
        outer = a * b
        outer = outer.view(outer.shape[:-2] + (-1,))
        outer = self.linear_out(outer)
        return outer

    def forward(
        self,
        m: torch.Tensor,
        nlist: torch.Tensor,
        op_mask: float,
        op_norm: float,
    ) -> torch.Tensor:
        ab = self.linear_in(m)
        ab = ab * op_mask
        a, b = ab.chunk(2, dim=-1)
        # [ncluster, natoms, natoms, d_pair]
        z = self._opm(a, b)
        z *= op_norm
        return z


class Attention(nn.Module):
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        total_dim = head_dim * self.num_heads
        self.total_dim = total_dim
        self.q_dim = q_dim
        self.gating = gating
        self.linear_q = Linear(q_dim, total_dim, bias=False, init="glorot")
        self.linear_k = Linear(k_dim, total_dim, bias=False, init="glorot")
        self.linear_v = Linear(v_dim, total_dim, bias=False, init="glorot")
        self.linear_o = Linear(total_dim, q_dim, init="final")
        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(q_dim, total_dim, init="gating")
        # precompute the 1/sqrt(head_dim)
        self.norm = head_dim**-0.5
        self.dropout = dropout

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        nframes, nloc, embed_dim = q.size()
        g = None
        if self.linear_g is not None:
            # gating, use raw query input
            # [nframes, nloc, total_dim]
            g = self.linear_g(q)
        # [nframes, nloc, total_dim]
        q = self.linear_q(q)
        q *= self.norm
        # [nframes, nloc, total_dim]
        k = self.linear_k(k)
        # [nframes, nloc, total_dim]
        v = self.linear_v(v)
        # global
        # q [nframes, h, nloc, d]
        # k [nframes, h, nloc, d]
        # v [nframes, h, nloc, d]
        # attn [nframes, h, nloc, nloc]
        # o [nframes, h, nloc, d]

        # [nframes, h, nloc, d]
        q = q.view(q.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3).contiguous()
        k = k.view(k.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3).contiguous()
        v = v.view(v.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3)
        # [nframes, h, nloc, nloc]
        attn = torch.matmul(q, k.transpose(-1, -2))
        del q, k
        # [nframes, h, nloc, nloc]
        attn = softmax_dropout(attn, self.dropout, self.training, mask=mask, bias=bias)
        # [nframes, h, nloc, d]
        o = torch.matmul(attn, v)
        del attn, v

        # local
        # q [nframes, h, nloc, 1, d]
        # k [nframes, h, nloc, nnei, d]
        # v [nframes, h, nloc, nnei, d]
        # attn [nframes, h, nloc, nnei]
        # o [nframes, h, nloc, d]

        assert list(o.size()) == [nframes, self.num_heads, nloc, self.head_dim]
        # [nframes, nloc, total_dim]
        o = o.transpose(-2, -3).contiguous()
        o = o.view(*o.shape[:-2], -1)

        if g is not None:
            o = torch.sigmoid(g) * o

        # merge heads
        o = self.linear_o(o)
        return o


class AtomAttention(nn.Module):
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        pair_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.mha = Attention(
            q_dim, k_dim, v_dim, head_dim, num_heads, gating=gating, dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(pair_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
        self.linear_bias = Linear(pair_dim, num_heads)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        nlist: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        pair = self.layer_norm(pair)
        bias = self.linear_bias(pair).permute(0, 3, 1, 2).contiguous()
        return self.mha(q, k, v, bias=bias, mask=mask)


class TriangleMultiplication(nn.Module):
    def __init__(self, d_pair, d_hid):
        super().__init__()

        self.linear_ab_p = Linear(d_pair, d_hid * 2)
        self.linear_ab_g = Linear(d_pair, d_hid * 2, init="gating")

        self.linear_g = Linear(d_pair, d_pair, init="gating")
        self.linear_z = Linear(d_hid, d_pair, init="final")

        self.layer_norm_out = nn.LayerNorm(d_hid, dtype=env.GLOBAL_PT_FLOAT_PRECISION)

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # z : [nframes, nloc, nloc, pair_dim]

        # [nframes, nloc, nloc, pair_dim]
        g = self.linear_g(z)
        if self.training:
            ab = self.linear_ab_p(z) * torch.sigmoid(self.linear_ab_g(z))
        else:
            ab = self.linear_ab_p(z)
            ab *= torch.sigmoid(self.linear_ab_g(z))
        # [nframes, nloc, nloc, d]
        a, b = torch.chunk(ab, 2, dim=-1)
        del z, ab

        # [nframes, d, nloc_i, nloc_k] row not trans
        a1 = a.permute(0, 3, 1, 2)
        # [nframes, d, nloc_k, nloc_j(i)]  trans
        b1 = b.transpose(-1, -3)
        # [nframes, d, nloc_i, nloc_j]
        x = torch.matmul(a1, b1)
        del a1, b1

        # [nframes, d, nloc_k, nloc_j(i)] not trans
        b2 = b.permute(0, 3, 1, 2)
        # [nframes, d, nloc_i, nloc_k]  col trans # check TODO
        a2 = a.transpose(-1, -3)

        # [nframes, d, nloc_i, nloc_j]
        x = x + torch.matmul(a2, b2)
        del a, b, a2, b2

        # [nframes, nloc_i, nloc_j, d]
        x = x.permute(0, 2, 3, 1)

        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        return g * x


class EvoformerEncoderLayer(nn.Module):
    def __init__(
        self,
        feature_dim: int = 768,
        ffn_dim: int = 2048,
        attn_head: int = 8,
        activation_fn: str = "gelu",
        post_ln: bool = False,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.ffn_dim = ffn_dim
        self.attn_head = attn_head
        self.activation_fn = (
            get_activation_fn(activation_fn) if activation_fn is not None else None
        )
        self.post_ln = post_ln
        self.self_attn_layer_norm = nn.LayerNorm(
            self.feature_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )

        self.self_attn = LocalSelfMultiheadAttention(
            self.feature_dim,
            self.attn_head,
        )
        self.final_layer_norm = nn.LayerNorm(
            self.feature_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )
        self.fc1 = SimpleLinear(self.feature_dim, self.ffn_dim)
        self.fc2 = SimpleLinear(self.ffn_dim, self.feature_dim)

    def forward(
        self,
        x,
        attn_bias: Optional[torch.Tensor] = None,
        nlist_mask: Optional[torch.Tensor] = None,
        nlist: Optional[torch.Tensor] = None,
        return_attn=True,
    ):
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            attn_bias=attn_bias,
            nlist_mask=nlist_mask,
            nlist=nlist,
            return_attn=return_attn,
        )
        if return_attn:
            x, attn_weights, attn_probs = x
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        if not return_attn:
            return x
        else:
            return x, attn_weights, attn_probs


# output: atomic_rep, transformed_atomic_rep, pair_rep, delta_pair_rep, norm_x, norm_delta_pair_rep,
class Evoformer2bEncoder(nn.Module):
    def __init__(
        self,
        nnei: int,
        layer_num: int = 6,
        attn_head: int = 8,
        atomic_dim: int = 1024,
        pair_dim: int = 100,
        feature_dim: int = 1024,
        ffn_dim: int = 2048,
        post_ln: bool = False,
        final_layer_norm: bool = True,
        final_head_layer_norm: bool = False,
        emb_layer_norm: bool = False,
        atomic_residual: bool = False,
        evo_residual: bool = False,
        residual_factor: float = 1.0,
        activation_function: str = "gelu",
    ):
        super().__init__()
        self.nnei = nnei
        self.layer_num = layer_num
        self.attn_head = attn_head
        self.atomic_dim = atomic_dim
        self.pair_dim = pair_dim
        self.feature_dim = feature_dim
        self.ffn_dim = ffn_dim
        self.post_ln = post_ln
        self._final_layer_norm = final_layer_norm
        self._final_head_layer_norm = final_head_layer_norm
        self._emb_layer_norm = emb_layer_norm
        self.activation_function = activation_function
        self.evo_residual = evo_residual
        self.residual_factor = residual_factor
        if atomic_residual and atomic_dim == feature_dim:
            self.atomic_residual = True
        else:
            self.atomic_residual = False
        self.in_proj = SimpleLinear(
            self.atomic_dim,
            self.feature_dim,
            bavg=0.0,
            stddev=1.0,
            use_timestep=False,
            activate="tanh",
        )  # TODO
        self.out_proj = SimpleLinear(
            self.feature_dim,
            self.atomic_dim,
            bavg=0.0,
            stddev=1.0,
            use_timestep=False,
            activate="tanh",
        )
        if self._emb_layer_norm:
            self.emb_layer_norm = nn.LayerNorm(
                self.feature_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION
            )

        ## TODO debug : self.in_proj_pair = NonLinearHead(self.pair_dim, self.attn_head, activation_fn=None)
        self.in_proj_pair = SimpleLinear(self.pair_dim, self.attn_head, activate=None)
        evoformer_encoder_layers = []
        for i in range(self.layer_num):
            evoformer_encoder_layers.append(
                EvoformerEncoderLayer(
                    feature_dim=self.feature_dim,
                    ffn_dim=self.ffn_dim,
                    attn_head=self.attn_head,
                    activation_fn=self.activation_function,
                    post_ln=self.post_ln,
                )
            )
        self.evoformer_encoder_layers = nn.ModuleList(evoformer_encoder_layers)
        if self._final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                self.feature_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION
            )
        if self._final_head_layer_norm:
            self.final_head_layer_norm = nn.LayerNorm(
                self.attn_head, dtype=env.GLOBAL_PT_FLOAT_PRECISION
            )

    def forward(self, atomic_rep, pair_rep, nlist, nlist_type, nlist_mask):
        """Encoder the atomic and pair representations.

        Args:
        - atomic_rep: Atomic representation with shape [nframes, nloc, atomic_dim].
        - pair_rep: Pair representation with shape [nframes, nloc, nnei, pair_dim].
        - nlist: Neighbor list with shape [nframes, nloc, nnei].
        - nlist_type: Neighbor types with shape [nframes, nloc, nnei].
        - nlist_mask: Neighbor mask with shape [nframes, nloc, nnei], `False` if blank.

        Returns
        -------
        - atomic_rep: Atomic representation after encoder with shape [nframes, nloc, feature_dim].
        - transformed_atomic_rep: Transformed atomic representation after encoder with shape [nframes, nloc, atomic_dim].
        - pair_rep: Pair representation after encoder with shape [nframes, nloc, nnei, attn_head].
        - delta_pair_rep: Delta pair representation after encoder with shape [nframes, nloc, nnei, attn_head].
        - norm_x: Normalization loss of atomic_rep.
        - norm_delta_pair_rep: Normalization loss of delta_pair_rep.
        """
        # Global branch
        nframes, nloc, _ = atomic_rep.size()
        nnei = pair_rep.shape[2]
        input_atomic_rep = atomic_rep
        # [nframes, nloc, feature_dim]
        if self.atomic_residual:
            atomic_rep = atomic_rep + self.in_proj(atomic_rep)
        else:
            atomic_rep = self.in_proj(atomic_rep)

        if self._emb_layer_norm:
            atomic_rep = self.emb_layer_norm(atomic_rep)

        # Local branch
        # [nframes, nloc, nnei, attn_head]
        pair_rep = self.in_proj_pair(pair_rep)
        # [nframes, attn_head, nloc, nnei]
        pair_rep = pair_rep.permute(0, 3, 1, 2).contiguous()
        input_pair_rep = pair_rep
        pair_rep = pair_rep.masked_fill(~nlist_mask.unsqueeze(1), float("-inf"))

        for i in range(self.layer_num):
            atomic_rep, pair_rep, _ = self.evoformer_encoder_layers[i](
                atomic_rep,
                attn_bias=pair_rep,
                nlist_mask=nlist_mask,
                nlist=nlist,
                return_attn=True,
            )

        def norm_loss(x, eps=1e-10, tolerance=1.0):
            # x = x.float()
            max_norm = x.shape[-1] ** 0.5
            norm = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
            error = F.relu((norm - max_norm).abs() - tolerance)
            return error

        def masked_mean(mask, value, dim=-1, eps=1e-10):
            return (
                torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
            ).mean()

        # atomic_rep shape: [nframes, nloc, feature_dim]
        # pair_rep shape: [nframes, attn_head, nloc, nnei]

        norm_x = torch.mean(norm_loss(atomic_rep))
        if self._final_layer_norm:
            atomic_rep = self.final_layer_norm(atomic_rep)

        delta_pair_rep = pair_rep - input_pair_rep
        delta_pair_rep = delta_pair_rep.masked_fill(~nlist_mask.unsqueeze(1), 0)
        # [nframes, nloc, nnei, attn_head]
        delta_pair_rep = (
            delta_pair_rep.view(nframes, self.attn_head, nloc, nnei)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        # [nframes, nloc, nnei]
        norm_delta_pair_rep = norm_loss(delta_pair_rep)
        norm_delta_pair_rep = masked_mean(mask=nlist_mask, value=norm_delta_pair_rep)
        if self._final_head_layer_norm:
            delta_pair_rep = self.final_head_layer_norm(delta_pair_rep)

        if self.atomic_residual:
            transformed_atomic_rep = atomic_rep + self.out_proj(atomic_rep)
        else:
            transformed_atomic_rep = self.out_proj(atomic_rep)

        if self.evo_residual:
            transformed_atomic_rep = (
                self.residual_factor * transformed_atomic_rep + input_atomic_rep
            ) * (1 / np.sqrt(2))

        return (
            atomic_rep,
            transformed_atomic_rep,
            pair_rep,
            delta_pair_rep,
            norm_x,
            norm_delta_pair_rep,
        )


class Evoformer3bEncoderLayer(nn.Module):
    def __init__(
        self,
        nnei,
        embedding_dim: int = 768,
        pair_dim: int = 64,
        pair_hidden_dim: int = 32,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        droppath_prob: float = 0.0,
        pair_dropout: float = 0.25,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        pre_ln: bool = True,
        tri_update: bool = True,
    ):
        super().__init__()
        # Initialize parameters
        self.nnei = nnei
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout

        # self.dropout = dropout
        self.activation_dropout = activation_dropout

        if droppath_prob > 0.0:
            self.dropout_module = DropPath(droppath_prob)
        else:
            self.dropout_module = Dropout(dropout)

        # self.self_attn = AtomAttentionLocal(embedding_dim, embedding_dim, embedding_dim, pair_dim,
        #                                     embedding_dim // num_attention_heads, num_attention_heads,
        #                                     gating=False, dropout=attention_dropout)
        self.self_attn = AtomAttention(
            embedding_dim,
            embedding_dim,
            embedding_dim,
            pair_dim,
            embedding_dim // num_attention_heads,
            num_attention_heads,
            gating=False,
            dropout=attention_dropout,
        )
        # layer norm associated with the self attention layer
        self.pre_ln = pre_ln
        self.self_attn_layer_norm = nn.LayerNorm(
            self.embedding_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )
        self.fc1 = nn.Linear(
            self.embedding_dim, ffn_embedding_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )
        self.fc2 = nn.Linear(
            ffn_embedding_dim, self.embedding_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )
        self.final_layer_norm = nn.LayerNorm(
            self.embedding_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )

        self.x_layer_norm_opm = nn.LayerNorm(
            self.embedding_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )
        # self.opm = OuterProductLocal(self.embedding_dim, pair_dim, d_hid=pair_hidden_dim)
        self.opm = OuterProduct(self.embedding_dim, pair_dim, d_hid=pair_hidden_dim)
        # self.pair_layer_norm_opm = nn.LayerNorm(pair_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
        self.pair_layer_norm_ffn = nn.LayerNorm(
            pair_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )
        self.pair_ffn = Transition(
            pair_dim,
            1,
            dropout=activation_dropout,
        )
        self.pair_dropout = pair_dropout
        self.tri_update = tri_update
        if self.tri_update:
            self.pair_layer_norm_trimul = nn.LayerNorm(
                pair_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION
            )
            self.pair_tri_mul = TriangleMultiplication(pair_dim, pair_hidden_dim)

    def update_pair(
        self,
        x,
        pair,
        nlist,
        op_mask,
        op_norm,
    ):
        # local:
        # [nframes, nloc, nnei, pair_dim]
        # global:
        # [nframes, nloc, nloc, pair_dim]
        pair = pair + self.dropout_module(
            self.opm(self.x_layer_norm_opm(x), nlist, op_mask, op_norm)
        )
        if not self.pre_ln:
            pair = self.pair_layer_norm_opm(pair)
        return x, pair

    def shared_dropout(self, x, shared_dim, dropout):
        shape = list(x.shape)
        shape[shared_dim] = 1
        with torch.no_grad():
            mask = x.new_ones(shape)
        return F.dropout(mask, p=dropout, training=self.training) * x

    def forward(
        self,
        x: torch.Tensor,
        pair: torch.Tensor,
        nlist: torch.Tensor = None,
        attn_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        op_mask: float = 1.0,
        op_norm: float = 1.0,
    ):
        """Encoder the atomic and pair representations.

        Args:
        - x: Atomic representation with shape [ncluster, natoms, embed_dim].
        - pair: Pair representation with shape [ncluster, natoms, natoms, pair_dim].
        - attn_mask: Attention mask with shape [ncluster, head, natoms, natoms].
        - pair_mask: Neighbor mask with shape [ncluster, natoms, natoms].

        """
        # [ncluster, natoms, embed_dim]
        residual = x
        if self.pre_ln:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            x,
            x,
            x,
            nlist=nlist,
            pair=pair,
            mask=attn_mask,
        )
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_ln:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_ln:
            x = self.final_layer_norm(x)
        x = F.linear(x, self.fc1.weight)
        # x = fused_ops.bias_torch_gelu(x, self.fc1.bias)
        x = nn.GELU()(x) + self.fc1.bias
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.dropout_module(x)

        x = residual + x
        if not self.pre_ln:
            x = self.final_layer_norm(x)

        block = [
            partial(
                self.update_pair,
                nlist=nlist,
                op_mask=op_mask,
                op_norm=op_norm,
            )
        ]

        x, pair = checkpoint_sequential(
            block,
            input_x=(x, pair),
        )

        if self.tri_update:
            residual_pair = pair
            if self.pre_ln:
                pair = self.pair_layer_norm_trimul(pair)

            pair = self.shared_dropout(
                self.pair_tri_mul(pair, pair_mask), -3, self.pair_dropout
            )
            pair = residual_pair + pair
            if not self.pre_ln:
                pair = self.pair_layer_norm_trimul(pair)

        residual_pair = pair
        if self.pre_ln:
            pair = self.pair_layer_norm_ffn(pair)
        pair = self.dropout_module(self.pair_ffn(pair))
        pair = residual_pair + pair
        if not self.pre_ln:
            pair = self.pair_layer_norm_ffn(pair)
        return x, pair


class Evoformer3bEncoder(nn.Module):
    def __init__(
        self,
        nnei,
        layer_num=6,
        attn_head=8,
        atomic_dim=768,
        pair_dim=64,
        pair_hidden_dim=32,
        ffn_embedding_dim=3072,
        dropout: float = 0.1,
        droppath_prob: float = 0.0,
        pair_dropout: float = 0.25,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        pre_ln: bool = True,
        tri_update: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.nnei = nnei
        if droppath_prob > 0:
            droppath_probs = [
                x.item() for x in torch.linspace(0, droppath_prob, layer_num)
            ]
        else:
            droppath_probs = None

        self.layers = nn.ModuleList(
            [
                Evoformer3bEncoderLayer(
                    nnei,
                    atomic_dim,
                    pair_dim,
                    pair_hidden_dim,
                    ffn_embedding_dim,
                    num_attention_heads=attn_head,
                    dropout=dropout,
                    droppath_prob=droppath_probs[_],
                    pair_dropout=pair_dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    pre_ln=pre_ln,
                    tri_update=tri_update,
                )
                for _ in range(layer_num)
            ]
        )

    def forward(self, x, pair, attn_mask=None, pair_mask=None, atom_mask=None):
        """Encoder the atomic and pair representations.

        Args:
            x: Atomic representation with shape [ncluster, natoms, atomic_dim].
            pair: Pair representation with shape [ncluster, natoms, natoms, pair_dim].
            attn_mask: Attention mask (with -inf for softmax) with shape [ncluster, head, natoms, natoms].
            pair_mask: Pair mask (with 1 for real atom pair and 0 for padding) with shape [ncluster, natoms, natoms].
            atom_mask: Atom mask (with 1 for real atom and 0 for padding) with shape [ncluster, natoms].

        Returns
        -------
        x: Atomic representation with shape [ncluster, natoms, atomic_dim].
        pair: Pair representation with shape [ncluster, natoms, natoms, pair_dim].

        """
        # [ncluster, natoms, 1]
        op_mask = atom_mask.unsqueeze(-1)
        op_mask = op_mask * (op_mask.size(-2) ** -0.5)
        eps = 1e-3
        # [ncluster, natoms, natoms, 1]
        op_norm = 1.0 / (eps + torch.einsum("...bc,...dc->...bdc", op_mask, op_mask))
        for layer in self.layers:
            x, pair = layer(
                x,
                pair,
                nlist=None,
                attn_mask=attn_mask,
                pair_mask=pair_mask,
                op_mask=op_mask,
                op_norm=op_norm,
            )
        return x, pair
