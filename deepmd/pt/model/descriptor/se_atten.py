# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as torch_func

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.descriptor.descriptor import (
    DescriptorBlock,
)
from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat,
)
from deepmd.pt.model.network.layernorm import (
    LayerNorm,
)
from deepmd.pt.model.network.mlp import (
    EmbeddingNet,
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
from deepmd.pt.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.pt.utils.exclude_mask import (
    PairExcludeMask,
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


@DescriptorBlock.register("se_atten")
class DescrptBlockSeAtten(DescriptorBlock):
    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: Union[List[int], int],
        ntypes: int,
        neuron: list = [25, 50, 100],
        axis_neuron: int = 16,
        tebd_dim: int = 8,
        tebd_input_mode: str = "concat",
        set_davg_zero: bool = True,
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
        smooth: bool = True,
        type_one_side: bool = False,
        exclude_types: List[Tuple[int, int]] = [],
        env_protection: float = 0.0,
        trainable_ln: bool = True,
        ln_eps: Optional[float] = 1e-5,
        seed: Optional[Union[int, List[int]]] = None,
        type: Optional[str] = None,
        old_impl: bool = False,
    ):
        r"""Construct an embedding net of type `se_atten`.

        Parameters
        ----------
        rcut : float
            The cut-off radius :math:`r_c`
        rcut_smth : float
            From where the environment matrix should be smoothed :math:`r_s`
        sel : list[int], int
            list[int]: sel[i] specifies the maxmum number of type i atoms in the cut-off radius
            int: the total maxmum number of atoms in the cut-off radius
        ntypes : int
            Number of element types
        neuron : list[int]
            Number of neurons in each hidden layers of the embedding net :math:`\mathcal{N}`
        axis_neuron : int
            Number of the axis neuron :math:`M_2` (number of columns of the sub-matrix of the embedding matrix)
        tebd_dim : int
            Dimension of the type embedding
        tebd_input_mode : str
            The input mode of the type embedding. Supported modes are ["concat", "strip"].
            - "concat": Concatenate the type embedding with the smoothed radial information as the union input for the embedding network.
            - "strip": Use a separated embedding network for the type embedding and combine the output with the radial embedding network output.
        resnet_dt : bool
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
        trainable_ln : bool
            Whether to use trainable shift and scale weights in layer normalization.
        ln_eps : float, Optional
            The epsilon value for layer normalization.
        type_one_side : bool
            If 'False', type embeddings of both neighbor and central atoms are considered.
            If 'True', only type embeddings of neighbor atoms are considered.
            Default is 'False'.
        attn : int
            Hidden dimension of the attention vectors
        attn_layer : int
            Number of attention layers
        attn_dotr : bool
            If dot the angular gate to the attention weights
        attn_mask : bool
            (Only support False to keep consistent with other backend references.)
            (Not used in this version.)
            If mask the diagonal of attention weights
        exclude_types : List[List[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
        env_protection : float
            Protection parameter to prevent division by zero errors during environment matrix calculations.
        set_davg_zero : bool
            Set the shift of embedding net input to zero.
        activation_function : str
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
        precision : str
            The precision of the embedding net parameters. Supported options are |PRECISION|
        scaling_factor : float
            The scaling factor of normalization in calculations of attention weights.
            If `temperature` is None, the scaling of attention weights is (N_dim * scaling_factor)**0.5
        normalize : bool
            Whether to normalize the hidden vectors in attention weights calculation.
        temperature : float
            If not None, the scaling of attention weights is `temperature` itself.
        seed : int, Optional
            Random seed for parameter initialization.
        """
        super().__init__()
        del type
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.neuron = neuron
        self.filter_neuron = self.neuron
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
        self.smooth = smooth
        self.type_one_side = type_one_side
        self.env_protection = env_protection
        self.trainable_ln = trainable_ln
        self.seed = seed
        #  to keep consistent with default value in this backends
        if ln_eps is None:
            ln_eps = 1e-5
        self.ln_eps = ln_eps
        self.old_impl = old_impl

        if isinstance(sel, int):
            sel = [sel]

        self.ntypes = ntypes
        self.sel = sel
        self.sec = self.sel
        self.split_sel = self.sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 4
        # order matters, placed after the assignment of self.ntypes
        self.reinit_exclude(exclude_types)
        if self.old_impl:
            assert self.tebd_input_mode in [
                "concat"
            ], "Old implementation does not support tebd_input_mode != 'concat'."
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
                smooth=self.smooth,
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
                trainable_ln=self.trainable_ln,
                ln_eps=self.ln_eps,
                smooth=self.smooth,
                precision=self.precision,
                seed=child_seed(self.seed, 0),
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
        self.tebd_dim_input = self.tebd_dim if self.type_one_side else self.tebd_dim * 2
        if self.tebd_input_mode in ["concat"]:
            self.embd_input_dim = 1 + self.tebd_dim_input
        else:
            self.embd_input_dim = 1

        self.filter_layers_old = None
        self.filter_layers = None
        self.filter_layers_strip = None
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
                ndim=0, ntypes=self.ntypes, network_type="embedding_network"
            )
            filter_layers[0] = EmbeddingNet(
                self.embd_input_dim,
                self.filter_neuron,
                activation_function=self.activation_function,
                precision=self.precision,
                resnet_dt=self.resnet_dt,
                seed=child_seed(self.seed, 1),
            )
            self.filter_layers = filter_layers
            if self.tebd_input_mode in ["strip"]:
                filter_layers_strip = NetworkCollection(
                    ndim=0, ntypes=self.ntypes, network_type="embedding_network"
                )
                filter_layers_strip[0] = EmbeddingNet(
                    self.tebd_dim_input,
                    self.filter_neuron,
                    activation_function=self.activation_function,
                    precision=self.precision,
                    resnet_dt=self.resnet_dt,
                    seed=child_seed(self.seed, 2),
                )
                self.filter_layers_strip = filter_layers_strip
        self.stats = None

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.rcut

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.rcut_smth

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return sum(self.sel)

    def get_sel(self) -> List[int]:
        """Returns the number of selected atoms for each type."""
        return self.sel

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def get_dim_in(self) -> int:
        """Returns the input dimension."""
        return self.dim_in

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        return self.dim_out

    def get_dim_emb(self) -> int:
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

    def mixed_types(self) -> bool:
        """If true, the discriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the discriptor
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
        return self.filter_neuron[-1] * self.axis_neuron

    @property
    def dim_in(self):
        """Returns the atomic input dimension of this descriptor."""
        return self.tebd_dim

    @property
    def dim_emb(self):
        """Returns the output dimension of embedding."""
        return self.get_dim_emb()

    def compute_input_stats(
        self,
        merged: Union[Callable[[], List[dict]], List[dict]],
        path: Optional[DPPath] = None,
    ):
        """
        Compute the input statistics (e.g. mean and stddev) for the descriptors from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], List[dict]], List[dict]]
            - List[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], List[dict]]: A lazy function that returns data samples in the above format
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
        if not self.set_davg_zero:
            self.mean.copy_(torch.tensor(mean, device=env.DEVICE))
        self.stddev.copy_(torch.tensor(stddev, device=env.DEVICE))

    def get_stats(self) -> Dict[str, StatItem]:
        """Get the statistics of the descriptor."""
        if self.stats is None:
            raise RuntimeError(
                "The statistics of the descriptor has not been computed."
            )
        return self.stats

    def reinit_exclude(
        self,
        exclude_types: List[Tuple[int, int]] = [],
    ):
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_atype_embd: Optional[torch.Tensor] = None,
        mapping: Optional[torch.Tensor] = None,
    ):
        """Compute the descriptor.

        Parameters
        ----------
        nlist
            The neighbor list. shape: nf x nloc x nnei
        extended_coord
            The extended coordinates of atoms. shape: nf x (nallx3)
        extended_atype
            The extended aotm types. shape: nf x nall x nt
        extended_atype_embd
            The extended type embedding of atoms. shape: nf x nall
        mapping
            The index mapping, not required by this descriptor.

        Returns
        -------
        result
            The descriptor. shape: nf x nloc x (ng x axis_neuron)
        g2
            The rotationally invariant pair-partical representation.
            shape: nf x nloc x nnei x ng
        h2
            The rotationally equivariant pair-partical representation.
            shape: nf x nloc x nnei x 3
        gr
            The rotationally equivariant and permutationally invariant single particle
            representation. shape: nf x nloc x ng x 3
        sw
            The smooth switch function. shape: nf x nloc x nnei

        """
        del mapping
        assert extended_atype_embd is not None
        nframes, nloc, nnei = nlist.shape
        atype = extended_atype[:, :nloc]
        nb = nframes
        nall = extended_coord.view(nb, -1, 3).shape[1]
        dmatrix, diff, sw = prod_env_mat(
            extended_coord,
            nlist,
            atype,
            self.mean,
            self.stddev,
            self.rcut,
            self.rcut_smth,
            protection=self.env_protection,
        )
        # nb x nloc x nnei
        exclude_mask = self.emask(nlist, extended_atype)
        nlist = torch.where(exclude_mask != 0, nlist, -1)
        nlist_mask = nlist != -1
        nlist = torch.where(nlist == -1, 0, nlist)
        sw = torch.squeeze(sw, -1)
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
        # beyond the cutoff sw should be 0.0
        sw = sw.masked_fill(~nlist_mask, 0.0)
        # (nb x nloc) x nnei
        exclude_mask = exclude_mask.view(nb * nloc, nnei)
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
            # nfnl x nnei x 4
            dmatrix = dmatrix.view(-1, self.nnei, 4)
            nfnl = dmatrix.shape[0]
            # nfnl x nnei x 4
            rr = dmatrix
            rr = rr * exclude_mask[:, :, None]
            ss = rr[:, :, :1]
            nlist_tebd = atype_tebd_nlist.reshape(nfnl, nnei, self.tebd_dim)
            atype_tebd = atype_tebd_nnei.reshape(nfnl, nnei, self.tebd_dim)
            if self.tebd_input_mode in ["concat"]:
                if not self.type_one_side:
                    # nfnl x nnei x (1 + tebd_dim * 2)
                    ss = torch.concat([ss, nlist_tebd, atype_tebd], dim=2)
                else:
                    # nfnl x nnei x (1 + tebd_dim)
                    ss = torch.concat([ss, nlist_tebd], dim=2)
                # nfnl x nnei x ng
                gg = self.filter_layers.networks[0](ss)
            elif self.tebd_input_mode in ["strip"]:
                # nfnl x nnei x ng
                gg_s = self.filter_layers.networks[0](ss)
                assert self.filter_layers_strip is not None
                if not self.type_one_side:
                    # nfnl x nnei x (tebd_dim * 2)
                    tt = torch.concat([nlist_tebd, atype_tebd], dim=2)
                else:
                    # nfnl x nnei x tebd_dim
                    tt = nlist_tebd
                # nfnl x nnei x ng
                gg_t = self.filter_layers_strip.networks[0](tt)
                if self.smooth:
                    gg_t = gg_t * sw.reshape(-1, self.nnei, 1)
                # nfnl x nnei x ng
                gg = gg_s * gg_t + gg_s
            else:
                raise NotImplementedError

            input_r = torch.nn.functional.normalize(
                rr.reshape(-1, self.nnei, 4)[:, :, 1:4], dim=-1
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
            rot_mat.view(-1, nloc, self.filter_neuron[-1], 3),
            sw,
        )

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor block has message passing."""
        return False


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
        trainable_ln: bool = True,
        ln_eps: float = 1e-5,
        smooth: bool = True,
        precision: str = DEFAULT_PRECISION,
        seed: Optional[Union[int, List[int]]] = None,
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
        self.trainable_ln = trainable_ln
        self.ln_eps = ln_eps
        self.smooth = smooth
        self.precision = precision
        self.seed = seed
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
                    trainable_ln=trainable_ln,
                    ln_eps=ln_eps,
                    smooth=smooth,
                    precision=precision,
                    seed=child_seed(seed, i),
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
        """Compute the multi-layer gated self-attention.

        Parameters
        ----------
        input_G
            inputs with shape: (nf x nloc) x nnei x embed_dim.
        nei_mask
            neighbor mask, with paddings being 0. shape: (nf x nloc) x nnei.
        input_r
            normalized radial. shape: (nf x nloc) x nnei x 3.
        sw
            The smooth switch function. shape: nf x nloc x nnei
        """
        out = input_G
        # https://github.com/pytorch/pytorch/issues/39165#issuecomment-635472592
        for layer in self.attention_layers:
            out = layer(out, nei_mask, input_r=input_r, sw=sw)
        return out

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.attention_layers[key]
        else:
            raise TypeError(key)

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise TypeError(key)
        if isinstance(value, self.network_type):
            pass
        elif isinstance(value, dict):
            value = self.network_type.deserialize(value)
        else:
            raise TypeError(value)
        self.attention_layers[key] = value

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        return {
            "@class": "NeighborGatedAttention",
            "@version": 1,
            "layer_num": self.layer_num,
            "nnei": self.nnei,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "dotr": self.dotr,
            "do_mask": self.do_mask,
            "scaling_factor": self.scaling_factor,
            "normalize": self.normalize,
            "temperature": self.temperature,
            "trainable_ln": self.trainable_ln,
            "ln_eps": self.ln_eps,
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
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
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
        smooth: bool = True,
        trainable_ln: bool = True,
        ln_eps: float = 1e-5,
        precision: str = DEFAULT_PRECISION,
        seed: Optional[Union[int, List[int]]] = None,
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
        self.trainable_ln = trainable_ln
        self.ln_eps = ln_eps
        self.seed = seed
        self.attention_layer = GatedAttentionLayer(
            nnei,
            embed_dim,
            hidden_dim,
            dotr=dotr,
            do_mask=do_mask,
            scaling_factor=scaling_factor,
            normalize=normalize,
            temperature=temperature,
            smooth=smooth,
            precision=precision,
            seed=child_seed(seed, 0),
        )
        self.attn_layer_norm = LayerNorm(
            self.embed_dim,
            eps=ln_eps,
            trainable=trainable_ln,
            precision=precision,
            seed=child_seed(seed, 1),
        )

    def forward(
        self,
        x,
        nei_mask,
        input_r: Optional[torch.Tensor] = None,
        sw: Optional[torch.Tensor] = None,
    ):
        residual = x
        x, _ = self.attention_layer(x, nei_mask, input_r=input_r, sw=sw)
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
            "trainable_ln": self.trainable_ln,
            "ln_eps": self.ln_eps,
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
        data = data.copy()
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
        num_heads: int = 1,
        dotr: bool = False,
        do_mask: bool = False,
        scaling_factor: float = 1.0,
        normalize: bool = True,
        temperature: Optional[float] = None,
        bias: bool = True,
        smooth: bool = True,
        precision: str = DEFAULT_PRECISION,
        seed: Optional[Union[int, List[int]]] = None,
    ):
        """Construct a multi-head neighbor-wise attention net."""
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.nnei = nnei
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dotr = dotr
        self.do_mask = do_mask
        self.bias = bias
        self.smooth = smooth
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        self.precision = precision
        self.seed = seed
        self.scaling = (
            (self.head_dim * scaling_factor) ** -0.5
            if temperature is None
            else temperature
        )
        self.normalize = normalize
        self.in_proj = MLPLayer(
            embed_dim,
            hidden_dim * 3,
            bias=bias,
            use_timestep=False,
            bavg=0.0,
            stddev=1.0,
            precision=precision,
            seed=child_seed(seed, 0),
        )
        self.out_proj = MLPLayer(
            hidden_dim,
            embed_dim,
            bias=bias,
            use_timestep=False,
            bavg=0.0,
            stddev=1.0,
            precision=precision,
            seed=child_seed(seed, 1),
        )

    def forward(
        self,
        query,
        nei_mask,
        input_r: Optional[torch.Tensor] = None,
        sw: Optional[torch.Tensor] = None,
        attnw_shift: float = 20.0,
    ):
        """Compute the multi-head gated self-attention.

        Parameters
        ----------
        query
            inputs with shape: (nf x nloc) x nnei x embed_dim.
        nei_mask
            neighbor mask, with paddings being 0. shape: (nf x nloc) x nnei.
        input_r
            normalized radial. shape: (nf x nloc) x nnei x 3.
        sw
            The smooth switch function. shape: (nf x nloc) x nnei
        attnw_shift : float
            The attention weight shift to preserve smoothness when doing padding before softmax.
        """
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        # Reshape for multi-head attention: (nf x nloc) x num_heads x nnei x head_dim
        q = q.view(-1, self.nnei, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(-1, self.nnei, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(-1, self.nnei, self.num_heads, self.head_dim).transpose(1, 2)

        if self.normalize:
            q = torch_func.normalize(q, dim=-1)
            k = torch_func.normalize(k, dim=-1)
            v = torch_func.normalize(v, dim=-1)

        q = q * self.scaling
        # (nf x nloc) x num_heads x head_dim x nnei
        k = k.transpose(-2, -1)

        # Compute attention scores
        # (nf x nloc) x num_heads x nnei x nnei
        attn_weights = torch.matmul(q, k)
        # (nf x nloc) x nnei
        nei_mask = nei_mask.view(-1, self.nnei)

        if self.smooth:
            assert sw is not None
            # (nf x nloc) x 1 x nnei
            sw = sw.view(-1, 1, self.nnei)
            attn_weights = (attn_weights + attnw_shift) * sw[:, :, :, None] * sw[
                :, :, None, :
            ] - attnw_shift
        else:
            # (nf x nloc) x 1 x 1 x nnei
            attn_weights = attn_weights.masked_fill(
                ~nei_mask.unsqueeze(1).unsqueeze(1), float("-inf")
            )

        attn_weights = torch_func.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.masked_fill(
            ~nei_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )
        if self.smooth:
            assert sw is not None
            attn_weights = attn_weights * sw[:, :, :, None] * sw[:, :, None, :]

        if self.dotr:
            # (nf x nloc) x nnei x 3
            assert input_r is not None, "input_r must be provided when dotr is True!"
            # (nf x nloc) x 1 x nnei x nnei
            angular_weight = torch.matmul(input_r, input_r.transpose(-2, -1)).view(
                -1, 1, self.nnei, self.nnei
            )
            attn_weights = attn_weights * angular_weight

        # Apply attention to values
        # (nf x nloc) x nnei x (num_heads x head_dim)
        o = (
            torch.matmul(attn_weights, v)
            .transpose(1, 2)
            .reshape(-1, self.nnei, self.hidden_dim)
        )
        output = self.out_proj(o)
        return output, attn_weights

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
            "num_heads": self.num_heads,
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
        data = data.copy()
        in_proj = data.pop("in_proj")
        out_proj = data.pop("out_proj")
        obj = cls(**data)
        obj.in_proj = MLPLayer.deserialize(in_proj)
        obj.out_proj = MLPLayer.deserialize(out_proj)
        return obj
