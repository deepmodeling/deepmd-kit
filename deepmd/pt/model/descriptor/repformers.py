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

from deepmd.pt.model.descriptor.descriptor import (
    DescriptorBlock,
)
from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.pt.utils.exclude_mask import (
    PairExcludeMask,
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

from .repformer_layer import (
    RepformerLayer,
)
from .repformer_layer_old_impl import RepformerLayer as RepformerLayerOld

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
            "border_op is not available since customized PyTorch OP library is not built when freezing the model."
        )

    # Note: this hack cannot actually save a model that can be runned using LAMMPS.
    torch.ops.deepmd.border_op = border_op


@DescriptorBlock.register("se_repformer")
@DescriptorBlock.register("se_uni")
class DescrptBlockRepformers(DescriptorBlock):
    def __init__(
        self,
        rcut,
        rcut_smth,
        sel: int,
        ntypes: int,
        nlayers: int = 3,
        g1_dim=128,
        g2_dim=16,
        axis_neuron: int = 4,
        direct_dist: bool = False,
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
        set_davg_zero: bool = True,
        smooth: bool = True,
        exclude_types: List[Tuple[int, int]] = [],
        env_protection: float = 0.0,
        precision: str = "float64",
        trainable_ln: bool = True,
        ln_eps: Optional[float] = 1e-5,
        old_impl: bool = False,
    ):
        r"""
        The repformer descriptor block.

        Parameters
        ----------
        rcut : float
            The cut-off radius.
        rcut_smth : float
            Where to start smoothing. For example the 1/r term is smoothed from rcut to rcut_smth.
        sel : int
            Maximally possible number of selected neighbors.
        ntypes : int
            Number of element types
        nlayers : int, optional
            Number of repformer layers.
        g1_dim : int, optional
            Dimension of the first graph convolution layer.
        g2_dim : int, optional
            Dimension of the second graph convolution layer.
        axis_neuron : int, optional
            Size of the submatrix of G (embedding matrix).
        direct_dist : bool, optional
            Whether to use direct distance information (1/r term) in the repformer block.
        update_g1_has_conv : bool, optional
            Whether to update the g1 rep with convolution term.
        update_g1_has_drrd : bool, optional
            Whether to update the g1 rep with the drrd term.
        update_g1_has_grrg : bool, optional
            Whether to update the g1 rep with the grrg term.
        update_g1_has_attn : bool, optional
            Whether to update the g1 rep with the localized self-attention.
        update_g2_has_g1g1 : bool, optional
            Whether to update the g2 rep with the g1xg1 term.
        update_g2_has_attn : bool, optional
            Whether to update the g2 rep with the gated self-attention.
        update_h2 : bool, optional
            Whether to update the h2 rep.
        attn1_hidden : int, optional
            The hidden dimension of localized self-attention to update the g1 rep.
        attn1_nhead : int, optional
            The number of heads in localized self-attention to update the g1 rep.
        attn2_hidden : int, optional
            The hidden dimension of gated self-attention to update the g2 rep.
        attn2_nhead : int, optional
            The number of heads in gated self-attention to update the g2 rep.
        attn2_has_gate : bool, optional
            Whether to use gate in the gated self-attention to update the g2 rep.
        activation_function : str, optional
            The activation function in the embedding net.
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
        set_davg_zero : bool, optional
            Set the normalization average to zero.
        precision : str, optional
            The precision of the embedding net parameters.
        smooth : bool, optional
            Whether to use smoothness in processes such as attention weights calculation.
        exclude_types : List[List[int]], optional
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
        env_protection : float, optional
            Protection parameter to prevent division by zero errors during environment matrix calculations.
            For example, when using paddings, there may be zero distances of neighbors, which may make division by zero error during environment matrix calculations without protection.
        trainable_ln : bool, optional
            Whether to use trainable shift and scale weights in layer normalization.
        ln_eps : float, optional
            The epsilon value for layer normalization.
        """
        super().__init__()
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.ntypes = ntypes
        self.nlayers = nlayers
        sel = [sel] if isinstance(sel, int) else sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 4  # use full descriptor.
        assert len(sel) == 1
        self.sel = sel
        self.sec = self.sel
        self.split_sel = self.sel
        self.axis_neuron = axis_neuron
        self.set_davg_zero = set_davg_zero
        self.g1_dim = g1_dim
        self.g2_dim = g2_dim
        self.update_g1_has_conv = update_g1_has_conv
        self.update_g1_has_drrd = update_g1_has_drrd
        self.update_g1_has_grrg = update_g1_has_grrg
        self.update_g1_has_attn = update_g1_has_attn
        self.update_g2_has_g1g1 = update_g2_has_g1g1
        self.update_g2_has_attn = update_g2_has_attn
        self.update_h2 = update_h2
        self.attn1_hidden = attn1_hidden
        self.attn1_nhead = attn1_nhead
        self.attn2_has_gate = attn2_has_gate
        self.attn2_hidden = attn2_hidden
        self.attn2_nhead = attn2_nhead
        self.activation_function = activation_function
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.direct_dist = direct_dist
        self.act = ActivationFn(activation_function)
        self.smooth = smooth
        # order matters, placed after the assignment of self.ntypes
        self.reinit_exclude(exclude_types)
        self.env_protection = env_protection
        self.precision = precision
        self.trainable_ln = trainable_ln
        self.ln_eps = ln_eps
        self.epsilon = 1e-4
        self.old_impl = old_impl

        self.g2_embd = MLPLayer(1, self.g2_dim, precision=precision)
        layers = []
        for ii in range(nlayers):
            if self.old_impl:
                layers.append(
                    RepformerLayerOld(
                        self.rcut,
                        self.rcut_smth,
                        self.sel,
                        self.ntypes,
                        self.g1_dim,
                        self.g2_dim,
                        axis_neuron=self.axis_neuron,
                        update_chnnl_2=(ii != nlayers - 1),
                        update_g1_has_conv=self.update_g1_has_conv,
                        update_g1_has_drrd=self.update_g1_has_drrd,
                        update_g1_has_grrg=self.update_g1_has_grrg,
                        update_g1_has_attn=self.update_g1_has_attn,
                        update_g2_has_g1g1=self.update_g2_has_g1g1,
                        update_g2_has_attn=self.update_g2_has_attn,
                        update_h2=self.update_h2,
                        attn1_hidden=self.attn1_hidden,
                        attn1_nhead=self.attn1_nhead,
                        attn2_has_gate=self.attn2_has_gate,
                        attn2_hidden=self.attn2_hidden,
                        attn2_nhead=self.attn2_nhead,
                        activation_function=self.activation_function,
                        update_style=self.update_style,
                        smooth=self.smooth,
                    )
                )
            else:
                layers.append(
                    RepformerLayer(
                        self.rcut,
                        self.rcut_smth,
                        self.sel,
                        self.ntypes,
                        self.g1_dim,
                        self.g2_dim,
                        axis_neuron=self.axis_neuron,
                        update_chnnl_2=(ii != nlayers - 1),
                        update_g1_has_conv=self.update_g1_has_conv,
                        update_g1_has_drrd=self.update_g1_has_drrd,
                        update_g1_has_grrg=self.update_g1_has_grrg,
                        update_g1_has_attn=self.update_g1_has_attn,
                        update_g2_has_g1g1=self.update_g2_has_g1g1,
                        update_g2_has_attn=self.update_g2_has_attn,
                        update_h2=self.update_h2,
                        attn1_hidden=self.attn1_hidden,
                        attn1_nhead=self.attn1_nhead,
                        attn2_has_gate=self.attn2_has_gate,
                        attn2_hidden=self.attn2_hidden,
                        attn2_nhead=self.attn2_nhead,
                        activation_function=self.activation_function,
                        update_style=self.update_style,
                        update_residual=self.update_residual,
                        update_residual_init=self.update_residual_init,
                        smooth=self.smooth,
                        trainable_ln=self.trainable_ln,
                        ln_eps=self.ln_eps,
                        precision=precision,
                    )
                )
        self.layers = torch.nn.ModuleList(layers)

        wanted_shape = (self.ntypes, self.nnei, 4)
        mean = torch.zeros(
            wanted_shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
        )
        stddev = torch.ones(
            wanted_shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
        )
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
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

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        return self.dim_out

    def get_dim_in(self) -> int:
        """Returns the input dimension."""
        return self.dim_in

    def get_dim_emb(self) -> int:
        """Returns the embedding dimension g2."""
        return self.g2_dim

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
        return self.g1_dim

    @property
    def dim_in(self):
        """Returns the atomic input dimension of this descriptor."""
        return self.g1_dim

    @property
    def dim_emb(self):
        """Returns the embedding dimension g2."""
        return self.get_dim_emb()

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
        comm_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if comm_dict is None:
            assert mapping is not None
            assert extended_atype_embd is not None
        nframes, nloc, nnei = nlist.shape
        nall = extended_coord.view(nframes, -1).shape[1] // 3
        atype = extended_atype[:, :nloc]
        # nb x nloc x nnei
        exclude_mask = self.emask(nlist, extended_atype)
        nlist = nlist * exclude_mask
        # nb x nloc x nnei x 4, nb x nloc x nnei x 3, nb x nloc x nnei x 1
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
        nlist_mask = nlist != -1
        sw = torch.squeeze(sw, -1)
        # beyond the cutoff sw should be 0.0
        sw = sw.masked_fill(~nlist_mask, 0.0)

        # [nframes, nloc, tebd_dim]
        if comm_dict is None:
            assert isinstance(extended_atype_embd, torch.Tensor)  # for jit
            atype_embd = extended_atype_embd[:, :nloc, :]
            assert list(atype_embd.shape) == [nframes, nloc, self.g1_dim]
        else:
            atype_embd = extended_atype_embd
        assert isinstance(atype_embd, torch.Tensor)  # for jit
        g1 = self.act(atype_embd)
        # nb x nloc x nnei x 1,  nb x nloc x nnei x 3
        if not self.direct_dist:
            g2, h2 = torch.split(dmatrix, [1, 3], dim=-1)
        else:
            g2, h2 = torch.linalg.norm(diff, dim=-1, keepdim=True), diff
            g2 = g2 / self.rcut
            h2 = h2 / self.rcut
        # nb x nloc x nnei x ng2
        g2 = self.act(self.g2_embd(g2))

        # set all padding positions to index of 0
        # if the a neighbor is real or not is indicated by nlist_mask
        nlist[nlist == -1] = 0
        # nb x nall x ng1
        if comm_dict is None:
            assert mapping is not None
            mapping = (
                mapping.view(nframes, nall).unsqueeze(-1).expand(-1, -1, self.g1_dim)
            )
        for idx, ll in enumerate(self.layers):
            # g1:     nb x nloc x ng1
            # g1_ext: nb x nall x ng1
            if comm_dict is None:
                assert mapping is not None
                g1_ext = torch.gather(g1, 1, mapping)
            else:
                n_padding = nall - nloc
                g1 = torch.nn.functional.pad(
                    g1.squeeze(0), (0, 0, 0, n_padding), value=0.0
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
                    g1,
                    comm_dict["communicator"],
                    torch.tensor(nloc),
                    torch.tensor(nall - nloc),
                )
                g1_ext = ret[0].unsqueeze(0)
            g1, g2, h2 = ll.forward(
                g1_ext,
                g2,
                h2,
                nlist,
                nlist_mask,
                sw,
            )

        # nb x nloc x 3 x ng2
        h2g2 = RepformerLayer._cal_hg(
            g2, h2, nlist_mask, sw, smooth=self.smooth, epsilon=self.epsilon
        )
        # (nb x nloc) x ng2 x 3
        rot_mat = torch.permute(h2g2, (0, 1, 3, 2))

        return g1, g2, h2, rot_mat.view(-1, nloc, self.dim_emb, 3), sw

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
