# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

import torch

from deepmd.pt.model.descriptor.descriptor import (
    DescriptorBlock,
)
from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat,
)
from deepmd.pt.model.network.network import (
    SimpleLinear,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.pt.utils.utils import (
    get_activation_fn,
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

mydtype = env.GLOBAL_PT_FLOAT_PRECISION
mydev = env.DEVICE


def torch_linear(*args, **kwargs):
    return torch.nn.Linear(*args, **kwargs, dtype=mydtype, device=mydev)


simple_linear = SimpleLinear
mylinear = simple_linear


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
        axis_dim: int = 4,
        direct_dist: bool = False,
        do_bn_mode: str = "no",
        bn_momentum: float = 0.1,
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
        set_davg_zero: bool = True,  # TODO
        smooth: bool = True,
        add_type_ebd_to_seq: bool = False,
        type: Optional[str] = None,
    ):
        """
        smooth:
            If strictly smooth, cannot be used with update_g1_has_attn
        add_type_ebd_to_seq:
            At the presence of seq_input (optional input to forward),
            whether or not add an type embedding to seq_input.
            If no seq_input is given, it has no effect.
        """
        super().__init__()
        del type
        self.epsilon = 1e-4  # protection of 1./nnei
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
        self.axis_dim = axis_dim
        self.set_davg_zero = set_davg_zero
        self.g1_dim = g1_dim
        self.g2_dim = g2_dim
        self.act = get_activation_fn(activation_function)
        self.direct_dist = direct_dist
        self.add_type_ebd_to_seq = add_type_ebd_to_seq

        self.g2_embd = mylinear(1, self.g2_dim)
        layers = []
        for ii in range(nlayers):
            layers.append(
                RepformerLayer(
                    rcut,
                    rcut_smth,
                    sel,
                    ntypes,
                    self.g1_dim,
                    self.g2_dim,
                    axis_dim=self.axis_dim,
                    update_chnnl_2=(ii != nlayers - 1),
                    do_bn_mode=do_bn_mode,
                    bn_momentum=bn_momentum,
                    update_g1_has_conv=update_g1_has_conv,
                    update_g1_has_drrd=update_g1_has_drrd,
                    update_g1_has_grrg=update_g1_has_grrg,
                    update_g1_has_attn=update_g1_has_attn,
                    update_g2_has_g1g1=update_g2_has_g1g1,
                    update_g2_has_attn=update_g2_has_attn,
                    update_h2=update_h2,
                    attn1_hidden=attn1_hidden,
                    attn1_nhead=attn1_nhead,
                    attn2_has_gate=attn2_has_gate,
                    attn2_hidden=attn2_hidden,
                    attn2_nhead=attn2_nhead,
                    activation_function=activation_function,
                    update_style=update_style,
                    smooth=smooth,
                )
            )
        self.layers = torch.nn.ModuleList(layers)

        sshape = (self.ntypes, self.nnei, 4)
        mean = torch.zeros(sshape, dtype=mydtype, device=mydev)
        stddev = torch.ones(sshape, dtype=mydtype, device=mydev)
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.stats = None

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.rcut

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

    def mixed_types(self) -> bool:
        """If true, the discriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the discriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return True

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

    def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_atype_embd: Optional[torch.Tensor] = None,
        mapping: Optional[torch.Tensor] = None,
    ):
        assert mapping is not None
        assert extended_atype_embd is not None
        nframes, nloc, nnei = nlist.shape
        nall = extended_coord.view(nframes, -1).shape[1] // 3
        atype = extended_atype[:, :nloc]
        # nb x nloc x nnei x 4, nb x nloc x nnei x 3, nb x nloc x nnei x 1
        dmatrix, diff, sw = prod_env_mat(
            extended_coord,
            nlist,
            atype,
            self.mean,
            self.stddev,
            self.rcut,
            self.rcut_smth,
        )
        nlist_mask = nlist != -1
        sw = torch.squeeze(sw, -1)
        # beyond the cutoff sw should be 0.0
        sw = sw.masked_fill(~nlist_mask, 0.0)

        # [nframes, nloc, tebd_dim]
        atype_embd = extended_atype_embd[:, :nloc, :]
        assert list(atype_embd.shape) == [nframes, nloc, self.g1_dim]

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
        mapping = mapping.view(nframes, nall).unsqueeze(-1).expand(-1, -1, self.g1_dim)
        for idx, ll in enumerate(self.layers):
            # g1:     nb x nloc x ng1
            # g1_ext: nb x nall x ng1
            g1_ext = torch.gather(g1, 1, mapping)
            g1, g2, h2 = ll.forward(
                g1_ext,
                g2,
                h2,
                nlist,
                nlist_mask,
                sw,
            )

        # uses the last layer.
        # nb x nloc x 3 x ng2
        h2g2 = ll._cal_h2g2(g2, h2, nlist_mask, sw)
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
