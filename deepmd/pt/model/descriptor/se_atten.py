# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np
import torch

from deepmd.pt.model.descriptor.descriptor import (
    DescriptorBlock,
)
from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat,
)
from deepmd.pt.model.network.network import (
    NeighborWiseAttention,
    TypeFilter,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.utils.env_mat_stat import (
    StatItem,
)
from deepmd.utils.path import (
    DPPath,
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
        post_ln=True,
        ffn=False,
        ffn_embed_dim=1024,
        activation_function="tanh",
        scaling_factor=1.0,
        head_num=1,
        normalize=True,
        temperature=None,
        return_rot=False,
        type: Optional[str] = None,
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
        self.filter_neuron = neuron
        self.axis_neuron = axis_neuron
        self.tebd_dim = tebd_dim
        self.tebd_input_mode = tebd_input_mode
        self.set_davg_zero = set_davg_zero
        self.attn_dim = attn
        self.attn_layer = attn_layer
        self.attn_dotr = attn_dotr
        self.attn_mask = attn_mask
        self.post_ln = post_ln
        self.ffn = ffn
        self.ffn_embed_dim = ffn_embed_dim
        self.activation = activation_function
        # TODO: To be fixed: precision should be given from inputs
        self.prec = torch.float64
        self.scaling_factor = scaling_factor
        self.head_num = head_num
        self.normalize = normalize
        self.temperature = temperature
        self.return_rot = return_rot

        if isinstance(sel, int):
            sel = [sel]

        self.ntypes = ntypes
        self.sel = sel
        self.sec = self.sel
        self.split_sel = self.sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 4
        self.dpa1_attention = NeighborWiseAttention(
            self.attn_layer,
            self.nnei,
            self.filter_neuron[-1],
            self.attn_dim,
            dotr=self.attn_dotr,
            do_mask=self.attn_mask,
            post_ln=self.post_ln,
            ffn=self.ffn,
            ffn_embed_dim=self.ffn_embed_dim,
            activation=self.activation,
            scaling_factor=self.scaling_factor,
            head_num=self.head_num,
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
        self.filter_layers = torch.nn.ModuleList(filter_layers)
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

    def get_dim_in(self) -> int:
        """Returns the output dimension."""
        return self.dim_in

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        return self.dim_out

    def get_dim_emb(self) -> int:
        """Returns the output dimension of embedding."""
        return self.filter_neuron[-1]

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
        dmatrix, diff, sw = prod_env_mat(
            extended_coord,
            nlist,
            atype,
            self.mean,
            self.stddev,
            self.rcut,
            self.rcut_smth,
        )
        # [nfxnlocxnnei, self.ndescrpt]
        dmatrix = dmatrix.view(-1, self.ndescrpt)
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
        ret = self.filter_layers[0](
            dmatrix,
            atype_tebd=atype_tebd_nnei,
            nlist_tebd=atype_tebd_nlist,
        )  # shape is [nframes*nall, self.neei, out_size]
        input_r = torch.nn.functional.normalize(
            dmatrix.reshape(-1, self.nnei, 4)[:, :, 1:4], dim=-1
        )
        ret = self.dpa1_attention(
            ret, nlist_mask, input_r=input_r, sw=sw
        )  # shape is [nframes*nloc, self.neei, out_size]
        inputs_reshape = dmatrix.view(-1, self.nnei, 4).permute(
            0, 2, 1
        )  # shape is [nframes*natoms[0], 4, self.neei]
        xyz_scatter = torch.matmul(
            inputs_reshape, ret
        )  # shape is [nframes*natoms[0], 4, out_size]
        xyz_scatter = xyz_scatter / self.nnei
        xyz_scatter_1 = xyz_scatter.permute(0, 2, 1)
        rot_mat = xyz_scatter_1[:, :, 1:4]
        xyz_scatter_2 = xyz_scatter[:, :, 0 : self.axis_neuron]
        result = torch.matmul(
            xyz_scatter_1, xyz_scatter_2
        )  # shape is [nframes*nloc, self.filter_neuron[-1], self.axis_neuron]
        return (
            result.view(-1, nloc, self.filter_neuron[-1] * self.axis_neuron),
            ret.view(-1, nloc, self.nnei, self.filter_neuron[-1]),
            dmatrix.view(-1, nloc, self.nnei, 4)[..., 1:],
            rot_mat.view(-1, nloc, self.filter_neuron[-1], 3),
            sw,
        )


def analyze_descrpt(matrix, ndescrpt, natoms, mixed_types=False, real_atype=None):
    """Collect avg, square avg and count of descriptors in a batch."""
    ntypes = natoms.shape[1] - 2
    if not mixed_types:
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
