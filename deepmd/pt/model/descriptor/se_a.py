# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    ClassVar,
    List,
    Optional,
)

import numpy as np
import torch

from deepmd.pt.model.descriptor import (
    Descriptor,
    DescriptorBlock,
    compute_std,
    prod_env_mat_se_a,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)

try:
    from typing import (
        Final,
    )
except ImportError:
    from torch.jit import Final

from deepmd.model_format import EnvMat as DPEnvMat
from deepmd.pt.model.network.mlp import (
    EmbeddingNet,
    NetworkCollection,
)
from deepmd.pt.model.network.network import (
    TypeFilter,
)


@Descriptor.register("se_e2_a")
class DescrptSeA(Descriptor):
    def __init__(
        self,
        rcut,
        rcut_smth,
        sel,
        neuron=[25, 50, 100],
        axis_neuron=16,
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = "float64",
        resnet_dt: bool = False,
        old_impl: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.sea = DescrptBlockSeA(
            rcut,
            rcut_smth,
            sel,
            neuron,
            axis_neuron,
            set_davg_zero,
            activation_function,
            precision,
            resnet_dt,
            old_impl,
            **kwargs,
        )

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.sea.get_rcut()

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return self.sea.get_nsel()

    def get_sel(self) -> List[int]:
        """Returns the number of selected atoms for each type."""
        return self.sea.get_sel()

    def get_ntype(self) -> int:
        """Returns the number of element types."""
        return self.sea.get_ntype()

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        return self.sea.get_dim_out()

    @property
    def dim_out(self):
        """Returns the output dimension of this descriptor."""
        return self.sea.dim_out

    def compute_input_stats(self, merged):
        """Update mean and stddev for descriptor elements."""
        return self.sea.compute_input_stats(merged)

    def init_desc_stat(self, sumr, suma, sumn, sumr2, suma2):
        self.sea.init_desc_stat(sumr, suma, sumn, sumr2, suma2)

    @classmethod
    def get_stat_name(cls, config):
        descrpt_type = config["type"]
        assert descrpt_type in ["se_e2_a"]
        return f'stat_file_sea_rcut{config["rcut"]:.2f}_smth{config["rcut_smth"]:.2f}_sel{config["sel"]}.npz'

    @classmethod
    def get_data_process_key(cls, config):
        descrpt_type = config["type"]
        assert descrpt_type in ["se_e2_a"]
        return {"sel": config["sel"], "rcut": config["rcut"]}

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
    ):
        return self.sea.forward(nlist, extended_coord, extended_atype, None, mapping)

    def set_stat_mean_and_stddev(
        self,
        mean: torch.Tensor,
        stddev: torch.Tensor,
    ) -> None:
        self.sea.mean = mean
        self.sea.stddev = stddev

    def serialize(self) -> dict:
        obj = self.sea
        return {
            "rcut": obj.rcut,
            "rcut_smth": obj.rcut_smth,
            "sel": obj.sel,
            "neuron": obj.neuron,
            "axis_neuron": obj.axis_neuron,
            "resnet_dt": obj.resnet_dt,
            "set_davg_zero": obj.set_davg_zero,
            "activation_function": obj.activation_function,
            "precision": obj.precision,
            "embeddings": obj.filter_layers.serialize(),
            "env_mat": DPEnvMat(obj.rcut, obj.rcut_smth).serialize(),
            "@variables": {
                "davg": obj["davg"].detach().cpu().numpy(),
                "dstd": obj["dstd"].detach().cpu().numpy(),
            },
            ## to be updated when the options are supported.
            "trainable": True,
            "type_one_side": True,
            "exclude_types": [],
            "spin": None,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptSeA":
        variables = data.pop("@variables")
        embeddings = data.pop("embeddings")
        env_mat = data.pop("env_mat")
        obj = cls(**data)

        def t_cvt(xx):
            return torch.tensor(xx, dtype=obj.sea.prec, device=env.DEVICE)

        obj.sea["davg"] = t_cvt(variables["davg"])
        obj.sea["dstd"] = t_cvt(variables["dstd"])
        obj.sea.filter_layers = NetworkCollection.deserialize(embeddings)
        return obj


@DescriptorBlock.register("se_e2_a")
class DescrptBlockSeA(DescriptorBlock):
    ndescrpt: Final[int]
    __constants__: ClassVar[list] = ["ndescrpt"]

    def __init__(
        self,
        rcut,
        rcut_smth,
        sel,
        neuron=[25, 50, 100],
        axis_neuron=16,
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = "float64",
        resnet_dt: bool = False,
        old_impl: bool = False,
        **kwargs,
    ):
        """Construct an embedding net of type `se_a`.

        Args:
        - rcut: Cut-off radius.
        - rcut_smth: Smooth hyper-parameter for pair force & energy.
        - sel: For each element type, how many atoms is selected as neighbors.
        - filter_neuron: Number of neurons in each hidden layers of the embedding net.
        - axis_neuron: Number of columns of the sub-matrix of the embedding matrix.
        """
        super().__init__()
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.neuron = neuron
        self.filter_neuron = self.neuron
        self.axis_neuron = axis_neuron
        self.set_davg_zero = set_davg_zero
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.resnet_dt = resnet_dt
        self.old_impl = old_impl

        self.ntypes = len(sel)  # 元素数量
        self.sel = sel  # 每种元素在邻居中的位移
        self.sec = torch.tensor(
            np.append([0], np.cumsum(self.sel)), dtype=int, device=env.DEVICE
        )
        self.split_sel = self.sel
        self.nnei = sum(sel)  # 总的邻居数量
        self.ndescrpt = self.nnei * 4  # 描述符的元素数量

        wanted_shape = (self.ntypes, self.nnei, 4)
        mean = torch.zeros(wanted_shape, dtype=self.prec, device=env.DEVICE)
        stddev = torch.ones(wanted_shape, dtype=self.prec, device=env.DEVICE)
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.filter_layers_old = None
        self.filter_layers = None

        if self.old_impl:
            filter_layers = []
            # TODO: remove
            start_index = 0
            for type_i in range(self.ntypes):
                one = TypeFilter(start_index, sel[type_i], self.filter_neuron)
                filter_layers.append(one)
                start_index += sel[type_i]
            self.filter_layers_old = torch.nn.ModuleList(filter_layers)
        else:
            filter_layers = NetworkCollection(
                ndim=1, ntypes=len(sel), network_type="embedding_network"
            )
            # TODO: ndim=2 if type_one_side=False
            for ii in range(self.ntypes):
                filter_layers[(ii,)] = EmbeddingNet(
                    1,
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

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        return self.dim_out

    def get_dim_in(self) -> int:
        """Returns the input dimension."""
        return self.dim_in

    @property
    def dim_out(self):
        """Returns the output dimension of this descriptor."""
        return self.filter_neuron[-1] * self.axis_neuron

    @property
    def dim_in(self):
        """Returns the atomic input dimension of this descriptor."""
        return 0

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
        for system in merged:  # 逐个 system 的分析
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
            sysr, sysr2, sysa, sysa2, sysn = analyze_descrpt(
                env_mat.detach().cpu().numpy(), self.ndescrpt, system["natoms"]
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
    ):
        """Calculate decoded embedding for each atom.

        Args:
        - coord: Tell atom coordinates with shape [nframes, natoms[1]*3].
        - atype: Tell atom types with shape [nframes, natoms[1]].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].
        - box: Tell simulation box with shape [nframes, 9].

        Returns
        -------
        - `torch.Tensor`: descriptor matrix with shape [nframes, natoms[0]*self.filter_neuron[-1]*self.axis_neuron].
        """
        del extended_atype_embd, mapping
        nloc = nlist.shape[1]
        atype = extended_atype[:, :nloc]
        dmatrix, diff, _ = prod_env_mat_se_a(
            extended_coord,
            nlist,
            atype,
            self.mean,
            self.stddev,
            self.rcut,
            self.rcut_smth,
        )

        if self.old_impl:
            assert self.filter_layers_old is not None
            dmatrix = dmatrix.view(
                -1, self.ndescrpt
            )  # shape is [nframes*nall, self.ndescrpt]
            xyz_scatter = torch.empty(
                1,
            )
            ret = self.filter_layers_old[0](dmatrix)
            xyz_scatter = ret
            for ii, transform in enumerate(self.filter_layers_old[1:]):
                # shape is [nframes*nall, 4, self.filter_neuron[-1]]
                ret = transform.forward(dmatrix)
                xyz_scatter = xyz_scatter + ret
        else:
            assert self.filter_layers is not None
            dmatrix = dmatrix.view(-1, self.nnei, 4)
            nfnl = dmatrix.shape[0]
            # pre-allocate a shape to pass jit
            xyz_scatter = torch.zeros(
                [nfnl, 4, self.filter_neuron[-1]], dtype=self.prec, device=env.DEVICE
            )
            for ii, ll in enumerate(self.filter_layers.networks):
                # nfnl x nt x 4
                rr = dmatrix[:, self.sec[ii] : self.sec[ii + 1], :]
                ss = rr[:, :, :1]
                # nfnl x nt x ng
                gg = ll.forward(ss)
                # nfnl x 4 x ng
                gr = torch.matmul(rr.permute(0, 2, 1), gg)
                xyz_scatter += gr

        xyz_scatter /= self.nnei
        xyz_scatter_1 = xyz_scatter.permute(0, 2, 1)
        rot_mat = xyz_scatter_1[:, :, 1:4]
        xyz_scatter_2 = xyz_scatter[:, :, 0 : self.axis_neuron]
        result = torch.matmul(
            xyz_scatter_1, xyz_scatter_2
        )  # shape is [nframes*nall, self.filter_neuron[-1], self.axis_neuron]
        return (
            result.view(-1, nloc, self.filter_neuron[-1] * self.axis_neuron),
            None,
            None,
            None,
            None,
        )


def analyze_descrpt(matrix, ndescrpt, natoms):
    """Collect avg, square avg and count of descriptors in a batch."""
    ntypes = natoms.shape[1] - 2
    start_index = 0
    sysr = []  # 每类元素的径向均值
    sysa = []  # 每类元素的轴向均值
    sysn = []  # 每类元素的样本数量
    sysr2 = []  # 每类元素的径向平方均值
    sysa2 = []  # 每类元素的轴向平方均值
    for type_i in range(ntypes):
        end_index = start_index + natoms[0, 2 + type_i]
        dd = matrix[:, start_index:end_index]  # 本元素所有原子的 descriptor
        start_index = end_index
        dd = np.reshape(
            dd, [-1, 4]
        )  # Shape is [nframes*natoms[2+type_id]*self.nnei, 4]
        ddr = dd[:, :1]  # 径向值
        dda = dd[:, 1:]  # XYZ 轴分量
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
    return sysr, sysr2, sysa, sysa2, sysn
