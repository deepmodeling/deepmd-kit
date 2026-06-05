# SPDX-License-Identifier: LGPL-3.0-or-later
"""Variational-Gaussian smooth descriptor (se_a_vg) for DeepMD PT backend."""

from __future__ import (
    annotations,
)

import itertools
from collections.abc import (
    Callable,
)
from typing import (
    Any,
    ClassVar,
    Final,
)

import numpy as np
import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.descriptor import (
    DescriptorBlock,
)
from deepmd.pt.model.descriptor.env_mat_vg import (
    VG_ENV_DIM,
    prod_env_mat_vg,
    tabulate_fusion_se_a_vg,
)
from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.network.mlp import (
    EmbeddingNet,
)
from deepmd.pt.model.network.network import (
    NetworkCollection,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
)
from deepmd.pt.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt.utils.tabulate import (
    DPTabulate,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.dpmodel.utils import EnvMat as DPEnvMat
from deepmd.utils.path import (
    DPPath,
)

from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)

if not hasattr(torch.ops.deepmd, "tabulate_fusion_se_a"):

    def tabulate_fusion_se_a(
        argument0: torch.Tensor,
        argument1: torch.Tensor,
        argument2: torch.Tensor,
        argument3: torch.Tensor,
        argument4: int,
    ) -> list[torch.Tensor]:
        raise NotImplementedError(
            "tabulate_fusion_se_a is not available since customized PyTorch OP library is not built when freezing the model. "
            "See documentation for model compression for details."
        )

    torch.ops.deepmd.tabulate_fusion_se_a = tabulate_fusion_se_a


@DescriptorBlock.register("se_a_vg")
class DescrptBlockSeAVg(DescriptorBlock):
    """DP-SE descriptor block with VGM Gaussian-averaged radial kernel (5-column env mat)."""

    ndescrpt: Final[int]
    __constants__: ClassVar[list] = ["ndescrpt"]

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: int | list[int],
        neuron: list[int] | None = None,
        axis_neuron: int = 16,
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = "float64",
        resnet_dt: bool = False,
        exclude_types: list[tuple[int, int]] | None = None,
        env_protection: float = 0.0,
        type_one_side: bool = True,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        if neuron is None:
            neuron = [25, 50, 100]
        if exclude_types is None:
            exclude_types = []
        self.rcut = float(rcut)
        self.rcut_smth = float(rcut_smth)
        self.neuron = neuron
        self.filter_neuron = self.neuron
        self.axis_neuron = axis_neuron
        self.set_davg_zero = set_davg_zero
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.resnet_dt = resnet_dt
        self.env_protection = env_protection
        self.ntypes = len(sel)
        self.type_one_side = type_one_side
        self.seed = seed
        self.reinit_exclude(exclude_types)

        self.sel = sel if isinstance(sel, list) else [sel]
        self.sec = [0, *np.cumsum(self.sel).tolist()]
        self.nnei = sum(self.sel)
        self.ndescrpt = self.nnei * VG_ENV_DIM

        wanted_shape = (self.ntypes, self.nnei, VG_ENV_DIM)
        mean = torch.zeros(wanted_shape, dtype=self.prec, device=env.DEVICE)
        stddev = torch.ones(wanted_shape, dtype=self.prec, device=env.DEVICE)
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)

        ndim = 1 if self.type_one_side else 2
        filter_layers = NetworkCollection(
            ndim=ndim, ntypes=len(self.sel), network_type="embedding_network"
        )
        for ii, embedding_idx in enumerate(
            itertools.product(range(self.ntypes), repeat=ndim)
        ):
            filter_layers[embedding_idx] = EmbeddingNet(
                1,
                self.filter_neuron,
                activation_function=self.activation_function,
                precision=self.precision,
                resnet_dt=self.resnet_dt,
                seed=child_seed(self.seed, ii),
                trainable=trainable,
            )
        self.filter_layers = filter_layers
        self.stats = None
        self.trainable = trainable
        for param in self.parameters():
            param.requires_grad = trainable
        self.compress = False
        self.compress_info = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(0, dtype=self.prec, device="cpu"))
                for _ in range(len(self.filter_layers.networks))
            ]
        )
        self.compress_data = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(0, dtype=self.prec, device=env.DEVICE))
                for _ in range(len(self.filter_layers.networks))
            ]
        )

    def get_rcut(self) -> float:
        return self.rcut

    def get_rcut_smth(self) -> float:
        return self.rcut_smth

    def get_nsel(self) -> int:
        return self.nnei

    def get_sel(self) -> list[int]:
        return self.sel

    def get_ntypes(self) -> int:
        return self.ntypes

    def get_dim_out(self) -> int:
        return self.dim_out

    def get_dim_emb(self) -> int:
        return self.neuron[-1]

    def get_dim_in(self) -> int:
        return 0

    def mixed_types(self) -> bool:
        return False

    def get_env_protection(self) -> float:
        return self.env_protection

    @property
    def dim_out(self) -> int:
        return self.filter_neuron[-1] * self.axis_neuron

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        if key in ("avg", "data_avg", "davg"):
            self.mean = value
        elif key in ("std", "data_std", "dstd"):
            self.stddev = value
        else:
            raise KeyError(key)

    def __getitem__(self, key: str) -> torch.Tensor:
        if key in ("avg", "data_avg", "davg"):
            return self.mean
        elif key in ("std", "data_std", "dstd"):
            return self.stddev
        else:
            raise KeyError(key)

    def reinit_exclude(
        self,
        exclude_types: list[tuple[int, int]] | None = None,
    ) -> None:
        if exclude_types is None:
            exclude_types = []
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        path: DPPath | None = None,
    ) -> None:
        if callable(merged):
            sampled = merged()
        else:
            sampled = merged
        sumv = np.zeros((self.ntypes, self.nnei, VG_ENV_DIM), dtype=np.float64)
        sumv2 = np.zeros_like(sumv)
        sumn = np.zeros((self.ntypes, self.nnei), dtype=np.float64)
        for system in sampled:
            coord = np.asarray(system["coord"], dtype=np.float64)
            atype = np.asarray(system["atype"], dtype=np.int32)
            box = system.get("box")
            nframes, nloc = atype.shape[:2]
            aparam = system.get("aparam")
            if aparam is None:
                aparam_np = np.zeros((nframes, nloc, 1), dtype=np.float64)
            else:
                aparam_np = np.asarray(aparam, dtype=np.float64).reshape(nframes, nloc, -1)
                if aparam_np.shape[-1] != 1:
                    aparam_np = aparam_np[..., :1]
            for ff in range(nframes):
                coord_t = torch.tensor(
                    coord[ff], dtype=self.prec, device=env.DEVICE
                ).reshape(1, -1)
                atype_t = torch.tensor(
                    atype[ff], dtype=torch.long, device=env.DEVICE
                ).reshape(1, -1)
                box_t = None
                if box is not None:
                    box_t = torch.tensor(
                        box[ff], dtype=self.prec, device=env.DEVICE
                    ).reshape(1, 9)
                aparam_t = torch.tensor(
                    aparam_np[ff], dtype=self.prec, device=env.DEVICE
                ).reshape(1, nloc, 1)
                extended_coord, extended_atype, _, nlist = (
                    extend_input_and_build_neighbor_list(
                        coord_t,
                        atype_t,
                        self.rcut,
                        self.sel,
                        mixed_types=False,
                        box=box_t,
                    )
                )
                env_mat, _, _ = prod_env_mat_vg(
                    extended_coord,
                    nlist,
                    extended_atype[:, :nloc],
                    aparam_t,
                    self.mean,
                    torch.ones_like(self.stddev),
                    self.rcut,
                    self.rcut_smth,
                    protection=self.env_protection,
                )
                env_mat = env_mat.detach().cpu().numpy().reshape(nloc, self.nnei, VG_ENV_DIM)
                for ii in range(nloc):
                    ti = int(atype[ff, ii])
                    sumv[ti] += env_mat[ii]
                    sumv2[ti] += env_mat[ii] * env_mat[ii]
                    sumn[ti] += 1.0
        sumn_safe = np.maximum(sumn, 1.0)[..., None]
        mean = sumv / sumn_safe
        var = sumv2 / sumn_safe - mean * mean
        stddev = np.sqrt(np.maximum(var, 1e-2))
        if not self.set_davg_zero:
            self.mean.copy_(torch.tensor(mean, dtype=self.prec, device=env.DEVICE))
        self.stddev.copy_(
            torch.tensor(stddev, dtype=self.prec, device=env.DEVICE)
        )

    def enable_compression(
        self,
        table_data: dict[str, torch.Tensor],
        table_config: list[int | float],
        lower: dict[str, int],
        upper: dict[str, int],
    ) -> None:
        for embedding_idx, ll in enumerate(self.filter_layers.networks):
            del ll
            if self.type_one_side:
                ii = embedding_idx
                ti = -1
            else:
                ii = embedding_idx // self.ntypes
                ti = embedding_idx % self.ntypes
            if self.type_one_side:
                net = "filter_-1_net_" + str(ii)
            else:
                net = "filter_" + str(ti) + "_net_" + str(ii)
            info_ii = torch.as_tensor(
                [
                    lower[net],
                    upper[net],
                    upper[net] * table_config[0],
                    table_config[1],
                    table_config[2],
                    table_config[3],
                ],
                dtype=self.prec,
                device="cpu",
            )
            tensor_data_ii = table_data[net].to(device=env.DEVICE, dtype=self.prec)
            self.compress_data[embedding_idx] = tensor_data_ii
            self.compress_info[embedding_idx] = info_ii
        self.compress = True

    def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_atype_embd: torch.Tensor | None = None,
        mapping: torch.Tensor | None = None,
        type_embedding: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        del extended_atype_embd, mapping, type_embedding
        nf = nlist.shape[0]
        nloc = nlist.shape[1]
        atype = extended_atype[:, :nloc]
        if aparam is None:
            aparam = torch.zeros(
                (nf, nloc, 1),
                dtype=self.prec,
                device=extended_coord.device,
            )
        else:
            aparam = aparam.to(dtype=self.prec, device=extended_coord.device)
            if aparam.shape[-1] != 1:
                aparam = aparam[..., :1]
            if aparam.shape[1] != nloc:
                aparam = aparam.reshape(nf, nloc, 1)

        dmatrix, diff, sw = prod_env_mat_vg(
            extended_coord,
            nlist,
            atype,
            aparam,
            self.mean,
            self.stddev,
            self.rcut,
            self.rcut_smth,
            protection=self.env_protection,
        )
        dmatrix = dmatrix.view(-1, self.nnei, VG_ENV_DIM)
        nfnl = dmatrix.shape[0]
        xyz_scatter = torch.zeros(
            [nfnl, VG_ENV_DIM, self.filter_neuron[-1]],
            dtype=self.prec,
            device=extended_coord.device,
        )
        exclude_mask = self.emask(nlist, extended_atype).view(nfnl, self.nnei)
        for embedding_idx, (ll, compress_data_ii, compress_info_ii) in enumerate(
            zip(
                self.filter_layers.networks,
                self.compress_data,
                self.compress_info,
                strict=True,
            )
        ):
            if self.type_one_side:
                ii = embedding_idx
                ti_mask = None
            else:
                ii = embedding_idx // self.ntypes
                ti = embedding_idx % self.ntypes
                ti_mask = atype.ravel().eq(ti)
            if ti_mask is not None:
                mm = exclude_mask[ti_mask, self.sec[ii] : self.sec[ii + 1]]
                rr = dmatrix[ti_mask, self.sec[ii] : self.sec[ii + 1], :]
            else:
                mm = exclude_mask[:, self.sec[ii] : self.sec[ii + 1]]
                rr = dmatrix[:, self.sec[ii] : self.sec[ii + 1], :]
            rr = rr * mm[:, :, None]
            ss = rr[:, :, :1]
            if self.compress:
                ss = ss.reshape(-1, 1)
                gr = tabulate_fusion_se_a_vg(
                    compress_data_ii.contiguous(),
                    compress_info_ii.cpu().contiguous(),
                    ss.contiguous(),
                    rr.contiguous(),
                    self.filter_neuron[-1],
                )
            else:
                gg = ll.forward(ss)
                gr = torch.matmul(rr.permute(0, 2, 1), gg)
            if ti_mask is not None:
                xyz_scatter[ti_mask] += gr
            else:
                xyz_scatter += gr

        xyz_scatter /= self.nnei
        xyz_scatter_1 = xyz_scatter.permute(0, 2, 1)
        rot_mat = xyz_scatter_1[:, :, 1:4]
        xyz_scatter_2 = xyz_scatter[:, :, 0 : self.axis_neuron]
        result = torch.matmul(xyz_scatter_1, xyz_scatter_2)
        result = result.view(nf, nloc, self.filter_neuron[-1] * self.axis_neuron)
        rot_mat = rot_mat.view([nf, nloc] + list(rot_mat.shape[1:]))
        return result, rot_mat, None, None, sw

    def has_message_passing(self) -> bool:
        return False

    def need_sorted_nlist_for_lower(self) -> bool:
        return False


@BaseDescriptor.register("se_a_vg")
@BaseDescriptor.register("se_e2_a_vg")
class DescrptSeAVg(DescrptSeA):
    """VG-aware wrapper around :class:`DescrptBlockSeAVg`."""

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: list[int] | int,
        neuron: list[int] | None = None,
        axis_neuron: int = 16,
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = "float64",
        resnet_dt: bool = False,
        exclude_types: list[tuple[int, int]] | None = None,
        env_protection: float = 0.0,
        type_one_side: bool = True,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        ntypes: int | None = None,
        type_map: list[str] | None = None,
        spin: Any | None = None,
    ) -> None:
        del ntypes, spin
        nn.Module.__init__(self)
        BaseDescriptor.__init__(self)
        self.type_map = type_map
        self.compress = False
        self.prec = PRECISION_DICT[precision]
        self.sea = DescrptBlockSeAVg(
            rcut,
            rcut_smth,
            sel,
            neuron=neuron or [25, 50, 100],
            axis_neuron=axis_neuron,
            set_davg_zero=set_davg_zero,
            activation_function=activation_function,
            precision=precision,
            resnet_dt=resnet_dt,
            exclude_types=exclude_types or [],
            env_protection=env_protection,
            type_one_side=type_one_side,
            trainable=trainable,
            seed=seed,
        )

    def forward(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
        fparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        del comm_dict, fparam, charge_spin
        coord_ext = coord_ext.to(dtype=self.prec)
        g1, rot_mat, g2, h2, sw = self.sea.forward(
            nlist,
            coord_ext,
            atype_ext,
            aparam=aparam,
            mapping=mapping,
        )
        return (
            g1.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            rot_mat.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION)
            if rot_mat is not None
            else None,
            None,
            None,
            sw.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION) if sw is not None else None,
        )

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        if self.compress:
            raise ValueError("Compression is already enabled.")
        data = self.serialize()
        self.table = DPTabulate(
            self,
            data["neuron"],
            data["type_one_side"],
            data["exclude_types"],
            ActivationFn(data["activation_function"]),
        )
        self.table_config = [
            table_extrapolate,
            table_stride_1,
            table_stride_2,
            check_frequency,
        ]
        self.lower, self.upper = self.table.build(
            min_nbor_dist, table_extrapolate, table_stride_1, table_stride_2
        )
        self.sea.enable_compression(
            self.table.data, self.table_config, self.lower, self.upper
        )
        self.compress = True

    def serialize(self) -> dict:
        obj = self.sea
        return {
            "@class": "Descriptor",
            "type": "se_a_vg",
            "@version": 2,
            "rcut": obj.rcut,
            "rcut_smth": obj.rcut_smth,
            "sel": obj.sel,
            "neuron": obj.neuron,
            "axis_neuron": obj.axis_neuron,
            "resnet_dt": obj.resnet_dt,
            "set_davg_zero": obj.set_davg_zero,
            "activation_function": obj.activation_function,
            "precision": RESERVED_PRECISION_DICT[obj.prec],
            "embeddings": obj.filter_layers.serialize(),
            "env_mat": DPEnvMat(
                obj.rcut, obj.rcut_smth, obj.env_protection
            ).serialize(),
            "exclude_types": obj.exclude_types,
            "env_protection": obj.env_protection,
            "@variables": {
                "davg": obj["davg"].detach().cpu().numpy(),
                "dstd": obj["dstd"].detach().cpu().numpy(),
            },
            "type_map": self.type_map,
            "trainable": True,
            "type_one_side": obj.type_one_side,
            "spin": None,
        }

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any | None = None
    ) -> None:
        raise NotImplementedError(
            "Descriptor se_a_vg does not support changing type related params yet."
        )

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float | None]:
        local_jdata_cpy = local_jdata.copy()
        min_nbor_dist, local_jdata_cpy["sel"] = UpdateSel().update_one_sel(
            train_data, type_map, local_jdata_cpy["rcut"], local_jdata_cpy["sel"], False
        )
        return local_jdata_cpy, min_nbor_dist
