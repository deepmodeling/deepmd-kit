# SPDX-License-Identifier: LGPL-3.0-or-later

import copy
import logging
from typing import (
    Callable,
    Optional,
    Union,
)

import numpy as np
import paddle

from deepmd.dpmodel.atomic_model import (
    make_base_atomic_model,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pd.utils import (
    AtomExcludeMask,
    PairExcludeMask,
    env,
)
from deepmd.pd.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pd.utils.stat import (
    compute_output_stats,
)
from deepmd.pd.utils.utils import (
    to_numpy_array,
    to_paddle_tensor,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
    map_atom_exclude_types,
    map_pair_exclude_types,
)
from deepmd.utils.path import (
    DPPath,
)

log = logging.getLogger(__name__)
dtype = env.GLOBAL_PD_FLOAT_PRECISION
device = env.DEVICE

BaseAtomicModel_ = make_base_atomic_model(paddle.Tensor)


class BaseAtomicModel(paddle.nn.Layer, BaseAtomicModel_):
    """The base of atomic model.

    Parameters
    ----------
    type_map
        Mapping atom type to the name (str) of the type.
        For example `type_map[1]` gives the name of the type 1.
    atom_exclude_types
        Exclude the atomic contribution of the given types
    pair_exclude_types
        Exclude the pair of atoms of the given types from computing the output
        of the atomic model. Implemented by removing the pairs from the nlist.
    rcond : float, optional
        The condition number for the regression of atomic energy.
    preset_out_bias : Dict[str, list[Optional[paddle.Tensor]]], optional
        Specifying atomic energy contribution in vacuum. Given by key:value pairs.
        The value is a list specifying the bias. the elements can be None or np.array of output shape.
        For example: [None, [2.]] means type 0 is not set, type 1 is set to [2.]
        The `set_davg_zero` key in the descriptor should be set.

    """

    def __init__(
        self,
        type_map: list[str],
        atom_exclude_types: list[int] = [],
        pair_exclude_types: list[tuple[int, int]] = [],
        rcond: Optional[float] = None,
        preset_out_bias: Optional[dict[str, np.ndarray]] = None,
    ):
        paddle.nn.Layer.__init__(self)
        BaseAtomicModel_.__init__(self)
        self.type_map = type_map
        self.reinit_atom_exclude(atom_exclude_types)
        self.reinit_pair_exclude(pair_exclude_types)
        self.rcond = rcond
        self.preset_out_bias = preset_out_bias

    def init_out_stat(self):
        """Initialize the output bias."""
        ntypes = self.get_ntypes()
        self.bias_keys: list[str] = list(self.fitting_output_def().keys())
        self.max_out_size = max(
            [self.atomic_output_def()[kk].size for kk in self.bias_keys]
        )
        self.n_out = len(self.bias_keys)
        out_bias_data = self._default_bias()
        out_std_data = self._default_std()
        self.register_buffer("out_bias", out_bias_data)
        self.register_buffer("out_std", out_std_data)

    def set_out_bias(self, out_bias: paddle.Tensor) -> None:
        self.out_bias = out_bias

    def __setitem__(self, key, value):
        if key in ["out_bias"]:
            self.out_bias = value
        elif key in ["out_std"]:
            self.out_std = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        if key in ["out_bias"]:
            return self.out_bias
        elif key in ["out_std"]:
            return self.out_std
        else:
            raise KeyError(key)

    def get_type_map(self) -> list[str]:
        """Get the type map."""
        return self.type_map

    def reinit_atom_exclude(
        self,
        exclude_types: list[int] = [],
    ):
        self.atom_exclude_types = exclude_types
        if exclude_types == []:
            self.atom_excl = None
        else:
            self.atom_excl = AtomExcludeMask(self.get_ntypes(), self.atom_exclude_types)

    def reinit_pair_exclude(
        self,
        exclude_types: list[tuple[int, int]] = [],
    ):
        self.pair_exclude_types = exclude_types
        if exclude_types == []:
            self.pair_excl = None
        else:
            self.pair_excl = PairExcludeMask(self.get_ntypes(), self.pair_exclude_types)

    # to make jit happy...
    def make_atom_mask(
        self,
        atype: paddle.Tensor,
    ) -> paddle.Tensor:
        """The atoms with type < 0 are treated as virtual atoms,
        which serves as place-holders for multi-frame calculations
        with different number of atoms in different frames.

        Parameters
        ----------
        atype
            Atom types. >= 0 for real atoms <0 for virtual atoms.

        Returns
        -------
        mask
            True for real atoms and False for virtual atoms.

        """
        # supposed to be supported by all backends
        return atype >= 0

    def atomic_output_def(self) -> FittingOutputDef:
        old_def = self.fitting_output_def()
        old_list = list(old_def.get_data().values())
        return FittingOutputDef(
            old_list  # noqa:RUF005
            + [
                OutputVariableDef(
                    name="mask",
                    shape=[1],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                )
            ]
        )

    def forward_common_atomic(
        self,
        extended_coord: paddle.Tensor,
        extended_atype: paddle.Tensor,
        nlist: paddle.Tensor,
        mapping: Optional[paddle.Tensor] = None,
        fparam: Optional[paddle.Tensor] = None,
        aparam: Optional[paddle.Tensor] = None,
        comm_dict: Optional[dict[str, paddle.Tensor]] = None,
    ) -> dict[str, paddle.Tensor]:
        """Common interface for atomic inference.

        This method accept extended coordinates, extended atom typs, neighbor list,
        and predict the atomic contribution of the fit property.

        Parameters
        ----------
        extended_coord
            extended coordinates, shape: nf x (nall x 3)
        extended_atype
            extended atom typs, shape: nf x nall
            for a type < 0 indicating the atomic is virtual.
        nlist
            neighbor list, shape: nf x nloc x nsel
        mapping
            extended to local index mapping, shape: nf x nall
        fparam
            frame parameters, shape: nf x dim_fparam
        aparam
            atomic parameter, shape: nf x nloc x dim_aparam
        comm_dict
            The data needed for communication for parallel inference.

        Returns
        -------
        ret_dict
            dict of output atomic properties.
            should implement the definition of `fitting_output_def`.
            ret_dict["mask"] of shape nf x nloc will be provided.
            ret_dict["mask"][ff,ii] == 1 indicating the ii-th atom of the ff-th frame is real.
            ret_dict["mask"][ff,ii] == 0 indicating the ii-th atom of the ff-th frame is virtual.

        """
        _, nloc, _ = nlist.shape
        atype = extended_atype[:, :nloc]

        if self.pair_excl is not None:
            pair_mask = self.pair_excl(nlist, extended_atype)
            # exclude neighbors in the nlist
            nlist = paddle.where(pair_mask == 1, nlist, -1)

        ext_atom_mask = self.make_atom_mask(extended_atype)
        ret_dict = self.forward_atomic(
            extended_coord,
            paddle.where(
                ext_atom_mask, extended_atype, paddle.zeros_like(extended_atype)
            ),
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            comm_dict=comm_dict,
        )
        ret_dict = self.apply_out_stat(ret_dict, atype)

        # nf x nloc
        atom_mask = ext_atom_mask[:, :nloc].astype(paddle.int32)
        if self.atom_excl is not None:
            atom_mask *= self.atom_excl(atype)

        for kk in ret_dict.keys():
            out_shape = ret_dict[kk].shape
            out_shape2 = 1
            for ss in out_shape[2:]:
                out_shape2 *= ss
            ret_dict[kk] = (
                ret_dict[kk].reshape([out_shape[0], out_shape[1], out_shape2])
                * atom_mask.unsqueeze(2).astype(ret_dict[kk].dtype)
            ).reshape(out_shape)
        ret_dict["mask"] = atom_mask

        return ret_dict

    def forward(
        self,
        extended_coord: paddle.Tensor,
        extended_atype: paddle.Tensor,
        nlist: paddle.Tensor,
        mapping: Optional[paddle.Tensor] = None,
        fparam: Optional[paddle.Tensor] = None,
        aparam: Optional[paddle.Tensor] = None,
        comm_dict: Optional[dict[str, paddle.Tensor]] = None,
    ) -> dict[str, paddle.Tensor]:
        return self.forward_common_atomic(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            comm_dict=comm_dict,
        )

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        remap_index, has_new_type = get_index_between_two_maps(self.type_map, type_map)
        self.type_map = type_map
        self.reinit_atom_exclude(
            map_atom_exclude_types(self.atom_exclude_types, remap_index)
        )
        self.reinit_pair_exclude(
            map_pair_exclude_types(self.pair_exclude_types, remap_index)
        )
        if has_new_type:
            extend_shape = [
                self.out_bias.shape[0],
                len(type_map),
                *list(self.out_bias.shape[2:]),
            ]
            extend_bias = paddle.zeros(extend_shape, dtype=self.out_bias.dtype).to(
                device=self.out_bias.place
            )
            self.out_bias = paddle.concat([self.out_bias, extend_bias], axis=1)
            extend_std = paddle.ones(extend_shape, dtype=self.out_std.dtype).to(
                device=self.out_std.place
            )
            self.out_std = paddle.concat([self.out_std, extend_std], axis=1)
        self.out_bias = self.out_bias[:, remap_index, :]
        self.out_std = self.out_std[:, remap_index, :]

    def serialize(self) -> dict:
        return {
            "type_map": self.type_map,
            "atom_exclude_types": self.atom_exclude_types,
            "pair_exclude_types": self.pair_exclude_types,
            "rcond": self.rcond,
            "preset_out_bias": self.preset_out_bias,
            "@variables": {
                "out_bias": to_numpy_array(self.out_bias),
                "out_std": to_numpy_array(self.out_std),
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> "BaseAtomicModel":
        data = copy.deepcopy(data)
        variables = data.pop("@variables", None)
        variables = (
            {"out_bias": None, "out_std": None} if variables is None else variables
        )
        obj = cls(**data)
        obj["out_bias"] = (
            to_paddle_tensor(variables["out_bias"])
            if variables["out_bias"] is not None
            else obj._default_bias()
        )
        obj["out_std"] = (
            to_paddle_tensor(variables["out_std"])
            if variables["out_std"] is not None
            else obj._default_std()
        )
        return obj

    def compute_or_load_stat(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        stat_file_path: Optional[DPPath] = None,
    ):
        """
        Compute the output statistics (e.g. energy bias) for the fitting net from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `paddle.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        stat_file_path : Optional[DPPath]
            The path to the stat file.

        """
        raise NotImplementedError

    def compute_or_load_out_stat(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        stat_file_path: Optional[DPPath] = None,
    ):
        """
        Compute the output statistics (e.g. energy bias) for the fitting net from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `paddle.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        stat_file_path : Optional[DPPath]
            The path to the stat file.

        """
        self.change_out_bias(
            merged,
            stat_file_path=stat_file_path,
            bias_adjust_mode="set-by-statistic",
        )

    def apply_out_stat(
        self,
        ret: dict[str, paddle.Tensor],
        atype: paddle.Tensor,
    ):
        """Apply the stat to each atomic output.
        The developer may override the method to define how the bias is applied
        to the atomic output of the model.

        Parameters
        ----------
        ret
            The returned dict by the forward_atomic method
        atype
            The atom types. nf x nloc

        """
        out_bias, out_std = self._fetch_out_stat(self.bias_keys)
        for kk in self.bias_keys:
            # nf x nloc x odims, out_bias: ntypes x odims
            ret[kk] = ret[kk] + out_bias[kk][atype]
        return ret

    def change_out_bias(
        self,
        sample_merged,
        stat_file_path: Optional[DPPath] = None,
        bias_adjust_mode="change-by-statistic",
    ) -> None:
        """Change the output bias according to the input data and the pretrained model.

        Parameters
        ----------
        sample_merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `paddle.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        bias_adjust_mode : str
            The mode for changing output bias : ['change-by-statistic', 'set-by-statistic']
            'change-by-statistic' : perform predictions on labels of target dataset,
                    and do least square on the errors to obtain the target shift as bias.
            'set-by-statistic' : directly use the statistic output bias in the target dataset.
        stat_file_path : Optional[DPPath]
            The path to the stat file.
        """
        if bias_adjust_mode == "change-by-statistic":
            delta_bias, out_std = compute_output_stats(
                sample_merged,
                self.get_ntypes(),
                keys=list(self.atomic_output_def().keys()),
                stat_file_path=stat_file_path,
                model_forward=self._get_forward_wrapper_func(),
                rcond=self.rcond,
                preset_bias=self.preset_out_bias,
                atomic_output=self.atomic_output_def(),
            )
            self._store_out_stat(delta_bias, out_std, add=True)
        elif bias_adjust_mode == "set-by-statistic":
            bias_out, std_out = compute_output_stats(
                sample_merged,
                self.get_ntypes(),
                keys=list(self.atomic_output_def().keys()),
                stat_file_path=stat_file_path,
                rcond=self.rcond,
                preset_bias=self.preset_out_bias,
                atomic_output=self.atomic_output_def(),
            )
            self._store_out_stat(bias_out, std_out)
        else:
            raise RuntimeError("Unknown bias_adjust_mode mode: " + bias_adjust_mode)

    def _get_forward_wrapper_func(self) -> Callable[..., paddle.Tensor]:
        """Get a forward wrapper of the atomic model for output bias calculation."""

        def model_forward(coord, atype, box, fparam=None, aparam=None):
            with (
                paddle.no_grad()
            ):  # it's essential for pure paddle forward function to use auto_batchsize
                (
                    extended_coord,
                    extended_atype,
                    mapping,
                    nlist,
                ) = extend_input_and_build_neighbor_list(
                    coord,
                    atype,
                    self.get_rcut(),
                    self.get_sel(),
                    mixed_types=self.mixed_types(),
                    box=box,
                )
                atomic_ret = self.forward_common_atomic(
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping=mapping,
                    fparam=fparam,
                    aparam=aparam,
                )
                return {kk: vv.detach() for kk, vv in atomic_ret.items()}

        return model_forward

    def _default_bias(self):
        ntypes = self.get_ntypes()
        return paddle.zeros([self.n_out, ntypes, self.max_out_size], dtype=dtype).to(
            device=device
        )

    def _default_std(self):
        ntypes = self.get_ntypes()
        return paddle.ones([self.n_out, ntypes, self.max_out_size], dtype=dtype).to(
            device=device
        )

    def _varsize(
        self,
        shape: list[int],
    ) -> int:
        output_size = 1
        len_shape = len(shape)
        for i in range(len_shape):
            output_size *= shape[i]
        return output_size

    def _get_bias_index(
        self,
        kk: str,
    ) -> int:
        res: list[int] = []
        for i, e in enumerate(self.bias_keys):
            if e == kk:
                res.append(i)
        assert len(res) == 1
        return res[0]

    def _store_out_stat(
        self,
        out_bias: dict[str, paddle.Tensor],
        out_std: dict[str, paddle.Tensor],
        add: bool = False,
    ):
        ntypes = self.get_ntypes()
        out_bias_data = paddle.clone(self.out_bias)
        out_std_data = paddle.clone(self.out_std)
        for kk in out_bias.keys():
            assert kk in out_std.keys()
            idx = self._get_bias_index(kk)
            size = self._varsize(self.atomic_output_def()[kk].shape)
            if not add:
                out_bias_data[idx, :, :size] = out_bias[kk].reshape([ntypes, size])
            else:
                out_bias_data[idx, :, :size] += out_bias[kk].reshape([ntypes, size])
            out_std_data[idx, :, :size] = out_std[kk].reshape([ntypes, size])
        paddle.assign(out_bias_data, self.out_bias)
        paddle.assign(out_std_data, self.out_std)

    def _fetch_out_stat(
        self,
        keys: list[str],
    ) -> tuple[dict[str, paddle.Tensor], dict[str, paddle.Tensor]]:
        ret_bias = {}
        ret_std = {}
        ntypes = self.get_ntypes()
        for kk in keys:
            idx = self._get_bias_index(kk)
            isize = self._varsize(self.atomic_output_def()[kk].shape)
            ret_bias[kk] = self.out_bias[idx, :, :isize].reshape(
                [ntypes] + list(self.atomic_output_def()[kk].shape)  # noqa: RUF005
            )
            ret_std[kk] = self.out_std[idx, :, :isize].reshape(
                [ntypes] + list(self.atomic_output_def()[kk].shape)  # noqa: RUF005
            )
        return ret_bias, ret_std
