# SPDX-License-Identifier: LGPL-3.0-or-later
import functools
import math
from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.array_api import (
    Array,
    xp_take_first_n,
)
from deepmd.dpmodel.common import (
    NativeOP,
    to_numpy_array,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.dpmodel.utils import (
    AtomExcludeMask,
    PairExcludeMask,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
    map_atom_exclude_types,
    map_pair_exclude_types,
)
from deepmd.utils.path import (
    DPPath,
)

from .make_base_atomic_model import (
    make_base_atomic_model,
)

BaseAtomicModel_ = make_base_atomic_model(np.ndarray)


class BaseAtomicModel(BaseAtomicModel_, NativeOP):
    def __init__(
        self,
        type_map: list[str],
        atom_exclude_types: list[int] = [],
        pair_exclude_types: list[tuple[int, int]] = [],
        rcond: float | None = None,
        preset_out_bias: dict[str, Array] | None = None,
        data_stat_protect: float = 1e-2,
    ) -> None:
        super().__init__()
        self.type_map = type_map
        self.reinit_atom_exclude(atom_exclude_types)
        self.reinit_pair_exclude(pair_exclude_types)
        self.rcond = rcond
        self.preset_out_bias = preset_out_bias
        self.data_stat_protect = data_stat_protect
        self._observed_type: list[str] | None = None

    @property
    def observed_type(self) -> list[str] | None:
        """Get the observed element type list from data statistics."""
        return self._observed_type

    def _collect_and_set_observed_type(
        self,
        sampled_func: Callable[[], list[dict]],
        stat_file_path: DPPath | None,
        preset_observed_type: list[str] | None,
    ) -> None:
        """Collect observed types with priority: preset > stat_file > compute.

        Parameters
        ----------
        sampled_func
            The lazy sampled function to get data frames.
        stat_file_path
            The path to the statistics files (should already include type_map suffix).
        preset_observed_type
            User-specified observed types that take highest priority.
        """
        from deepmd.dpmodel.utils.stat import (
            _restore_observed_type_from_file,
            _save_observed_type_to_file,
            collect_observed_types,
        )

        if preset_observed_type is not None:
            self._observed_type = preset_observed_type
        else:
            observed = _restore_observed_type_from_file(stat_file_path)
            if observed is None:
                sampled = sampled_func()
                observed = collect_observed_types(sampled, self.type_map)
                _save_observed_type_to_file(stat_file_path, observed)
            self._observed_type = observed

    def init_out_stat(self) -> None:
        """Initialize the output bias."""
        ntypes = self.get_ntypes()
        self.bias_keys: list[str] = list(self.fitting_output_def().keys())
        self.max_out_size = max(
            [self.atomic_output_def()[kk].size for kk in self.bias_keys]
        )
        self.n_out = len(self.bias_keys)
        out_bias_data = np.zeros(
            [self.n_out, ntypes, self.max_out_size], dtype=GLOBAL_NP_FLOAT_PRECISION
        )
        out_std_data = np.ones(
            [self.n_out, ntypes, self.max_out_size], dtype=GLOBAL_NP_FLOAT_PRECISION
        )
        self.out_bias = out_bias_data
        self.out_std = out_std_data

    def get_out_bias(self) -> Array:
        """Get the output bias."""
        return self.out_bias

    def set_out_bias(self, out_bias: Array) -> None:
        """Set the output bias."""
        self.out_bias = out_bias

    def __setitem__(self, key: str, value: Array) -> None:
        if key in ["out_bias"]:
            self.out_bias = value
        elif key in ["out_std"]:
            self.out_std = value
        else:
            raise KeyError(key)

    def __getitem__(self, key: str) -> Array:
        if key in ["out_bias"]:
            return self.out_bias
        elif key in ["out_std"]:
            return self.out_std
        else:
            raise KeyError(key)

    def get_type_map(self) -> list[str]:
        """Get the type map."""
        return self.type_map

    def has_default_fparam(self) -> bool:
        """Check if the model has default frame parameters."""
        return False

    def get_default_fparam(self) -> list[float] | None:
        """Get the default frame parameters."""
        return None

    def reinit_atom_exclude(
        self,
        exclude_types: list[int] = [],
    ) -> None:
        self.atom_exclude_types = exclude_types
        if exclude_types == []:
            self.atom_excl = None
        else:
            self.atom_excl = AtomExcludeMask(self.get_ntypes(), self.atom_exclude_types)

    def reinit_pair_exclude(
        self,
        exclude_types: list[tuple[int, int]] = [],
    ) -> None:
        self.pair_exclude_types = exclude_types
        if exclude_types == []:
            self.pair_excl = None
        else:
            self.pair_excl = PairExcludeMask(self.get_ntypes(), self.pair_exclude_types)

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

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any | None = None
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
        self.out_bias = self.out_bias[:, remap_index, :]
        self.out_std = self.out_std[:, remap_index, :]

    def forward_common_atomic(
        self,
        extended_coord: Array,
        extended_atype: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
    ) -> dict[str, Array]:
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

        Returns
        -------
        ret_dict
            dict of output atomic properties.
            should implement the definition of `fitting_output_def`.
            ret_dict["mask"] of shape nf x nloc will be provided.
            ret_dict["mask"][ff,ii] == 1 indicating the ii-th atom of the ff-th frame is real.
            ret_dict["mask"][ff,ii] == 0 indicating the ii-th atom of the ff-th frame is virtual.

        """
        xp = array_api_compat.array_namespace(extended_coord, extended_atype, nlist)
        _, nloc, _ = nlist.shape
        atype = xp_take_first_n(extended_atype, 1, nloc)
        if self.pair_excl is not None:
            pair_mask = self.pair_excl.build_type_exclude_mask(nlist, extended_atype)
            # exclude neighbors in the nlist
            nlist = xp.where(pair_mask == 1, nlist, -1)

        ext_atom_mask = self.make_atom_mask(extended_atype)
        ret_dict = self.forward_atomic(
            extended_coord,
            xp.where(ext_atom_mask, extended_atype, 0),
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
        )
        ret_dict = self.apply_out_stat(ret_dict, atype)

        # nf x nloc
        atom_mask = xp_take_first_n(ext_atom_mask, 1, nloc)
        if self.atom_excl is not None:
            atom_mask = xp.logical_and(
                atom_mask, self.atom_excl.build_type_exclude_mask(atype)
            )

        for kk in ret_dict.keys():
            out_shape = ret_dict[kk].shape
            out_shape2 = math.prod(out_shape[2:])
            tmp_arr = ret_dict[kk].reshape([out_shape[0], out_shape[1], out_shape2])
            tmp_arr = xp.where(atom_mask[:, :, None], tmp_arr, xp.zeros_like(tmp_arr))
            ret_dict[kk] = xp.reshape(tmp_arr, out_shape)
        ret_dict["mask"] = xp.astype(atom_mask, xp.int32)

        return ret_dict

    def call(
        self,
        extended_coord: Array,
        extended_atype: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
    ) -> dict[str, Array]:
        return self.forward_common_atomic(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
        )

    def get_intensive(self) -> bool:
        """Whether the fitting property is intensive."""
        return False

    def get_compute_stats_distinguish_types(self) -> bool:
        """Get whether the fitting net computes stats which are not distinguished between different types of atoms."""
        return True

    def compute_or_load_stat(
        self,
        sampled_func: Callable[[], list[dict]],
        stat_file_path: DPPath | None = None,
        compute_or_load_out_stat: bool = True,
        preset_observed_type: list[str] | None = None,
    ) -> None:
        """Compute or load the statistics parameters of the model,
        such as mean and standard deviation of descriptors or the energy bias of the fitting net.

        Parameters
        ----------
        sampled_func
            The lazy sampled function to get data frames from different data systems.
        stat_file_path
            The path to the stat file.
        compute_or_load_out_stat : bool
            Whether to compute the output statistics.
            If False, it will only compute the input statistics
            (e.g. mean and standard deviation of descriptors).
        """
        raise NotImplementedError

    def compute_or_load_out_stat(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        stat_file_path: DPPath | None = None,
    ) -> None:
        """
        Compute the output statistics (e.g. energy bias) for the fitting net from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `np.ndarray`
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

    def _make_wrapped_sampler(
        self,
        sampled_func: Callable[[], list[dict]],
    ) -> Callable[[], list[dict]]:
        """Wrap the sampled function with exclusion types and default fparam.

        The returned callable is cached so that the sampling (which may be
        expensive) is performed at most once.

        Parameters
        ----------
        sampled_func
            The lazy sampled function to get data frames from different data
            systems.

        Returns
        -------
        Callable[[], list[dict]]
            A cached wrapper around *sampled_func* that additionally sets
            ``pair_exclude_types``, ``atom_exclude_types`` and default
            ``fparam`` on every sample dict when applicable.
        """

        @functools.lru_cache
        def wrapped_sampler() -> list[dict]:
            sampled = sampled_func()
            if self.pair_excl is not None:
                pair_exclude_types = self.pair_excl.get_exclude_types()
                for sample in sampled:
                    sample["pair_exclude_types"] = list(pair_exclude_types)
            if self.atom_excl is not None:
                atom_exclude_types = self.atom_excl.get_exclude_types()
                for sample in sampled:
                    sample["atom_exclude_types"] = list(atom_exclude_types)
            # For systems where fparam is missing (find_fparam == 0),
            # fill with default fparam if available and mark as found.
            if self.has_default_fparam():
                default_fparam = self.get_default_fparam()
                if default_fparam is not None:
                    default_fparam_np = np.array(default_fparam)
                    for sample in sampled:
                        if "find_fparam" in sample and not sample["find_fparam"]:
                            nframe = sample["atype"].shape[0]
                            sample["fparam"] = np.tile(
                                default_fparam_np.reshape(1, -1), (nframe, 1)
                            )
                            sample["find_fparam"] = np.bool_(True)
            return sampled

        return wrapped_sampler

    def change_out_bias(
        self,
        sample_merged: Callable[[], list[dict]] | list[dict],
        stat_file_path: DPPath | None = None,
        bias_adjust_mode: str = "change-by-statistic",
    ) -> None:
        """Change the output bias according to the input data and the pretrained model.

        Parameters
        ----------
        sample_merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `np.ndarray`
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
        from deepmd.dpmodel.utils.stat import (
            compute_output_stats,
        )

        if bias_adjust_mode == "change-by-statistic":
            delta_bias, out_std = compute_output_stats(
                sample_merged,
                self.get_ntypes(),
                keys=list(self.atomic_output_def().keys()),
                stat_file_path=stat_file_path,
                model_forward=self._get_forward_wrapper_func(),
                rcond=self.rcond,
                preset_bias=self.preset_out_bias,
                stats_distinguish_types=self.get_compute_stats_distinguish_types(),
                intensive=self.get_intensive(),
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
                stats_distinguish_types=self.get_compute_stats_distinguish_types(),
                intensive=self.get_intensive(),
            )
            self._store_out_stat(bias_out, std_out)
        else:
            raise RuntimeError("Unknown bias_adjust_mode mode: " + bias_adjust_mode)

    def _store_out_stat(
        self,
        out_bias: dict[str, np.ndarray],
        out_std: dict[str, np.ndarray],
        add: bool = False,
    ) -> None:
        """Store output bias and std into the model."""
        ntypes = self.get_ntypes()
        out_bias_data = np.array(to_numpy_array(self.out_bias))
        out_std_data = np.array(to_numpy_array(self.out_std))
        for kk in out_bias.keys():
            assert kk in out_std.keys()
            idx = self._get_bias_index(kk)
            size = self._varsize(self.atomic_output_def()[kk].shape)
            if not add:
                out_bias_data[idx, :, :size] = out_bias[kk].reshape(ntypes, size)
            else:
                out_bias_data[idx, :, :size] += out_bias[kk].reshape(ntypes, size)
            out_std_data[idx, :, :size] = out_std[kk].reshape(ntypes, size)
        self.out_bias = out_bias_data
        self.out_std = out_std_data

    def _get_forward_wrapper_func(self) -> Callable[..., dict[str, np.ndarray]]:
        """Get a forward wrapper of the atomic model for output bias calculation."""
        import array_api_compat

        from deepmd.dpmodel.utils.nlist import (
            extend_input_and_build_neighbor_list,
        )

        def model_forward(
            coord: np.ndarray,
            atype: np.ndarray,
            box: np.ndarray | None,
            fparam: np.ndarray | None = None,
            aparam: np.ndarray | None = None,
        ) -> dict[str, np.ndarray]:
            # Get reference array to determine the target array type and device
            # Use out_bias as reference since it's always present
            ref_array = self.out_bias
            xp = array_api_compat.array_namespace(ref_array)

            # Convert numpy inputs to the model's array type with correct device
            device = array_api_compat.device(ref_array)
            coord = xp.asarray(coord, device=device)
            atype = xp.asarray(atype, device=device)
            if box is not None:
                if np.allclose(box, 0.0):
                    box = None
                else:
                    box = xp.asarray(box, device=device)
            if fparam is not None:
                fparam = xp.asarray(fparam, device=device)
            if aparam is not None:
                aparam = xp.asarray(aparam, device=device)

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
            # Convert outputs back to numpy arrays
            return {kk: to_numpy_array(vv) for kk, vv in atomic_ret.items()}

        return model_forward

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
        # do not deep copy Descriptor and Fitting class
        data = data.copy()
        variables = data.pop("@variables")
        obj = cls(**data)
        for kk in variables.keys():
            obj[kk] = variables[kk]
        return obj

    def apply_out_stat(
        self,
        ret: dict[str, Array],
        atype: Array,
    ) -> dict[str, Array]:
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

    def _fetch_out_stat(
        self,
        keys: list[str],
    ) -> tuple[dict[str, Array], dict[str, Array]]:
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
