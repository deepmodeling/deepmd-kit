# SPDX-License-Identifier: LGPL-3.0-or-later
import collections
import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import socket
import threading
import time
import warnings
from functools import (
    cached_property,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
    Self,
)

import numpy as np

import deepmd.utils.random as dp_random
from deepmd.common import (
    expand_sys_str,
    make_default_mesh,
    rglob_sys_str,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.utils.data import (
    DataRequirementItem,
    DeepmdData,
)
from deepmd.utils.out_stat import (
    compute_stats_from_redu,
)

log = logging.getLogger(__name__)

_DPDATA_CACHE_DIR = ".deepmd_dpdata_cache"
_DPDATA_DEFAULT_OUT_FORMAT = "deepmd/lmdb"
_DPDATA_CONVERSION_SCHEMA_VERSION = "2"
_DPDATA_CONVERSION_CACHE: dict[tuple[str, str, str, str, str], list[str]] = {}
_DPDATA_SOURCE_MTIME_CACHE: dict[tuple[str, str], tuple[float, float]] = {}
# Neighbor-stat, trainer construction, and multi-task routing commonly query
# the same directory within one startup. Reuse that O(file-count) scan while
# still letting a long-lived process notice later source rewrites.
_DPDATA_SOURCE_MTIME_CACHE_TTL = 60.0
_CONVERSION_LOCK_HEARTBEAT_SECONDS = 5.0
_CONVERSION_LOCK_STALE_SECONDS = 30.0


def validate_lmdb_systems(
    systems: list[str],
    *,
    backend_name: str,
    supported: bool = True,
) -> str | None:
    """Validate expanded systems and return the sole resolved LMDB path.

    LMDB stores multiple logical systems inside one database, so mixing an
    LMDB path with other expanded paths is ambiguous and unsupported.
    """
    # Import after data_system has initialized. Importing the dpmodel package
    # at module load time re-enters this module through descriptor utilities.
    from deepmd.dpmodel.utils.lmdb_data import (
        is_lmdb,
    )

    lmdb_paths = [path for path in systems if is_lmdb(path)]
    if not lmdb_paths:
        return None
    if not supported:
        raise NotImplementedError(
            f"{backend_name} backend does not support LMDB data yet. "
            "Choose out_format='deepmd/hdf5' for automatic conversion."
        )
    if len(systems) != 1:
        raise ValueError(
            f"{backend_name} backend requires an LMDB dataset to resolve to "
            "exactly one path; LMDB paths cannot be mixed with other systems."
        )
    return lmdb_paths[0]


class DeepmdDataSystem:
    """Class for manipulating many data systems.

    It is implemented with the help of DeepmdData
    """

    def __init__(
        self,
        systems: list[str],
        batch_size: int,
        test_size: int,
        rcut: float | None = None,
        set_prefix: str = "set",
        shuffle_test: bool = True,
        type_map: list[str] | None = None,
        optional_type_map: bool = True,
        modifier: Any | None = None,
        trn_all_set: bool = False,
        sys_probs: list[float] | None = None,
        auto_prob_style: str = "prob_sys_size",
        sort_atoms: bool = True,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        systems
            Specifying the paths to systems
        batch_size
            The batch size
        test_size
            The size of test data
        rcut
            The cut-off radius. Not used.
        set_prefix
            Prefix for the directories of different sets
        shuffle_test
            If the test data are shuffled
        type_map
            Gives the name of different atom types
        optional_type_map
            If the type_map.raw in each system is optional
        modifier
            Data modifier that has the method `modify_data`
        trn_all_set
            Use all sets as training dataset. Otherwise, if the number of sets is more than 1, the last set is left for test.
        sys_probs : list of float
            The probabilitis of systems to get the batch.
            Summation of positive elements of this list should be no greater than 1.
            Element of this list can be negative, the probability of the corresponding system is determined
                automatically by the number of batches in the system.
        auto_prob_style : str
            Determine the probability of systems automatically. The method is assigned by this key and can be
            - "prob_uniform"  : the probability all the systems are equal, namely 1.0/self.get_nsystems()
            - "prob_sys_size" : the probability of a system is proportional to the number of batches in the system
            - "prob_sys_size;stt_idx:end_idx:weight;stt_idx:end_idx:weight;..." :
                                the list of systems is divided into blocks. A block is specified by `stt_idx:end_idx:weight`,
                                where `stt_idx` is the starting index of the system, `end_idx` is then ending (not including) index of the system,
                                the probabilities of the systems in this block sums up to `weight`, and the relatively probabilities within this block is proportional
                to the number of batches in the system.
        sort_atoms : bool
            Sort atoms by atom types. Required to enable when the data is directly fed to
            descriptors except mixed types.
        """
        # init data
        del rcut
        self.system_dirs = systems
        self.nsystems = len(self.system_dirs)
        if self.nsystems <= 0:
            raise ValueError("No systems provided")
        self.data_systems = []
        for ii in self.system_dirs:
            self.data_systems.append(
                DeepmdData(
                    ii,
                    set_prefix=set_prefix,
                    shuffle_test=shuffle_test,
                    type_map=type_map,
                    optional_type_map=optional_type_map,
                    modifier=modifier,
                    trn_all_set=trn_all_set,
                    sort_atoms=sort_atoms,
                )
            )
        # check mix_type format
        error_format_msg = (
            "if one of the system is of mixed_type format, "
            "then all of the systems should be of mixed_type format!"
        )
        if self.data_systems[0].mixed_type:
            for data_sys in self.data_systems[1:]:
                assert data_sys.mixed_type, error_format_msg
            self.mixed_type = True
        else:
            for data_sys in self.data_systems[1:]:
                assert not data_sys.mixed_type, error_format_msg
            self.mixed_type = False
        # batch size
        self.batch_size = batch_size
        is_auto_bs = False
        self.mixed_systems = False
        if isinstance(self.batch_size, int):
            self.batch_size = self.batch_size * np.ones(self.nsystems, dtype=int)
        elif isinstance(self.batch_size, str):
            words = self.batch_size.split(":")
            if "auto" == words[0]:
                is_auto_bs = True
                rule = 32
                if len(words) == 2:
                    rule = int(words[1])
                self.batch_size = self._make_auto_bs(rule)
            elif "mixed" == words[0]:
                self.mixed_type = True
                self.mixed_systems = True
                if len(words) == 2:
                    rule = int(words[1])
                else:
                    raise RuntimeError("batch size must be specified for mixed systems")
                self.batch_size = rule * np.ones(self.nsystems, dtype=int)
            elif "max" == words[0]:
                # Determine batch size so that batch_size * natoms <= rule, at least 1
                if len(words) != 2:
                    raise RuntimeError("batch size must be specified for max systems")
                rule = int(words[1])
                bs = []
                for ii in self.data_systems:
                    ni = ii.get_natoms()
                    bsi = rule // ni
                    if bsi == 0:
                        bsi = 1
                    bs.append(bsi)
                self.batch_size = bs
            elif "filter" == words[0]:
                # Remove systems with natoms > rule, then set batch size like "max:rule"
                if len(words) != 2:
                    raise RuntimeError(
                        "batch size must be specified for filter systems"
                    )
                rule = int(words[1])
                filtered_data_systems = []
                filtered_system_dirs = []
                for sys_dir, data_sys in zip(
                    self.system_dirs, self.data_systems, strict=True
                ):
                    if data_sys.get_natoms() <= rule:
                        filtered_data_systems.append(data_sys)
                        filtered_system_dirs.append(sys_dir)
                if len(filtered_data_systems) == 0:
                    raise RuntimeError(
                        f"No system left after removing systems with more than {rule} atoms"
                    )
                if len(filtered_data_systems) != len(self.data_systems):
                    warnings.warn(
                        f"Remove {len(self.data_systems) - len(filtered_data_systems)} systems with more than {rule} atoms"
                    )
                self.data_systems = filtered_data_systems
                self.system_dirs = filtered_system_dirs
                self.nsystems = len(self.data_systems)
                bs = []
                for ii in self.data_systems:
                    ni = ii.get_natoms()
                    bsi = rule // ni
                    if bsi == 0:
                        bsi = 1
                    bs.append(bsi)
                self.batch_size = bs
            elif words[0] == "mix":
                raise RuntimeError(
                    "the 'mix' batch_size rule packs frames of unequal atom "
                    "count into one batch and is only available for LMDB "
                    "datasets on the pt and pt_expt backends"
                )
            else:
                raise RuntimeError("unknown batch_size rule " + words[0])
        elif isinstance(self.batch_size, list):
            pass
        else:
            raise RuntimeError("invalid batch_size")
        assert isinstance(self.batch_size, (list, np.ndarray))
        assert len(self.batch_size) == self.nsystems

        # natoms, nbatches
        ntypes = []
        for ii in self.data_systems:
            ntypes.append(ii.get_ntypes())
        self.sys_ntypes = max(ntypes)
        self.natoms = []
        self.natoms_vec = []
        self.nbatches = []
        type_map_list = []
        for ii in range(self.nsystems):
            self.natoms.append(self.data_systems[ii].get_natoms())
            self.natoms_vec.append(
                self.data_systems[ii].get_natoms_vec(self.sys_ntypes).astype(int)
            )
            self.nbatches.append(
                self.data_systems[ii].get_sys_numb_batch(self.batch_size[ii])
            )
            type_map_list.append(self.data_systems[ii].get_type_map())
        self.type_map = self._check_type_map_consistency(type_map_list)

        # ! altered by Marián Rynik
        # test size
        # now test size can be set as a percentage of systems data or test size
        # can be set for each system individually in the same manner as batch
        # size. This enables one to use systems with diverse number of
        # structures and different number of atoms.
        self.test_size = test_size
        if isinstance(self.test_size, int):
            self.test_size = self.test_size * np.ones(self.nsystems, dtype=int)
        elif isinstance(self.test_size, str):
            words = self.test_size.split("%")
            try:
                percent = int(words[0])
            except ValueError as e:
                raise RuntimeError("unknown test_size rule " + words[0]) from e
            self.test_size = self._make_auto_ts(percent)
        elif isinstance(self.test_size, list):
            pass
        else:
            raise RuntimeError("invalid test_size")
        assert isinstance(self.test_size, (list, np.ndarray))
        assert len(self.test_size) == self.nsystems

        # init pick idx
        self.pick_idx = 0

        # derive system probabilities
        self.sys_probs = None
        self.set_sys_probs(sys_probs, auto_prob_style)

        # check batch and test size
        for ii in range(self.nsystems):
            chk_ret = self.data_systems[ii].check_batch_size(self.batch_size[ii])
            if chk_ret is not None and not is_auto_bs and not self.mixed_systems:
                warnings.warn(
                    f"system {self.system_dirs[ii]} required batch size is larger than the size of the dataset {chk_ret[0]} ({self.batch_size[ii]} > {chk_ret[1]})"
                )
            chk_ret = self.data_systems[ii].check_test_size(self.test_size[ii])
            if chk_ret is not None and not is_auto_bs and not self.mixed_systems:
                warnings.warn(
                    f"system {self.system_dirs[ii]} required test size is larger than the size of the dataset {chk_ret[0]} ({self.test_size[ii]} > {chk_ret[1]})"
                )

    def _load_test(self, ntests: int = -1) -> None:
        self.test_data = collections.defaultdict(list)
        for ii in range(self.nsystems):
            test_system_data = self.data_systems[ii].get_test(ntests=ntests)
            for nn in test_system_data:
                self.test_data[nn].append(test_system_data[nn])

    @cached_property
    def default_mesh(self) -> list[np.ndarray]:
        """Mesh for each system."""
        return [
            make_default_mesh(
                self.data_systems[ii].pbc, self.data_systems[ii].mixed_type
            )
            for ii in range(self.nsystems)
        ]

    def compute_energy_shift(
        self, rcond: float | None = None, key: str = "energy"
    ) -> tuple[np.ndarray, np.ndarray]:
        sys_ener = []
        for ss in self.data_systems:
            sys_ener.append(ss.avg(key))
        sys_ener = np.concatenate(sys_ener)
        sys_tynatom = np.array(self.natoms_vec, dtype=GLOBAL_NP_FLOAT_PRECISION)
        sys_tynatom = np.reshape(sys_tynatom, [self.nsystems, -1])
        sys_tynatom = sys_tynatom[:, 2:]
        energy_shift, _ = compute_stats_from_redu(
            sys_ener.reshape(-1, 1),
            sys_tynatom,
            rcond=rcond,
        )
        return energy_shift.ravel()

    def add_dict(self, adict: dict[str, dict[str, Any]]) -> None:
        """Add items to the data system by a `dict`.
        `adict` should have items like
        .. code-block:: python.

           adict[key] = {
               "ndof": ndof,
               "atomic": atomic,
               "must": must,
               "high_prec": high_prec,
               "type_sel": type_sel,
               "repeat": repeat,
           }

        For the explanation of the keys see `add`
        """
        for kk in adict:
            self.add(
                kk,
                adict[kk]["ndof"],
                atomic=adict[kk]["atomic"],
                must=adict[kk]["must"],
                high_prec=adict[kk]["high_prec"],
                type_sel=adict[kk]["type_sel"],
                repeat=adict[kk]["repeat"],
                default=adict[kk]["default"],
                dtype=adict[kk].get("dtype"),
                output_natoms_for_type_sel=adict[kk].get(
                    "output_natoms_for_type_sel", False
                ),
                special_shape=adict[kk].get("special_shape"),
            )

    def add_data_requirements(
        self, data_requirements: list[DataRequirementItem]
    ) -> None:
        """Add items to the data system by a list of `DataRequirementItem`."""
        self.add_dict({rr.key: rr.dict for rr in data_requirements})

    def add(
        self,
        key: str,
        ndof: int,
        atomic: bool = False,
        must: bool = False,
        high_prec: bool = False,
        type_sel: list[int] | None = None,
        repeat: int = 1,
        default: float = 0.0,
        dtype: np.dtype | None = None,
        output_natoms_for_type_sel: bool = False,
        special_shape: str | None = None,
    ) -> None:
        """Add a data item that to be loaded.

        Parameters
        ----------
        key
            The key of the item. The corresponding data is stored in `sys_path/set.*/key.npy`
        ndof
            The number of dof
        atomic
            The item is an atomic property.
            If False, the size of the data should be nframes x ndof
            If True, the size of data should be nframes x natoms x ndof
        must
            The data file `sys_path/set.*/key.npy` must exist.
            If must is False and the data file does not exist, the `data_dict[find_key]` is set to 0.0
        high_prec
            Load the data and store in float64, otherwise in float32
        type_sel
            Select certain type of atoms
        repeat
            The data will be repeated `repeat` times.
        default, default=0.
            Default value of data
        dtype
            The dtype of data, overwrites `high_prec` if provided
        output_natoms_for_type_sel : bool
            If True and type_sel is True, the atomic dimension will be natoms instead of nsel
        special_shape : str, optional
            Name of a loader-defined non-standard shape contract.
        """
        for ii in self.data_systems:
            ii.add(
                key,
                ndof,
                atomic=atomic,
                must=must,
                high_prec=high_prec,
                repeat=repeat,
                type_sel=type_sel,
                default=default,
                dtype=dtype,
                output_natoms_for_type_sel=output_natoms_for_type_sel,
                special_shape=special_shape,
            )

    def reduce(self, key_out: str, key_in: str) -> None:
        """Generate a new item from the reduction of another atom.

        Parameters
        ----------
        key_out
            The name of the reduced item
        key_in
            The name of the data item to be reduced
        """
        for ii in self.data_systems:
            ii.reduce(key_out, key_in)

    def get_data_dict(self, ii: int = 0) -> dict:
        return self.data_systems[ii].get_data_dict()

    def set_sys_probs(
        self,
        sys_probs: list[float] | None = None,
        auto_prob_style: str = "prob_sys_size",
    ) -> None:
        if sys_probs is None:
            if auto_prob_style == "prob_uniform":
                prob_v = 1.0 / float(self.nsystems)
                probs = [prob_v for ii in range(self.nsystems)]
            elif auto_prob_style[:13] == "prob_sys_size":
                if auto_prob_style == "prob_sys_size":
                    prob_style = f"prob_sys_size;0:{self.get_nsystems()}:1.0"
                else:
                    prob_style = auto_prob_style
                probs = prob_sys_size_ext(
                    prob_style, self.get_nsystems(), self.nbatches
                )
            else:
                raise RuntimeError("Unknown auto prob style: " + auto_prob_style)
        else:
            probs = process_sys_probs(sys_probs, self.nbatches)
        self.sys_probs = probs

    def get_batch(self, sys_idx: int | None = None) -> dict:
        # batch generation style altered by Ziyao Li:
        # one should specify the "sys_prob" and "auto_prob_style" params
        # via set_sys_prob() function. The sys_probs this function uses is
        # defined as a private variable, self.sys_probs, initialized in __init__().
        # This is to optimize the (vain) efforts in evaluating sys_probs every batch.
        """Get a batch of data from the data systems.

        Parameters
        ----------
        sys_idx : int
            The index of system from which the batch is get.
            If sys_idx is not None, `sys_probs` and `auto_prob_style` are ignored
            If sys_idx is None, automatically determine the system according to `sys_probs` or `auto_prob_style`, see the following.
            This option does not work for mixed systems.

        Returns
        -------
        dict
            The batch data
        """
        if not self.mixed_systems:
            b_data = self.get_batch_standard(sys_idx)
        else:
            b_data = self.get_batch_mixed()
        return b_data

    def get_batch_standard(self, sys_idx: int | None = None) -> dict:
        """Get a batch of data from the data systems in the standard way.

        Parameters
        ----------
        sys_idx : int
            The index of system from which the batch is get.
            If sys_idx is not None, `sys_probs` and `auto_prob_style` are ignored
            If sys_idx is None, automatically determine the system according to `sys_probs` or `auto_prob_style`, see the following.

        Returns
        -------
        dict
            The batch data
        """
        if sys_idx is not None:
            self.pick_idx = sys_idx
        else:
            # prob = self._get_sys_probs(sys_probs, auto_prob_style)
            self.pick_idx = dp_random.choice(
                np.arange(self.nsystems, dtype=np.int32), p=self.sys_probs
            )
        b_data = self.data_systems[self.pick_idx].get_batch(
            self.batch_size[self.pick_idx]
        )
        b_data["natoms_vec"] = self.natoms_vec[self.pick_idx]
        b_data["default_mesh"] = self.default_mesh[self.pick_idx]
        return b_data

    def get_batch_mixed(self) -> dict:
        """Get a batch of data from the data systems in the mixed way.

        Returns
        -------
        dict
            The batch data
        """
        # mixed systems have a global batch size
        batch_size = self.batch_size[0]
        batch_data = []
        for _ in range(batch_size):
            self.pick_idx = dp_random.choice(
                np.arange(self.nsystems, dtype=np.int32), p=self.sys_probs
            )
            bb_data = self.data_systems[self.pick_idx].get_batch(1)
            bb_data["natoms_vec"] = self.natoms_vec[self.pick_idx]
            bb_data["default_mesh"] = self.default_mesh[self.pick_idx]
            batch_data.append(bb_data)
        b_data = self._merge_batch_data(batch_data)
        return b_data

    def _merge_batch_data(self, batch_data: list[dict]) -> dict:
        """Merge batch data from different systems.

        Parameters
        ----------
        batch_data : list of dict
            A list of batch data from different systems.

        Returns
        -------
        dict
            The merged batch data.
        """
        b_data = {}
        max_natoms = max(bb["natoms_vec"][0] for bb in batch_data)
        # natoms_vec
        natoms_vec = np.zeros(2 + self.get_ntypes(), dtype=int)
        natoms_vec[0:3] = max_natoms
        b_data["natoms_vec"] = natoms_vec
        # real_natoms_vec
        real_natoms_vec = np.vstack([bb["natoms_vec"] for bb in batch_data])
        b_data["real_natoms_vec"] = real_natoms_vec
        # type
        type_vec = np.full((len(batch_data), max_natoms), -1, dtype=int)
        for ii, bb in enumerate(batch_data):
            type_vec[ii, : bb["type"].shape[1]] = bb["type"][0]
        b_data["type"] = type_vec
        # default_mesh
        default_mesh = np.mean([bb["default_mesh"] for bb in batch_data], axis=0)
        b_data["default_mesh"] = default_mesh
        # other data
        data_dict = self.get_data_dict(0)
        for kk, vv in data_dict.items():
            if kk not in batch_data[0]:
                continue
            b_data["find_" + kk] = batch_data[0]["find_" + kk]
            if vv.get("special_shape") == "hessian" or kk == "hessian":
                # A Hessian is a (3 * natoms, 3 * natoms) matrix, so neither
                # branch below pads it correctly: concatenating raises on
                # ragged systems and copying a flat prefix would scatter the
                # rows. Embed each frame's block in the top-left corner of the
                # padded square instead; the padded rows and columns stay zero
                # and are dropped by the loss mask.
                padded_dof = max_natoms * 3
                merged = np.zeros(
                    (len(batch_data), padded_dof, padded_dof),
                    dtype=batch_data[0][kk].dtype,
                )
                for ii, bb in enumerate(batch_data):
                    frame_dof = bb["natoms_vec"][0] * 3
                    merged[ii, :frame_dof, :frame_dof] = bb[kk][0].reshape(
                        frame_dof, frame_dof
                    )
                b_data[kk] = merged.reshape(len(batch_data), -1)
            elif not vv["atomic"]:
                b_data[kk] = np.concatenate([bb[kk] for bb in batch_data], axis=0)
            else:
                b_data[kk] = np.zeros(
                    (len(batch_data), max_natoms * vv["ndof"] * vv["repeat"]),
                    dtype=batch_data[0][kk].dtype,
                )
                for ii, bb in enumerate(batch_data):
                    b_data[kk][ii, : bb[kk].shape[1]] = bb[kk][0]
        return b_data

    # ! altered by Marián Rynik
    def get_test(
        self, sys_idx: int | None = None, n_test: int = -1
    ) -> dict[str, np.ndarray]:  # depreciated
        """Get test data from the the data systems.

        Parameters
        ----------
        sys_idx
            The test dat of system with index `sys_idx` will be returned.
            If is None, the currently selected system will be returned.
        n_test
            Number of test data. If set to -1 all test data will be get.
        """
        if not hasattr(self, "test_data"):
            self._load_test(ntests=n_test)
        if sys_idx is not None:
            idx = sys_idx
        else:
            idx = self.pick_idx

        test_system_data = {}
        for nn in self.test_data:
            test_system_data[nn] = self.test_data[nn][idx]
        test_system_data["natoms_vec"] = self.natoms_vec[idx]
        test_system_data["default_mesh"] = self.default_mesh[idx]
        return test_system_data

    def get_sys_ntest(self, sys_idx: int | None = None) -> int:
        """Get number of tests for the currently selected system,
        or one defined by sys_idx.
        """
        if sys_idx is not None:
            return self.test_size[sys_idx]
        else:
            return self.test_size[self.pick_idx]

    def get_type_map(self) -> list[str]:
        """Get the type map."""
        return self.type_map

    def get_nbatches(self) -> int:
        """Get the total number of batches."""
        return self.nbatches

    def get_ntypes(self) -> int:
        """Get the number of types."""
        return self.sys_ntypes

    def get_nsystems(self) -> int:
        """Get the number of data systems."""
        return self.nsystems

    def get_sys(self, idx: int) -> DeepmdData:
        """Get a certain data system."""
        return self.data_systems[idx]

    def get_batch_size(self) -> int:
        """Get the batch size."""
        return self.batch_size

    def print_summary(self, name: str) -> None:
        print_summary(
            name,
            self.nsystems,
            self.system_dirs,
            self.natoms,
            self.batch_size,
            self.nbatches,
            self.sys_probs,
            [ii.pbc for ii in self.data_systems],
        )

    def _make_auto_bs(self, rule: int) -> list[int]:
        bs = []
        for ii in self.data_systems:
            ni = ii.get_natoms()
            bsi = rule // ni
            if bsi * ni < rule:
                bsi += 1
            bs.append(bsi)
        return bs

    # ! added by Marián Rynik
    def _make_auto_ts(self, percent: float) -> list[int]:
        ts = []
        for ii in range(self.nsystems):
            ni = self.batch_size[ii] * self.nbatches[ii]
            tsi = int(ni * percent / 100)
            ts.append(tsi)

        return ts

    def _check_type_map_consistency(
        self, type_map_list: list[list[str] | None]
    ) -> list[str]:
        ret = []
        for ii in type_map_list:
            if ii is not None:
                min_len = min([len(ii), len(ret)])
                for idx in range(min_len):
                    if ii[idx] != ret[idx]:
                        raise RuntimeError(
                            f"Inconsistent type map: {ret!s} {ii!s} in different data systems. "
                            "If you didn't set model/type_map, please set it, "
                            "since the type map of the model cannot be decided by data."
                        )
                if len(ii) > len(ret):
                    ret = ii
        return ret


class LmdbDataSystem:
    """A DeepmdDataSystem-compatible adapter for LMDB datasets.

    The adapter returns raw DeePMD-style numpy batches (``type``,
    ``natoms_vec``, ``default_mesh``) so it can be consumed by the legacy
    TensorFlow/JAX training paths. Consumers that need the dpmodel canonical
    format can still call ``normalize_batch`` on its output.
    """

    def __init__(
        self,
        lmdb_path: str,
        type_map: list[str],
        batch_size: int | str | list[int | str] = "auto",
        auto_prob_style: str | None = None,
        seed: int | None = None,
    ) -> None:
        # Keep the framework-agnostic LMDB implementation lazy so importing a
        # legacy backend cannot create a data_system <-> dpmodel import cycle.
        from deepmd.dpmodel.utils.lmdb_data import (
            LmdbBatchSampler,
            LmdbDataReader,
            LmdbTestData,
            compute_block_targets,
        )

        if not type_map:
            raise ValueError(
                "LMDB datasets require a non-empty model/type_map because "
                "LMDB stores atom type indices and the training data adapter "
                "must map them to element names."
            )

        self.lmdb_path = lmdb_path
        self._type_map = list(type_map)
        self._closed = False
        self._data_dict = {
            "box": DataRequirementItem(
                "box",
                9,
                atomic=False,
                must=False,
                default=np.zeros(9, dtype=GLOBAL_NP_FLOAT_PRECISION),
            ).dict,
            "coord": {
                "ndof": 3,
                "atomic": True,
                "must": True,
                "high_prec": False,
                "type_sel": None,
                "repeat": 1,
                "default": 0.0,
                "dtype": None,
                "output_natoms_for_type_sel": False,
            },
            "numb_copy": {
                "ndof": 1,
                "atomic": False,
                "must": False,
                "high_prec": False,
                "type_sel": None,
                "repeat": 1,
                "default": 1,
                "dtype": int,
                "output_natoms_for_type_sel": False,
            },
        }

        self._reader = LmdbDataReader(lmdb_path, type_map, batch_size)
        # Box availability is part of stack compatibility. Register it before
        # any grouping so periodic and non-periodic frames never share the
        # scalar ``find_box`` flag of one legacy batch.
        box_requirement = DataRequirementItem(
            "box",
            9,
            atomic=False,
            must=False,
            default=np.zeros(9, dtype=GLOBAL_NP_FLOAT_PRECISION),
        )
        self._reader.add_data_requirement([box_requirement])
        self._test_data = LmdbTestData(
            lmdb_path,
            type_map=type_map,
            shuffle_test=False,
        )
        self._test_data.add_data_requirement([box_requirement])
        # LMDB is defined as mixed-type by its reader contract; determining
        # this must not scan every frame during data-system initialization.
        self.mixed_type = self._reader.mixed_type
        self.nsystems = 1
        self.natoms = [
            int(self._reader.frame_nlocs.max()) if len(self._reader.frame_nlocs) else 0
        ]
        self.batch_size = [self._reader.batch_size]
        self.sys_probs = [1.0]

        block_targets = None
        if auto_prob_style is not None and self._reader.frame_system_ids is not None:
            block_targets = compute_block_targets(
                auto_prob_style,
                self._reader.nsystems,
                self._reader.system_nframes,
            )
        self._sampler = LmdbBatchSampler(
            self._reader,
            shuffle=True,
            seed=seed,
            block_targets=block_targets,
        )
        self.nbatches = [self._sampler.total_batches]
        self._iter = iter(self._sampler)
        self._refresh_groups()

    def _refresh_groups(self) -> None:
        """Refresh bounded statistics chunks and full-validation views."""
        from deepmd.dpmodel.utils.lmdb_data import (
            LmdbTestDataNlocView,
            collect_lmdb_sampling_groups,
        )

        groups = collect_lmdb_sampling_groups(self._reader)
        self._stat_groups = groups
        self._stat_offsets = [0] * len(groups)

        # Neighbor statistics are a bounded sample, matching the dedicated
        # LMDB path. Chunks additionally cap decoded atoms so one large-nloc
        # group cannot create a large transient Python/NumPy allocation.
        selected = np.zeros(len(self._reader), dtype=bool)
        max_frames = min(len(self._reader), 2000)
        if max_frames:
            rng = np.random.RandomState(42)
            chosen = (
                rng.choice(len(self._reader), max_frames, replace=False)
                if max_frames < len(self._reader)
                else np.arange(len(self._reader), dtype=np.int64)
            )
            selected[np.asarray(chosen, dtype=np.int64)] = True

        self._nloc_set_indices: dict[str, np.ndarray] = {}
        data_systems = []
        system_dirs: list[str] = []
        any_pbc = False
        for group_idx, (nloc, indices) in enumerate(groups):
            group_label = f"{self.lmdb_path}#group={group_idx}:nloc={nloc}"
            original_indices = np.asarray(
                self._reader.original_keys(indices), dtype=np.int64
            )
            frame = self._reader.peek_frame(int(indices[0]))
            group_pbc = bool(float(frame.get("find_box", 0.0)) > 0.5)
            any_pbc = any_pbc or group_pbc

            stat_groups: dict[str, np.ndarray] = {}
            sampled_indices = np.asarray(indices)[selected[np.asarray(indices)]]
            chunk_size = max(1, min(128, 20000 // max(int(nloc), 1)))
            for chunk_idx, start in enumerate(
                range(0, len(sampled_indices), chunk_size)
            ):
                chunk = sampled_indices[start : start + chunk_size]
                set_name = f"{group_label}:chunk={chunk_idx}"
                self._nloc_set_indices[set_name] = np.asarray(chunk, dtype=np.int64)
                stat_groups[set_name] = np.asarray(
                    self._reader.original_keys(chunk), dtype=np.int64
                )

            data_systems.append(
                LmdbTestDataNlocView(
                    self._test_data,
                    int(nloc),
                    original_indices,
                    pbc=group_pbc,
                    stat_groups=stat_groups,
                )
            )
            system_dirs.append(group_label)

        # These views do not point back to this adapter, avoiding the
        # ``data_systems=[self]`` reference cycle while satisfying both the
        # neighbor-stat and JAX/TF2 full-validation contracts.
        self.data_systems = data_systems
        self.system_dirs = system_dirs
        self.dirs = list(self._nloc_set_indices)
        self.pbc = any_pbc

    def _detect_pbc(self) -> bool:
        """Return True when LMDB frames contain a non-zero simulation box."""
        if len(self._reader) == 0:
            return False
        frame = self._reader.peek_frame(0)
        return bool(float(frame.get("find_box", 0.0)) > 0.5)

    def add_data_requirements(
        self, data_requirements: list[DataRequirementItem]
    ) -> None:
        """Add label/auxiliary data requirements."""
        self._reader.add_data_requirement(data_requirements)
        self._test_data.add_data_requirement(data_requirements)
        for item in data_requirements:
            self._data_dict[item.key] = item.dict
        self._refresh_groups()
        self.nbatches = [self._sampler.total_batches]
        self._iter = iter(self._sampler)

    def add_data_requirement(self, data_requirement: list[DataRequirementItem]) -> None:
        """Alias used by DataLoader-style backends."""
        self.add_data_requirements(data_requirement)

    def add(
        self,
        key: str,
        ndof: int,
        atomic: bool = False,
        must: bool = False,
        high_prec: bool = False,
        type_sel: list[int] | None = None,
        repeat: int = 1,
        default: float = 0.0,
        dtype: np.dtype | None = None,
        output_natoms_for_type_sel: bool = False,
    ) -> None:
        item = DataRequirementItem(
            key,
            ndof,
            atomic=atomic,
            must=must,
            high_prec=high_prec,
            type_sel=type_sel,
            repeat=repeat,
            default=default,
            dtype=dtype,
            output_natoms_for_type_sel=output_natoms_for_type_sel,
        )
        self.add_data_requirements([item])

    def get_data_dict(self, ii: int = 0) -> dict[str, dict[str, Any]]:
        del ii
        return self._data_dict

    def _load_set(self, set_name: str) -> dict[str, Any]:
        """Load one bounded same-nloc chunk for legacy neighbor statistics."""
        indices = self._nloc_set_indices[str(set_name)]
        return self._legacy_batch(self._reader.decode_batch(indices, ragged=False))

    def _next_indices(self) -> list[int]:
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._sampler)
            return next(self._iter)

    def _legacy_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Translate a canonical LMDB batch to the legacy data-system shape."""
        coord = np.asarray(batch["coord"])
        nframes = coord.shape[0]
        out: dict[str, Any] = {}
        structural_keys = {"coord", "box"}
        for key, value in batch.items():
            if key in {
                "atype",
                "natoms",
                "real_natoms_vec",
                "fid",
                "sid",
                "n_node",
            }:
                continue
            if key.startswith("find_") and key[5:] not in self._data_dict:
                continue
            if (
                not key.startswith("find_")
                and key not in structural_keys
                and key not in self._data_dict
            ):
                continue
            if value is None:
                out[key] = None
                continue
            array = np.asarray(value)
            data_info = self._data_dict.get(key)
            if key == "coord" or (
                data_info is not None and data_info["atomic"] and array.ndim >= 3
            ):
                array = array.reshape(nframes, -1)
            out[key] = array

        atype = np.asarray(batch["atype"], dtype=np.int32)
        real_natoms_vec = np.asarray(
            batch.get("real_natoms_vec", batch["natoms"]), dtype=np.int32
        )
        if real_natoms_vec.ndim == 1:
            real_natoms_vec = np.tile(real_natoms_vec, (nframes, 1))
        pad_nloc = int(atype.shape[1])
        natoms_vec = np.concatenate(
            (
                np.array([pad_nloc, pad_nloc], dtype=np.int32),
                real_natoms_vec[:, 2:].max(axis=0).astype(np.int32),
            )
        )

        out["type"] = atype
        out["natoms_vec"] = natoms_vec
        out["real_natoms_vec"] = real_natoms_vec
        if "box" not in out or out["box"] is None:
            out["box"] = np.zeros((nframes, 9), dtype=GLOBAL_NP_FLOAT_PRECISION)
            out["find_box"] = np.float32(0.0)
        elif "find_box" not in out:
            out["find_box"] = np.float32(0.0 if np.allclose(out["box"], 0.0) else 1.0)
        out.setdefault("find_coord", np.float32(1.0))
        if "numb_copy" not in out:
            out["numb_copy"] = np.ones((nframes, 1), dtype=np.int64)
            out["find_numb_copy"] = np.float32(0.0)
        out["default_mesh"] = np.asarray(
            make_default_mesh(bool(float(out["find_box"]) > 0.5), self.mixed_type),
            dtype=np.int32,
        )
        return out

    def _stack_frames(self, frames: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate already-decoded frames with the reader's flag semantics."""
        if not frames:
            raise ValueError("Cannot stack an empty LMDB frame batch.")
        from deepmd.dpmodel.utils.lmdb_data import (
            collate_lmdb_frames,
            resolve_per_atom_keys,
        )

        per_atom_keys = resolve_per_atom_keys(frames[0], self._reader.decode_config)
        return self._legacy_batch(collate_lmdb_frames(frames, per_atom_keys))

    def get_batch(self, sys_idx: int | None = None) -> dict[str, Any]:
        del sys_idx
        indices = self._next_indices()
        return self._legacy_batch(self._reader.decode_batch(indices, ragged=False))

    def get_stat_batch(self, sys_idx: int) -> dict[str, Any]:
        """Return one bounded batch from a homogeneous statistical group."""
        if not 0 <= sys_idx < len(self._stat_groups):
            raise IndexError(f"Statistical system index {sys_idx} is out of range")
        nloc, indices = self._stat_groups[sys_idx]
        batch_size = self._get_stat_batch_size(nloc)
        start = self._stat_offsets[sys_idx]
        if start >= len(indices):
            start = 0
        stop = min(start + batch_size, len(indices))
        self._stat_offsets[sys_idx] = stop
        return self._legacy_batch(
            self._reader.decode_batch(indices[start:stop], ragged=False)
        )

    def get_stat_nsystems(self) -> int:
        """Return the number of stack-compatible statistical groups."""
        return len(self._stat_groups)

    def _get_stat_batch_size(self, nloc: int) -> int:
        """Cap model-stat decoding by both frames and decoded atom rows."""
        configured = self._reader.get_batch_size_for_nloc(nloc)
        return max(1, min(configured, 128, 20000 // max(int(nloc), 1)))

    def get_stat_numb_batches(self, sys_idx: int) -> int:
        """Return the finite batch count of one statistical group."""
        if not 0 <= sys_idx < len(self._stat_groups):
            raise IndexError(f"Statistical system index {sys_idx} is out of range")
        nloc, indices = self._stat_groups[sys_idx]
        batch_size = self._get_stat_batch_size(nloc)
        return (len(indices) + batch_size - 1) // batch_size

    def get_nsystems(self) -> int:
        return self.nsystems

    def get_natoms(self) -> int:
        return self.natoms[0]

    def get_ntypes(self) -> int:
        return len(self._type_map)

    def get_type_map(self) -> list[str]:
        return self._type_map

    @property
    def type_map(self) -> list[str]:
        """Model-side atom names exposed by the legacy data-system API."""
        return self._type_map

    def get_batch_size(self) -> list[int]:
        return self.batch_size

    def print_summary(self, name: str, prob: Any | None = None) -> None:
        del prob
        self._reader.print_summary(name, self.sys_probs)

    def close(self) -> None:
        """Release LMDB readers idempotently."""
        if getattr(self, "_closed", True):
            return
        self.data_systems = []
        test_data = getattr(self, "_test_data", None)
        if test_data is not None:
            test_data.close()
        reader = getattr(self, "_reader", None)
        if reader is not None:
            reader.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def _format_name_length(name: str, width: int) -> str:
    if len(name) <= width:
        return "{: >{}}".format(name, width)
    else:
        name = name[-(width - 3) :]
        name = "-- " + name
        return name


def print_summary(
    name: str,
    nsystems: int,
    system_dirs: list[str],
    natoms: list[int],
    batch_size: list[int],
    nbatches: list[int],
    sys_probs: list[float],
    pbc: list[bool],
    e_max: list[int] | None = None,
) -> None:
    """Print summary of systems.

    Parameters
    ----------
    name : str
        The name of the system
    nsystems : int
        The number of systems
    system_dirs : list of str
        The directories of the systems
    natoms : list of int
        The number of atoms
    batch_size : list of int
        The batch size
    nbatches : list of int
        The number of batches
    sys_probs : list of float
        The probabilities
    pbc : list of bool
        The periodic boundary conditions
    e_max : list of int, optional
        The maximal number of valid edges per frame for each system.
    """
    # width 65
    sys_width = 42
    log.info(
        f"---Summary of DataSystem: {name.capitalize():13s}-----------------------------------------------"
    )
    log.info("Found %d System(s):", nsystems)
    use_e_max = e_max is not None and len(e_max) == nsystems
    if use_e_max:
        emax_width = max(5, len(str(max(e_max))))
        log.info(
            "%s  %-6s  %-*s  %-6s  %-6s  %-9s  %-3s",
            _format_name_length("system", sys_width),
            "natoms",
            emax_width,
            "e_max",
            "bch_sz",
            "n_bch",
            "prob",
            "pbc",
        )
    else:
        log.info(
            "%s  %-6s  %-6s  %-6s  %-9s  %-3s",
            _format_name_length("system", sys_width),
            "natoms",
            "bch_sz",
            "n_bch",
            "prob",
            "pbc",
        )
    for ii in range(nsystems):
        if use_e_max:
            log.info(
                "%s  %6d  %*d  %6d  %6d  %9.3e  %3s",
                _format_name_length(system_dirs[ii], sys_width),
                natoms[ii],
                emax_width,
                e_max[ii],
                batch_size[ii],
                nbatches[ii],
                sys_probs[ii],
                "T" if pbc[ii] else "F",
            )
        else:
            log.info(
                "%s  %6d  %6d  %6d  %9.3e  %3s",
                _format_name_length(system_dirs[ii], sys_width),
                natoms[ii],
                batch_size[ii],
                nbatches[ii],
                sys_probs[ii],
                "T" if pbc[ii] else "F",
            )
    log.info(
        "--------------------------------------------------------------------------------------"
    )


def process_sys_probs(sys_probs: list[float], nbatch: int) -> np.ndarray:
    sys_probs = np.array(sys_probs)
    type_filter = sys_probs >= 0
    assigned_sum_prob = np.sum(type_filter * sys_probs)
    # 1e-8 is to handle floating point error; See #1917
    assert assigned_sum_prob <= 1.0 + 1e-8, (
        "the sum of assigned probability should be less than 1"
    )
    rest_sum_prob = 1.0 - assigned_sum_prob
    if not np.isclose(rest_sum_prob, 0):
        rest_nbatch = (1 - type_filter) * nbatch
        rest_prob = rest_sum_prob * rest_nbatch / np.sum(rest_nbatch)
        ret_prob = rest_prob + type_filter * sys_probs
    else:
        ret_prob = sys_probs
    assert np.isclose(np.sum(ret_prob), 1), "sum of probs should be 1"
    return ret_prob


def prob_sys_size_ext(keywords: str, nsystems: int, nbatch: int) -> list[float]:
    block_str = keywords.split(";")[1:]
    block_stt = []
    block_end = []
    block_weights = []
    for ii in block_str:
        stt = int(ii.split(":")[0])
        end = int(ii.split(":")[1])
        weight = float(ii.split(":")[2])
        assert weight >= 0, "the weight of a block should be no less than 0"
        block_stt.append(stt)
        block_end.append(end)
        block_weights.append(weight)
    nblocks = len(block_str)
    block_probs = np.array(block_weights) / np.sum(block_weights)
    sys_probs = np.zeros([nsystems], dtype=np.float64)
    for ii in range(nblocks):
        nbatch_block = nbatch[block_stt[ii] : block_end[ii]]
        tmp_prob = [float(i) for i in nbatch_block] / np.sum(nbatch_block)
        sys_probs[block_stt[ii] : block_end[ii]] = tmp_prob * block_probs[ii]
    return sys_probs


def _is_deepmd_data_format(fmt: str) -> bool:
    return fmt in {
        "deepmd",
        "deepmd/raw",
        "deepmd/npy",
        "deepmd/comp",
        "deepmd/npy/mixed",
        "deepmd/hdf5",
        "deepmd/lmdb",
        "lmdb",
    }


def _is_dpdata_lmdb_format(fmt: str) -> bool:
    """Return whether *fmt* names dpdata's DeePMD-compatible LMDB format."""
    return fmt in {"deepmd/lmdb", "lmdb"}


def _canonical_dpdata_out_format(out_fmt: str | None) -> str:
    """Return the canonical dpdata output format used by conversion caches."""
    if out_fmt is None:
        return _DPDATA_DEFAULT_OUT_FORMAT
    out_fmt = out_fmt.lower()
    return _DPDATA_DEFAULT_OUT_FORMAT if out_fmt == "lmdb" else out_fmt


def conversion_will_write_lmdb(data_config: dict[str, Any]) -> bool:
    """Whether a non-DeePMD input config will be converted to LMDB."""
    data_format = data_config.get("format")
    if data_format is None or _is_deepmd_data_format(data_format.lower()):
        return False
    out_format = data_config.get("out_format", data_config.get("output_format"))
    return _is_dpdata_lmdb_format(_canonical_dpdata_out_format(out_format))


def validate_backend_data_config(
    data_config: dict[str, Any],
    *,
    backend_name: str,
    lmdb_supported: bool,
) -> None:
    """Reject unsupported converted output before dpdata performs any I/O."""
    if not lmdb_supported and conversion_will_write_lmdb(data_config):
        raise NotImplementedError(
            f"{backend_name} backend does not support LMDB data yet. "
            "Choose out_format='deepmd/hdf5' for automatic conversion."
        )


def validate_lmdb_sampling_options(data_config: dict[str, Any]) -> None:
    """Reject sampling options that cannot be represented by one LMDB route."""
    if data_config.get("sys_probs") is not None:
        raise ValueError(
            "LMDB data does not support explicit sys_probs yet. Use auto_prob "
            "('prob_sys_size', 'prob_uniform', or block weights) so sampling "
            "can be derived from LMDB frame_system_ids."
        )


def close_data_systems(*values: Any) -> None:
    """Close nested data-system mappings/sequences, ignoring shared objects."""
    seen: set[int] = set()

    def close_one(value: Any) -> None:
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict):
            for child in value.values():
                close_one(child)
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                close_one(child)
            return
        close = getattr(value, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                log.warning(
                    "Failed to close data system %r during cleanup",
                    value,
                    exc_info=True,
                )

    for value in values:
        close_one(value)


def _looks_like_extxyz(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open() as fp:
            fp.readline()
            comment = fp.readline()
    except OSError:
        return False
    return "Properties=" in comment or "Lattice=" in comment


def _normalize_dpdata_format(fmt: str, source: Path) -> str:
    fmt = fmt.lower()
    if fmt == "ase":
        return "ase/structure"
    if fmt != "auto":
        return fmt
    suffix = source.suffix.lower().lstrip(".")
    if suffix == "traj":
        return "ase/traj"
    if suffix == "extxyz" or (suffix == "xyz" and _looks_like_extxyz(source)):
        return "extxyz"
    return suffix or fmt


def _iter_conversion_inputs(path: str, patterns: list[str] | None) -> list[str]:
    if patterns is None:
        return [path]
    root = Path(path)
    if not root.is_dir():
        return [path]
    matches = []
    for pattern in patterns:
        matches.extend(str(match) for match in root.rglob(pattern))
    return sorted(set(matches))


def _conversion_cache_path(source: Path, fmt: str, out_fmt: str) -> Path:
    source_resolved = source.resolve(strict=False)
    try:
        dpdata_version = importlib.metadata.version("dpdata")
    except importlib.metadata.PackageNotFoundError:
        dpdata_version = "unknown"
    digest = hashlib.sha1(
        (
            f"{source_resolved}|{fmt}|{out_fmt}|"
            f"schema={_DPDATA_CONVERSION_SCHEMA_VERSION}|dpdata={dpdata_version}"
        ).encode()
    ).hexdigest()[:16]
    stem = source_resolved.stem or source_resolved.name or "dataset"
    safe_out_fmt = out_fmt.replace("/", "-")
    suffix = ".lmdb" if _is_dpdata_lmdb_format(out_fmt) else ""
    return Path.cwd() / _DPDATA_CACHE_DIR / f"{stem}-{safe_out_fmt}-{digest}{suffix}"


def _source_mtime(source: Path, cache_file: Path, *, force: bool = False) -> float:
    if source.is_file():
        return source.stat().st_mtime
    if not source.is_dir():
        return 0.0
    cache_key = (
        str(source.resolve(strict=False)),
        str(cache_file.parent.resolve(strict=False)),
    )
    now = time.monotonic()
    cached = _DPDATA_SOURCE_MTIME_CACHE.get(cache_key)
    if (
        not force
        and cached is not None
        and now - cached[0] < _DPDATA_SOURCE_MTIME_CACHE_TTL
    ):
        return cached[1]
    cache_dir = cache_file.parent.resolve(strict=False)
    latest = source.stat().st_mtime
    for item in source.rglob("*"):
        try:
            item_resolved = item.resolve(strict=False)
            if item_resolved == cache_file or cache_dir in item_resolved.parents:
                continue
            latest = max(latest, item.stat().st_mtime)
        except OSError:
            continue
    _DPDATA_SOURCE_MTIME_CACHE[cache_key] = (now, latest)
    return latest


def _is_conversion_current(
    source: Path, output: Path, *, force_source_scan: bool = False
) -> bool:
    if not output.exists():
        return False
    return output.stat().st_mtime >= _source_mtime(
        source, output, force=force_source_scan
    )


def _process_start_time(pid: int) -> str | None:
    """Return Linux's stable process start token, if available."""
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields_after_name = stat_text.rsplit(")", 1)[1].split()
    return fields_after_name[19] if len(fields_after_name) > 19 else None


def _same_lock_file(lock_path: Path, expected: os.stat_result) -> bool:
    """Whether *lock_path* still names the inode originally acquired/read."""
    try:
        current = lock_path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    return (current.st_dev, current.st_ino) == (expected.st_dev, expected.st_ino)


class _ConversionLock:
    """Owned conversion lock with a heartbeat for cross-host stale recovery."""

    def __init__(self, lock_path: Path, lock_fd: int) -> None:
        self.path = lock_path
        self._stat = os.fstat(lock_fd)
        payload = {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "process_start": _process_start_time(os.getpid()),
            "created": time.time(),
        }
        with os.fdopen(lock_fd, "w") as fp:
            json.dump(payload, fp)
            fp.flush()
            os.fsync(fp.fileno())
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._heartbeat,
            name="deepmd-dpdata-conversion-lock",
            daemon=True,
        )
        try:
            self._thread.start()
        except Exception:
            if _same_lock_file(self.path, self._stat):
                self.path.unlink(missing_ok=True)
            raise

    def _heartbeat(self) -> None:
        while not self._stop.wait(_CONVERSION_LOCK_HEARTBEAT_SECONDS):
            if not _same_lock_file(self.path, self._stat):
                return
            try:
                os.utime(self.path, None, follow_symlinks=False)
            except FileNotFoundError:
                return

    def release(self) -> None:
        """Stop heartbeating and remove only the lock inode we own."""
        self._stop.set()
        self._thread.join()
        if _same_lock_file(self.path, self._stat):
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def _lock_owner_is_alive(payload: dict[str, Any]) -> bool | None:
    """Return owner liveness locally, or None for another host/invalid data."""
    if payload.get("hostname") != socket.gethostname():
        return None
    try:
        pid = int(payload["pid"])
    except (KeyError, TypeError, ValueError):
        return None
    expected_start = payload.get("process_start")
    current_start = _process_start_time(pid)
    if expected_start is not None and current_start is not None:
        return str(expected_start) == current_start
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _recover_stale_conversion_lock(lock_path: Path) -> bool:
    """Remove a dead-owner or expired-lease lock without touching a successor."""
    try:
        lock_stat = lock_path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    try:
        payload = json.loads(lock_path.read_text())
    except (OSError, json.JSONDecodeError):
        payload = {}

    owner_alive = _lock_owner_is_alive(payload)
    lease_expired = time.time() - lock_stat.st_mtime > _CONVERSION_LOCK_STALE_SECONDS
    stale = owner_alive is False or (owner_alive is None and lease_expired)
    if not stale or not _same_lock_file(lock_path, lock_stat):
        return False
    log.warning("Recovering stale dpdata conversion lock %s", lock_path)
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    return True


def _wait_for_conversion(source: Path, output: Path, lock_path: Path) -> bool:
    """Wait without rescanning the source tree while a valid writer owns it."""
    while lock_path.exists():
        if _recover_stale_conversion_lock(lock_path):
            continue
        time.sleep(1.0)
    # Freshness is checked once after publication, not once per waiter-second.
    return _is_conversion_current(source, output, force_source_scan=True)


def _remove_path(path: Path) -> None:
    # Check links before directories: Path.is_dir follows a directory symlink,
    # while cache cleanup must never recurse into a target outside the cache.
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _publish_conversion_output(tmp_output: Path, output: Path) -> None:
    """Publish a non-LMDB conversion without discarding a valid old cache."""
    if not output.exists() or output.is_symlink() or not output.is_dir():
        os.replace(tmp_output, output)
        backup = output.with_name(f".{output.name}.backup")
        _remove_path(backup)
        return

    # POSIX cannot replace a non-empty directory directly. Preserve the old
    # cache under a sibling name and restore it if the second rename fails.
    backup = output.with_name(f".{output.name}.backup")
    _remove_path(backup)
    os.replace(output, backup)
    try:
        os.replace(tmp_output, output)
    except Exception:
        os.replace(backup, output)
        raise
    else:
        _remove_path(backup)


def _write_dpdata_conversion(
    source: Path, fmt: str, out_fmt: str, output: Path
) -> None:
    """Load *source* with dpdata and publish it in a DeePMD format.

    dpdata 1.1.0 makes ``deepmd/lmdb`` writes transactional: it stages and
    validates the complete database before atomically publishing it. Use that
    writer directly so DeePMD-kit does not duplicate or weaken its overwrite
    guarantees. Other dpdata formats do not share that contract, so they keep
    the cache-level temporary output used for failure isolation.
    """
    try:
        import dpdata
    except ImportError as exc:
        raise ImportError(
            "dpdata is required when training_data.format or "
            "validation_data.format is specified. Install dpdata to enable "
            "automatic dataset conversion."
        ) from exc

    multi_systems = dpdata.MultiSystems()
    try:
        multi_systems.load_systems_from_file(str(source), fmt=fmt)
    except (NotImplementedError, ValueError) as labeled_error:
        # dpdata 1.1 exposes an explicit unlabeled path. This matters for
        # structure-only EXTXYZ/ASE inputs, which are valid descriptor data
        # even though a supervised loss may later require labels.
        unlabeled_systems = dpdata.MultiSystems()
        try:
            unlabeled_systems.load_systems_from_file(
                str(source), fmt=fmt, labeled=False
            )
        except (NotImplementedError, TypeError, ValueError):
            try:
                labeled_system = dpdata.LabeledSystem(str(source), fmt=fmt)
            except Exception:
                raise labeled_error from None
            multi_systems = dpdata.MultiSystems(labeled_system)
        else:
            log.info("Loaded unlabeled dpdata input %s using format %s", source, fmt)
            multi_systems = unlabeled_systems
    if len(multi_systems) == 0:
        raise RuntimeError(f"No frames were loaded by dpdata from {source}")

    if _is_dpdata_lmdb_format(out_fmt):
        multi_systems.to(out_fmt, str(output), overwrite=True)
        return

    tmp_output = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    _remove_path(tmp_output)
    try:
        multi_systems.to(out_fmt, str(tmp_output))
        _publish_conversion_output(tmp_output, output)
    except Exception:
        _remove_path(tmp_output)
        raise


def _convert_system_by_dpdata(
    source_path: str, fmt: str, out_fmt: str | None
) -> list[str]:
    source = Path(source_path)
    fmt = _normalize_dpdata_format(fmt, source)
    out_fmt = _canonical_dpdata_out_format(out_fmt)
    output = _conversion_cache_path(source, fmt, out_fmt)
    cache_key = (
        str(Path.cwd().resolve(strict=False)),
        str(source.resolve(strict=False)),
        fmt,
        out_fmt,
        str(output),
    )
    cached_systems = _DPDATA_CONVERSION_CACHE.get(cache_key)
    if cached_systems is not None:
        if _is_conversion_current(source, output):
            return cached_systems
        # A long-lived training/validation process may observe source files
        # rewritten in place. Drop the fast-path entry so the normal locked
        # conversion flow refreshes the on-disk result before it is reused.
        del _DPDATA_CONVERSION_CACHE[cache_key]

    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_suffix(output.suffix + ".lock")
    if not _is_conversion_current(source, output):
        while True:
            try:
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if _wait_for_conversion(source, output, lock_path):
                    break
                continue
            else:
                conversion_lock = _ConversionLock(lock_path, lock_fd)
                try:
                    if not _is_conversion_current(
                        source, output, force_source_scan=True
                    ):
                        log.info(
                            "Converting %s from dpdata format %s to %s at %s",
                            source,
                            fmt,
                            out_fmt,
                            output,
                        )
                        _write_dpdata_conversion(source, fmt, out_fmt, output)
                finally:
                    conversion_lock.release()
                break

    if _is_dpdata_lmdb_format(out_fmt):
        converted_systems = [str(output)]
    else:
        converted_systems = expand_sys_str(str(output))
    if not converted_systems:
        raise RuntimeError(f"No DeePMD systems were found in converted file {output}")
    _DPDATA_CONVERSION_CACHE[cache_key] = converted_systems
    return converted_systems


def process_systems(
    systems: str | list[str],
    patterns: list[str] | None = None,
    fmt: str | None = None,
    out_fmt: str | None = None,
) -> list[str]:
    """Process the user-input systems.

    If it is a single directory, search for all the systems in the directory.
    If it is a list, each item in the list is treated as a directory to search.
    If it is a single LMDB path, return it directly without expansion.
    If fmt is specified and is not a DeePMD data format, each input path is
    converted by dpdata and the converted systems are returned.
    Check if the systems are valid.

    Parameters
    ----------
    systems : str or list of str
        The user-input systems
    patterns : list of str, optional
        The patterns to match the systems, by default None
    fmt : str, optional
        The dpdata input format. If None, no conversion is performed.
    out_fmt : str, optional
        The dpdata output format. If None, ``deepmd/lmdb`` is used when fmt
        triggers conversion.

    Returns
    -------
    result_systems: list of str
        The valid systems
    """
    # See validate_lmdb_systems: this must remain a local import because
    # deepmd.dpmodel initializes descriptors that depend on data_system.
    from deepmd.dpmodel.utils.lmdb_data import (
        is_lmdb,
    )

    # Normalize input to a list of paths to search
    if isinstance(systems, str):
        search_paths = [systems]
    elif isinstance(systems, list):
        search_paths = systems
    else:
        # Handle unsupported input types
        raise ValueError(
            f"Invalid systems type: {type(systems)}. Must be str or list[str]."
        )

    if fmt is not None:
        fmt = fmt.lower()
        if _is_deepmd_data_format(fmt):
            fmt = None

    conversion_inputs: list[str] = []
    if fmt is not None:
        for path in search_paths:
            conversion_inputs.extend(_iter_conversion_inputs(path, patterns))
        if (
            _is_dpdata_lmdb_format(_canonical_dpdata_out_format(out_fmt))
            and len(conversion_inputs) != 1
        ):
            raise ValueError(
                "Automatic LMDB conversion requires exactly one resolved input "
                "path. Merge multiple inputs with dpdata first or choose "
                "out_format='deepmd/hdf5'."
            )

    # Iterate over the search_paths list and apply expansion logic to each path
    result_systems = []
    if fmt is not None:
        for input_path in conversion_inputs:
            result_systems.extend(_convert_system_by_dpdata(input_path, fmt, out_fmt))
    else:
        for path in search_paths:
            if is_lmdb(path):
                result_systems.append(path)
            elif patterns is None:
                expanded_paths = expand_sys_str(path)
                result_systems.extend(expanded_paths)
            else:
                expanded_paths = rglob_sys_str(path, patterns)
                result_systems.extend(expanded_paths)

    return result_systems


def get_data(
    jdata: dict[str, Any],
    rcut: float,
    type_map: list[str] | None,
    modifier: Any | None,
    multi_task_mode: bool = False,
) -> DeepmdDataSystem | LmdbDataSystem:
    """Get the data system.

    Parameters
    ----------
    jdata
        The json data
    rcut
        The cut-off radius, not used
    type_map
        The type map
    modifier
        The data modifier
    multi_task_mode
        If in multi task mode

    Returns
    -------
    DeepmdDataSystem
        The data system
    """
    systems = jdata["systems"]
    rglob_patterns = jdata.get("rglob_patterns")
    data_format = jdata.get("format")
    out_format = jdata.get("out_format", jdata.get("output_format"))
    if conversion_will_write_lmdb(jdata):
        validate_lmdb_sampling_options(jdata)
    systems = process_systems(
        systems, patterns=rglob_patterns, fmt=data_format, out_fmt=out_format
    )

    batch_size = jdata["batch_size"]
    sys_probs = jdata.get("sys_probs")
    auto_prob = jdata.get("auto_prob", "prob_sys_size")
    optional_type_map = not multi_task_mode

    lmdb_path = validate_lmdb_systems(systems, backend_name="legacy data loader")
    if lmdb_path is not None:
        validate_lmdb_sampling_options(jdata)
        if type_map is None:
            raise ValueError(
                "LMDB training data requires model/type_map to be set. "
                "Set model/type_map or choose training_data.out_format="
                "'deepmd/hdf5' for automatic conversion."
            )
        return LmdbDataSystem(
            lmdb_path=lmdb_path,
            type_map=type_map,
            batch_size=batch_size,
            auto_prob_style=auto_prob,
        )

    data = DeepmdDataSystem(
        systems=systems,
        batch_size=batch_size,
        test_size=1,  # to satisfy the old api
        shuffle_test=True,  # to satisfy the old api
        rcut=rcut,
        type_map=type_map,
        optional_type_map=optional_type_map,
        modifier=modifier,
        trn_all_set=True,  # sample from all sets
        sys_probs=sys_probs,
        auto_prob_style=auto_prob,
    )

    return data
