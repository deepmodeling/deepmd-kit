#!/usr/bin/env python3

# SPDX-License-Identifier: LGPL-3.0-or-later
import bisect
import copy
import functools
import logging
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np

from deepmd.env import (
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_NP_FLOAT_PRECISION,
    LRU_CACHE_SIZE,
)
from deepmd.utils import random as dp_random
from deepmd.utils.path import (
    DPH5Path,
    DPPath,
)

log = logging.getLogger(__name__)


class DeepmdData:
    """Class for a data system.

    It loads data from hard disk, and maintains the data as a `data_dict`

    Parameters
    ----------
    sys_path
            Path to the data system
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
            [DEPRECATED] Deprecated. Now all sets are trained and tested.
    sort_atoms : bool
            Sort atoms by atom types. Required to enable when the data is directly fed to
            descriptors except mixed types.
    """

    def __init__(
        self,
        sys_path: str,
        set_prefix: str = "set",
        shuffle_test: bool = True,
        type_map: list[str] | None = None,
        optional_type_map: bool = True,
        modifier: Any | None = None,
        trn_all_set: bool = False,
        sort_atoms: bool = True,
    ) -> None:
        """Constructor."""
        root = DPPath(sys_path)
        if not root.is_dir():
            raise FileNotFoundError(f"System {sys_path} is not found!")
        self.dirs = root.glob(set_prefix + ".*")
        if not len(self.dirs):
            raise FileNotFoundError(f"No {set_prefix}.* is found in {sys_path}")
        self.dirs.sort()
        # check mix_type format
        error_format_msg = "if one of the set is of mixed_type format, then all of the sets in this system should be of mixed_type format!"
        self.mixed_type = self._check_mode(self.dirs[0])
        for set_item in self.dirs[1:]:
            assert self._check_mode(set_item) == self.mixed_type, error_format_msg
        # load atom type
        self.atom_type = self._load_type(root)
        self.natoms = len(self.atom_type)
        # load atom type map
        self.type_map = self._load_type_map(root)
        assert optional_type_map or self.type_map is not None, (
            f"System {sys_path} must have type_map.raw in this mode! "
        )
        if self.type_map is not None:
            assert len(self.type_map) >= max(self.atom_type) + 1
        # check pbc
        self.pbc = self._check_pbc(root)
        # enforce type_map if necessary
        self.enforce_type_map = False
        if type_map is not None and self.type_map is not None and len(type_map):
            missing_elements = [elem for elem in self.type_map if elem not in type_map]
            if missing_elements:
                raise ValueError(
                    f"Elements {missing_elements} are not present in the provided `type_map`."
                )
            if not self.mixed_type:
                old_to_new_type_idx = np.array(
                    [type_map.index(name) for name in self.type_map], dtype=np.int32
                )
                self.atom_type = old_to_new_type_idx[self.atom_type].astype(np.int32)
            else:
                self.enforce_type_map = True
                sorter = np.argsort(type_map)
                self.type_idx_map = np.array(
                    sorter[np.searchsorted(type_map, self.type_map, sorter=sorter)]
                )
                # padding for virtual atom
                self.type_idx_map = np.append(
                    self.type_idx_map, np.array([-1], dtype=np.int32)
                )
            self.type_map = type_map
        if type_map is None and self.type_map is None and self.mixed_type:
            raise RuntimeError("mixed_type format must have type_map!")
        # make idx map
        self.sort_atoms = sort_atoms
        self.idx_map = self._make_idx_map(self.atom_type)
        self.data_dict = {}
        # add box and coord
        self.add("box", 9, must=self.pbc)
        self.add("coord", 3, atomic=True, must=True)
        # the training times of each frame
        self.add("numb_copy", 1, must=False, default=1, dtype=int)
        # set counters
        self.set_count = 0
        self.iterator = 0
        self.shuffle_test = shuffle_test
        # set modifier
        self.modifier = modifier
        frames_list = [self._get_nframes(set_name) for set_name in self.dirs]
        self.nframes = np.sum(frames_list)
        # The prefix sum stores the range of indices contained in each directory, which is needed by get_item method
        self.prefix_sum = np.cumsum(frames_list).tolist()

        self.use_modifier_cache = True
        if self.modifier is not None:
            if hasattr(self.modifier, "use_cache"):
                self.use_modifier_cache = self.modifier.use_cache
            # Cache for modified frames when use_modifier_cache is True
            self._modified_frame_cache = {}

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
    ) -> "DeepmdData":
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
        default : float, default=0.
            default value of data
        dtype : np.dtype, optional
            the dtype of data, overwrites `high_prec` if provided
        output_natoms_for_type_sel : bool, optional
            if True and type_sel is True, the atomic dimension will be natoms instead of nsel
        """
        self.data_dict[key] = {
            "ndof": ndof,
            "atomic": atomic,
            "must": must,
            "high_prec": high_prec,
            "type_sel": type_sel,
            "repeat": repeat,
            "reduce": None,
            "default": default,
            "dtype": dtype,
            "output_natoms_for_type_sel": output_natoms_for_type_sel,
        }
        return self

    def reduce(self, key_out: str, key_in: str) -> "DeepmdData":
        """Generate a new item from the reduction of another atom.

        Parameters
        ----------
        key_out
            The name of the reduced item
        key_in
            The name of the data item to be reduced
        """
        assert key_in in self.data_dict, "cannot find input key"
        assert self.data_dict[key_in]["atomic"], "reduced property should be atomic"
        assert key_out not in self.data_dict, "output key should not have been added"
        assert self.data_dict[key_in]["repeat"] == 1, (
            "reduced properties should not have been repeated"
        )

        self.data_dict[key_out] = {
            "ndof": self.data_dict[key_in]["ndof"],
            "atomic": False,
            "must": True,
            "high_prec": True,
            "type_sel": None,
            "repeat": 1,
            "reduce": key_in,
        }
        return self

    def get_data_dict(self) -> dict:
        """Get the `data_dict`."""
        return self.data_dict

    def check_batch_size(self, batch_size: int) -> bool:
        """Check if the system can get a batch of data with `batch_size` frames."""
        for ii in self.dirs:
            if self.data_dict["coord"]["high_prec"]:
                tmpe = (
                    (ii / "coord.npy").load_numpy().astype(GLOBAL_ENER_FLOAT_PRECISION)
                )
            else:
                tmpe = (ii / "coord.npy").load_numpy().astype(GLOBAL_NP_FLOAT_PRECISION)
            if tmpe.ndim == 1:
                tmpe = tmpe.reshape([1, -1])
            if tmpe.shape[0] < batch_size:
                return ii, tmpe.shape[0]
        return None

    def check_test_size(self, test_size: int) -> bool:
        """Check if the system can get a test dataset with `test_size` frames."""
        return self.check_batch_size(test_size)

    def get_item_torch(
        self,
        index: int,
        num_worker: int = 1,
    ) -> dict:
        """Get a single frame data . The frame is picked from the data system by index. The index is coded across all the sets.

        Parameters
        ----------
        index
            index of the frame
        num_worker
            number of workers for parallel data modification
        """
        return self.get_single_frame(index, num_worker)

    def get_item_paddle(
        self,
        index: int,
        num_worker: int = 1,
    ) -> dict:
        """Get a single frame data . The frame is picked from the data system by index. The index is coded across all the sets.
        Same with PyTorch backend.

        Parameters
        ----------
        index
            index of the frame
        num_worker
            number of workers for parallel data modification
        """
        return self.get_single_frame(index, num_worker)

    def get_batch(self, batch_size: int) -> dict:
        """Get a batch of data with `batch_size` frames. The frames are randomly picked from the data system.

        Parameters
        ----------
        batch_size
            size of the batch
        """
        if hasattr(self, "batch_set"):
            set_size = self.batch_set["coord"].shape[0]
        else:
            set_size = 0
        if self.iterator + batch_size > set_size:
            self._load_batch_set(self.dirs[self.set_count % self.get_numb_set()])
            self.set_count += 1
            set_size = self.batch_set["coord"].shape[0]
        iterator_1 = self.iterator + batch_size
        if iterator_1 >= set_size:
            iterator_1 = set_size
        idx = np.arange(self.iterator, iterator_1, dtype=np.int64)
        self.iterator += batch_size
        ret = self._get_subdata(self.batch_set, idx)
        return ret

    def get_test(self, ntests: int = -1) -> dict:
        """Get the test data with `ntests` frames.

        Parameters
        ----------
        ntests
            Size of the test data set. If `ntests` is -1, all test data will be get.
        """
        if not hasattr(self, "test_set"):
            self._load_test_set(self.shuffle_test)
        if ntests == -1:
            idx = None
        else:
            ntests_ = (
                ntests
                if ntests < self.test_set["type"].shape[0]
                else self.test_set["type"].shape[0]
            )
            # print('ntest', self.test_set['type'].shape[0], ntests, ntests_)
            idx = np.arange(ntests_, dtype=np.int64)
        ret = self._get_subdata(self.test_set, idx=idx)
        if self.modifier is not None:
            self.modifier.modify_data(ret, self)
        return ret

    def get_ntypes(self) -> int:
        """Number of atom types in the system."""
        if self.type_map is not None:
            return len(self.type_map)
        else:
            return max(self.get_atom_type()) + 1

    def get_type_map(self) -> list[str]:
        """Get the type map."""
        return self.type_map

    def get_atom_type(self) -> list[int]:
        """Get atom types."""
        return self.atom_type

    def get_numb_set(self) -> int:
        """Get number of training sets."""
        return len(self.dirs)

    def get_numb_batch(self, batch_size: int, set_idx: int) -> int:
        """Get the number of batches in a set."""
        set_name = self.dirs[set_idx]
        # Directly obtain the number of frames to avoid loading the entire dataset
        nframes = self._get_nframes(set_name)
        ret = nframes // batch_size
        if ret == 0:
            ret = 1
        return ret

    def get_sys_numb_batch(self, batch_size: int) -> int:
        """Get the number of batches in the data system."""
        ret = 0
        for ii in range(len(self.dirs)):
            ret += self.get_numb_batch(batch_size, ii)
        return ret

    def get_natoms(self) -> int:
        """Get number of atoms."""
        return len(self.atom_type)

    def get_natoms_vec(self, ntypes: int) -> np.ndarray:
        """Get number of atoms and number of atoms in different types.

        Parameters
        ----------
        ntypes
            Number of types (may be larger than the actual number of types in the system).

        Returns
        -------
        natoms
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        """
        natoms, natoms_vec = self._get_natoms_2(ntypes)
        tmp = [natoms, natoms]
        tmp = np.append(tmp, natoms_vec)
        return tmp.astype(np.int32)

    def get_single_frame(self, index: int, num_worker: int) -> dict:
        """Orchestrates loading a single frame efficiently using memmap."""
        # Check if we have a cached modified frame and use_modifier_cache is True
        if (
            self.use_modifier_cache
            and self.modifier is not None
            and index in self._modified_frame_cache
        ):
            return self._modified_frame_cache[index]

        if index < 0 or index >= self.nframes:
            raise IndexError(f"Frame index {index} out of range [0, {self.nframes})")
        # 1. Find the correct set directory and local frame index
        set_idx = bisect.bisect_right(self.prefix_sum, index)
        set_dir = self.dirs[set_idx]
        if not isinstance(set_dir, DPPath):
            set_dir = DPPath(set_dir)
        # Calculate local index within the set.* directory
        local_idx = index - (0 if set_idx == 0 else self.prefix_sum[set_idx - 1])
        # Calculate the number of frames in this set to avoid redundant _get_nframes calls
        set_nframes = (
            self.prefix_sum[set_idx]
            if set_idx == 0
            else self.prefix_sum[set_idx] - self.prefix_sum[set_idx - 1]
        )

        frame_data = {}
        # 2. Concurrently load all non-reduced items
        non_reduced_keys = [k for k, v in self.data_dict.items() if v["reduce"] is None]
        reduced_keys = [k for k, v in self.data_dict.items() if v["reduce"] is not None]
        # Use a thread pool to parallelize loading
        if non_reduced_keys:
            with ThreadPoolExecutor(max_workers=len(non_reduced_keys)) as executor:
                future_to_key = {
                    executor.submit(
                        self._load_single_data, set_dir, key, local_idx, set_nframes
                    ): key
                    for key in non_reduced_keys
                }
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    frame_data["find_" + key], frame_data[key] = future.result()

        # 3. Compute reduced items from already loaded data
        for key in reduced_keys:
            vv = self.data_dict[key]
            k_in = vv["reduce"]
            ndof = vv["ndof"]
            frame_data["find_" + key] = frame_data["find_" + k_in]
            # Reshape to (natoms, ndof) and sum over atom axis
            tmp_in = (
                frame_data[k_in].reshape(-1, ndof).astype(GLOBAL_ENER_FLOAT_PRECISION)
            )
            frame_data[key] = np.sum(tmp_in, axis=0)

        # 4. Handle atom types (mixed or standard)
        if self.mixed_type:
            type_path = set_dir / "real_atom_types.npy"
            # For HDF5 files, use load_numpy; for filesystem, use memmap
            if isinstance(type_path, DPH5Path):
                mmap_types = type_path.load_numpy()
            else:
                mmap_types = self._get_memmap(type_path)
            real_type = mmap_types[local_idx].copy().astype(np.int32)

            if self.enforce_type_map:
                try:
                    real_type = self.type_idx_map[real_type].astype(np.int32)
                except IndexError as e:
                    raise IndexError(
                        f"some types in 'real_atom_types.npy' of set {set_dir} are not contained in {self.get_ntypes()} types!"
                    ) from e

            frame_data["type"] = real_type
            ntypes = self.get_ntypes()
            natoms = len(real_type)
            # Use bincount for efficient counting of each type
            natoms_vec = np.bincount(
                real_type[real_type >= 0], minlength=ntypes
            ).astype(np.int32)
            frame_data["real_natoms_vec"] = np.concatenate(
                (np.array([natoms, natoms], dtype=np.int32), natoms_vec)
            )
        else:
            frame_data["type"] = self.atom_type[self.idx_map]

        # 5. Standardize keys
        frame_data = {kk.replace("atomic", "atom"): vv for kk, vv in frame_data.items()}

        # 6. Reshape atomic data to match expected format [natoms, ndof]
        for kk in self.data_dict.keys():
            if (
                "find_" not in kk
                and kk in frame_data
                and not self.data_dict[kk]["atomic"]
            ):
                frame_data[kk] = frame_data[kk].reshape(-1)
        frame_data["atype"] = frame_data["type"]

        if not self.pbc:
            frame_data["box"] = None

        frame_data["fid"] = index

        if self.modifier is not None:
            with ThreadPoolExecutor(max_workers=num_worker) as executor:
                # Apply modifier if it exists
                executor.submit(
                    self.modifier.modify_data,
                    frame_data,
                    self,
                )
            if self.use_modifier_cache:
                # Cache the modified frame to avoid recomputation
                self._modified_frame_cache[index] = copy.deepcopy(frame_data)
        return frame_data

    def preload_and_modify_all_data_torch(self, num_worker: int) -> None:
        """Preload all frames and apply modifier to cache them.

        This method is useful when use_modifier_cache is True and you want to
        avoid applying the modifier repeatedly during training.
        """
        if not self.use_modifier_cache or self.modifier is None:
            return

        log.info("Preloading and modifying all data frames...")
        for i in range(self.nframes):
            if i not in self._modified_frame_cache:
                self.get_single_frame(i, num_worker)
                if (i + 1) % 100 == 0:
                    log.info(f"Processed {i + 1}/{self.nframes} frames")
        log.info("All frames preloaded and modified.")

    def avg(self, key: str) -> float:
        """Return the average value of an item."""
        if key not in self.data_dict.keys():
            raise RuntimeError(f"key {key} has not been added")
        info = self.data_dict[key]
        ndof = info["ndof"]
        eners = []
        for ii in self.dirs:
            data = self._load_set(ii)
            ei = data[key].reshape([-1, ndof])
            eners.append(ei)
        eners = np.concatenate(eners, axis=0)
        if eners.size == 0:
            return 0
        else:
            return np.average(eners, axis=0)

    def _idx_map_sel(self, atom_type: np.ndarray, type_sel: list[int]) -> np.ndarray:
        # Use vectorized operations instead of Python loop
        sel_mask = np.isin(atom_type, type_sel)
        new_types = atom_type[sel_mask]
        natoms = new_types.shape[0]
        idx = np.arange(natoms, dtype=np.int64)
        idx_map = np.lexsort((idx, new_types))
        return idx_map

    def _get_natoms_2(self, ntypes: int) -> tuple[int, np.ndarray]:
        sample_type = self.atom_type
        natoms = len(sample_type)
        natoms_vec = np.zeros(ntypes, dtype=np.int64)
        for ii in range(ntypes):
            natoms_vec[ii] = np.count_nonzero(sample_type == ii)
        return natoms, natoms_vec

    def _get_memmap(self, path: DPPath) -> np.memmap:
        """Get or create a memory-mapped object for a given npy file.
        Uses file path and modification time as cache keys to detect file changes
        and invalidate cache when files are modified.
        """
        abs_path = Path(str(path)).absolute()
        file_mtime = abs_path.stat().st_mtime
        return self._create_memmap(str(abs_path), str(file_mtime))

    def _get_subdata(
        self, data: dict[str, Any], idx: np.ndarray | None = None
    ) -> dict[str, Any]:
        new_data = {}
        for ii in data:
            dd = data[ii]
            if "find_" in ii:
                new_data[ii] = dd
            else:
                if idx is not None:
                    new_data[ii] = dd[idx]
                else:
                    new_data[ii] = dd
        return new_data

    def _load_batch_set(self, set_name: DPPath) -> None:
        if not hasattr(self, "batch_set") or self.get_numb_set() > 1:
            self.batch_set = self._load_set(set_name)
            if self.modifier is not None:
                self.modifier.modify_data(self.batch_set, self)
        self.batch_set, _ = self._shuffle_data(self.batch_set)
        self.reset_get_batch()

    def reset_get_batch(self) -> None:
        self.iterator = 0

    def _load_test_set(self, shuffle_test: bool) -> None:
        test_sets = []
        for ii in self.dirs:
            test_set = self._load_set(ii)
            test_sets.append(test_set)
        # merge test sets
        self.test_set = {}
        assert len(test_sets) > 0
        for kk in test_sets[0]:
            if "find_" in kk:
                self.test_set[kk] = test_sets[0][kk]
            else:
                self.test_set[kk] = np.concatenate(
                    [test_set[kk] for test_set in test_sets], axis=0
                )
        if shuffle_test:
            self.test_set, _ = self._shuffle_data(self.test_set)

    def _shuffle_data(self, data: dict[str, Any]) -> dict[str, Any]:
        ret = {}
        nframes = data["coord"].shape[0]
        idx = np.arange(nframes, dtype=np.int64)
        # the training times of each frame
        idx = np.repeat(idx, np.reshape(data["numb_copy"], (nframes,)))
        dp_random.shuffle(idx)
        for kk in data:
            if (
                isinstance(data[kk], np.ndarray)
                and len(data[kk].shape) == 2
                and data[kk].shape[0] == nframes
                and "find_" not in kk
            ):
                ret[kk] = data[kk][idx]
            else:
                ret[kk] = data[kk]
        return ret, idx

    def _get_nframes(self, set_name: DPPath | str) -> int:
        if not isinstance(set_name, DPPath):
            set_name = DPPath(set_name)
        path = set_name / "coord.npy"
        if isinstance(set_name, DPH5Path):
            nframes = path.root[path._name].shape[0]
        else:
            # Read only the header to get shape
            with open(str(path), "rb") as f:
                version = np.lib.format.read_magic(f)
                if version[0] == 1:
                    shape, _fortran_order, _dtype = np.lib.format.read_array_header_1_0(
                        f
                    )
                elif version[0] in [2, 3]:
                    shape, _fortran_order, _dtype = np.lib.format.read_array_header_2_0(
                        f
                    )
                else:
                    raise ValueError(f"Unsupported .npy file version: {version}")
            nframes = shape[0] if len(shape) > 1 else 1
        return nframes

    def reformat_data_torch(self, data: dict[str, Any]) -> dict[str, Any]:
        """Modify the data format for the requirements of Torch backend.

        Parameters
        ----------
        data
            original data
        """
        for kk in self.data_dict.keys():
            if "find_" in kk:
                pass
            else:
                if kk in data and self.data_dict[kk]["atomic"]:
                    data[kk] = data[kk].reshape(-1, self.data_dict[kk]["ndof"])
        data["atype"] = data["type"]
        if not self.pbc:
            data["box"] = None
        return data

    def _load_set(self, set_name: DPPath) -> dict[str, Any]:
        # get nframes
        if not isinstance(set_name, DPPath):
            set_name = DPPath(set_name)
        path = set_name / "coord.npy"
        if self.data_dict["coord"]["high_prec"]:
            coord = path.load_numpy().astype(GLOBAL_ENER_FLOAT_PRECISION)
        else:
            coord = path.load_numpy().astype(GLOBAL_NP_FLOAT_PRECISION)
        if coord.ndim == 1:
            coord = coord.reshape([1, -1])
        nframes = coord.shape[0]
        assert coord.shape[1] == self.data_dict["coord"]["ndof"] * self.natoms
        # load keys
        data = {}
        for kk in self.data_dict.keys():
            if self.data_dict[kk]["reduce"] is None:
                data["find_" + kk], data[kk] = self._load_data(
                    set_name,
                    kk,
                    nframes,
                    self.data_dict[kk]["ndof"],
                    atomic=self.data_dict[kk]["atomic"],
                    high_prec=self.data_dict[kk]["high_prec"],
                    must=self.data_dict[kk]["must"],
                    type_sel=self.data_dict[kk]["type_sel"],
                    repeat=self.data_dict[kk]["repeat"],
                    default=self.data_dict[kk]["default"],
                    dtype=self.data_dict[kk]["dtype"],
                    output_natoms_for_type_sel=self.data_dict[kk][
                        "output_natoms_for_type_sel"
                    ],
                )
        for kk in self.data_dict.keys():
            if self.data_dict[kk]["reduce"] is not None:
                k_in = self.data_dict[kk]["reduce"]
                ndof = self.data_dict[kk]["ndof"]
                data["find_" + kk] = data["find_" + k_in]
                tmp_in = data[k_in].astype(GLOBAL_ENER_FLOAT_PRECISION)
                data[kk] = np.sum(
                    np.reshape(tmp_in, [nframes, self.natoms, ndof]), axis=1
                )

        if self.mixed_type:
            # nframes x natoms
            atom_type_mix = self._load_type_mix(set_name)
            if self.enforce_type_map:
                try:
                    atom_type_mix_ = self.type_idx_map[atom_type_mix].astype(np.int32)
                except IndexError as e:
                    raise IndexError(
                        f"some types in 'real_atom_types.npy' of set {set_name} are not contained in {self.get_ntypes()} types!"
                    ) from e
                atom_type_mix = atom_type_mix_
            real_type = atom_type_mix.reshape([nframes, self.natoms])
            data["type"] = real_type
            natoms = data["type"].shape[1]
            # nframes x ntypes
            atom_type_nums = np.array(
                [(real_type == i).sum(axis=-1) for i in range(self.get_ntypes())],
                dtype=np.int32,
            ).T
            ghost_nums = np.array(
                [(real_type == -1).sum(axis=-1)],
                dtype=np.int32,
            ).T
            assert (
                atom_type_nums.sum(axis=-1) + ghost_nums.sum(axis=-1) == natoms
            ).all(), (
                f"some types in 'real_atom_types.npy' of set {set_name} are not contained in {self.get_ntypes()} types!"
            )
            data["real_natoms_vec"] = np.concatenate(
                (
                    np.tile(np.array([natoms, natoms], dtype=np.int32), (nframes, 1)),
                    atom_type_nums,
                ),
                axis=-1,
            )
        else:
            data["type"] = np.tile(self.atom_type[self.idx_map], (nframes, 1))

        # standardize keys
        data = {kk.replace("atomic", "atom"): vv for kk, vv in data.items()}
        return data

    def _load_data(
        self,
        set_name: str,
        key: str,
        nframes: int,
        ndof_: int,
        atomic: bool = False,
        must: bool = True,
        repeat: int = 1,
        high_prec: bool = False,
        type_sel: list[int] | None = None,
        default: float = 0.0,
        dtype: np.dtype | None = None,
        output_natoms_for_type_sel: bool = False,
    ) -> np.ndarray:
        if atomic:
            natoms = self.natoms
            idx_map = self.idx_map
            # if type_sel, then revise natoms and idx_map
            if type_sel is not None:
                # Use vectorized operations for better performance
                sel_mask = np.isin(self.atom_type, type_sel)
                natoms_sel = np.sum(sel_mask)
                idx_map_sel = self._idx_map_sel(self.atom_type, type_sel)
            else:
                natoms_sel = natoms
                idx_map_sel = idx_map
            ndof = ndof_ * natoms
        else:
            ndof = ndof_
            natoms_sel = 0
            idx_map_sel = None
        if dtype is not None:
            pass
        elif high_prec:
            dtype = GLOBAL_ENER_FLOAT_PRECISION
        else:
            dtype = GLOBAL_NP_FLOAT_PRECISION
        path = set_name / (key + ".npy")
        if path.is_file():
            data = path.load_numpy().astype(dtype)
            try:  # YWolfeee: deal with data shape error
                if atomic:
                    if type_sel is not None:
                        # check the data shape is nsel or natoms
                        if data.size == nframes * natoms_sel * ndof_:
                            if output_natoms_for_type_sel:
                                tmp = np.zeros(
                                    [nframes, natoms, ndof_], dtype=data.dtype
                                )
                                tmp[:, sel_mask] = data.reshape(
                                    [nframes, natoms_sel, ndof_]
                                )
                                data = tmp
                            else:
                                natoms = natoms_sel
                                idx_map = idx_map_sel
                                ndof = ndof_ * natoms
                        elif data.size == nframes * natoms * ndof_:
                            if output_natoms_for_type_sel:
                                pass
                            else:
                                data = data.reshape([nframes, natoms, ndof_])
                                data = data[:, sel_mask]
                                natoms = natoms_sel
                                idx_map = idx_map_sel
                                ndof = ndof_ * natoms
                        else:
                            raise ValueError(
                                f"The shape of the data {key} in {set_name}"
                                f"is {data.shape}, which doesn't match either"
                                f"({nframes}, {natoms_sel}, {ndof_}) or"
                                f"({nframes}, {natoms}, {ndof_})"
                            )
                    if key == "hessian":
                        data = data.reshape(nframes, 3 * natoms, 3 * natoms)
                        # get idx_map for hessian
                        num_chunks, chunk_size = len(idx_map), 3
                        idx_map_hess = np.arange(num_chunks * chunk_size)  # pylint: disable=no-explicit-dtype
                        idx_map_hess = idx_map_hess.reshape(num_chunks, chunk_size)
                        idx_map_hess = idx_map_hess[idx_map]
                        idx_map_hess = idx_map_hess.flatten()
                        data = data[:, idx_map_hess, :]
                        data = data[:, :, idx_map_hess]
                        data = data.reshape([nframes, -1])
                        ndof = (
                            3 * ndof * 3 * ndof
                        )  # size of hessian is 3Natoms * 3Natoms
                    else:
                        data = data.reshape([nframes, natoms, -1])
                        data = data[:, idx_map, :]
                        data = data.reshape([nframes, -1])
                data = np.reshape(data, [nframes, ndof])
            except ValueError as err_message:
                explanation = "This error may occur when your label mismatch its name, i.e. you might store global tensor in `atomic_tensor.npy` or atomic tensor in `tensor.npy`."
                log.error(str(err_message))
                log.error(explanation)
                raise ValueError(str(err_message) + ". " + explanation) from err_message
            if repeat != 1:
                data = np.repeat(data, repeat).reshape([nframes, -1])
            return np.float32(1.0), data
        elif must:
            raise RuntimeError(f"{path} not found!")
        else:
            if atomic and type_sel is not None and not output_natoms_for_type_sel:
                ndof = ndof_ * natoms_sel
            data = np.full([nframes, ndof], default, dtype=dtype)
            if repeat != 1:
                data = np.repeat(data, repeat).reshape([nframes, -1])
            return np.float32(0.0), data

    def _load_single_data(
        self, set_dir: DPPath, key: str, frame_idx: int, set_nframes: int
    ) -> tuple[np.float32, np.ndarray]:
        """
        Loads and processes data for a SINGLE frame from a SINGLE key,
        fully replicating the logic from the original _load_data method.

        Parameters
        ----------
        set_dir : DPPath
            The directory path of the set
        key : str
            The key name of the data to load
        frame_idx : int
            The local frame index within the set
        set_nframes : int
            The total number of frames in this set (to avoid redundant _get_nframes calls)
        """
        vv = self.data_dict[key]
        path = set_dir / (key + ".npy")

        if vv["atomic"]:
            natoms = self.natoms
            idx_map = self.idx_map
            # if type_sel, then revise natoms and idx_map
            if vv["type_sel"] is not None:
                # Use vectorized operations for better performance
                sel_mask = np.isin(self.atom_type, vv["type_sel"])
                natoms_sel = np.sum(sel_mask)
                idx_map_sel = self._idx_map_sel(self.atom_type, vv["type_sel"])
            else:
                natoms_sel = natoms
                idx_map_sel = idx_map
        else:
            natoms = 1
            natoms_sel = 0
            idx_map_sel = None
        ndof = vv["ndof"]

        # Determine target data type from requirements
        dtype = vv.get("dtype")
        if dtype is None:
            dtype = (
                GLOBAL_ENER_FLOAT_PRECISION
                if vv.get("high_prec")
                else GLOBAL_NP_FLOAT_PRECISION
            )

        # Branch 1: File does not exist
        if not path.is_file():
            if vv.get("must"):
                raise RuntimeError(f"{path} not found!")

            # Create a default array based on requirements
            if vv["atomic"]:
                if vv["type_sel"] is not None and not vv["output_natoms_for_type_sel"]:
                    natoms = natoms_sel
                data = np.full([natoms, ndof], vv["default"], dtype=dtype)
            else:
                # For non-atomic data, shape should be [ndof]
                data = np.full([ndof], vv["default"], dtype=dtype)
            return np.float32(0.0), data

        # Branch 2: Data loading
        if isinstance(path, DPH5Path):
            # For HDF5 files, use load_numpy which handles HDF5 datasets
            mmap_obj = path.load_numpy().astype(dtype)
        else:
            # For filesystem paths, use memmap for better performance
            mmap_obj = self._get_memmap(path)

        # corner case: single frame
        if set_nframes == 1:
            mmap_obj = mmap_obj[None, ...]
        # Slice the single frame and make an in-memory copy for modification
        data = mmap_obj[frame_idx].copy().astype(dtype, copy=False)

        try:
            if vv["atomic"]:
                # Handle type_sel logic
                if vv["type_sel"] is not None:
                    if mmap_obj.shape[1] == natoms_sel * ndof:
                        if vv["output_natoms_for_type_sel"]:
                            tmp = np.zeros([natoms, ndof], dtype=data.dtype)
                            # sel_mask needs to be applied to the original atom layout
                            tmp[sel_mask] = data.reshape([natoms_sel, ndof])
                            data = tmp
                        else:  # output is natoms_sel
                            natoms = natoms_sel
                            idx_map = idx_map_sel
                    elif mmap_obj.shape[1] == natoms * ndof:
                        data = data.reshape([natoms, ndof])
                        if vv["output_natoms_for_type_sel"]:
                            pass
                        else:
                            data = data[sel_mask]
                            idx_map = idx_map_sel
                            natoms = natoms_sel
                    else:  # Shape mismatch error
                        raise ValueError(
                            f"The shape of the data {key} in {set_dir} has width {mmap_obj.shape[1]}, which doesn't match either ({natoms_sel * ndof}) or ({natoms * ndof})"
                        )

                # Handle special case for Hessian
                if key == "hessian":
                    data = data.reshape(3 * natoms, 3 * natoms)
                    num_chunks, chunk_size = len(idx_map), 3
                    idx_map_hess = np.arange(
                        num_chunks * chunk_size, dtype=int
                    ).reshape(num_chunks, chunk_size)
                    idx_map_hess = idx_map_hess[idx_map].flatten()
                    data = data[idx_map_hess, :]
                    data = data[:, idx_map_hess]
                    data = data.reshape(-1)
                    # size of hessian is 3Natoms * 3Natoms
                    # ndof = 3 * ndof * 3 * ndof
                else:
                    # data should be 2D here: [natoms, ndof]
                    data = data.reshape([natoms, -1])
                    data = data[idx_map, :]
            else:
                data = data.reshape([ndof])

            # Atomic: return [natoms, ndof] or flattened hessian above
            # Non-atomic: return [ndof]
            return np.float32(1.0), data

        except ValueError as err_message:
            explanation = (
                "This error may occur when your label mismatches its name, "
                "e.g., global tensor stored in `atomic_tensor.npy` or atomic tensor in `tensor.npy`."
            )
            log.exception(
                "Single-frame load failed for key=%s, set=%s, frame=%d. %s",
                key,
                set_dir,
                frame_idx,
                explanation,
            )
            raise ValueError(f"{err_message}. {explanation}") from err_message

    def _load_type(self, sys_path: DPPath) -> np.ndarray:
        atom_type = (sys_path / "type.raw").load_txt(ndmin=1).astype(np.int32)
        return atom_type

    def _load_type_mix(self, set_name: DPPath) -> np.ndarray:
        type_path = set_name / "real_atom_types.npy"
        real_type = type_path.load_numpy().astype(np.int32).reshape([-1, self.natoms])
        return real_type

    def _make_idx_map(self, atom_type: np.ndarray) -> np.ndarray:
        natoms = atom_type.shape[0]
        idx = np.arange(natoms, dtype=np.int64)
        if self.sort_atoms:
            idx_map = np.lexsort((idx, atom_type))
        else:
            idx_map = idx
        return idx_map

    def _load_type_map(self, sys_path: DPPath) -> list[str] | None:
        fname = sys_path / "type_map.raw"
        if fname.is_file():
            return fname.load_txt(dtype=str, ndmin=1).tolist()
        else:
            return None

    def _check_pbc(self, sys_path: DPPath) -> bool:
        pbc = True
        if (sys_path / "nopbc").is_file():
            pbc = False
        return pbc

    def _check_mode(self, set_path: DPPath) -> bool:
        return (set_path / "real_atom_types.npy").is_file()

    @staticmethod
    @functools.lru_cache(maxsize=LRU_CACHE_SIZE)
    def _create_memmap(path_str: str, mtime_str: str) -> np.memmap:
        """A cached helper function to create memmap objects.
        Using lru_cache to limit the number of open file handles.

        Parameters
        ----------
        path_str
            The file path as a string.
        mtime_str
            The modification time as a string, used for cache invalidation.
        """
        with open(path_str, "rb") as f:
            version = np.lib.format.read_magic(f)
            if version[0] == 1:
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
            elif version[0] in [2, 3]:
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                raise ValueError(f"Unsupported .npy file version: {version}")
            offset = f.tell()
        order = "F" if fortran_order else "C"
        # Create a read-only memmap
        return np.memmap(
            path_str, dtype=dtype, mode="r", shape=shape, order=order, offset=offset
        )


class DataRequirementItem:
    """A class to store the data requirement for data systems.

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
    default : float, default=0.
        default value of data
    dtype : np.dtype, optional
        the dtype of data, overwrites `high_prec` if provided
    output_natoms_for_type_sel : bool, optional
        if True and type_sel is True, the atomic dimension will be natoms instead of nsel
    """

    def __init__(
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
        self.key = key
        self.ndof = ndof
        self.atomic = atomic
        self.must = must
        self.high_prec = high_prec
        self.type_sel = type_sel
        self.repeat = repeat
        self.default = default
        self.dtype = dtype
        self.output_natoms_for_type_sel = output_natoms_for_type_sel
        self.dict = self.to_dict()

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "ndof": self.ndof,
            "atomic": self.atomic,
            "must": self.must,
            "high_prec": self.high_prec,
            "type_sel": self.type_sel,
            "repeat": self.repeat,
            "default": self.default,
            "dtype": self.dtype,
            "output_natoms_for_type_sel": self.output_natoms_for_type_sel,
        }

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self.dict:
            raise KeyError(key)
        return self.dict[key]

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, DataRequirementItem):
            return False
        return self.dict == value.dict

    def __repr__(self) -> str:
        return f"DataRequirementItem({self.dict})"
