#!/usr/bin/env python3

# SPDX-License-Identifier: LGPL-3.0-or-later
import bisect
import logging
from typing import (
    List,
    Optional,
)

import numpy as np

from deepmd.env import (
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.utils import random as dp_random
from deepmd.utils.path import (
    DPPath,
)

log = logging.getLogger(__name__)


class DeepmdData:
    """Class for a data system.

    It loads data from hard disk, and mantains the data as a `data_dict`

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
            Use all sets as training dataset. Otherwise, if the number of sets is more than 1, the last set is left for test.
    sort_atoms : bool
            Sort atoms by atom types. Required to enable when the data is directly feeded to
            descriptors except mixed types.
    """

    def __init__(
        self,
        sys_path: str,
        set_prefix: str = "set",
        shuffle_test: bool = True,
        type_map: Optional[List[str]] = None,
        optional_type_map: bool = True,
        modifier=None,
        trn_all_set: bool = False,
        sort_atoms: bool = True,
    ):
        """Constructor."""
        root = DPPath(sys_path)
        self.dirs = root.glob(set_prefix + ".*")
        if not len(self.dirs):
            raise FileNotFoundError(f"No {set_prefix}.* is found in {sys_path}")
        self.dirs.sort()
        # check mix_type format
        error_format_msg = (
            "if one of the set is of mixed_type format, "
            "then all of the sets in this system should be of mixed_type format!"
        )
        self.mixed_type = self._check_mode(self.dirs[0])
        for set_item in self.dirs[1:]:
            assert self._check_mode(set_item) == self.mixed_type, error_format_msg
        # load atom type
        self.atom_type = self._load_type(root)
        self.natoms = len(self.atom_type)
        # load atom type map
        self.type_map = self._load_type_map(root)
        assert (
            optional_type_map or self.type_map is not None
        ), f"System {sys_path} must have type_map.raw in this mode! "
        if self.type_map is not None:
            assert len(self.type_map) >= max(self.atom_type) + 1
        # check pbc
        self.pbc = self._check_pbc(root)
        # enforce type_map if necessary
        self.enforce_type_map = False
        if type_map is not None and self.type_map is not None and len(type_map):
            if not self.mixed_type:
                atom_type_ = [
                    type_map.index(self.type_map[ii]) for ii in self.atom_type
                ]
                self.atom_type = np.array(atom_type_, dtype=np.int32)
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
        # train dirs
        self.test_dir = self.dirs[-1]
        if trn_all_set:
            self.train_dirs = self.dirs
        else:
            if len(self.dirs) == 1:
                self.train_dirs = self.dirs
            else:
                self.train_dirs = self.dirs[:-1]
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
        # calculate prefix sum for get_item method
        frames_list = [self._get_nframes(item) for item in self.dirs]
        self.nframes = np.sum(frames_list)
        # The prefix sum stores the range of indices contained in each directory, which is needed by get_item method
        self.prefix_sum = np.cumsum(frames_list).tolist()

    def add(
        self,
        key: str,
        ndof: int,
        atomic: bool = False,
        must: bool = False,
        high_prec: bool = False,
        type_sel: Optional[List[int]] = None,
        repeat: int = 1,
        default: float = 0.0,
        dtype: Optional[np.dtype] = None,
        output_natoms_for_type_sel: bool = False,
    ):
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

    def reduce(self, key_out: str, key_in: str):
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
        assert (
            self.data_dict[key_in]["repeat"] == 1
        ), "reduced proerties should not have been repeated"

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

    def check_batch_size(self, batch_size):
        """Check if the system can get a batch of data with `batch_size` frames."""
        for ii in self.train_dirs:
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

    def check_test_size(self, test_size):
        """Check if the system can get a test dataset with `test_size` frames."""
        if self.data_dict["coord"]["high_prec"]:
            tmpe = (
                (self.test_dir / "coord.npy")
                .load_numpy()
                .astype(GLOBAL_ENER_FLOAT_PRECISION)
            )
        else:
            tmpe = (
                (self.test_dir / "coord.npy")
                .load_numpy()
                .astype(GLOBAL_NP_FLOAT_PRECISION)
            )
        if tmpe.ndim == 1:
            tmpe = tmpe.reshape([1, -1])
        if tmpe.shape[0] < test_size:
            return self.test_dir, tmpe.shape[0]
        else:
            return None

    def get_item_torch(self, index: int) -> dict:
        """Get a single frame data . The frame is picked from the data system by index. The index is coded across all the sets.

        Parameters
        ----------
        index
            index of the frame
        """
        i = bisect.bisect_right(self.prefix_sum, index)
        frames = self._load_set(self.dirs[i])
        frame = self._get_subdata(frames, index - self.prefix_sum[i])
        frame = self.reformat_data_torch(frame)
        frame["fid"] = index
        return frame

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
            self._load_batch_set(self.train_dirs[self.set_count % self.get_numb_set()])
            self.set_count += 1
            set_size = self.batch_set["coord"].shape[0]
        iterator_1 = self.iterator + batch_size
        if iterator_1 >= set_size:
            iterator_1 = set_size
        idx = np.arange(self.iterator, iterator_1)
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
            self._load_test_set(self.test_dir, self.shuffle_test)
        if ntests == -1:
            idx = None
        else:
            ntests_ = (
                ntests
                if ntests < self.test_set["type"].shape[0]
                else self.test_set["type"].shape[0]
            )
            # print('ntest', self.test_set['type'].shape[0], ntests, ntests_)
            idx = np.arange(ntests_)
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

    def get_type_map(self) -> List[str]:
        """Get the type map."""
        return self.type_map

    def get_atom_type(self) -> List[int]:
        """Get atom types."""
        return self.atom_type

    def get_numb_set(self) -> int:
        """Get number of training sets."""
        return len(self.train_dirs)

    def get_numb_batch(self, batch_size: int, set_idx: int) -> int:
        """Get the number of batches in a set."""
        data = self._load_set(self.train_dirs[set_idx])
        ret = data["coord"].shape[0] // batch_size
        if ret == 0:
            ret = 1
        return ret

    def get_sys_numb_batch(self, batch_size: int) -> int:
        """Get the number of batches in the data system."""
        ret = 0
        for ii in range(len(self.train_dirs)):
            ret += self.get_numb_batch(batch_size, ii)
        return ret

    def get_natoms(self):
        """Get number of atoms."""
        return len(self.atom_type)

    def get_natoms_vec(self, ntypes: int):
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

    def avg(self, key):
        """Return the average value of an item."""
        if key not in self.data_dict.keys():
            raise RuntimeError("key %s has not been added" % key)
        info = self.data_dict[key]
        ndof = info["ndof"]
        eners = []
        for ii in self.train_dirs:
            data = self._load_set(ii)
            ei = data[key].reshape([-1, ndof])
            eners.append(ei)
        eners = np.concatenate(eners, axis=0)
        if eners.size == 0:
            return 0
        else:
            return np.average(eners, axis=0)

    def _idx_map_sel(self, atom_type, type_sel):
        new_types = []
        for ii in atom_type:
            if ii in type_sel:
                new_types.append(ii)
        new_types = np.array(new_types, dtype=int)
        natoms = new_types.shape[0]
        idx = np.arange(natoms)
        idx_map = np.lexsort((idx, new_types))
        return idx_map

    def _get_natoms_2(self, ntypes):
        sample_type = self.atom_type
        natoms = len(sample_type)
        natoms_vec = np.zeros(ntypes).astype(int)
        for ii in range(ntypes):
            natoms_vec[ii] = np.count_nonzero(sample_type == ii)
        return natoms, natoms_vec

    def _get_subdata(self, data, idx=None):
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

    def _load_batch_set(self, set_name: DPPath):
        if not hasattr(self, "batch_set") or self.get_numb_set() > 1:
            self.batch_set = self._load_set(set_name)
            if self.modifier is not None:
                self.modifier.modify_data(self.batch_set, self)
        self.batch_set, _ = self._shuffle_data(self.batch_set)
        self.reset_get_batch()

    def reset_get_batch(self):
        self.iterator = 0

    def _load_test_set(self, set_name: DPPath, shuffle_test):
        self.test_set = self._load_set(set_name)
        if shuffle_test:
            self.test_set, _ = self._shuffle_data(self.test_set)

    def _shuffle_data(self, data):
        ret = {}
        nframes = data["coord"].shape[0]
        idx = np.arange(nframes)
        # the training times of each frame
        idx = np.repeat(idx, np.reshape(data["numb_copy"], (nframes,)))
        dp_random.shuffle(idx)
        for kk in data:
            if (
                type(data[kk]) == np.ndarray
                and len(data[kk].shape) == 2
                and data[kk].shape[0] == nframes
                and "find_" not in kk
            ):
                ret[kk] = data[kk][idx]
            else:
                ret[kk] = data[kk]
        return ret, idx

    def _get_nframes(self, set_name: DPPath):
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
        return nframes

    def reformat_data_torch(self, data):
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
                if self.data_dict[kk]["atomic"]:
                    data[kk] = data[kk].reshape(-1, self.data_dict[kk]["ndof"])
        data["atype"] = data["type"]
        if not self.pbc:
            data["box"] = None
        return data

    def _load_set(self, set_name: DPPath):
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
                        "some types in 'real_atom_types.npy' of set {} are not contained in {} types!".format(
                            set_name, self.get_ntypes()
                        )
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
            ).all(), "some types in 'real_atom_types.npy' of set {} are not contained in {} types!".format(
                set_name, self.get_ntypes()
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

        return data

    def _load_data(
        self,
        set_name,
        key,
        nframes,
        ndof_,
        atomic=False,
        must=True,
        repeat=1,
        high_prec=False,
        type_sel=None,
        default: float = 0.0,
        dtype: Optional[np.dtype] = None,
        output_natoms_for_type_sel: bool = False,
    ):
        if atomic:
            natoms = self.natoms
            idx_map = self.idx_map
            # if type_sel, then revise natoms and idx_map
            if type_sel is not None:
                natoms_sel = 0
                for jj in type_sel:
                    natoms_sel += np.sum(self.atom_type == jj)
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
                                sel_mask = np.isin(self.atom_type, type_sel)
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
                                sel_mask = np.isin(self.atom_type, type_sel)
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
                    data = data.reshape([nframes, natoms, -1])
                    data = data[:, idx_map, :]
                    data = data.reshape([nframes, -1])
                data = np.reshape(data, [nframes, ndof])
            except ValueError as err_message:
                explanation = "This error may occur when your label mismatch it's name, i.e. you might store global tensor in `atomic_tensor.npy` or atomic tensor in `tensor.npy`."
                log.error(str(err_message))
                log.error(explanation)
                raise ValueError(str(err_message) + ". " + explanation) from err_message
            if repeat != 1:
                data = np.repeat(data, repeat).reshape([nframes, -1])
            return np.float32(1.0), data
        elif must:
            raise RuntimeError("%s not found!" % path)
        else:
            if type_sel is not None and not output_natoms_for_type_sel:
                ndof = ndof_ * natoms_sel
            data = np.full([nframes, ndof], default, dtype=dtype)
            if repeat != 1:
                data = np.repeat(data, repeat).reshape([nframes, -1])
            return np.float32(0.0), data

    def _load_type(self, sys_path: DPPath):
        atom_type = (sys_path / "type.raw").load_txt(ndmin=1).astype(np.int32)
        return atom_type

    def _load_type_mix(self, set_name: DPPath):
        type_path = set_name / "real_atom_types.npy"
        real_type = type_path.load_numpy().astype(np.int32).reshape([-1, self.natoms])
        return real_type

    def _make_idx_map(self, atom_type):
        natoms = atom_type.shape[0]
        idx = np.arange(natoms)
        if self.sort_atoms:
            idx_map = np.lexsort((idx, atom_type))
        else:
            idx_map = idx
        return idx_map

    def _load_type_map(self, sys_path: DPPath):
        fname = sys_path / "type_map.raw"
        if fname.is_file():
            return fname.load_txt(dtype=str, ndmin=1).tolist()
        else:
            return None

    def _check_pbc(self, sys_path: DPPath):
        pbc = True
        if (sys_path / "nopbc").is_file():
            pbc = False
        return pbc

    def _check_mode(self, set_path: DPPath):
        return (set_path / "real_atom_types.npy").is_file()


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
        type_sel: Optional[List[int]] = None,
        repeat: int = 1,
        default: float = 0.0,
        dtype: Optional[np.dtype] = None,
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

    def __getitem__(self, key: str):
        if key not in self.dict:
            raise KeyError(key)
        return self.dict[key]
