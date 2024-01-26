# SPDX-License-Identifier: LGPL-3.0-or-later
import glob
import os
from typing import (
    List,
    Optional,
)

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import (
    Dataset,
)
from tqdm import (
    trange,
)

from deepmd.pt.utils import (
    dp_random,
    env,
)
from deepmd.pt.utils.cache import (
    lru_cache,
)
from deepmd.pt.utils.preprocess import (
    Region3D,
    make_env_mat,
    normalize_coord,
)


class DeepmdDataSystem:
    def __init__(
        self,
        sys_path: str,
        rcut,
        sec,
        type_map: Optional[List[str]] = None,
        type_split=True,
        noise_settings=None,
        shuffle=True,
    ):
        """Construct DeePMD-style frame collection of one system.

        Args:
        - sys_path: Paths to the system.
        - type_map: Atom types.
        """
        sys_path = sys_path.replace("#", "")
        if ".hdf5" in sys_path:
            tmp = sys_path.split("/")
            path = "/".join(tmp[:-1])
            sys = tmp[-1]
            self.file = h5py.File(path)[sys]
            self._dirs = []
            for item in self.file.keys():
                if "set." in item:
                    self._dirs.append(item)
            self._dirs.sort()
        else:
            self.file = None
            self._dirs = glob.glob(os.path.join(sys_path, "set.*"))
            self._dirs.sort()
        self.type_split = type_split
        self.noise_settings = noise_settings
        self._check_pbc(sys_path)
        self.shuffle = shuffle
        if noise_settings is not None:
            self.noise_type = noise_settings.get("noise_type", "uniform")
            self.noise = float(noise_settings.get("noise", 1.0))
            self.noise_mode = noise_settings.get("noise_mode", "fix_num")
            self.mask_num = int(noise_settings.get("mask_num", 1))
            self.mask_prob = float(noise_settings.get("mask_prob", 0.15))
            self.same_mask = noise_settings.get("same_mask", False)
            self.mask_coord = noise_settings.get("mask_coord", False)
            self.mask_type = noise_settings.get("mask_type", False)
            self.mask_type_idx = int(noise_settings.get("mask_type_idx", 0))
            self.max_fail_num = int(noise_settings.get("max_fail_num", 10))

        # check mixed type
        error_format_msg = (
            "if one of the set is of mixed_type format, "
            "then all of the sets in this system should be of mixed_type format!"
        )
        if len(self._dirs) == 0:
            raise RuntimeError(f"No set found in system {sys_path}.")

        self.mixed_type = self._check_mode(self._dirs[0])
        for set_item in self._dirs[1:]:
            assert self._check_mode(set_item) == self.mixed_type, error_format_msg

        self._atom_type = self._load_type(sys_path)
        self._natoms = len(self._atom_type)

        self._type_map = self._load_type_map(sys_path)
        self.enforce_type_map = False
        if type_map is not None and self._type_map is not None:
            if not self.mixed_type:
                atom_type = [
                    type_map.index(self._type_map[ii]) for ii in self._atom_type
                ]
                self._atom_type = np.array(atom_type, dtype=np.int32)

            else:
                self.enforce_type_map = True
                sorter = np.argsort(type_map)
                self.type_idx_map = np.array(
                    sorter[np.searchsorted(type_map, self._type_map, sorter=sorter)]
                )
                # padding for virtual atom
                self.type_idx_map = np.append(
                    self.type_idx_map, np.array([-1], dtype=np.int32)
                )
            self._type_map = type_map
        if type_map is None and self.type_map is None and self.mixed_type:
            raise RuntimeError("mixed_type format must have type_map!")
        self._idx_map = _make_idx_map(self._atom_type)

        self._data_dict = {}
        self.add("box", 9, must=self.pbc)
        self.add("coord", 3, atomic=True, must=True)
        self.add("energy", 1, atomic=False, must=False, high_prec=True)
        self.add("force", 3, atomic=True, must=False, high_prec=False)
        self.add("virial", 9, atomic=False, must=False, high_prec=False)

        self._sys_path = sys_path
        self.rcut = rcut
        self.sec = sec
        if isinstance(rcut, float):
            self.hybrid = False
        elif isinstance(rcut, list):
            self.hybrid = True
        else:
            RuntimeError("Unkown rcut type!")
        self.sets = [None for i in range(len(self._sys_path))]

        self.nframes = 0
        i = 1
        self.prefix_sum = [0] * (len(self._dirs) + 1)
        for item in self._dirs:
            frames = self._load_set(item, fast=True)
            self.prefix_sum[i] = self.prefix_sum[i - 1] + frames
            i += 1
            self.nframes += frames

    def _check_pbc(self, sys_path):
        pbc = True
        if os.path.isfile(os.path.join(sys_path, "nopbc")):
            pbc = False
        self.pbc = pbc

    def set_noise(self, noise_settings):
        # noise_settings['noise_type'] # "trunc_normal", "normal", "uniform"
        # noise_settings['noise'] # float, default 1.0
        # noise_settings['noise_mode'] # "prob", "fix_num"
        # noise_settings['mask_num'] # if "fix_num", int
        # noise_settings['mask_prob'] # if "prob", float
        # noise_settings['same_mask'] # coord and type same mask?
        self.noise_settings = noise_settings
        self.noise_type = noise_settings.get("noise_type", "uniform")
        self.noise = float(noise_settings.get("noise", 1.0))
        self.noise_mode = noise_settings.get("noise_mode", "fix_num")
        self.mask_num = int(noise_settings.get("mask_num", 1))
        self.mask_coord = noise_settings.get("mask_coord", False)
        self.mask_type = noise_settings.get("mask_type", False)
        self.mask_prob = float(noise_settings.get("mask_prob", 0.15))
        self.same_mask = noise_settings.get("noise_type", False)

    def add(
        self,
        key: str,
        ndof: int,
        atomic: bool = False,
        must: bool = False,
        high_prec: bool = False,
    ):
        """Add a data item that to be loaded.

        Args:
        - key: The key of the item. The corresponding data is stored in `sys_path/set.*/key.npy`
        - ndof: The number of dof
        - atomic: The item is an atomic property.
        - must: The data file `sys_path/set.*/key.npy` must exist. Otherwise, value is set to zero.
        - high_prec: Load the data and store in float64, otherwise in float32.
        """
        self._data_dict[key] = {
            "ndof": ndof,
            "atomic": atomic,
            "must": must,
            "high_prec": high_prec,
        }

    # deprecated TODO
    def get_batch_for_train(self, batch_size: int):
        """Get a batch of data with at most `batch_size` frames. The frames are randomly picked from the data system.

        Args:
        - batch_size: Frame count.
        """
        if not hasattr(self, "_frames"):
            self.set_size = 0
            self._set_count = 0
            self._iterator = 0
        if batch_size == "auto":
            batch_size = -(-32 // self._natoms)
        if self._iterator + batch_size > self.set_size:
            set_idx = self._set_count % len(self._dirs)
            if self.sets[set_idx] is None:
                frames = self._load_set(self._dirs[set_idx])
                frames = self.preprocess(frames)
                cnt = 0
                for item in self.sets:
                    if item is not None:
                        cnt += 1
                if cnt < env.CACHE_PER_SYS:
                    self.sets[set_idx] = frames
            else:
                frames = self.sets[set_idx]
            self._frames = frames
            self._shuffle_data()
            if dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                ssize = self._frames["coord"].shape[0]
                subsize = ssize // world_size
                self._iterator = rank * subsize
                self.set_size = min((rank + 1) * subsize, ssize)
            else:
                self.set_size = self._frames["coord"].shape[0]
                self._iterator = 0
            self._set_count += 1
        iterator = min(self._iterator + batch_size, self.set_size)
        idx = np.arange(self._iterator, iterator)
        self._iterator += batch_size
        return self._get_subdata(idx)

    # deprecated TODO
    def get_batch(self, batch_size: int):
        """Get a batch of data with at most `batch_size` frames. The frames are randomly picked from the data system.
        Args:
        - batch_size: Frame count.
        """
        if not hasattr(self, "_frames"):
            self.set_size = 0
            self._set_count = 0
            self._iterator = 0
        if batch_size == "auto":
            batch_size = -(-32 // self._natoms)
        if self._iterator + batch_size > self.set_size:
            set_idx = self._set_count % len(self._dirs)
            if self.sets[set_idx] is None:
                frames = self._load_set(self._dirs[set_idx])
                frames = self.preprocess(frames)
                cnt = 0
                for item in self.sets:
                    if item is not None:
                        cnt += 1
                if cnt < env.CACHE_PER_SYS:
                    self.sets[set_idx] = frames
            else:
                frames = self.sets[set_idx]
            self._frames = frames
            self._shuffle_data()
            self.set_size = self._frames["coord"].shape[0]
            self._iterator = 0
            self._set_count += 1
        iterator = min(self._iterator + batch_size, self.set_size)
        idx = np.arange(self._iterator, iterator)
        self._iterator += batch_size
        return self._get_subdata(idx)

    def get_ntypes(self):
        """Number of atom types in the system."""
        if self._type_map is not None:
            return len(self._type_map)
        else:
            return max(self._atom_type) + 1

    def get_natoms_vec(self, ntypes: int):
        """Get number of atoms and number of atoms in different types.

        Args:
        - ntypes: Number of types (may be larger than the actual number of types in the system).
        """
        natoms = len(self._atom_type)
        natoms_vec = np.zeros(ntypes).astype(int)
        for ii in range(ntypes):
            natoms_vec[ii] = np.count_nonzero(self._atom_type == ii)
        tmp = [natoms, natoms]
        tmp = np.append(tmp, natoms_vec)
        return tmp.astype(np.int32)

    def _load_type(self, sys_path):
        if self.file is not None:
            return self.file["type.raw"][:]
        else:
            return np.loadtxt(
                os.path.join(sys_path, "type.raw"), dtype=np.int32, ndmin=1
            )

    def _load_type_map(self, sys_path):
        if self.file is not None:
            tmp = self.file["type_map.raw"][:].tolist()
            tmp = [item.decode("ascii") for item in tmp]
            return tmp
        else:
            fname = os.path.join(sys_path, "type_map.raw")
            if os.path.isfile(fname):
                with open(fname) as fin:
                    content = fin.read()
                return content.split()
            else:
                return None

    def _check_mode(self, sys_path):
        return os.path.isfile(sys_path + "/real_atom_types.npy")

    def _load_type_mix(self, set_name):
        type_path = set_name + "/real_atom_types.npy"
        real_type = np.load(type_path).astype(np.int32).reshape([-1, self._natoms])
        return real_type

    @lru_cache(maxsize=16, copy=True)
    def _load_set(self, set_name, fast=False):
        if self.file is None:
            path = os.path.join(set_name, "coord.npy")
            if self._data_dict["coord"]["high_prec"]:
                coord = np.load(path).astype(env.GLOBAL_ENER_FLOAT_PRECISION)
            else:
                coord = np.load(path).astype(env.GLOBAL_NP_FLOAT_PRECISION)
            if coord.ndim == 1:
                coord = coord.reshape([1, -1])
            assert coord.shape[1] == self._data_dict["coord"]["ndof"] * self._natoms
            nframes = coord.shape[0]
            if fast:
                return nframes
            data = {"type": np.tile(self._atom_type[self._idx_map], (nframes, 1))}
            for kk in self._data_dict.keys():
                data["find_" + kk], data[kk] = self._load_data(
                    set_name,
                    kk,
                    nframes,
                    self._data_dict[kk]["ndof"],
                    atomic=self._data_dict[kk]["atomic"],
                    high_prec=self._data_dict[kk]["high_prec"],
                    must=self._data_dict[kk]["must"],
                )
            if self.mixed_type:
                # nframes x natoms
                atom_type_mix = self._load_type_mix(set_name)
                if self.enforce_type_map:
                    try:
                        atom_type_mix_ = self.type_idx_map[atom_type_mix].astype(
                            np.int32
                        )
                    except IndexError as e:
                        raise IndexError(
                            "some types in 'real_atom_types.npy' of set {} are not contained in {} types!".format(
                                set_name, self.get_ntypes()
                            )
                        ) from e
                    atom_type_mix = atom_type_mix_
                real_type = atom_type_mix.reshape([nframes, self._natoms])
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
                        np.tile(
                            np.array([natoms, natoms], dtype=np.int32), (nframes, 1)
                        ),
                        atom_type_nums,
                    ),
                    axis=-1,
                )

            return data
        else:
            data = {}
            nframes = self.file[set_name]["coord.npy"].shape[0]
            if fast:
                return nframes
            for key in ["coord", "energy", "force", "box"]:
                data[key] = self.file[set_name][f"{key}.npy"][:]
                if self._data_dict[key]["atomic"]:
                    data[key] = data[key].reshape(nframes, self._natoms, -1)[
                        :, self._idx_map, :
                    ]
            if self.mixed_type:
                # nframes x natoms
                atom_type_mix = self._load_type_mix(set_name)
                if self.enforce_type_map:
                    try:
                        atom_type_mix_ = self.type_idx_map[atom_type_mix].astype(
                            np.int32
                        )
                    except IndexError as e:
                        raise IndexError(
                            "some types in 'real_atom_types.npy' of set {} are not contained in {} types!".format(
                                set_name, self.get_ntypes()
                            )
                        ) from e
                    atom_type_mix = atom_type_mix_
                real_type = atom_type_mix.reshape([nframes, self._natoms])
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
                        np.tile(
                            np.array([natoms, natoms], dtype=np.int32), (nframes, 1)
                        ),
                        atom_type_nums,
                    ),
                    axis=-1,
                )
            else:
                data["type"] = np.tile(self._atom_type[self._idx_map], (nframes, 1))
            return data

    def _load_data(
        self, set_name, key, nframes, ndof, atomic=False, must=True, high_prec=False
    ):
        if atomic:
            ndof *= self._natoms
        path = os.path.join(set_name, key + ".npy")
        # logging.info('Loading data from: %s', path)
        if os.path.isfile(path):
            if high_prec:
                data = np.load(path).astype(env.GLOBAL_ENER_FLOAT_PRECISION)
            else:
                data = np.load(path).astype(env.GLOBAL_NP_FLOAT_PRECISION)
            if atomic:
                data = data.reshape([nframes, self._natoms, -1])
                data = data[:, self._idx_map, :]
                data = data.reshape([nframes, -1])
            data = np.reshape(data, [nframes, ndof])
            return np.float32(1.0), data
        elif must:
            raise RuntimeError("%s not found!" % path)
        else:
            if high_prec:
                data = np.zeros([nframes, ndof]).astype(env.GLOBAL_ENER_FLOAT_PRECISION)
            else:
                data = np.zeros([nframes, ndof]).astype(env.GLOBAL_NP_FLOAT_PRECISION)
            return np.float32(0.0), data

    # deprecated TODO
    def preprocess(self, batch):
        n_frames = batch["coord"].shape[0]
        for kk in self._data_dict.keys():
            if "find_" in kk:
                pass
            else:
                batch[kk] = torch.tensor(
                    batch[kk],
                    dtype=env.GLOBAL_PT_FLOAT_PRECISION,
                    device=env.PREPROCESS_DEVICE,
                )
                if self._data_dict[kk]["atomic"]:
                    batch[kk] = batch[kk].view(
                        n_frames, -1, self._data_dict[kk]["ndof"]
                    )

        for kk in ["type", "real_natoms_vec"]:
            if kk in batch.keys():
                batch[kk] = torch.tensor(
                    batch[kk], dtype=torch.long, device=env.PREPROCESS_DEVICE
                )
        batch["atype"] = batch.pop("type")

        keys = ["nlist", "nlist_loc", "nlist_type", "shift", "mapping"]
        coord = batch["coord"]
        atype = batch["atype"]
        box = batch["box"]
        rcut = self.rcut
        sec = self.sec
        assert batch["atype"].max() < len(self._type_map)
        nlist, nlist_loc, nlist_type, shift, mapping = [], [], [], [], []

        for sid in trange(n_frames, disable=env.DISABLE_TQDM):
            region = Region3D(box[sid])
            nloc = atype[sid].shape[0]
            _coord = normalize_coord(coord[sid], region, nloc)
            coord[sid] = _coord
            a, b, c, d, e = make_env_mat(
                _coord, atype[sid], region, rcut, sec, type_split=self.type_split
            )
            nlist.append(a)
            nlist_loc.append(b)
            nlist_type.append(c)
            shift.append(d)
            mapping.append(e)
        nlist = torch.stack(nlist)
        nlist_loc = torch.stack(nlist_loc)
        nlist_type = torch.stack(nlist_type)
        batch["nlist"] = nlist
        batch["nlist_loc"] = nlist_loc
        batch["nlist_type"] = nlist_type
        natoms_extended = max([item.shape[0] for item in shift])
        batch["shift"] = torch.zeros(
            (n_frames, natoms_extended, 3),
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.PREPROCESS_DEVICE,
        )
        batch["mapping"] = torch.zeros(
            (n_frames, natoms_extended), dtype=torch.long, device=env.PREPROCESS_DEVICE
        )
        for i in range(len(shift)):
            natoms_tmp = shift[i].shape[0]
            batch["shift"][i, :natoms_tmp] = shift[i]
            batch["mapping"][i, :natoms_tmp] = mapping[i]
        return batch

    def _shuffle_data(self):
        nframes = self._frames["coord"].shape[0]
        idx = np.arange(nframes)
        if self.shuffle:
            dp_random.shuffle(idx)
        self.idx_mapping = idx

    def _get_subdata(self, idx=None):
        data = self._frames
        idx = self.idx_mapping[idx]
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

    # note: this function needs to be optimized for single frame process
    def single_preprocess(self, batch, sid):
        for kk in self._data_dict.keys():
            if "find_" in kk:
                pass
            else:
                batch[kk] = torch.tensor(
                    batch[kk][sid],
                    dtype=env.GLOBAL_PT_FLOAT_PRECISION,
                    device=env.PREPROCESS_DEVICE,
                )
                if self._data_dict[kk]["atomic"]:
                    batch[kk] = batch[kk].view(-1, self._data_dict[kk]["ndof"])
        for kk in ["type", "real_natoms_vec"]:
            if kk in batch.keys():
                batch[kk] = torch.tensor(
                    batch[kk][sid], dtype=torch.long, device=env.PREPROCESS_DEVICE
                )
        clean_coord = batch.pop("coord")
        clean_type = batch.pop("type")
        nloc = clean_type.shape[0]
        rcut = self.rcut
        sec = self.sec
        nlist, nlist_loc, nlist_type, shift, mapping = [], [], [], [], []
        if self.pbc:
            box = batch["box"]
            region = Region3D(box)
        else:
            box = None
            batch["box"] = None
            region = None
        if self.noise_settings is None:
            batch["atype"] = clean_type
            batch["coord"] = clean_coord
            coord = clean_coord
            atype = batch["atype"]
            if self.pbc:
                _coord = normalize_coord(coord, region, nloc)

            else:
                _coord = coord.clone()
            batch["coord"] = _coord
            nlist, nlist_loc, nlist_type, shift, mapping = make_env_mat(
                _coord,
                atype,
                region,
                rcut,
                sec,
                pbc=self.pbc,
                type_split=self.type_split,
            )
            batch["nlist"] = nlist
            batch["nlist_loc"] = nlist_loc
            batch["nlist_type"] = nlist_type
            batch["shift"] = shift
            batch["mapping"] = mapping
            return batch
        else:
            batch["clean_type"] = clean_type
            if self.pbc:
                _clean_coord = normalize_coord(clean_coord, region, nloc)
            else:
                _clean_coord = clean_coord.clone()
            batch["clean_coord"] = _clean_coord
            # add noise
            for i in range(self.max_fail_num):
                mask_num = 0
                if self.noise_mode == "fix_num":
                    mask_num = self.mask_num
                    if len(batch["clean_type"]) < mask_num:
                        mask_num = len(batch["clean_type"])
                elif self.noise_mode == "prob":
                    mask_num = int(self.mask_prob * nloc)
                    if mask_num == 0:
                        mask_num = 1
                else:
                    NotImplementedError(f"Unknown noise mode {self.noise_mode}!")
                rng = np.random.default_rng()
                coord_mask_res = rng.choice(
                    range(nloc), mask_num, replace=False
                ).tolist()
                coord_mask = np.isin(range(nloc), coord_mask_res)
                if self.same_mask:
                    type_mask = coord_mask.copy()
                else:
                    rng = np.random.default_rng()
                    type_mask_res = rng.choice(
                        range(nloc), mask_num, replace=False
                    ).tolist()
                    type_mask = np.isin(range(nloc), type_mask_res)

                # add noise for coord
                if self.mask_coord:
                    noise_on_coord = 0.0
                    rng = np.random.default_rng()
                    if self.noise_type == "trunc_normal":
                        noise_on_coord = np.clip(
                            rng.standard_normal((mask_num, 3)) * self.noise,
                            a_min=-self.noise * 2.0,
                            a_max=self.noise * 2.0,
                        )
                    elif self.noise_type == "normal":
                        noise_on_coord = rng.standard_normal((mask_num, 3)) * self.noise
                    elif self.noise_type == "uniform":
                        noise_on_coord = rng.uniform(
                            low=-self.noise, high=self.noise, size=(mask_num, 3)
                        )
                    else:
                        NotImplementedError(f"Unknown noise type {self.noise_type}!")
                    noised_coord = _clean_coord.clone().detach()
                    noised_coord[coord_mask] += noise_on_coord
                    batch["coord_mask"] = torch.tensor(
                        coord_mask, dtype=torch.bool, device=env.PREPROCESS_DEVICE
                    )
                else:
                    noised_coord = _clean_coord
                    batch["coord_mask"] = torch.tensor(
                        np.zeros_like(coord_mask, dtype=bool),
                        dtype=torch.bool,
                        device=env.PREPROCESS_DEVICE,
                    )

                # add mask for type
                if self.mask_type:
                    masked_type = clean_type.clone().detach()
                    masked_type[type_mask] = self.mask_type_idx
                    batch["type_mask"] = torch.tensor(
                        type_mask, dtype=torch.bool, device=env.PREPROCESS_DEVICE
                    )
                else:
                    masked_type = clean_type
                    batch["type_mask"] = torch.tensor(
                        np.zeros_like(type_mask, dtype=bool),
                        dtype=torch.bool,
                        device=env.PREPROCESS_DEVICE,
                    )
                if self.pbc:
                    _coord = normalize_coord(noised_coord, region, nloc)
                else:
                    _coord = noised_coord.clone()
                try:
                    nlist, nlist_loc, nlist_type, shift, mapping = make_env_mat(
                        _coord,
                        masked_type,
                        region,
                        rcut,
                        sec,
                        pbc=self.pbc,
                        type_split=self.type_split,
                        min_check=True,
                    )
                except RuntimeError as e:
                    if i == self.max_fail_num - 1:
                        RuntimeError(
                            f"Add noise times beyond max tries {self.max_fail_num}!"
                        )
                    continue
                batch["atype"] = masked_type
                batch["coord"] = noised_coord
                batch["nlist"] = nlist
                batch["nlist_loc"] = nlist_loc
                batch["nlist_type"] = nlist_type
                batch["shift"] = shift
                batch["mapping"] = mapping
                return batch

    def _get_item(self, index):
        for i in range(
            0, len(self._dirs) + 1
        ):  # note: if different sets can be merged, prefix sum is unused to calculate
            if index < self.prefix_sum[i]:
                break
        frames = self._load_set(self._dirs[i - 1])
        frame = self.single_preprocess(frames, index - self.prefix_sum[i - 1])
        frame["fid"] = index
        return frame


def _make_idx_map(atom_type):
    natoms = atom_type.shape[0]
    idx = np.arange(natoms)
    idx_map = np.lexsort((idx, atom_type))
    return idx_map


class DeepmdDataSetForLoader(Dataset):
    def __init__(
        self,
        system: str,
        type_map: str,
        rcut,
        sel,
        weight=None,
        type_split=True,
        noise_settings=None,
        shuffle=True,
    ):
        """Construct DeePMD-style dataset containing frames cross different systems.

        Args:
        - systems: Paths to systems.
        - batch_size: Max frame count in a batch.
        - type_map: Atom types.
        """
        self._type_map = type_map
        if not isinstance(rcut, list):
            if isinstance(sel, int):
                sel = [sel]
            sec = torch.cumsum(torch.tensor(sel), dim=0)
        else:
            sec = []
            for sel_item in sel:
                if isinstance(sel_item, int):
                    sel_item = [sel_item]
                sec.append(torch.cumsum(torch.tensor(sel_item), dim=0))
        self._data_system = DeepmdDataSystem(
            system,
            rcut,
            sec,
            type_map=self._type_map,
            type_split=type_split,
            noise_settings=noise_settings,
            shuffle=shuffle,
        )
        self.mixed_type = self._data_system.mixed_type
        self._ntypes = self._data_system.get_ntypes()
        self._natoms = self._data_system._natoms
        self._natoms_vec = self._data_system.get_natoms_vec(self._ntypes)

    def set_noise(self, noise_settings):
        # noise_settings['noise_type'] # "trunc_normal", "normal", "uniform"
        # noise_settings['noise'] # float, default 1.0
        # noise_settings['noise_mode'] # "prob", "fix_num"
        # noise_settings['mask_num'] # if "fix_num", int
        # noise_settings['mask_prob'] # if "prob", float
        # noise_settings['same_mask'] # coord and type same mask?
        self._data_system.set_noise(noise_settings)

    def __len__(self):
        return self._data_system.nframes

    def __getitem__(self, index):
        """Get a frame from the selected system."""
        b_data = self._data_system._get_item(index)
        b_data["natoms"] = torch.tensor(self._natoms_vec, device=env.PREPROCESS_DEVICE)
        return b_data


# deprecated TODO
class DeepmdDataSet(Dataset):
    def __init__(
        self,
        systems: List[str],
        batch_size: int,
        type_map: List[str],
        rcut=None,
        sel=None,
        weight=None,
        type_split=True,
    ):
        """Construct DeePMD-style dataset containing frames cross different systems.

        Args:
        - systems: Paths to systems.
        - batch_size: Max frame count in a batch.
        - type_map: Atom types.
        """
        self._batch_size = batch_size
        self._type_map = type_map
        if sel is not None:
            if isinstance(sel, int):
                sel = [sel]
            sec = torch.cumsum(torch.tensor(sel), dim=0)
        if isinstance(systems, str):
            with h5py.File(systems) as file:
                systems = [os.path.join(systems, item) for item in file.keys()]
        self._data_systems = [
            DeepmdDataSystem(
                ii, rcut, sec, type_map=self._type_map, type_split=type_split
            )
            for ii in systems
        ]
        # check mix_type format
        error_format_msg = (
            "if one of the system is of mixed_type format, "
            "then all of the systems in this dataset should be of mixed_type format!"
        )
        self.mixed_type = self._data_systems[0].mixed_type
        for sys_item in self._data_systems[1:]:
            assert sys_item.mixed_type == self.mixed_type, error_format_msg

        if weight is None:

            def weight(name, sys):
                return sys.nframes

        self.probs = [
            weight(item, self._data_systems[i]) for i, item in enumerate(systems)
        ]
        self.probs = np.array(self.probs, dtype=float)
        self.probs /= self.probs.sum()
        self._ntypes = max([ii.get_ntypes() for ii in self._data_systems])
        self._natoms_vec = [
            ii.get_natoms_vec(self._ntypes) for ii in self._data_systems
        ]
        self.cache = [{} for sys in self._data_systems]

    @property
    def nsystems(self):
        return len(self._data_systems)

    def __len__(self):
        return self.nsystems

    def __getitem__(self, index=None):
        """Get a batch of frames from the selected system."""
        if index is None:
            index = dp_random.choice(np.arange(self.nsystems), self.probs)
        b_data = self._data_systems[index].get_batch(self._batch_size)
        b_data["natoms"] = torch.tensor(
            self._natoms_vec[index], device=env.PREPROCESS_DEVICE
        )
        batch_size = b_data["coord"].shape[0]
        b_data["natoms"] = b_data["natoms"].unsqueeze(0).expand(batch_size, -1)
        return b_data

    # deprecated TODO
    def get_training_batch(self, index=None):
        """Get a batch of frames from the selected system."""
        if index is None:
            index = dp_random.choice(np.arange(self.nsystems), self.probs)
        b_data = self._data_systems[index].get_batch_for_train(self._batch_size)
        b_data["natoms"] = torch.tensor(
            self._natoms_vec[index], device=env.PREPROCESS_DEVICE
        )
        batch_size = b_data["coord"].shape[0]
        b_data["natoms"] = b_data["natoms"].unsqueeze(0).expand(batch_size, -1)
        return b_data

    def get_batch(self, sys_idx=None):
        """TF-compatible batch for testing."""
        pt_batch = self[sys_idx]
        np_batch = {}
        for key in ["coord", "box", "force", "energy", "virial"]:
            if key in pt_batch.keys():
                np_batch[key] = pt_batch[key].cpu().numpy()
        for key in ["atype", "natoms"]:
            if key in pt_batch.keys():
                np_batch[key] = pt_batch[key].cpu().numpy()
        batch_size = pt_batch["coord"].shape[0]
        np_batch["coord"] = np_batch["coord"].reshape(batch_size, -1)
        np_batch["natoms"] = np_batch["natoms"][0]
        np_batch["force"] = np_batch["force"].reshape(batch_size, -1)
        return np_batch, pt_batch
