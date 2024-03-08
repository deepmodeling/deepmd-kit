# SPDX-License-Identifier: LGPL-3.0-or-later
import os
from abc import (
    ABC,
    abstractmethod,
)
from functools import (
    lru_cache,
)
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
)

import h5py
import numpy as np
from wcmatch.glob import (
    globfilter,
)


class DPPath(ABC):
    """The path class to data system (DeepmdData).

    Parameters
    ----------
    path : str
        path
    mode : str, optional
        mode, by default "r"
    """

    def __new__(cls, path: str, mode: str = "r"):
        if cls is DPPath:
            if os.path.isdir(path):
                return super().__new__(DPOSPath)
            elif os.path.isfile(path.split("#")[0]):
                # assume h5 if it is not dir
                # TODO: check if it is a real h5? or just check suffix?
                return super().__new__(DPH5Path)
            raise FileNotFoundError("%s not found" % path)
        return super().__new__(cls)

    @abstractmethod
    def load_numpy(self) -> np.ndarray:
        """Load NumPy array.

        Returns
        -------
        np.ndarray
            loaded NumPy array
        """

    @abstractmethod
    def load_txt(self, **kwargs) -> np.ndarray:
        """Load NumPy array from text.

        Returns
        -------
        np.ndarray
            loaded NumPy array
        """

    @abstractmethod
    def save_numpy(self, arr: np.ndarray) -> None:
        """Save NumPy array.

        Parameters
        ----------
        arr : np.ndarray
            NumPy array
        """

    @abstractmethod
    def glob(self, pattern: str) -> List["DPPath"]:
        """Search path using the glob pattern.

        Parameters
        ----------
        pattern : str
            glob pattern

        Returns
        -------
        List[DPPath]
            list of paths
        """

    @abstractmethod
    def rglob(self, pattern: str) -> List["DPPath"]:
        """This is like calling :meth:`DPPath.glob()` with `**/` added in front
        of the given relative pattern.

        Parameters
        ----------
        pattern : str
            glob pattern

        Returns
        -------
        List[DPPath]
            list of paths
        """

    @abstractmethod
    def is_file(self) -> bool:
        """Check if self is file."""

    @abstractmethod
    def is_dir(self) -> bool:
        """Check if self is directory."""

    @abstractmethod
    def __truediv__(self, key: str) -> "DPPath":
        """Used for / operator."""

    @abstractmethod
    def __lt__(self, other: "DPPath") -> bool:
        """Whether this DPPath is less than other for sorting."""

    @abstractmethod
    def __str__(self) -> str:
        """Represent string."""

    def __repr__(self) -> str:
        return f"{type(self)} ({self!s})"

    def __eq__(self, other) -> bool:
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the path."""

    @abstractmethod
    def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
        """Make directory.

        Parameters
        ----------
        parents : bool, optional
            If true, any missing parents of this directory are created as well.
        exist_ok : bool, optional
            If true, no error will be raised if the target directory already exists.
        """


class DPOSPath(DPPath):
    """The OS path class to data system (DeepmdData) for real directories.

    Parameters
    ----------
    path : str
        path
    mode : str, optional
        mode, by default "r"
    """

    def __init__(self, path: str, mode: str = "r") -> None:
        super().__init__()
        self.mode = mode
        if isinstance(path, Path):
            self.path = path
        else:
            self.path = Path(path)

    def load_numpy(self) -> np.ndarray:
        """Load NumPy array.

        Returns
        -------
        np.ndarray
            loaded NumPy array
        """
        return np.load(str(self.path))

    def load_txt(self, **kwargs) -> np.ndarray:
        """Load NumPy array from text.

        Returns
        -------
        np.ndarray
            loaded NumPy array
        """
        return np.loadtxt(str(self.path), **kwargs)

    def save_numpy(self, arr: np.ndarray) -> None:
        """Save NumPy array.

        Parameters
        ----------
        arr : np.ndarray
            NumPy array
        """
        if self.mode == "r":
            raise ValueError("Cannot save to read-only path")
        np.save(str(self.path), arr)

    def glob(self, pattern: str) -> List["DPPath"]:
        """Search path using the glob pattern.

        Parameters
        ----------
        pattern : str
            glob pattern

        Returns
        -------
        List[DPPath]
            list of paths
        """
        # currently DPOSPath will only derivative DPOSPath
        # TODO: discuss if we want to mix DPOSPath and DPH5Path?
        return [type(self)(p, mode=self.mode) for p in self.path.glob(pattern)]

    def rglob(self, pattern: str) -> List["DPPath"]:
        """This is like calling :meth:`DPPath.glob()` with `**/` added in front
        of the given relative pattern.

        Parameters
        ----------
        pattern : str
            glob pattern

        Returns
        -------
        List[DPPath]
            list of paths
        """
        return [type(self)(p, mode=self.mode) for p in self.path.rglob(pattern)]

    def is_file(self) -> bool:
        """Check if self is file."""
        return self.path.is_file()

    def is_dir(self) -> bool:
        """Check if self is directory."""
        return self.path.is_dir()

    def __truediv__(self, key: str) -> "DPPath":
        """Used for / operator."""
        return type(self)(self.path / key, mode=self.mode)

    def __lt__(self, other: "DPOSPath") -> bool:
        """Whether this DPPath is less than other for sorting."""
        return self.path < other.path

    def __str__(self) -> str:
        """Represent string."""
        return str(self.path)

    @property
    def name(self) -> str:
        """Name of the path."""
        return self.path.name

    def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
        """Make directory.

        Parameters
        ----------
        parents : bool, optional
            If true, any missing parents of this directory are created as well.
        exist_ok : bool, optional
            If true, no error will be raised if the target directory already exists.
        """
        if self.mode == "r":
            raise ValueError("Cannot mkdir to read-only path")
        self.path.mkdir(parents=parents, exist_ok=exist_ok)


class DPH5Path(DPPath):
    """The path class to data system (DeepmdData) for HDF5 files.

    Notes
    -----
    OS - HDF5 relationship:
        directory - Group
        file - Dataset

    Parameters
    ----------
    path : str
        path
    mode : str, optional
        mode, by default "r"
    """

    def __init__(self, path: str, mode: str = "r") -> None:
        super().__init__()
        self.mode = mode
        # we use "#" to split path
        # so we do not support file names containing #...
        s = path.split("#")
        self.root_path = s[0]
        self.root = self._load_h5py(s[0], mode)
        # h5 path: default is the root path
        self._name = s[1] if len(s) > 1 else "/"

    @classmethod
    @lru_cache(None)
    def _load_h5py(cls, path: str, mode: str = "r") -> h5py.File:
        """Load hdf5 file.

        Parameters
        ----------
        path : str
            path to hdf5 file
        mode : str, optional
            mode, by default 'r'
        """
        # this method has cache to avoid duplicated
        # loading from different DPH5Path
        # However the file will be never closed?
        return h5py.File(path, mode)

    def load_numpy(self) -> np.ndarray:
        """Load NumPy array.

        Returns
        -------
        np.ndarray
            loaded NumPy array
        """
        return self.root[self._name][:]

    def load_txt(self, dtype: Optional[np.dtype] = None, **kwargs) -> np.ndarray:
        """Load NumPy array from text.

        Returns
        -------
        np.ndarray
            loaded NumPy array
        """
        arr = self.load_numpy()
        if dtype:
            arr = arr.astype(dtype)
        return arr

    def save_numpy(self, arr: np.ndarray) -> None:
        """Save NumPy array.

        Parameters
        ----------
        arr : np.ndarray
            NumPy array
        """
        if self._name in self._keys:
            del self.root[self._name]
        self.root.create_dataset(self._name, data=arr)
        self.root.flush()

    def glob(self, pattern: str) -> List["DPPath"]:
        """Search path using the glob pattern.

        Parameters
        ----------
        pattern : str
            glob pattern

        Returns
        -------
        List[DPPath]
            list of paths
        """
        # got paths starts with current path first, which is faster
        subpaths = [ii for ii in self._keys if ii.startswith(self._name)]
        return [
            type(self)(f"{self.root_path}#{pp}", mode=self.mode)
            for pp in globfilter(subpaths, self._connect_path(pattern))
        ]

    def rglob(self, pattern: str) -> List["DPPath"]:
        """This is like calling :meth:`DPPath.glob()` with `**/` added in front
        of the given relative pattern.

        Parameters
        ----------
        pattern : str
            glob pattern

        Returns
        -------
        List[DPPath]
            list of paths
        """
        return self.glob("**" + pattern)

    @property
    def _keys(self) -> List[str]:
        """Walk all groups and dataset."""
        return self._file_keys(self.root)

    @classmethod
    @lru_cache(None)
    def _file_keys(cls, file: h5py.File) -> List[str]:
        """Walk all groups and dataset."""
        l = []
        file.visit(lambda x: l.append("/" + x))
        return l

    def is_file(self) -> bool:
        """Check if self is file."""
        if self._name not in self._keys:
            return False
        return isinstance(self.root[self._name], h5py.Dataset)

    def is_dir(self) -> bool:
        """Check if self is directory."""
        if self._name == "/":
            return True
        if self._name not in self._keys:
            return False
        return isinstance(self.root[self._name], h5py.Group)

    def __truediv__(self, key: str) -> "DPPath":
        """Used for / operator."""
        return type(self)(f"{self.root_path}#{self._connect_path(key)}", mode=self.mode)

    def _connect_path(self, path: str) -> str:
        """Connect self with path."""
        if self._name.endswith("/"):
            return f"{self._name}{path}"
        return f"{self._name}/{path}"

    def __lt__(self, other: "DPH5Path") -> bool:
        """Whether this DPPath is less than other for sorting."""
        if self.root_path == other.root_path:
            return self._name < other._name
        return self.root_path < other.root_path

    def __str__(self) -> str:
        """Returns path of self."""
        return f"{self.root_path}#{self._name}"

    @property
    def name(self) -> str:
        """Name of the path."""
        return self._name.split("/")[-1]

    def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
        """Make directory.

        Parameters
        ----------
        parents : bool, optional
            If true, any missing parents of this directory are created as well.
        exist_ok : bool, optional
            If true, no error will be raised if the target directory already exists.
        """
        if self._name in self._keys:
            if not exist_ok:
                raise FileExistsError(f"{self} already exists")
            return
        if parents:
            self.root.require_group(self._name)
        else:
            self.root.create_group(self._name)
