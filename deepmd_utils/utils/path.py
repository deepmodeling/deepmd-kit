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
    """

    def __new__(cls, path: str):
        if cls is DPPath:
            if os.path.isdir(path):
                return super().__new__(DPOSPath)
            elif os.path.isfile(path.split("#")[0]):
                # assume h5 if it is not dir
                # TODO: check if it is a real h5? or just check suffix?
                return super().__new__(DPH5Path)
            raise FileNotFoundError(f"{path} not found")
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


class DPOSPath(DPPath):
    """The OS path class to data system (DeepmdData) for real directories.

    Parameters
    ----------
    path : str
        path
    """

    def __init__(self, path: str) -> None:
        super().__init__()
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
        return [type(self)(p) for p in self.path.glob(pattern)]

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
        return [type(self)(p) for p in self.path.rglob(pattern)]

    def is_file(self) -> bool:
        """Check if self is file."""
        return self.path.is_file()

    def is_dir(self) -> bool:
        """Check if self is directory."""
        return self.path.is_dir()

    def __truediv__(self, key: str) -> "DPPath":
        """Used for / operator."""
        return type(self)(self.path / key)

    def __lt__(self, other: "DPOSPath") -> bool:
        """Whether this DPPath is less than other for sorting."""
        return self.path < other.path

    def __str__(self) -> str:
        """Represent string."""
        return str(self.path)


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
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        # we use "#" to split path
        # so we do not support file names containing #...
        s = path.split("#")
        self.root_path = s[0]
        self.root = self._load_h5py(s[0])
        # h5 path: default is the root path
        self.name = s[1] if len(s) > 1 else "/"

    @classmethod
    @lru_cache(None)
    def _load_h5py(cls, path: str) -> h5py.File:
        """Load hdf5 file.

        Parameters
        ----------
        path : str
            path to hdf5 file
        """
        # this method has cache to avoid duplicated
        # loading from different DPH5Path
        # However the file will be never closed?
        return h5py.File(path, "r")

    def load_numpy(self) -> np.ndarray:
        """Load NumPy array.

        Returns
        -------
        np.ndarray
            loaded NumPy array
        """
        return self.root[self.name][:]

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
        subpaths = [ii for ii in self._keys if ii.startswith(self.name)]
        return [
            type(self)(f"{self.root_path}#{pp}")
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
        if self.name not in self._keys:
            return False
        return isinstance(self.root[self.name], h5py.Dataset)

    def is_dir(self) -> bool:
        """Check if self is directory."""
        if self.name not in self._keys:
            return False
        return isinstance(self.root[self.name], h5py.Group)

    def __truediv__(self, key: str) -> "DPPath":
        """Used for / operator."""
        return type(self)(f"{self.root_path}#{self._connect_path(key)}")

    def _connect_path(self, path: str) -> str:
        """Connect self with path."""
        if self.name.endswith("/"):
            return f"{self.name}{path}"
        return f"{self.name}/{path}"

    def __lt__(self, other: "DPH5Path") -> bool:
        """Whether this DPPath is less than other for sorting."""
        if self.root_path == other.root_path:
            return self.name < other.name
        return self.root_path < other.root_path

    def __str__(self) -> str:
        """Returns path of self."""
        return f"{self.root_path}#{self.name}"
