# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import os
from abc import (
    ABC,
    abstractmethod,
)
from pathlib import (
    Path,
)
from threading import (
    Lock,
)
from typing import (
    Any,
    ClassVar,
)
from weakref import (
    WeakKeyDictionary,
    WeakValueDictionary,
)

import h5py
import numpy as np
from typing_extensions import (
    Self,
)
from wcmatch.glob import (
    GLOBSTAR,
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

    def __new__(cls, path: str, mode: str = "r") -> Self:
        if cls is DPPath:
            if os.path.isdir(path):
                return super().__new__(DPOSPath)
            elif os.path.isfile(path.split("#")[0]):
                # assume h5 if it is not dir
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
    def load_txt(self, **kwargs: Any) -> np.ndarray:
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
    def glob(self, pattern: str) -> list["DPPath"]:
        """Search path using the glob pattern.

        Parameters
        ----------
        pattern : str
            glob pattern

        Returns
        -------
        list[DPPath]
            list of paths
        """

    @abstractmethod
    def rglob(self, pattern: str) -> list["DPPath"]:
        """This is like calling :meth:`DPPath.glob()` with `**/` added in front
        of the given relative pattern.

        Parameters
        ----------
        pattern : str
            glob pattern

        Returns
        -------
        list[DPPath]
            list of paths
        """

    @abstractmethod
    def is_file(self) -> bool:
        """Check if self is file."""

    @abstractmethod
    def is_dir(self) -> bool:
        """Check if self is directory."""

    @abstractmethod
    def __getnewargs__(self) -> tuple[str, str]:
        """Return the arguments to be passed to __new__ when unpickling an instance."""

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

    def __eq__(self, other: object) -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
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
    path : Union[str, Path]
        path
    mode : str, optional
        mode, by default "r"
    """

    def __init__(self, path: str | Path, mode: str = "r") -> None:
        super().__init__()
        self.mode = mode
        self.path = Path(path)

    def __getnewargs__(self) -> tuple[str, str]:
        return (self.path, self.mode)

    def load_numpy(self) -> np.ndarray:
        """Load NumPy array.

        Returns
        -------
        np.ndarray
            loaded NumPy array
        """
        return np.load(str(self.path))

    def load_txt(self, **kwargs: Any) -> np.ndarray:
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
        with self.path.open("wb") as f:
            np.save(f, arr)

    def glob(self, pattern: str) -> list["DPPath"]:
        """Search path using the glob pattern.

        Parameters
        ----------
        pattern : str
            glob pattern

        Returns
        -------
        list[DPPath]
            list of paths
        """
        # currently DPOSPath will only derivative DPOSPath
        return [type(self)(p, mode=self.mode) for p in self.path.glob(pattern)]

    def rglob(self, pattern: str) -> list["DPPath"]:
        """This is like calling :meth:`DPPath.glob()` with `**/` added in front
        of the given relative pattern.

        Parameters
        ----------
        pattern : str
            glob pattern

        Returns
        -------
        list[DPPath]
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

    def __getnewargs__(self) -> tuple[str, str]:
        return (self.root_path, self.mode)

    _open_files: ClassVar[WeakValueDictionary[tuple[str, str, int, int], h5py.File]] = (
        WeakValueDictionary()
    )
    _open_lock = Lock()

    @classmethod
    def _load_h5py(cls, path: str, mode: str = "r") -> h5py.File:
        """Share a handle only for the current file and while paths own it."""
        path = os.path.realpath(path)
        with cls._open_lock:
            info = os.stat(path)
            # Atomic publication replaces the inode, not the existing handle.
            key = (path, mode, info.st_dev, info.st_ino)
            file = cls._open_files.get(key)
            if file is None or not file:
                file = h5py.File(path, mode)
                cls._open_files[key] = file
            return file

    def _with_name(self, name: str) -> "DPH5Path":
        """Keep descendants pinned to the parent's HDF5 snapshot."""
        path = object.__new__(type(self))
        path.mode = self.mode
        path.root_path = self.root_path
        path.root = self.root
        path._name = name
        return path

    def load_numpy(self) -> np.ndarray:
        """Load NumPy array.

        Returns
        -------
        np.ndarray
            loaded NumPy array
        """
        return self.root[self._name][:]

    def load_txt(self, dtype: np.dtype | None = None, **kwargs: Any) -> np.ndarray:
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
        if self.mode == "r":
            raise ValueError("Cannot save to read-only path")
        if self._name in self._keys:
            del self.root[self._name]
        self.root.create_dataset(self._name, data=arr)
        self.root.flush()
        self._new_keys.append(self._name)

    def glob(self, pattern: str) -> list["DPPath"]:
        """Search path using the glob pattern.

        Parameters
        ----------
        pattern : str
            glob pattern

        Returns
        -------
        list[DPPath]
            list of paths
        """
        # got paths starts with current path first, which is faster
        subpaths = [
            ii
            for ii in itertools.chain(self._keys, self._new_keys)
            if ii.startswith(self._name)
        ]
        return [
            self._with_name(pp)
            for pp in globfilter(subpaths, self._connect_path(pattern), flags=GLOBSTAR)
        ]

    def rglob(self, pattern: str) -> list["DPPath"]:
        """This is like calling :meth:`DPPath.glob()` with `**/` added in front
        of the given relative pattern.

        Parameters
        ----------
        pattern : str
            glob pattern

        Returns
        -------
        list[DPPath]
            list of paths
        """
        return self.glob("**/" + pattern)

    @property
    def _keys(self) -> list[str]:
        """Walk all groups and dataset."""
        if self.root not in self._file_keys:
            keys = []
            self.root.visit(lambda name: keys.append("/" + name))
            self._file_keys[self.root] = keys
        return self._file_keys[self.root]

    _file_keys: ClassVar[WeakKeyDictionary[h5py.File, list[str]]] = WeakKeyDictionary()
    _file_new_keys: ClassVar[WeakKeyDictionary[h5py.File, list[str]]] = (
        WeakKeyDictionary()
    )

    @property
    def _new_keys(self) -> list[str]:
        """New keys that haven't been cached."""
        return self._file_new_keys.setdefault(self.root, [])

    def _invalidate_keys(self) -> None:
        """Forget cached names after a direct edit through the HDF5 handle."""
        self._file_keys.pop(self.root, None)
        self._file_new_keys.pop(self.root, None)

    def is_file(self) -> bool:
        """Check if self is file."""
        if self._name not in self._keys and self._name not in self._new_keys:
            return False
        return isinstance(self.root[self._name], h5py.Dataset)

    def is_dir(self) -> bool:
        """Check if self is directory."""
        if self._name == "/":
            return True
        if self._name not in self._keys and self._name not in self._new_keys:
            return False
        return isinstance(self.root[self._name], h5py.Group)

    def __truediv__(self, key: str) -> "DPPath":
        """Used for / operator."""
        return self._with_name(self._connect_path(key))

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
        if self.mode == "r":
            raise ValueError("Cannot mkdir to read-only path")
        if self._name in self._keys:
            if not exist_ok:
                raise FileExistsError(f"{self} already exists")
            return
        if parents:
            self.root.require_group(self._name)
        else:
            self.root.create_group(self._name)
        self._new_keys.append(self._name)
