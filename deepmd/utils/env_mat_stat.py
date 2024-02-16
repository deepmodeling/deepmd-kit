# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from collections import (
    defaultdict,
)
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
)

import numpy as np

from deepmd.utils.path import (
    DPPath,
)


class StatItem:
    """A class to store the statistics of the environment matrix.

    Parameters
    ----------
    number : int
        The total size of given array.
    sum : float
        The sum value of the matrix.
    squared_sum : float
        The sum squared value of the matrix.
    """

    def __init__(self, number: int = 0, sum: float = 0, squared_sum: float = 0) -> None:
        self.number = number
        self.sum = sum
        self.squared_sum = squared_sum

    def __add__(self, other: "StatItem") -> "StatItem":
        return StatItem(
            number=self.number + other.number,
            sum=self.sum + other.sum,
            squared_sum=self.squared_sum + other.squared_sum,
        )

    def compute_std(self, default: float = 1e-1, protection: float = 1e-2) -> float:
        """Compute the standard deviation of the environment matrix.

        Parameters
        ----------
        default : float, optional
            The default value of the standard deviation, by default 1e-1.
        protection : float, optional
            The protection value for the standard deviation, by default 1e-2.

        Returns
        -------
        float
            The standard deviation of the environment matrix.
        """
        if self.number == 0:
            return default
        val = np.sqrt(
            self.squared_sum / self.number
            - np.multiply(self.sum / self.number, self.sum / self.number)
        )
        if np.abs(val) < protection:
            val = protection
        return val


class EnvMatStat(ABC):
    """A base class to store and calculate the statistics of the environment matrix."""

    def __init__(self) -> None:
        super().__init__()
        self.stats = defaultdict(StatItem)

    def compute_stats(self, data: List[Dict[str, np.ndarray]]) -> None:
        """Compute the statistics of the environment matrix.

        Parameters
        ----------
        data : List[Dict[str, np.ndarray]]
            The environment matrix.
        """
        if len(self.stats) > 0:
            raise ValueError("The statistics has already been computed.")
        for iter_stats in self.iter(data):
            for kk in iter_stats:
                self.stats[kk] += iter_stats[kk]

    @abstractmethod
    def iter(self, data: List[Dict[str, np.ndarray]]) -> Iterator[Dict[str, StatItem]]:
        """Get the iterator of the environment matrix.

        Parameters
        ----------
        data : List[Dict[str, np.ndarray]]
            The environment matrix.

        Yields
        ------
        Dict[str, StatItem]
            The statistics of the environment matrix.
        """

    def save_stats(self, path: DPPath) -> None:
        """Save the statistics of the environment matrix.

        Parameters
        ----------
        path : DPH5Path
            The path to save the statistics of the environment matrix.
        """
        if len(self.stats) == 0:
            raise ValueError("The statistics hasn't been computed.")
        for kk, vv in self.stats.items():
            (path / kk / "number").save(vv.number)
            (path / kk / "sum").save(vv.sum)
            (path / kk / "squared_sum").save(vv.squared_sum)

    def load_stats(self, path: DPPath) -> None:
        """Load the statistics of the environment matrix.

        Parameters
        ----------
        path : DPH5Path
            The path to load the statistics of the environment matrix.
        """
        if len(self.stats) > 0:
            raise ValueError("The statistics has already been computed.")
        for kk in path.glob("*"):
            self.stats[kk.name] = StatItem(
                number=(kk / "number").load_numpy().item(),
                sum=(kk / "sum").load_numpy().item(),
                squared_sum=(kk / "squared_sum").load_numpy().item(),
            )

    def load_or_compute_stats(
        self, data: List[Dict[str, np.ndarray]], path: Optional[DPPath] = None
    ) -> None:
        """Load the statistics of the environment matrix if it exists, otherwise compute and save it.

        Parameters
        ----------
        path : DPH5Path
            The path to load the statistics of the environment matrix.
        data : List[Dict[str, np.ndarray]]
            The environment matrix.
        """
        if path is not None and path.is_dir():
            self.load_stats(path)
        else:
            self.compute_stats(data)
            if path is not None:
                self.save_stats(path)
