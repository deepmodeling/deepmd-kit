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
)

import numpy as np


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

    def compute_std(self, default: float = 1e-1) -> float:
        """Compute the standard deviation of the environment matrix.

        Parameters
        ----------
        default : float, optional
            The default value of the standard deviation, by default 1e-1.

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
        if np.abs(val) < 1e-2:
            val = 1e-2
        return val


class EnvMatStat(ABC):
    """A base class to store and calculate the statistics of the environment matrix."""

    def compute_stats(self, data: List[Dict[str, np.ndarray]]) -> Dict[str, StatItem]:
        """Compute the statistics of the environment matrix.

        Parameters
        ----------
        data : List[Dict[str, np.ndarray]]
            The environment matrix.

        Returns
        -------
        Dict[str, StatItem]
            The statistics of the environment matrix.
        """
        stats = defaultdict(StatItem)
        for iter_stats in self.iter(data):
            for kk in iter_stats:
                stats[kk] += iter_stats[kk]
        return stats

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
