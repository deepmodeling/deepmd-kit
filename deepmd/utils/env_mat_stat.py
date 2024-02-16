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
    mean : float
        The mean value of the matrix.
    squared_mean : float
        The mean squared value of the matrix.
    """

    def __init__(
        self, number: int = 0, mean: float = 0, squared_mean: float = 0
    ) -> None:
        self.number = number
        self.mean = mean
        self.squared_mean = squared_mean

    def __add__(self, other: "StatItem") -> "StatItem":
        self_frac = self.number / (self.number + other.number)
        other_frac = 1 - self_frac
        return StatItem(
            number=self.number + other.number,
            mean=self_frac * self.mean + other_frac * other.mean,
            squared_mean=self_frac * self.squared_mean
            + other_frac * other.squared_mean,
        )


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
