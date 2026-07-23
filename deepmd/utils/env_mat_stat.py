# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from abc import (
    ABC,
    abstractmethod,
)
from collections import (
    defaultdict,
)
from collections.abc import (
    Callable,
    Iterator,
)

import numpy as np

from deepmd.utils.path import (
    DPPath,
)

log = logging.getLogger(__name__)


class StatItem:
    """A class to store the statistics of the environment matrix.

    Parameters
    ----------
    number : float
        The total size of given array.
    sum : float
        The sum value of the matrix.
    squared_sum : float
        The sum squared value of the matrix.
    """

    def __init__(
        self, number: float = 0, sum: float = 0, squared_sum: float = 0
    ) -> None:
        self.number = number
        self.sum = sum
        self.squared_sum = squared_sum

    def __add__(self, other: "StatItem") -> "StatItem":
        return StatItem(
            number=self.number + other.number,
            sum=self.sum + other.sum,
            squared_sum=self.squared_sum + other.squared_sum,
        )

    def __mul__(self, scalar: float) -> "StatItem":
        return StatItem(
            number=self.number * scalar,
            sum=self.sum * scalar,
            squared_sum=self.squared_sum * scalar,
        )

    def compute_avg(self, default: float = 0) -> float:
        """Compute the average of the environment matrix.

        Parameters
        ----------
        default : float, optional
            The default value of the average, by default 0.

        Returns
        -------
        float
            The average of the environment matrix.
        """
        if self.number == 0:
            return default
        return self.sum / self.number

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
            np.clip(
                self.squared_sum / self.number
                - np.multiply(self.sum / self.number, self.sum / self.number),
                a_min=0,
                a_max=None,
            )
        )
        if np.abs(val) < protection:
            val = protection
        return val


class EnvMatStat(ABC):
    """A base class to store and calculate the statistics of the environment matrix."""

    def __init__(self) -> None:
        super().__init__()
        self.stats = defaultdict(StatItem)

    def compute_stats(self, data: list[dict[str, np.ndarray]]) -> None:
        """Compute the statistics of the environment matrix.

        Parameters
        ----------
        data : list[dict[str, np.ndarray]]
            The environment matrix.
        """
        if len(self.stats) > 0:
            raise ValueError("The statistics has already been computed.")
        for iter_stats in self.iter(data):
            for kk in iter_stats:
                self.stats[kk] += iter_stats[kk]

    @abstractmethod
    def iter(self, data: list[dict[str, np.ndarray]]) -> Iterator[dict[str, StatItem]]:
        """Get the iterator of the environment matrix.

        Parameters
        ----------
        data : list[dict[str, np.ndarray]]
            The environment matrix.

        Yields
        ------
        dict[str, StatItem]
            The statistics of the environment matrix.
        """

    def get_stat_keys(self) -> list[str]:
        """Get the dataset names required for a complete statistics cache."""
        return []

    def save_stats(self, path: DPPath) -> None:
        """Save the statistics of the environment matrix.

        Parameters
        ----------
        path : DPPath
            The path to save the statistics of the environment matrix.
        """
        if len(self.stats) == 0:
            raise ValueError("The statistics hasn't been computed.")
        for kk, vv in self.stats.items():
            path.mkdir(parents=True, exist_ok=True)
            (path / kk).save_numpy(np.array([vv.number, vv.sum, vv.squared_sum]))

    def load_stats(self, path: DPPath) -> None:
        """Load the statistics of the environment matrix.

        Parameters
        ----------
        path : DPPath
            The path to load the statistics of the environment matrix.
        """
        if len(self.stats) > 0:
            raise ValueError("The statistics has already been computed.")
        for kk in path.glob("*"):
            arr = kk.load_numpy()
            self.stats[kk.name] = StatItem(
                number=arr[0],
                sum=arr[1],
                squared_sum=arr[2],
            )

    def load_or_compute_stats(
        self,
        data: (Callable[[], list[dict[str, np.ndarray]]] | list[dict[str, np.ndarray]]),
        path: DPPath | None = None,
    ) -> None:
        """Load the statistics of the environment matrix if it exists, otherwise compute and save it.

        Parameters
        ----------
        data : Callable or list[dict[str, np.ndarray]]
            The environment-matrix data or a lazy callable that returns it.
        path : DPPath
            The path to load the statistics of the environment matrix.

        Raises
        ------
        FileNotFoundError
            If a read-only cache is missing its statistics group or any
            required dataset.
        """
        cache_exists = path is not None and path.is_dir()
        missing = (
            [key for key in self.get_stat_keys() if not (path / key).is_file()]
            if cache_exists
            else []
        )
        if cache_exists and not missing:
            self.load_stats(path)
            log.info(f"Load stats from {path}.")
            return
        if path is not None and getattr(path, "mode", None) == "r":
            if not cache_exists:
                raise FileNotFoundError(
                    f"Read-only statistics cache {path} is missing the required "
                    "environment statistics group."
                )
            missing_items = ", ".join(repr(item) for item in missing)
            raise FileNotFoundError(
                f"Read-only statistics cache {path} is missing required "
                f"environment statistics item(s): {missing_items}."
            )

        sampled = data() if callable(data) else data
        self.compute_stats(sampled)
        if path is not None:
            self.save_stats(path)
            log.info(f"Save stats to {path}.")

    def get_avg(self, default: float = 0) -> dict[str, float]:
        """Get the average of the environment matrix.

        Parameters
        ----------
        default : float, optional
            The default value of the average, by default 0.

        Returns
        -------
        dict[str, float]
            The average of the environment matrix.
        """
        return {kk: vv.compute_avg(default=default) for kk, vv in self.stats.items()}

    def get_std(
        self, default: float = 1e-1, protection: float = 1e-2
    ) -> dict[str, float]:
        """Get the standard deviation of the environment matrix.

        Parameters
        ----------
        default : float, optional
            The default value of the standard deviation, by default 1e-1.
        protection : float, optional
            The protection value for the standard deviation, by default 1e-2.

        Returns
        -------
        dict[str, float]
            The standard deviation of the environment matrix.
        """
        return {
            kk: vv.compute_std(default=default, protection=protection)
            for kk, vv in self.stats.items()
        }
