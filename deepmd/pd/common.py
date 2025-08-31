# SPDX-License-Identifier: LGPL-3.0-or-later
"""Common functionality shared across Paddle descriptor implementations."""

from abc import (
    abstractmethod,
)
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)


class ComputeInputStatsMixin:
    """Mixin class providing common compute_input_stats implementation for Paddle backend.

    This mixin implements the shared logic for computing input statistics
    while allowing backend-specific tensor assignment through abstract methods.
    """

    def compute_input_stats(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        path: Optional[Any] = None,
    ) -> None:
        """
        Compute the input statistics (e.g. mean and stddev) for the descriptors from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: paddle.Tensor
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        path : Optional[DPPath]
            The path to the stat file.

        """
        from deepmd.pd.utils.env_mat_stat import (
            EnvMatStatSe,
        )

        env_mat_stat = EnvMatStatSe(self)
        if path is not None:
            path = path / env_mat_stat.get_hash()
        if path is None or not path.is_dir():
            if callable(merged):
                # only get data for once
                sampled = merged()
            else:
                sampled = merged
        else:
            sampled = []
        env_mat_stat.load_or_compute_stats(sampled, path)
        self.stats = env_mat_stat.stats
        mean, stddev = env_mat_stat()

        # Backend-specific tensor assignment
        self._set_stat_mean_and_stddev(mean, stddev)

    @abstractmethod
    def _set_stat_mean_and_stddev(self, mean, stddev) -> None:
        """Set the computed statistics to the descriptor's mean and stddev attributes.

        This method should be implemented by each descriptor to handle the specific
        tensor assignment logic for Paddle backend.

        Parameters
        ----------
        mean : array-like
            The computed mean values
        stddev : array-like
            The computed standard deviation values
        """
        raise NotImplementedError

    def get_stats(self) -> dict[str, Any]:
        """Get the statistics of the descriptor."""
        if self.stats is None:
            raise RuntimeError(
                "The statistics of the descriptor has not been computed."
            )
        return self.stats
