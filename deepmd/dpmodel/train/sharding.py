# SPDX-License-Identifier: LGPL-3.0-or-later
"""Distribution strategy of a training run.

``training.zero_stage`` selects how much of the replicated training state is
sharded across ranks, following the ZeRO stages: the optimizer state at stage
one, the gradients as well at stage two, and the parameters on top of that at
stage three. Every consequence of that choice -- which wrapper holds the
model, how the optimizer is built, how a checkpoint is collected, whether a
gradient norm may be reduced locally -- follows from the stage alone. The
stage is therefore resolved into a policy object once, which the backends
then query for the individual decisions.
"""

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)

__all__ = ["ShardingPolicy"]

_MAX_STAGE = 3


@dataclass(frozen=True)
class ShardingPolicy:
    """What a ZeRO stage implies for the mechanics of a training run.

    Attributes
    ----------
    stage : int
        The ZeRO stage in effect, between zero and three. A run that is not
        distributed is always stage zero: there is nothing to shard across.
    """

    stage: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.stage <= _MAX_STAGE:
            raise ValueError(
                f"training.zero_stage must be 0, 1, 2, or 3, got {self.stage}"
            )

    @classmethod
    def from_training_params(
        cls,
        training_params: dict,
        *,
        is_distributed: bool,
    ) -> ShardingPolicy:
        """Read the policy from a normalized ``training`` section.

        Parameters
        ----------
        training_params : dict
            The normalized ``training`` section.
        is_distributed : bool
            Whether the run spans several ranks. A single-process run cannot
            shard anything, so any requested stage is dropped rather than
            rejected, which keeps one configuration usable in both settings.

        Returns
        -------
        ShardingPolicy
            The policy in effect for the run.
        """
        # Construct first, so that an out-of-range stage is rejected whether or
        # not this run is in a position to honour it.
        policy = cls(stage=int(training_params.get("zero_stage", 0)))
        return policy if is_distributed else cls()

    @property
    def enabled(self) -> bool:
        """Whether any training state is sharded."""
        return self.stage > 0

    @property
    def shards_optimizer_state(self) -> bool:
        """Whether the optimizer state is split across ranks."""
        return self.stage >= 1

    @property
    def shards_parameters(self) -> bool:
        """Whether parameters and gradients live as shards of a whole.

        A sharded parameter is a ``DTensor``, which rules out any reduction
        that assumes a rank holds the complete tensor.
        """
        return self.stage >= 2

    @property
    def reshards_after_forward(self) -> bool:
        """Whether parameters are released again once the forward is done."""
        return self.stage >= 3

    def describe(self) -> str:
        """Return a one-line description of the strategy in effect."""
        if not self.enabled:
            return "Distributed data parallel without state sharding."
        if self.stage == 1:
            return "Enabled DDP + ZeRO Stage-1 Optimizer State Sharding."
        stage = "FULL_SHARD (Stage 3)" if self.stage >= 3 else "SHARD_GRAD_OP (Stage 2)"
        return f"Enabled FSDP2 {stage}."
