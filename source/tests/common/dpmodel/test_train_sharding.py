# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the distribution strategy of a training run."""

import unittest

from deepmd.dpmodel.train import (
    ShardingPolicy,
)


class TestShardingPolicy(unittest.TestCase):
    def test_each_stage_widens_what_is_sharded(self) -> None:
        # What a stage shards is cumulative, and each of the three decisions
        # it drives switches over at a different stage.
        expected = {
            0: (False, False, False, False),
            1: (True, True, False, False),
            2: (True, True, True, False),
            3: (True, True, True, True),
        }
        for stage, entry in expected.items():
            with self.subTest(stage=stage):
                policy = ShardingPolicy(stage=stage)
                self.assertEqual(
                    (
                        policy.enabled,
                        policy.shards_optimizer_state,
                        policy.shards_parameters,
                        policy.reshards_after_forward,
                    ),
                    entry,
                )

    def test_a_single_process_run_drops_the_requested_stage(self) -> None:
        # One configuration has to remain usable whether or not it is launched
        # across ranks, so an unusable stage is dropped rather than rejected.
        policy = ShardingPolicy.from_training_params(
            {"zero_stage": 3}, is_distributed=False
        )
        self.assertEqual(policy.stage, 0)

    def test_an_out_of_range_stage_is_rejected_even_without_ranks(self) -> None:
        for is_distributed in (True, False):
            with self.subTest(is_distributed=is_distributed):
                with self.assertRaisesRegex(ValueError, "must be 0, 1, 2, or 3"):
                    ShardingPolicy.from_training_params(
                        {"zero_stage": 4}, is_distributed=is_distributed
                    )


if __name__ == "__main__":
    unittest.main()
