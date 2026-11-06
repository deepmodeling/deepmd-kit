# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-neutral training cursor and selection-RNG checkpoint contract."""

import unittest

from deepmd.dpmodel.train.data_progress import (
    collect_training_data_progress,
    preserve_training_data_progress,
    restore_training_data_progress,
)
from deepmd.utils import random as dp_random


class _Source:
    def __init__(self) -> None:
        self.cursor = 0

    def state_dict(self) -> dict:
        return {"cursor": self.cursor}

    def load_state_dict(self, state: dict) -> None:
        self.cursor = state["cursor"]


class TestDataProgress(unittest.TestCase):
    def test_task_cursors_and_selection_rng_roundtrip(self) -> None:
        sources = {"water": _Source(), "ice": _Source()}
        sources["water"].cursor = 2
        sources["ice"].cursor = 5
        dp_random.seed(17)
        progress = collect_training_data_progress(sources)
        expected = [int(dp_random.choice(4)) for _ in range(20)]
        sources["water"].cursor = 20
        sources["ice"].cursor = 50
        restore_training_data_progress(sources, progress)
        self.assertEqual(sources["water"].cursor, 2)
        self.assertEqual(sources["ice"].cursor, 5)
        self.assertEqual([int(dp_random.choice(4)) for _ in range(20)], expected)

    def test_diagnostics_restore_progress_even_on_failure(self) -> None:
        source = _Source()
        dp_random.seed(11)
        before = collect_training_data_progress({"water": source})
        with self.assertRaisesRegex(RuntimeError, "diagnostic failed"):
            with preserve_training_data_progress({"water": source}):
                source.cursor += 1
                dp_random.choice(10)
                raise RuntimeError("diagnostic failed")
        self.assertEqual(collect_training_data_progress({"water": source}), before)

    def test_legacy_and_sources_without_protocol(self) -> None:
        source = _Source()
        source.cursor = 3
        restore_training_data_progress({"water": source}, None)
        self.assertEqual(source.cursor, 3)
        sources = {"unsupported": object(), "missing": None}
        progress = collect_training_data_progress(sources)
        self.assertEqual(progress["tasks"], {})
        restore_training_data_progress(sources, progress)

    def test_incompatible_payload_rejected_before_source_restore(self) -> None:
        sources = {"water": _Source()}
        progress = collect_training_data_progress(sources)
        for changed in (
            {"version": 999},
            {"tasks": []},
            {"tasks": {"ice": {"cursor": 5}}},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                restore_training_data_progress(sources, progress | changed)
        del progress["rng"]
        with self.assertRaises(ValueError):
            restore_training_data_progress(sources, progress)
        self.assertEqual(sources["water"].cursor, 0)
