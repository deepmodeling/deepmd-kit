# SPDX-License-Identifier: LGPL-3.0-or-later
"""Keep fixture-sharing test units together while balancing CI shards.

``pytest-split`` balances individual test items well, but assigning individual
methods independently repeats class- and module-scoped setup in multiple CI
processes.  This plugin first groups items into fixture-sharing units, then
uses longest-processing-time (LPT) bin packing to balance those units by their
recorded durations.

Classes are kept together by default.  Module-level tests from the same file
form one unit.  Tests that need to share a module-scoped fixture across class
boundaries can use ``@pytest.mark.ci_split_group("name")`` to opt into the same
explicit unit.
"""

from __future__ import (
    annotations,
)

import heapq
import json
import math
from collections import (
    OrderedDict,
)
from dataclasses import (
    dataclass,
    field,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Protocol,
)

import pytest

if TYPE_CHECKING:
    from _pytest import (
        nodes,
    )
    from _pytest.config import (
        Config,
    )
    from _pytest.config.argparsing import (
        Parser,
    )


class _SplitItem(Protocol):
    """The subset of a pytest item used by the grouping algorithm."""

    nodeid: str
    cls: type[object] | None

    def get_closest_marker(self, name: str) -> pytest.Mark | None:
        """Return the closest marker with the requested name."""
        raise NotImplementedError


@dataclass
class _TestUnit:
    """A set of test items that must execute in the same CI shard."""

    key: str
    items: list[_SplitItem] = field(default_factory=list)
    estimated_duration: float = 0.0


@dataclass
class _TestGroup:
    """One CI shard selected by the grouped splitting algorithm."""

    items: list[_SplitItem]
    estimated_duration: float
    unit_count: int


def _explicit_group_name(item: _SplitItem) -> str | None:
    """Return and validate an explicitly requested cross-class group name."""
    marker = item.get_closest_marker("ci_split_group")
    if marker is None:
        return None
    if len(marker.args) != 1 or marker.kwargs or not isinstance(marker.args[0], str):
        raise pytest.UsageError(
            "ci_split_group requires exactly one non-empty string argument"
        )
    name = marker.args[0].strip()
    if not name:
        raise pytest.UsageError(
            "ci_split_group requires exactly one non-empty string argument"
        )
    return name


def _base_unit_key(item: _SplitItem) -> str:
    """Return the class/module unit that must never be split."""
    # nodeid contains a repository-relative path, unlike item.path which may
    # contain runner-specific checkout prefixes.
    path = item.nodeid.split("::", maxsplit=1)[0]
    if item.cls is not None:
        return f"class:{path}::{item.cls.__qualname__}"
    return f"module:{path}"


def _relevant_durations(
    items: list[_SplitItem], durations: dict[str, float]
) -> dict[str, float]:
    """Discard stale cache entries before deriving the unknown-test fallback."""
    nodeids = {item.nodeid for item in items}
    return {
        nodeid: float(duration)
        for nodeid, duration in durations.items()
        if nodeid in nodeids and duration >= 0 and math.isfinite(duration)
    }


def _build_units(
    items: list[_SplitItem],
    durations: dict[str, float],
) -> list[_TestUnit]:
    """Build complete class/module units, then merge explicit unit groups."""
    relevant = _relevant_durations(items, durations)
    fallback_duration = sum(relevant.values()) / len(relevant) if relevant else 1.0
    base_units: OrderedDict[str, _TestUnit] = OrderedDict()
    for item in items:
        key = _base_unit_key(item)
        unit = base_units.setdefault(key, _TestUnit(key=key))
        unit.items.append(item)

    for unit in base_units.values():
        known_durations = [
            relevant[item.nodeid] for item in unit.items if item.nodeid in relevant
        ]
        if known_durations:
            missing_count = len(unit.items) - len(known_durations)
            unit.estimated_duration = sum(known_durations) + (
                missing_count * fallback_duration
            )
        else:
            unit.estimated_duration = len(unit.items) * fallback_duration

    merged_units: OrderedDict[str, _TestUnit] = OrderedDict()
    for unit in base_units.values():
        explicit_names = [
            name
            for item in unit.items
            if (name := _explicit_group_name(item)) is not None
        ]
        if explicit_names and len(explicit_names) != len(unit.items):
            raise pytest.UsageError(
                "ci_split_group must mark every test in its class/module unit"
            )
        if len(set(explicit_names)) > 1:
            raise pytest.UsageError(
                "a class/module unit cannot use multiple ci_split_group names"
            )
        key = f"explicit:{explicit_names[0]}" if explicit_names else unit.key
        merged = merged_units.setdefault(key, _TestUnit(key=key))
        merged.items.extend(unit.items)
        merged.estimated_duration += unit.estimated_duration
    return list(merged_units.values())


def _split_items(
    items: list[_SplitItem],
    durations: dict[str, float],
    splits: int,
) -> list[_TestGroup]:
    """Assign indivisible units using deterministic LPT bin packing."""
    units = _build_units(items, durations)
    item_index = {id(item): index for index, item in enumerate(items)}

    # The group index breaks equal-duration ties deterministically.  Sorting
    # units by key gives all CI runners the same assignment even when several
    # units have identical estimates, including a completely empty cache.
    heap: list[tuple[float, int, list[_TestUnit]]] = [
        (0.0, group_index, []) for group_index in range(splits)
    ]
    heapq.heapify(heap)
    for unit in sorted(units, key=lambda value: (-value.estimated_duration, value.key)):
        duration, group_index, assigned = heapq.heappop(heap)
        assigned.append(unit)
        heapq.heappush(
            heap,
            (duration + unit.estimated_duration, group_index, assigned),
        )

    groups: list[_TestGroup | None] = [None] * splits
    for duration, group_index, assigned in heap:
        selected = [item for unit in assigned for item in unit.items]
        selected.sort(key=lambda item: item_index[id(item)])
        groups[group_index] = _TestGroup(
            items=selected,
            estimated_duration=duration,
            unit_count=len(assigned),
        )
    return [group for group in groups if group is not None]


def _load_durations(path: Path) -> dict[str, float]:
    """Load pytest-split's duration cache, including its legacy list format."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise pytest.UsageError(f"cannot read duration cache {path}: {exc}") from exc

    if isinstance(data, list):
        try:
            data = dict(data)
        except (TypeError, ValueError) as exc:
            raise pytest.UsageError(
                f"duration cache {path} contains an invalid legacy list"
            ) from exc
    if not isinstance(data, dict):
        raise pytest.UsageError(f"duration cache {path} must contain a JSON object")
    try:
        return {str(nodeid): float(duration) for nodeid, duration in data.items()}
    except (TypeError, ValueError) as exc:
        raise pytest.UsageError(
            f"duration cache {path} contains a non-numeric duration"
        ) from exc


def pytest_addoption(parser: Parser) -> None:
    """Declare CI-specific split options without activating pytest-split."""
    group = parser.getgroup("DeePMD grouped CI splitting")
    group.addoption(
        "--ci-splits",
        type=int,
        help="Number of grouped CI test shards.",
    )
    group.addoption(
        "--ci-group",
        type=int,
        help="One-based grouped CI shard to execute.",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: Config) -> int | None:
    """Validate grouped split options before pytest starts collection."""
    splits = config.getoption("ci_splits")
    group = config.getoption("ci_group")
    if splits is None and group is None:
        return None
    if splits is None or group is None:
        raise pytest.UsageError("--ci-splits and --ci-group must be used together")
    if splits < 1:
        raise pytest.UsageError("--ci-splits must be at least 1")
    if group < 1 or group > splits:
        raise pytest.UsageError("--ci-group must be between 1 and --ci-splits")
    return None


def pytest_configure(config: Config) -> None:
    """Register the marker used to join classes sharing expensive fixtures."""
    config.addinivalue_line(
        "markers",
        "ci_split_group(name): keep marked tests in one duration-balanced CI shard",
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: Config, items: list[nodes.Item]) -> None:
    """Select one grouped shard after the complete test suite is collected."""
    splits = config.getoption("ci_splits")
    group_index = config.getoption("ci_group")
    if splits is None or group_index is None:
        return

    durations_path = Path(config.getoption("durations_path"))
    durations = _load_durations(durations_path)
    groups = _split_items(items, durations, splits)
    selected = groups[group_index - 1]
    selected_ids = {id(item) for item in selected.items}
    deselected = [item for item in items if id(item) not in selected_ids]
    items[:] = selected.items
    config.hook.pytest_deselected(items=deselected)

    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        estimate_source = "runtime cache" if durations else "test-count fallback"
        reporter.write_line(
            "[deepmd-ci-split] "
            f"group {group_index}/{splits}: {len(selected.items)} tests in "
            f"{selected.unit_count} units, estimated "
            f"{selected.estimated_duration:.2f}s ({estimate_source})"
        )
