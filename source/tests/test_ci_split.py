# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for fixture-aware CI test splitting."""

from dataclasses import (
    dataclass,
)

import pytest

from .ci_split import (
    _split_items,
    _TestGroup,
    _unit_key,
)


class FirstClass:
    pass


class SecondClass:
    pass


@dataclass(frozen=True)
class FakeMarker:
    args: tuple[object, ...]
    kwargs: dict[str, object]


@dataclass
class FakeItem:
    nodeid: str
    cls: type[object] | None = None
    explicit_group: str | None = None

    def get_closest_marker(self, name: str) -> FakeMarker | None:
        if name == "ci_split_group" and self.explicit_group is not None:
            return FakeMarker((self.explicit_group,), {})
        return None


def _item(
    nodeid: str,
    *,
    cls: type[object] | None = None,
    explicit_group: str | None = None,
) -> FakeItem:
    return FakeItem(
        nodeid=nodeid,
        cls=cls,
        explicit_group=explicit_group,
    )


def _owner(groups: list[_TestGroup], item: FakeItem) -> int:
    return next(index for index, group in enumerate(groups) if item in group.items)


class TestUnitConstruction:
    def test_class_and_module_items_are_indivisible(self) -> None:
        class_items = [
            _item(f"tests/test_a.py::FirstClass::test_{index}", cls=FirstClass)
            for index in range(3)
        ]
        module_items = [
            _item(f"tests/test_a.py::test_module_{index}") for index in range(2)
        ]
        groups = _split_items(class_items + module_items, {}, splits=2)

        assert len({_owner(groups, item) for item in class_items}) == 1
        assert len({_owner(groups, item) for item in module_items}) == 1

    def test_explicit_group_joins_classes_across_modules(self) -> None:
        first = _item(
            "tests/test_a.py::FirstClass::test_one",
            cls=FirstClass,
            explicit_group="shared-model",
        )
        second = _item(
            "tests/test_b.py::SecondClass::test_two",
            cls=SecondClass,
            explicit_group="shared-model",
        )
        groups = _split_items([first, second], {}, splits=2)

        assert _owner(groups, first) == _owner(groups, second)

    def test_unit_key_rejects_empty_explicit_group(self) -> None:
        item = _item(
            "tests/test_a.py::FirstClass::test_one",
            cls=FirstClass,
            explicit_group="   ",
        )

        with pytest.raises(pytest.UsageError, match="non-empty string"):
            _unit_key(item)


class TestUnitBalancing:
    def test_lpt_balances_recorded_unit_durations(self) -> None:
        items = [_item(f"tests/test_{index}.py::test_value") for index in range(4)]
        durations = dict(
            zip(
                (item.nodeid for item in items),
                [9.0, 8.0, 7.0, 6.0],
                strict=True,
            )
        )
        groups = _split_items(items, durations, splits=2)

        assert [group.estimated_duration for group in groups] == [15.0, 15.0]

    def test_empty_cache_balances_by_number_of_tests(self) -> None:
        items = [
            *[
                _item(f"tests/test_a.py::FirstClass::test_{index}", cls=FirstClass)
                for index in range(5)
            ],
            *[
                _item(f"tests/test_b.py::SecondClass::test_{index}", cls=SecondClass)
                for index in range(4)
            ],
            *[_item(f"tests/test_c.py::test_{index}") for index in range(3)],
            *[_item(f"tests/test_d.py::test_{index}") for index in range(2)],
        ]
        groups = _split_items(items, {}, splits=2)

        assert [len(group.items) for group in groups] == [7, 7]

    def test_assignment_is_deterministic_and_preserves_collection_order(self) -> None:
        items = [_item(f"tests/test_{index}.py::test_value") for index in range(8)]

        first = _split_items(items, {}, splits=3)
        second = _split_items(items, {}, splits=3)

        assert [[item.nodeid for item in group.items] for group in first] == [
            [item.nodeid for item in group.items] for group in second
        ]
        for group in first:
            indices = [items.index(item) for item in group.items]
            assert indices == sorted(indices)
