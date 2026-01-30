# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    Any,
)

from .base_modifier import (
    BaseModifier,
)

__all__ = [
    "BaseModifier",
    "get_data_modifier",
]


def get_data_modifier(_modifier_params: dict[str, Any]) -> BaseModifier:
    modifier_params = copy.deepcopy(_modifier_params)
    try:
        modifier_type = modifier_params.pop("type")
    except KeyError:
        raise ValueError("Data modifier type not specified!") from None
    return BaseModifier.get_class_by_type(modifier_type).get_modifier(modifier_params)
