# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.utils.path import (
    DPPath,
)


def add_type_map_to_stat_path(
    stat_file_path: DPPath | None,
    type_map: list[str] | None,
) -> DPPath | None:
    if stat_file_path is not None and type_map is not None:
        return stat_file_path / " ".join(type_map)
    return stat_file_path
