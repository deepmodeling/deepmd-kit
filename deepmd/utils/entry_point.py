# SPDX-License-Identifier: LGPL-3.0-or-later
from importlib import (
    metadata,
)


def load_entry_point(group: str) -> list:
    """Load entry points from a group.

    Parameters
    ----------
    group : str
        The group name.

    Returns
    -------
    list
        A list of loaded entry points.
    """
    # https://setuptools.readthedocs.io/en/latest/userguide/entry_point.html
    try:
        eps = metadata.entry_points(group=group)
    except TypeError:
        eps = metadata.entry_points().get(group, [])
    return [ep.load() for ep in eps]
