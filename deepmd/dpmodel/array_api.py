# SPDX-License-Identifier: LGPL-3.0-or-later
"""Utilities for the array API."""


def support_array_api(version: str) -> callable:
    """Mark a function as supporting the specific version of the array API.

    Parameters
    ----------
    version : str
        The version of the array API

    Returns
    -------
    callable
        The decorated function

    Examples
    --------
    >>> @support_array_api(version="2022.12")
    ... def f(x):
    ...     pass
    """

    def set_version(func: callable) -> callable:
        func.array_api_version = version
        return func

    return set_version
