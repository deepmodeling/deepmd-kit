# SPDX-License-Identifier: LGPL-3.0-or-later
"""DP-GUI entrypoint."""


def start_dpgui(*, port: int, bind_all: bool, **kwargs):
    """Host DP-GUI server.

    Parameters
    ----------
    port : int
        The port to serve DP-GUI on.
    bind_all : bool
        Serve on all public interfaces. This will expose your DP-GUI instance
        to the network on both IPv4 and IPv6 (where available).
    **kwargs
        additional arguments

    Raises
    ------
    ModuleNotFoundError
        The dpgui package is not installed
    """
    try:
        from dpgui import (
            start_dpgui,
        )
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "To use DP-GUI, please install the dpgui package:\npip install dpgui"
        ) from e
    start_dpgui(port=port, bind_all=bind_all)
