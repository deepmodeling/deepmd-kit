# SPDX-License-Identifier: LGPL-3.0-or-later
def check_version_compatibility(
    current_version: int,
    maximum_supported_version: int,
    minimal_supported_version: int = 1,
):
    """Check if the current version is compatible with the supported versions.

    Parameters
    ----------
    current_version : int
        The current version.
    maximum_supported_version : int
        The maximum supported version.
    minimal_supported_version : int, optional
        The minimal supported version. Default is 1.

    Raises
    ------
    ValueError
        If the current version is not compatible with the supported versions.
    """
    if not minimal_supported_version <= current_version <= maximum_supported_version:
        raise ValueError(
            f"Current version {current_version} is not compatible with supported versions "
            f"[{minimal_supported_version}, {maximum_supported_version}]."
        )
