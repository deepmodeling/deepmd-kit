# SPDX-License-Identifier: LGPL-3.0-or-later
import functools
import operator
from pathlib import (
    Path,
)
from typing import (
    Optional,
    Type,
    Union,
)

from deepmd.backend.backend import (
    Backend,
)


def format_model_suffix(
    filename: str,
    feature: Optional[Backend.Feature] = None,
    preferred_backend: Optional[Union[str, Type["Backend"]]] = None,
    strict_prefer: Optional[bool] = None,
) -> str:
    """Check and format the suffixes of a filename.

    When preferred_backend is not given, this method checks the suffix of the filename
    is within the suffixes of the any backends (with the given feature) and doesn't do formating.
    When preferred_backend is given, strict_prefer must be given.
    If strict_prefer is True and the suffix is not within the suffixes of the preferred backend,
    or strict_prefer is False and the suffix is not within the suffixes of the any backend with the given feature,
    the filename will be formatted with the preferred suffix of the preferred backend.

    Parameters
    ----------
    filename : str
        The filename to be formatted.
    feature : Backend.Feature, optional
        The feature of the backend, by default None
    preferred_backend : str or type of Backend, optional
        The preferred backend, by default None
    strict_prefer : bool, optional
        Whether to strictly prefer the preferred backend, by default None

    Returns
    -------
    str
        The formatted filename with the correct suffix.

    Raises
    ------
    ValueError
        When preferred_backend is not given and the filename is not supported by any backend.
    """
    if preferred_backend is not None and strict_prefer is None:
        raise ValueError("strict_prefer must be given when preferred_backend is given.")
    if isinstance(preferred_backend, str):
        preferred_backend = Backend.get_backend(preferred_backend)
    if preferred_backend is not None and strict_prefer:
        all_backends = [preferred_backend]
    elif feature is None:
        all_backends = list(Backend.get_backends().values())
    else:
        all_backends = list(Backend.get_backends_by_feature(feature).values())

    all_suffixes = set(
        functools.reduce(
            operator.iconcat, [backend.suffixes for backend in all_backends], []
        )
    )
    pp = Path(filename)
    current_suffix = pp.suffix
    if current_suffix not in all_suffixes:
        if preferred_backend is not None:
            return str(pp) + preferred_backend.suffixes[0]
        raise ValueError(f"Unsupported model file format: {filename}")
    return filename
