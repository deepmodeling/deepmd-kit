# SPDX-License-Identifier: LGPL-3.0-or-later
"""DeePMD-kit is a package written in Python/C++, designed to
minimize the effort required to build deep learning-based model
of interatomic potential energy and force field and to perform
molecular dynamics (MD).

The top module (deepmd.__init__) should not import any third-party
modules for performance.
"""
try:
    from deepmd._version import version as __version__
except ImportError:
    from .__about__ import (
        __version__,
    )


def DeepPotential(*args, **kwargs):
    """Factory function that forwards to DeepEval (for compatbility
    and performance).

    Parameters
    ----------
    *args
        positional arguments
    **kwargs
        keyword arguments

    Returns
    -------
    DeepEval
        potentials
    """
    from deepmd.infer import (
        DeepPotential,
    )

    return DeepPotential(*args, **kwargs)


__all__ = [
    "__version__",
    "DeepPotential",
]
