# SPDX-License-Identifier: LGPL-3.0-or-later
try:
    from ._version import version as __version__
except ImportError:
    from .__about__ import __version__ as __version__
