# SPDX-License-Identifier: LGPL-3.0-or-later
"""Synchronize with deepmd.jax for test purpose only."""

import array_api_strict

# this is the default version in the latest array_api_strict,
# but in old versions it may be 2022.12
array_api_strict.set_array_api_strict_flags(api_version="2023.12")
