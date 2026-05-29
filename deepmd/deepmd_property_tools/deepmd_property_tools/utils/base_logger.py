# SPDX-License-Identifier: LGPL-3.0-or-later
"""Logging helpers."""

import logging

logger = logging.getLogger("deepmd_property_tools")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
