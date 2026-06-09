# SPDX-License-Identifier: LGPL-3.0-or-later
"""Entry point for the ``dpa`` CLI.

This is the console_script target registered in pyproject.toml.
"""

from dpa_adapt.cli import main

if __name__ == "__main__":
    main()
