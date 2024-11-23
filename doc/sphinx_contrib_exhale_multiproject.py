# SPDX-License-Identifier: LGPL-3.0-or-later
# The initial version of this file comes from
# https://github.com/mithro/sphinx-contrib-mithro/tree/master/sphinx-contrib-exhale-multiproject
# under the following license:
#
# Copyright (C) 2017  The Project X-Ray Authors. All rights reserved.
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#
# Any modification to the initial version still applies LGPL v3 license.

# -- Breathe + Exhale config for C++ API Documentation --------------------
'''Example configuration of exhale with multiple projects;
.. highlight::python
    breathe_projects = {
        "firmware":     "_doxygen/firmware/xml",
        "edid-decode":  "_doxygen/edid-decode/xml",
        "libuip":       "_doxygen/libuip/xml",
    }
    breathe_default_project = "firmware"
    breathe_projects_source = {
        "firmware":     "../firmware",
        "edid-decode":  "../third_party/edid-decode",
        "libuip":       "../third_party/libuip",
    }
    # Setup the exhale extension
    exhale_args = {
        'verboseBuild': True,
        "rootFileTitle":        "Unknown",
        "containmentFolder":    "unknown",
        # These arguments are required
        "rootFileName":          "root.rst",
        "doxygenStripFromPath":  "../",
        # Suggested optional arguments
        "createTreeView":        True,
        # TIP: if using the sphinx-bootstrap-theme, you need
        # "treeViewIsBootstrap": True,
        "exhaleExecutesDoxygen": True,
        #"exhaleUseDoxyfile":     True,
        "exhaleDoxygenStdin":    """
    EXCLUDE     = ../doc ../third_party/litex/litex/soc/software/compiler_rt ../third_party/litex/litex/soc/software/libcompiler_rt */__pycache__
    """,
    }
    # Monkey patch exhale.environment_ready to allow multiple doxygen runs with
    # different configs.
    exhale_projects_args = {
        "firmware": {
            "exhaleDoxygenStdin":   "INPUT = ../firmware"+exhale_args["exhaleDoxygenStdin"],
            "containmentFolder":    "firmware-api",
            "rootFileTitle":        "Firmware",
        },
        # Third Party Project Includes
        "edid-decode": {
            "exhaleDoxygenStdin":   "INPUT = ../third_party/edid-decode"+exhale_args["exhaleDoxygenStdin"],
            "containmentFolder":    "third_party-edid-decode-api",
            "rootFileTitle":        "edid-decode",
        },
        "libuip": {
            "exhaleDoxygenStdin":   "INPUT = ../third_party/libuip"+exhale_args["exhaleDoxygenStdin"],
            "containmentFolder":    "third_party-libuip-api",
            "rootFileTitle":        "libuip",
        },
    }.
'''

import os
import os.path
from pprint import (
    pprint,
)

import exhale
import exhale.configs
import exhale.deploy
import exhale.utils


def exhale_environment_ready(app) -> None:
    default_project = app.config.breathe_default_project
    default_exhale_args = dict(app.config.exhale_args)

    exhale_projects_args = dict(app.config._raw_config["exhale_projects_args"])
    breathe_projects = dict(app.config._raw_config["breathe_projects"])

    for project in breathe_projects:
        app.config.breathe_default_project = project
        os.makedirs(breathe_projects[project], exist_ok=True)

        project_exhale_args = exhale_projects_args.get(project, {})

        app.config.exhale_args = dict(default_exhale_args)
        app.config.exhale_args.update(project_exhale_args)
        app.config.exhale_args["containmentFolder"] = os.path.realpath(
            app.config.exhale_args["containmentFolder"]
        )
        print("=" * 75)  # noqa: T201
        print(project)  # noqa: T201
        print("-" * 50)  # noqa: T201
        pprint(app.config.exhale_args)  # noqa: T203
        print("=" * 75)  # noqa: T201

        # First, setup the extension and verify all of the configurations.
        exhale.configs.apply_sphinx_configurations(app)
        ####### Next, perform any cleanup

        # Generate the full API!
        try:
            exhale.deploy.explode()
        except Exception:
            exhale.utils.fancyError(
                "Exhale: could not generate reStructuredText documents :/"
            )

    app.config.breathe_default_project = default_project


exhale.environment_ready = exhale_environment_ready
