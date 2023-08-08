# SPDX-License-Identifier: LGPL-3.0-or-later
"""Prepublish script for nodejs package.

The NPM package downloads the C library binary from GitHub releases.
This script changes the package.json to make it work.
"""
import json
import shutil

with open("package.json") as f:
    package = json.load(f)
# check version
version = package["version"]
if version.startswith("0."):
    raise ValueError("Update version to actual release version")

download_url = f"https://github.com/deepmodeling/deepmd-kit/releases/download/v{version}/libdeepmd_c.tar.gz"

# change install script
package["scripts"]["install"] = (
    f"curl -L {download_url} -o libdeepmd_c.tar.gz && tar -vxzf libdeepmd_c.tar.gz && "
    + package["scripts"]["install"]
)

with open("package.json", "w") as f:
    json.dump(package, f, indent=2)

# copy binding.gyp.pack to binding.gyp
shutil.copy("binding.gyp.pack", "binding.gyp")
