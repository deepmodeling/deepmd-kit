# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regressions for LAMMPS source selection in the C++ build wrapper."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import (
    Path,
)


class TestBuildCCLammpsArgs(unittest.TestCase):
    def _configure_args(self, source_root=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_dir = root / "source" / "install"
            install_dir.mkdir(parents=True)
            shutil.copy2(
                Path(__file__).resolve().parents[3]
                / "source"
                / "install"
                / "build_cc.sh",
                install_dir / "build_cc.sh",
            )
            bin_dir = root / "bin"
            bin_dir.mkdir()
            cmake = bin_dir / "cmake"
            cmake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['CMAKE_ARGS_LOG'], 'a', encoding='utf-8') as log:\n"
                "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n",
                encoding="utf-8",
            )
            cmake.chmod(0o755)
            log_path = root / "cmake-args.jsonl"
            env = {
                **os.environ,
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "CMAKE_ARGS_LOG": str(log_path),
                "INSTALL_PREFIX": str(root / "install"),
                "LAMMPS_SOURCE_ROOT": source_root or "",
                "DP_VARIANT": "",
            }
            subprocess.run(
                ["bash", str(install_dir / "build_cc.sh")],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            records = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertEqual(len(records), 3)  # configure, build, install
            return records[0]

    def test_source_override_with_spaces(self):
        source = "/opt/lammps source"
        args = self._configure_args(source)
        self.assertIn(f"-DLAMMPS_SOURCE_ROOT={source}", args)
        self.assertIn("-ULAMMPS_VERSION", args)
        self.assertFalse(any(arg == "source" for arg in args))

    def test_default_clears_previous_source_override(self):
        args = self._configure_args()
        self.assertIn("-ULAMMPS_SOURCE_ROOT", args)
        self.assertIn("-DLAMMPS_VERSION=stable_22Jul2025_update2", args)

    @unittest.skipUnless(shutil.which("cmake"), "requires CMake")
    def test_cmake_reused_build_switches_to_explicit_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.20)\n"
                "project(selector NONE)\n"
                "if(DEFINED LAMMPS_VERSION)\n"
                '  set(selected "version")\n'
                "else()\n"
                '  set(selected "source")\n'
                "endif()\n"
                'file(WRITE "${CMAKE_BINARY_DIR}/selected.txt" "${selected}")\n',
                encoding="utf-8",
            )
            build = root / "build"
            subprocess.run(
                ["cmake", "-S", str(root), "-B", str(build), "-DLAMMPS_VERSION=old"],
                check=True,
                capture_output=True,
            )
            self.assertEqual((build / "selected.txt").read_text(), "version")
            source_args = self._configure_args("/opt/lammps source")
            selectors = [
                value
                for value in source_args
                if value.startswith("-ULAMMPS_") or value.startswith("-DLAMMPS_")
            ]
            subprocess.run(
                ["cmake", "-S", str(root), "-B", str(build), *selectors],
                check=True,
                capture_output=True,
            )
            self.assertEqual((build / "selected.txt").read_text(), "source")
            self.assertNotIn("LAMMPS_VERSION:", (build / "CMakeCache.txt").read_text())


if __name__ == "__main__":
    unittest.main()
