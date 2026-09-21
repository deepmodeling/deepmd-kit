# SPDX-License-Identifier: LGPL-3.0-or-later
"""The PyTorch backend must import without the optional ``e3nn`` package.

``e3nn`` ships in the ``dpa-adapt`` extra, but ``sezm_nn.projection`` imported
it at module scope, so ``import deepmd.pt.model.model`` failed outright on an
installation without that extra -- the whole PyTorch backend was unusable even
for models that never build a SeZM projector.

The import is checked in a subprocess: modules already imported by the test
session cannot be un-imported, and a meta-path hook installed here would not
undo them.
"""

import subprocess
import sys
import textwrap
import unittest

BLOCK_E3NN = textwrap.dedent(
    """
    import sys


    class _BlockE3nn:
        \"\"\"Make e3nn look absent, whether or not it is installed.\"\"\"

        def find_spec(self, name, path=None, target=None):
            if name == "e3nn" or name.startswith("e3nn."):
                raise ModuleNotFoundError(f"No module named '{name}'")
            return None


    sys.meta_path.insert(0, _BlockE3nn())
    assert "e3nn" not in sys.modules, "e3nn was already imported; the guard is void"
    """
)


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", BLOCK_E3NN + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        check=False,
    )


class TestOptionalE3nnImport(unittest.TestCase):
    def test_pt_models_import_without_e3nn(self) -> None:
        """The backend's model package must not need e3nn to be imported."""
        proc = _run(
            """
            import deepmd.pt.model.model  # noqa: F401
            print("imported")
            """
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("imported", proc.stdout)

    def test_entrypoint_imports_without_e3nn(self) -> None:
        """`dp --pt` must reach its entry point without the extra installed."""
        proc = _run(
            """
            import deepmd.pt.entrypoints.main  # noqa: F401
            print("imported")
            """
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("imported", proc.stdout)

    def test_projector_reports_the_missing_extra(self) -> None:
        """Building a projector without e3nn must say which package to install."""
        proc = _run(
            """
            from deepmd.pt.model.descriptor.sezm_nn.projection import (
                _import_e3nn_o3,
            )

            try:
                _import_e3nn_o3()
            except ImportError as e:
                print("MESSAGE:", e)
            else:
                raise AssertionError("expected ImportError without e3nn")
            """
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("e3nn", proc.stdout)
        self.assertIn("dpa-adapt", proc.stdout)


if __name__ == "__main__":
    unittest.main()
