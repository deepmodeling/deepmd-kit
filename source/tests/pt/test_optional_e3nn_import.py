# SPDX-License-Identifier: LGPL-3.0-or-later
"""The PyTorch backend must import without the optional ``e3nn`` package.

``e3nn`` ships with the ``torch`` and ``dpa-adapt`` extras, but
``sezm_nn.projection`` imported it at module scope, so
``import deepmd.pt.model.model`` failed outright on an installation that uses
neither -- such as the PyTorch CPU install documented in
``doc/install/easy-install.md`` -- leaving the whole PyTorch backend unusable
even for models that never build a SeZM projector.

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

    # Make e3nn unavailable, whether or not it is installed, and let CPython's
    # own import machinery raise the error. Constructing the exception here
    # instead would prove nothing about what the code actually meets: a
    # hand-built ModuleNotFoundError is easy to give the wrong ``name``, and
    # ``name`` is what tells a missing package from a broken one.
    sys.modules["e3nn"] = None
    """
)


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", BLOCK_E3NN + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        check=False,
    )


# e3nn is importable, but something it imports is not: the failure a broken
# environment produces, as opposed to an absent optional dependency.
BREAK_E3NN_DEPENDENCY = textwrap.dedent(
    """
    import sys
    import types


    class _E3nnWithMissingDependency:
        \"\"\"Serve an ``e3nn`` whose own import fails on a missing module.\"\"\"

        def find_spec(self, name, path=None, target=None):
            if name == "e3nn" or name.startswith("e3nn."):
                import importlib.machinery

                return importlib.machinery.ModuleSpec(name, _Loader())
            return None


    class _Loader:
        def create_module(self, spec):
            return types.ModuleType(spec.name)

        def exec_module(self, module):
            raise ModuleNotFoundError("No module named 'a_dependency_of_e3nn'",
                                      name="a_dependency_of_e3nn")


    sys.meta_path.insert(0, _E3nnWithMissingDependency())
    """
)


def _run_broken(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", BREAK_E3NN_DEPENDENCY + textwrap.dedent(body)],
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

    def test_a_broken_e3nn_is_not_reported_as_a_missing_one(self) -> None:
        """An installed e3nn that fails to import must not say "not installed".

        Translating every ImportError sends a user whose environment is broken
        to reinstall a package they already have, and hides the module that is
        actually missing.
        """
        proc = _run_broken(
            """
            from deepmd.pt.model.descriptor.sezm_nn.projection import (
                _import_e3nn_o3,
            )

            try:
                _import_e3nn_o3()
            except ModuleNotFoundError as e:
                print("PROPAGATED:", e.name)
            except ImportError as e:
                print("TRANSLATED:", e)
            """
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("PROPAGATED: a_dependency_of_e3nn", proc.stdout)
        # the misleading advice must not appear
        self.assertNotIn("TRANSLATED", proc.stdout)
        self.assertNotIn("not installed", proc.stdout)


if __name__ == "__main__":
    unittest.main()
