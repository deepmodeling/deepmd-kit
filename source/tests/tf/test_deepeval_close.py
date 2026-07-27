# SPDX-License-Identifier: LGPL-3.0-or-later
"""Session-close behavior of the TensorFlow ``DeepEval`` evaluators.

Every TF ``DeepEval`` backend caches a ``tf.Session`` via ``sess`` (a
``cached_property``). These tests verify the session is released by ``close()``,
by the context manager, and that ``close()`` never *materializes* a session that
was not already created.
"""

from unittest import (
    TestCase,
    main,
    mock,
)

from deepmd.infer.deep_eval import DeepEval as DeepEvalWrapper
from deepmd.infer.deep_eval import (
    DeepEvalBackend,
)
from deepmd.tf.infer.deep_eval import DeepEval as DeepEvalTF
from deepmd.tf.infer.deep_eval import (
    DeepEvalOld,
)


class TestTFDeepEvalClose(TestCase):
    """The TF backends own a cached ``tf.Session`` that must be closeable."""

    # both the modern backend and the legacy one used by DeepDipole/DeepPolar
    backends = (DeepEvalTF, DeepEvalOld)

    def _bare(self, cls: type) -> object:
        # bypass __init__/__new__ (which need a frozen model); we only exercise
        # the close()/context-manager logic, not graph loading.
        return object.__new__(cls)

    def test_close_closes_materialized_session(self) -> None:
        for cls in self.backends:
            with self.subTest(cls=cls.__name__):
                obj = self._bare(cls)
                fake_sess = mock.Mock()
                obj.__dict__["sess"] = fake_sess  # simulate the cached_property
                obj.close()
                fake_sess.close.assert_called_once()
                # cache dropped so a later access can recreate the session
                self.assertNotIn("sess", obj.__dict__)

    def test_close_without_session_does_not_create_one(self) -> None:
        for cls in self.backends:
            with self.subTest(cls=cls.__name__):
                obj = self._bare(cls)
                self.assertNotIn("sess", obj.__dict__)
                obj.close()  # must be a no-op, not materialize a session
                self.assertNotIn("sess", obj.__dict__)

    def test_context_manager_closes_session(self) -> None:
        for cls in self.backends:
            with self.subTest(cls=cls.__name__):
                obj = self._bare(cls)
                fake_sess = mock.Mock()
                obj.__dict__["sess"] = fake_sess
                with obj as entered:
                    self.assertIs(entered, obj)
                fake_sess.close.assert_called_once()


class TestDeepEvalWrapperClose(TestCase):
    """The high-level ``DeepEval`` wrapper forwards close() to its backend."""

    def _bare_wrapper(self) -> DeepEvalWrapper:
        class _ConcreteEval(DeepEvalWrapper):
            @property
            def output_def(self) -> None:  # the sole abstract member
                return None

        return object.__new__(_ConcreteEval)

    def test_close_forwards_to_backend(self) -> None:
        obj = self._bare_wrapper()
        obj.deep_eval = mock.Mock()
        obj.close()
        obj.deep_eval.close.assert_called_once()

    def test_context_manager_forwards_to_backend(self) -> None:
        obj = self._bare_wrapper()
        obj.deep_eval = mock.Mock()
        with obj as entered:
            self.assertIs(entered, obj)
        obj.deep_eval.close.assert_called_once()


class TestDeepEvalBackendBaseClose(TestCase):
    """The backend base close() is a no-op so session-less backends comply."""

    def test_base_close_is_noop(self) -> None:
        # call the unbound base method on an arbitrary object: it must do
        # nothing and not raise (backends without a session inherit this).
        self.assertIsNone(DeepEvalBackend.close(object()))


if __name__ == "__main__":
    main()
