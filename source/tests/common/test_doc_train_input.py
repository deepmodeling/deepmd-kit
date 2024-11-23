# SPDX-License-Identifier: LGPL-3.0-or-later
import io
import json
import unittest
from contextlib import (
    redirect_stdout,
)

from deepmd.entrypoints.doc import (
    doc_train_input,
)

from ..consistent.common import (
    parameterized,
)


@parameterized(
    (False, True)  # multi_task
)
class TestDocTrainInput(unittest.TestCase):
    @property
    def multi_task(self):
        return self.param[0]

    def test_rst(self) -> None:
        f = io.StringIO()
        with redirect_stdout(f):
            doc_train_input(out_type="rst", multi_task=self.multi_task)
        self.assertNotEqual(f.getvalue(), "")

    def test_json(self) -> None:
        f = io.StringIO()
        with redirect_stdout(f):
            doc_train_input(out_type="json", multi_task=self.multi_task)
        # validate json
        json.loads(f.getvalue())

    def test_json_schema(self) -> None:
        f = io.StringIO()
        with redirect_stdout(f):
            doc_train_input(out_type="json_schema", multi_task=self.multi_task)
        # validate json
        json.loads(f.getvalue())
