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


class TestDocTrainInput(unittest.TestCase):
    def test_rst(self):
        f = io.StringIO()
        with redirect_stdout(f):
            doc_train_input(out_type="rst")
        self.assertTrue(f.getvalue() != "")

    def test_json(self):
        f = io.StringIO()
        with redirect_stdout(f):
            doc_train_input(out_type="json")
        # validate json
        json.loads(f.getvalue())

    def test_json_schema(self):
        f = io.StringIO()
        with redirect_stdout(f):
            doc_train_input(out_type="json_schema")
        # validate json
        json.loads(f.getvalue())
