# SPDX-License-Identifier: LGPL-3.0-or-later


from .utils import (
    LossTestCase,
)


class LossTest(LossTestCase):
    def setUp(self) -> None:
        LossTestCase.setUp(self)
