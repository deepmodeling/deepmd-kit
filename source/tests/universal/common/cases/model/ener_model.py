# SPDX-License-Identifier: LGPL-3.0-or-later


from .utils import (
    ModelTestCase,
)


class EnerModelTest(ModelTestCase):
    def setUp(self) -> None:
        self.expected_rcut = 5.0
        self.expected_type_map = ["foo", "bar"]
        self.expected_dim_fparam = 0
        self.expected_dim_aparam = 0
        self.expected_sel_type = [0, 1]
        self.expected_aparam_nall = False
        self.expected_model_output_type = ["energy", "mask"]
        self.expected_sel = [8, 12]
        self.expected_has_message_passing = False
