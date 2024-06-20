# SPDX-License-Identifier: LGPL-3.0-or-later

import os

from .utils import (
    AtomicModelTestCase,
)

CUR_DIR = os.path.dirname(__file__)


class EnerAtomicModelTest(AtomicModelTestCase):
    def setUp(self) -> None:
        self.expected_rcut = 5.0
        self.expected_type_map = ["O", "H"]
        self.expected_dim_fparam = 0
        self.expected_dim_aparam = 0
        self.expected_sel_type = [0, 1]
        self.expected_aparam_nall = False
        self.expected_model_output_type = ["energy", "mask"]
        self.model_output_equivariant = []
        self.expected_sel = [46, 92]
        self.expected_sel_mix = sum(self.expected_sel)
        self.expected_has_message_passing = False
        self.aprec_dict = {}
        self.rprec_dict = {}
        self.epsilon_dict = {}


class DipoleAtomicModelTest(AtomicModelTestCase):
    def setUp(self) -> None:
        self.expected_rcut = 5.0
        self.expected_type_map = ["O", "H"]
        self.expected_dim_fparam = 0
        self.expected_dim_aparam = 0
        self.expected_sel_type = [0, 1]
        self.expected_aparam_nall = False
        self.expected_model_output_type = ["dipole", "mask"]
        self.model_output_equivariant = ["dipole"]
        self.expected_sel = [46, 92]
        self.expected_sel_mix = sum(self.expected_sel)
        self.expected_has_message_passing = False
        self.aprec_dict = {}
        self.rprec_dict = {}
        self.epsilon_dict = {}


class PolarAtomicModelTest(AtomicModelTestCase):
    def setUp(self) -> None:
        self.expected_rcut = 5.0
        self.expected_type_map = ["O", "H"]
        self.expected_dim_fparam = 0
        self.expected_dim_aparam = 0
        self.expected_sel_type = [0, 1]
        self.expected_aparam_nall = False
        self.expected_model_output_type = ["polarizability", "mask"]
        self.model_output_equivariant = ["polarizability"]
        self.expected_sel = [46, 92]
        self.expected_sel_mix = sum(self.expected_sel)
        self.expected_has_message_passing = False
        self.aprec_dict = {}
        self.rprec_dict = {}
        self.epsilon_dict = {}


class DosAtomicModelTest(AtomicModelTestCase):
    def setUp(self) -> None:
        self.expected_rcut = 5.0
        self.expected_type_map = ["O", "H"]
        self.expected_dim_fparam = 0
        self.expected_dim_aparam = 0
        self.expected_sel_type = [0, 1]
        self.expected_aparam_nall = False
        self.expected_model_output_type = ["dos", "mask"]
        self.model_output_equivariant = []
        self.expected_sel = [46, 92]
        self.expected_sel_mix = sum(self.expected_sel)
        self.expected_has_message_passing = False
        self.aprec_dict = {}
        self.rprec_dict = {}
        self.epsilon_dict = {}


class ZBLAtomicModelTest(AtomicModelTestCase):
    def setUp(self) -> None:
        self.expected_rcut = 5.0
        self.expected_type_map = ["O", "H", "B"]
        self.expected_dim_fparam = 0
        self.expected_dim_aparam = 0
        self.expected_sel_type = []
        self.expected_aparam_nall = False
        self.expected_model_output_type = ["energy", "mask"]
        self.model_output_equivariant = []
        self.expected_sel = [46, 92, 10]
        self.expected_sel_mix = sum(self.expected_sel)
        self.expected_has_message_passing = False
        self.aprec_dict = {}
        self.rprec_dict = {}
        self.epsilon_dict = {}
        self.tab_file = {
            "use_srtab": f"{CUR_DIR}/../data/zbl_tab_potential/H2O_tab_potential.txt",
            "smin_alpha": 0.1,
            "sw_rmin": 0.2,
            "sw_rmax": 4.0,
        }
