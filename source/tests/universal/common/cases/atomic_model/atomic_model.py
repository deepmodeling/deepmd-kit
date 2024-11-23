# SPDX-License-Identifier: LGPL-3.0-or-later

import os

from .utils import (
    AtomicModelTestCase,
)

CUR_DIR = os.path.dirname(__file__)


class EnerAtomicModelTest(AtomicModelTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expected_rcut = 5.0
        cls.expected_type_map = ["O", "H"]
        cls.expected_dim_fparam = 0
        cls.expected_dim_aparam = 0
        cls.expected_sel_type = [0, 1]
        cls.expected_aparam_nall = False
        cls.expected_model_output_type = ["energy", "mask"]
        cls.model_output_equivariant = []
        cls.expected_sel = [46, 92]
        cls.expected_sel_mix = sum(cls.expected_sel)
        cls.expected_has_message_passing = False
        cls.aprec_dict = {}
        cls.rprec_dict = {}
        cls.epsilon_dict = {}


class DipoleAtomicModelTest(AtomicModelTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expected_rcut = 5.0
        cls.expected_type_map = ["O", "H"]
        cls.expected_dim_fparam = 0
        cls.expected_dim_aparam = 0
        cls.expected_sel_type = [0, 1]
        cls.expected_aparam_nall = False
        cls.expected_model_output_type = ["dipole", "mask"]
        cls.model_output_equivariant = ["dipole"]
        cls.expected_sel = [46, 92]
        cls.expected_sel_mix = sum(cls.expected_sel)
        cls.expected_has_message_passing = False
        cls.aprec_dict = {}
        cls.rprec_dict = {}
        cls.epsilon_dict = {}


class PolarAtomicModelTest(AtomicModelTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expected_rcut = 5.0
        cls.expected_type_map = ["O", "H"]
        cls.expected_dim_fparam = 0
        cls.expected_dim_aparam = 0
        cls.expected_sel_type = [0, 1]
        cls.expected_aparam_nall = False
        cls.expected_model_output_type = ["polarizability", "mask"]
        cls.model_output_equivariant = ["polarizability"]
        cls.expected_sel = [46, 92]
        cls.expected_sel_mix = sum(cls.expected_sel)
        cls.expected_has_message_passing = False
        cls.aprec_dict = {}
        cls.rprec_dict = {}
        cls.epsilon_dict = {}


class DosAtomicModelTest(AtomicModelTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expected_rcut = 5.0
        cls.expected_type_map = ["O", "H"]
        cls.expected_dim_fparam = 0
        cls.expected_dim_aparam = 0
        cls.expected_sel_type = [0, 1]
        cls.expected_aparam_nall = False
        cls.expected_model_output_type = ["dos", "mask"]
        cls.model_output_equivariant = []
        cls.expected_sel = [46, 92]
        cls.expected_sel_mix = sum(cls.expected_sel)
        cls.expected_has_message_passing = False
        cls.aprec_dict = {}
        cls.rprec_dict = {}
        cls.epsilon_dict = {}


class ZBLAtomicModelTest(AtomicModelTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expected_rcut = 5.0
        cls.expected_type_map = ["O", "H", "B"]
        cls.expected_dim_fparam = 0
        cls.expected_dim_aparam = 0
        cls.expected_sel_type = []
        cls.expected_aparam_nall = False
        cls.expected_model_output_type = ["energy", "mask"]
        cls.model_output_equivariant = []
        cls.expected_sel = [46, 92, 10]
        cls.expected_sel_mix = sum(cls.expected_sel)
        cls.expected_has_message_passing = False
        cls.aprec_dict = {}
        cls.rprec_dict = {}
        cls.epsilon_dict = {}
        cls.tab_file = {
            "use_srtab": f"{CUR_DIR}/../data/zbl_tab_potential/H2O_tab_potential.txt",
            "smin_alpha": 0.1,
            "sw_rmin": 0.2,
            "sw_rmax": 4.0,
        }


class PropertyAtomicModelTest(AtomicModelTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expected_rcut = 5.0
        cls.expected_type_map = ["O", "H"]
        cls.expected_dim_fparam = 0
        cls.expected_dim_aparam = 0
        cls.expected_sel_type = [0, 1]
        cls.expected_aparam_nall = False
        cls.expected_model_output_type = ["property", "mask"]
        cls.model_output_equivariant = []
        cls.expected_sel = [46, 92]
        cls.expected_sel_mix = sum(cls.expected_sel)
        cls.expected_has_message_passing = False
        cls.aprec_dict = {}
        cls.rprec_dict = {}
        cls.epsilon_dict = {}
