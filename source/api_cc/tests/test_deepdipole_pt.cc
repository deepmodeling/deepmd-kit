// SPDX-License-Identifier: LGPL-3.0-or-later
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

#include "DeepTensor.h"
#include "neighbor_list.h"
#include "test_utils.h"

template <class VALUETYPE>
class TestInferDeepTensorPt : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};

  // Expected global tensor values from Python inference
  std::vector<VALUETYPE> expected_global_tensor = {0.2338104, 0.23701073,
                                                   0.2334505};

  // Expected atomic tensor values from Python inference (flattened)
  std::vector<VALUETYPE> expected_atom_tensor = {-0.1808925408386811,
                                                 0.3190798607195795,
                                                 0.04760079958216837,
                                                 -0.0,
                                                 -0.0,
                                                 0.0,
                                                 0.0,
                                                 0.0,
                                                 -0.0,
                                                 0.4147029447879755,
                                                 -0.08206913353381971,
                                                 0.1858497008385067,
                                                 0.0,
                                                 -0.0,
                                                 0.0,
                                                 0.0,
                                                 0.0,
                                                 -0.0};

  int natoms = 6;
  int output_dim = 3;

  deepmd::DeepTensor dt;

  void SetUp() override {
    std::string file_name = "../../tests/infer/deepdipole_pt.pth";
    dt.init(file_name);
  };

  void TearDown() override {};
};

TYPED_TEST_SUITE(TestInferDeepTensorPt, ValueTypes);

TYPED_TEST(TestInferDeepTensorPt, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_global_tensor = this->expected_global_tensor;
  std::vector<VALUETYPE>& expected_atom_tensor = this->expected_atom_tensor;
  int& natoms = this->natoms;
  int& output_dim = this->output_dim;
  deepmd::DeepTensor& dt = this->dt;
  // Use reasonable tolerance for minimal trained model
  double tensor_tol = 1e-6;

  std::vector<VALUETYPE> global_tensor, force, virial, atom_tensor, atom_virial;

  dt.compute(global_tensor, force, virial, atom_tensor, atom_virial, coord,
             atype, box);

  EXPECT_EQ(global_tensor.size(), output_dim);
  EXPECT_EQ(atom_tensor.size(), natoms * output_dim);
  EXPECT_EQ(force.size(), natoms * output_dim * 3);
  EXPECT_EQ(virial.size(), output_dim * 9);
  EXPECT_EQ(atom_virial.size(), natoms * output_dim * 9);

  for (int ii = 0; ii < output_dim; ++ii) {
    EXPECT_LT(fabs(global_tensor[ii] - expected_global_tensor[ii]), tensor_tol);
  }

  for (int ii = 0; ii < natoms * output_dim; ++ii) {
    EXPECT_LT(fabs(atom_tensor[ii] - expected_atom_tensor[ii]), tensor_tol);
  }
}

TYPED_TEST(TestInferDeepTensorPt, cpu_lmp_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_global_tensor = this->expected_global_tensor;
  std::vector<VALUETYPE>& expected_atom_tensor = this->expected_atom_tensor;
  int& natoms = this->natoms;
  int& output_dim = this->output_dim;
  deepmd::DeepTensor& dt = this->dt;
  double ener_tol = 1e-10;

  float rc = dt.cutoff();
  int nloc = coord.size() / 3;
  std::vector<VALUETYPE> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int> > nlist_data;
  _build_nlist<VALUETYPE>(nlist_data, coord_cpy, atype_cpy, mapping, coord,
                          atype, box, rc);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);

  std::vector<VALUETYPE> global_tensor, force, virial, atom_tensor, atom_virial;

  dt.compute(global_tensor, force, virial, atom_tensor, atom_virial, coord_cpy,
             atype_cpy, box, nall - nloc, inlist);

  EXPECT_EQ(global_tensor.size(), output_dim);
  EXPECT_EQ(atom_tensor.size(), natoms * output_dim);

  for (int ii = 0; ii < output_dim; ++ii) {
    EXPECT_LT(fabs(global_tensor[ii] - expected_global_tensor[ii]), ener_tol);
  }

  for (int ii = 0; ii < natoms * output_dim; ++ii) {
    EXPECT_LT(fabs(atom_tensor[ii] - expected_atom_tensor[ii]), ener_tol);
  }
}

TYPED_TEST(TestInferDeepTensorPt, print_summary) {
  deepmd::DeepTensor& dt = this->dt;
  dt.print_summary("");
}

TYPED_TEST(TestInferDeepTensorPt, get_type_map) {
  deepmd::DeepTensor& dt = this->dt;
  std::string type_map_str;
  dt.get_type_map(type_map_str);
  // Parse the type map string manually
  std::vector<std::string> type_map;
  std::istringstream iss(type_map_str);
  std::string token;
  while (iss >> token) {
    type_map.push_back(token);
  }
  EXPECT_EQ(type_map.size(), 2);
  EXPECT_EQ(type_map[0], "O");
  EXPECT_EQ(type_map[1], "H");
}

TYPED_TEST(TestInferDeepTensorPt, get_properties) {
  deepmd::DeepTensor& dt = this->dt;

  EXPECT_EQ(dt.numb_types(), 2);
  EXPECT_EQ(dt.output_dim(), 3);
  EXPECT_DOUBLE_EQ(dt.cutoff(), 4.0);

  std::vector<int> sel_types = dt.sel_types();
  EXPECT_EQ(sel_types.size(), 2);  // PyTorch models always return all types
  EXPECT_EQ(sel_types[0], 0);      // Type 0 (O)
  EXPECT_EQ(sel_types[1],
            1);  // Type 1 (H) - included but may have zero results
}
