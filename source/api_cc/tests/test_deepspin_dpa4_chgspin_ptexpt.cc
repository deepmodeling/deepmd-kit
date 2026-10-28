// SPDX-License-Identifier: LGPL-3.0-or-later
// Functional test for the RUNTIME charge_spin argument of the DeepSpin
// (.pt2 / pt_expt) inference path.
//
// Every other spin fixture in this tree has dim_chg_spin == 0, so the
// charge_spin argument threaded through DeepSpin::compute -> DeepSpinPTExpt
// is inert on them: a dead ingestion seam would pass all of them. The model
// here (source/tests/infer/gen_dpa4_spin_chgspin.py) is the only one that is
// BOTH is_spin and dim_chg_spin == 2 -- a native-spin DPA4 whose descriptor
// also carries add_chg_spin_ebd=True -- so the argument is load-bearing.
//
// Asserted here:
//   * two DIFFERENT charge_spin vectors give DIFFERENT energies, each
//     matching its own reference section (the seam is live AND correct);
//   * an EMPTY charge_spin reproduces the model's stored default_chg_spin
//     (backward compatibility -- the property most likely to regress, since
//     every pre-existing caller passes nothing);
//   * passing the default value explicitly equals passing nothing (the two
//     ways of selecting the default agree);
//   * both DeepSpinPTExpt::compute overloads are covered: the standalone
//     (build-nlist) one and the LAMMPS (InputNlist) one, each of which does
//     its own charge_spin -> tensor conversion.
//
// Modeled on test_deepspin_dpa4_graph_ptexpt.cc (same native-spin graph
// fixture conventions) and test_deeppot_chg_spin_ptexpt.cc (the non-spin
// DeepPot twin of this charge_spin coverage).
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "DeepPotPTExpt.h"
#include "DeepSpin.h"
#include "DeepSpinPTExpt.h"
#include "expected_ref.h"
#include "neighbor_list.h"
#include "test_utils.h"

// The two BUILD_PT_EXPT* macros are defined by the two PTExpt headers above
// and by nothing else. If a refactor drops either include, the #if guards
// below would silently evaluate to "skip" and this whole suite would report
// PASSED while testing nothing (that exact regression cost 8 dead cases in
// another suite). Turn that failure mode into a compile error.
#ifdef BUILD_PYTORCH
#ifndef BUILD_PT_EXPT_SPIN
#error "BUILD_PT_EXPT_SPIN undefined -- DeepSpinPTExpt.h include was dropped"
#endif
#ifndef BUILD_PT_EXPT
#error "BUILD_PT_EXPT undefined -- DeepPotPTExpt.h include was dropped"
#endif
#endif

// Spin models need relaxed epsilon (same bound as
// test_deepspin_dpa4_graph_ptexpt.cc).
#undef EPSILON
#define EPSILON (std::is_same<VALUETYPE, double>::value ? 1e-10 : 1e-4)

namespace {
constexpr const char* kRefPath =
    "../../tests/infer/deeppot_dpa4_spin_chgspin.expected";
constexpr const char* kModelPath =
    "../../tests/infer/deeppot_dpa4_spin_chgspin.pt2";

// Minimum energy separation that the two charge_spin probes must produce.
// Far above the float32 comparison bound (1e-4) so the "different energies"
// assertion means something for BOTH instantiations; the generator asserts
// the same property at 1e-6 on the fp64 reference.
constexpr double kMinChgSpinGap = 1e-3;

// One reference section (PBC or NoPbc, default or explicit charge_spin).
template <class VALUETYPE>
struct SpinRefSection {
  std::vector<VALUETYPE> e, f, fm, tot_v, atom_v;
  double tot_e = 0.;
  int natoms = 0;

  void load(deepmd_test::ExpectedRef& ref, const char* section) {
    e = ref.template get<VALUETYPE>(section, "expected_e");
    f = ref.template get<VALUETYPE>(section, "expected_f");
    fm = ref.template get<VALUETYPE>(section, "expected_fm");
    tot_v = ref.template get<VALUETYPE>(section, "expected_tot_v");
    atom_v = ref.template get<VALUETYPE>(section, "expected_atom_v");
    natoms = static_cast<int>(e.size());
    EXPECT_EQ(natoms * 3, static_cast<int>(f.size()));
    EXPECT_EQ(natoms * 3, static_cast<int>(fm.size()));
    EXPECT_EQ(9, static_cast<int>(tot_v.size()));
    EXPECT_EQ(natoms * 9, static_cast<int>(atom_v.size()));
    tot_e = 0.;
    for (int ii = 0; ii < natoms; ++ii) {
      tot_e += e[ii];
    }
  }
};

// Compare one compute() result against a reference section.
template <class VALUETYPE>
void expect_matches(const SpinRefSection<VALUETYPE>& ref,
                    double ener,
                    const std::vector<VALUETYPE>& force,
                    const std::vector<VALUETYPE>& force_mag,
                    const std::vector<VALUETYPE>& virial,
                    double eps) {
  EXPECT_EQ(force.size(), static_cast<size_t>(ref.natoms * 3));
  EXPECT_EQ(force_mag.size(), static_cast<size_t>(ref.natoms * 3));
  EXPECT_LT(fabs(ener - ref.tot_e), eps);
  for (int ii = 0; ii < ref.natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - ref.f[ii]), eps);
    EXPECT_LT(fabs(force_mag[ii] - ref.fm[ii]), eps);
  }
  EXPECT_FALSE(virial.empty()) << "Virial should not be empty";
  EXPECT_EQ(virial.size(), 9u);
  for (int ii = 0; ii < 9; ++ii) {
    EXPECT_LT(fabs(virial[ii] - ref.tot_v[ii]), eps);
  }
}
}  // namespace

template <class VALUETYPE>
class TestInferDeepSpinDpa4ChgSpinPtExpt : public ::testing::Test {
 protected:
  // 6-atom system (3 Ni, spin-active; 3 O, non-magnetic) -- verbatim from
  // gen_dpa4_spin_chgspin.py's _COORDS/_CELL/_SPINS/_ATYPES.
  std::vector<VALUETYPE> coord = {1.0, 1.0, 1.0, 3.2, 1.4, 1.1, 1.3, 1.8, 1.0,
                                  0.4, 1.2, 1.6, 3.6, 2.0, 1.3, 3.4, 0.7, 1.7};
  std::vector<VALUETYPE> spin = {0.11,  0.05,  -0.02, -0.07, 0.09,  0.03,
                                 0.02,  -0.06, 0.08,  0.01,  -0.01, 0.02,
                                 -0.02, 0.03,  -0.01, 0.015, 0.02,  -0.03};
  std::vector<int> atype = {0, 0, 0, 1, 1, 1};
  std::vector<VALUETYPE> box = {6., 0., 0., 0., 6., 0., 0., 0., 6.};
  std::vector<VALUETYPE> nobox = {};

  // charge_spin is always double regardless of VALUETYPE.
  // The FiLM embedding is CATEGORICAL (charge -> index charge+100, spin ->
  // index spin), so both probes are integer-valued and differ in BOTH
  // components: [0.0, 1.0] -> (100, 1) is the model's stored default,
  // [1.0, 2.0] -> (101, 2) is the explicit runtime probe.
  std::vector<double> charge_spin_default = {0.0, 1.0};
  std::vector<double> charge_spin_explicit = {1.0, 2.0};

  SpinRefSection<VALUETYPE> pbc_default, pbc_explicit;
  SpinRefSection<VALUETYPE> nopbc_default, nopbc_explicit;

  deepmd::DeepSpin dp;

  void SetUp() override {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT_SPIN
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    std::ifstream model_file(kModelPath);
    if (!model_file.good()) {
      GTEST_SKIP() << "Skip because " << kModelPath
                   << " was not generated (run "
                      "source/tests/infer/gen_dpa4_spin_chgspin.py).";
    }
    dp.init(kModelPath);

    deepmd_test::ExpectedRef ref;
    ref.load(kRefPath);
    pbc_default.load(ref, "pbc_default");
    pbc_explicit.load(ref, "pbc_explicit");
    nopbc_default.load(ref, "nopbc_default");
    nopbc_explicit.load(ref, "nopbc_explicit");

    // The references themselves must be anti-vacuous: if the two sections
    // carried the same energy, every "charge_spin changes the output" check
    // below would pass for the wrong reason.
    EXPECT_GT(fabs(pbc_explicit.tot_e - pbc_default.tot_e), kMinChgSpinGap)
        << "reference sections pbc_default/pbc_explicit are degenerate; "
           "regenerate with source/tests/infer/gen_dpa4_spin_chgspin.py";
    EXPECT_GT(fabs(nopbc_explicit.tot_e - nopbc_default.tot_e), kMinChgSpinGap)
        << "reference sections nopbc_default/nopbc_explicit are degenerate";
  };

  void TearDown() override {};
};

TYPED_TEST_SUITE(TestInferDeepSpinDpa4ChgSpinPtExpt, ValueTypes);

TYPED_TEST(TestInferDeepSpinDpa4ChgSpinPtExpt, dim_chg_spin) {
  deepmd::DeepSpin& dp = this->dp;
  // 0 here would mean the archive does not carry the charge-spin slot at all
  // and every other case in this file is testing nothing.
  EXPECT_EQ(dp.dim_chg_spin(), 2);
}

TYPED_TEST(TestInferDeepSpinDpa4ChgSpinPtExpt, test_get_use_spin) {
  deepmd::DeepSpin& dp = this->dp;
  std::vector<bool> use_spin = dp.get_use_spin();
  EXPECT_EQ(use_spin.size(), 2);
  EXPECT_TRUE(use_spin[0]);   // Ni carries a magnetic moment
  EXPECT_FALSE(use_spin[1]);  // O does not
}

// ============================================================================
// Standalone (build-nlist) path -- DeepSpinPTExpt::compute, no InputNlist
// ============================================================================

// THE core assertion: two different runtime charge_spin vectors must give two
// different energies, and each must equal its own reference.
TYPED_TEST(TestInferDeepSpinDpa4ChgSpinPtExpt,
           cpu_build_nlist_two_charge_spin) {
  using VALUETYPE = TypeParam;
  deepmd::DeepSpin& dp = this->dp;

  double ener_def, ener_exp;
  std::vector<VALUETYPE> force_def, force_mag_def, virial_def;
  std::vector<VALUETYPE> force_exp, force_mag_exp, virial_exp;

  dp.compute(ener_def, force_def, force_mag_def, virial_def, this->coord,
             this->spin, this->atype, this->box, {}, {},
             this->charge_spin_default);
  dp.compute(ener_exp, force_exp, force_mag_exp, virial_exp, this->coord,
             this->spin, this->atype, this->box, {}, {},
             this->charge_spin_explicit);

  // The runtime argument reaches the model at all ...
  EXPECT_GT(fabs(ener_exp - ener_def), kMinChgSpinGap)
      << "charge_spin " << this->charge_spin_default[0] << ","
      << this->charge_spin_default[1] << " and "
      << this->charge_spin_explicit[0] << "," << this->charge_spin_explicit[1]
      << " produced the same energy (" << ener_def << " vs " << ener_exp
      << "): the runtime charge_spin is being ignored by the DeepSpin path.";
  // ... and lands on the right values, per charge_spin.
  expect_matches(this->pbc_default, ener_def, force_def, force_mag_def,
                 virial_def, EPSILON);
  expect_matches(this->pbc_explicit, ener_exp, force_exp, force_mag_exp,
                 virial_exp, EPSILON);
}

// Backward compatibility: an EMPTY charge_spin must reproduce the model's
// stored default_chg_spin -- this is what every pre-existing caller does.
TYPED_TEST(TestInferDeepSpinDpa4ChgSpinPtExpt,
           cpu_build_nlist_empty_is_default) {
  using VALUETYPE = TypeParam;
  deepmd::DeepSpin& dp = this->dp;

  double ener_empty, ener_default_value;
  std::vector<VALUETYPE> f_empty, fm_empty, v_empty;
  std::vector<VALUETYPE> f_val, fm_val, v_val;

  // No charge_spin argument at all (the pre-existing call shape).
  dp.compute(ener_empty, f_empty, fm_empty, v_empty, this->coord, this->spin,
             this->atype, this->box);
  expect_matches(this->pbc_default, ener_empty, f_empty, fm_empty, v_empty,
                 EPSILON);

  // Passing the stored default explicitly must select the same behaviour.
  dp.compute(ener_default_value, f_val, fm_val, v_val, this->coord, this->spin,
             this->atype, this->box, {}, {}, this->charge_spin_default);
  EXPECT_LT(fabs(ener_empty - ener_default_value), EPSILON)
      << "an empty charge_spin and an explicit default_chg_spin disagree";
  for (int ii = 0; ii < this->pbc_default.natoms * 3; ++ii) {
    EXPECT_LT(fabs(f_empty[ii] - f_val[ii]), EPSILON);
    EXPECT_LT(fabs(fm_empty[ii] - fm_val[ii]), EPSILON);
  }

  // And it must NOT accidentally be the explicit-probe behaviour.
  EXPECT_GT(fabs(ener_empty - this->pbc_explicit.tot_e), kMinChgSpinGap)
      << "the empty-charge_spin result equals the explicit-probe reference; "
         "the stored default is not being used.";
}

TYPED_TEST(TestInferDeepSpinDpa4ChgSpinPtExpt,
           cpu_build_nlist_atomic_explicit) {
  using VALUETYPE = TypeParam;
  deepmd::DeepSpin& dp = this->dp;
  SpinRefSection<VALUETYPE>& ref = this->pbc_explicit;

  double ener;
  std::vector<VALUETYPE> force, force_mag, virial, atom_ener, atom_vir;
  dp.compute(ener, force, force_mag, virial, atom_ener, atom_vir, this->coord,
             this->spin, this->atype, this->box, {}, {},
             this->charge_spin_explicit);

  expect_matches(ref, ener, force, force_mag, virial, EPSILON);
  EXPECT_EQ(atom_ener.size(), static_cast<size_t>(ref.natoms));
  for (int ii = 0; ii < ref.natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - ref.e[ii]), EPSILON);
  }
  EXPECT_FALSE(atom_vir.empty()) << "Atomic virial should not be empty";
  EXPECT_EQ(atom_vir.size(), static_cast<size_t>(ref.natoms * 9));
  for (int ii = 0; ii < ref.natoms * 9; ++ii) {
    EXPECT_LT(fabs(atom_vir[ii] - ref.atom_v[ii]), EPSILON);
  }
}

// The size check on the runtime vector (the other branch of "charge_spin is
// non-empty"): a wrong-width charge_spin must be rejected, not silently
// truncated/padded into the model. Mirrors
// test_deeppot_chg_spin_jax.cc::rejects_invalid_input_size for DeepSpin.
TYPED_TEST(TestInferDeepSpinDpa4ChgSpinPtExpt,
           rejects_invalid_charge_spin_size) {
  using VALUETYPE = TypeParam;
  deepmd::DeepSpin& dp = this->dp;
  const std::vector<double> invalid_charge_spin = {1.0, 2.0, 3.0};

  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;
  EXPECT_THROW(
      dp.compute(ener, force, force_mag, virial, this->coord, this->spin,
                 this->atype, this->box, {}, {}, invalid_charge_spin),
      deepmd::deepmd_exception);
}

// ============================================================================
// LAMMPS path (explicit InputNlist, nghost=0) -- the SECOND
// DeepSpinPTExpt::compute overload, with its own charge_spin conversion.
// NoPBC only (no ghost atoms needed for a nghost=0 InputNlist), matching
// test_deepspin_dpa4_graph_ptexpt.cc.
// ============================================================================

TYPED_TEST(TestInferDeepSpinDpa4ChgSpinPtExpt, cpu_lmp_nlist_two_charge_spin) {
  using VALUETYPE = TypeParam;
  deepmd::DeepSpin& dp = this->dp;
  const int natoms = this->nopbc_default.natoms;

  std::vector<std::vector<int> > nlist_data = {
      {1, 2, 3, 4, 5}, {0, 2, 3, 4, 5}, {0, 1, 3, 4, 5},
      {0, 1, 2, 4, 5}, {0, 1, 2, 3, 5}, {0, 1, 2, 3, 4}};
  std::vector<int> ilist(natoms), numneigh(natoms);
  std::vector<int*> firstneigh(natoms);
  deepmd::InputNlist inlist(natoms, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);

  double ener_def, ener_exp;
  std::vector<VALUETYPE> f_def, fm_def, v_def, f_exp, fm_exp, v_exp;

  // Empty charge_spin -> stored default_chg_spin.
  dp.compute(ener_def, f_def, fm_def, v_def, this->coord, this->spin,
             this->atype, this->nobox, 0, inlist, 0);
  dp.compute(ener_exp, f_exp, fm_exp, v_exp, this->coord, this->spin,
             this->atype, this->nobox, 0, inlist, 0, {}, {},
             this->charge_spin_explicit);

  EXPECT_GT(fabs(ener_exp - ener_def), kMinChgSpinGap)
      << "the LAMMPS-nlist overload ignores the runtime charge_spin";
  expect_matches(this->nopbc_default, ener_def, f_def, fm_def, v_def, EPSILON);
  expect_matches(this->nopbc_explicit, ener_exp, f_exp, fm_exp, v_exp, EPSILON);
}

TYPED_TEST(TestInferDeepSpinDpa4ChgSpinPtExpt, cpu_lmp_nlist_atomic_explicit) {
  using VALUETYPE = TypeParam;
  deepmd::DeepSpin& dp = this->dp;
  SpinRefSection<VALUETYPE>& ref = this->nopbc_explicit;
  const int natoms = ref.natoms;

  std::vector<std::vector<int> > nlist_data = {
      {1, 2, 3, 4, 5}, {0, 2, 3, 4, 5}, {0, 1, 3, 4, 5},
      {0, 1, 2, 4, 5}, {0, 1, 2, 3, 5}, {0, 1, 2, 3, 4}};
  std::vector<int> ilist(natoms), numneigh(natoms);
  std::vector<int*> firstneigh(natoms);
  deepmd::InputNlist inlist(natoms, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);

  double ener;
  std::vector<VALUETYPE> force, force_mag, virial, atom_ener, atom_vir;
  dp.compute(ener, force, force_mag, virial, atom_ener, atom_vir, this->coord,
             this->spin, this->atype, this->nobox, 0, inlist, 0, {}, {},
             this->charge_spin_explicit);

  expect_matches(ref, ener, force, force_mag, virial, EPSILON);
  EXPECT_EQ(atom_ener.size(), static_cast<size_t>(natoms));
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - ref.e[ii]), EPSILON);
  }
  EXPECT_FALSE(atom_vir.empty()) << "Atomic virial should not be empty";
  EXPECT_EQ(atom_vir.size(), static_cast<size_t>(natoms * 9));
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atom_vir[ii] - ref.atom_v[ii]), EPSILON);
  }
}
