// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>
#include <sys/stat.h>

#ifdef BUILD_TENSORFLOW
#define TF_PRIVATE

#include <cstdio>
#include <string>

#include "DataModifierTF.h"
#ifdef BUILD_JAX
#include "DeepPotJAX.h"
#endif
#include "DeepPotTF.h"
#include "DeepSpinTF.h"
#include "DeepTensorTF.h"
#include "common.h"
#include "errors.h"

namespace {

std::string empty_model_content() {
  tensorflow::GraphDef graph;
  graph.mutable_versions()->set_producer(1);
  return graph.SerializeAsString();
}

bool path_exists(const std::string& path) {
  struct stat stat_buffer;
  return stat(path.c_str(), &stat_buffer) == 0;
}

TEST(TestTensorFlowSessionLifecycle, failed_init_can_be_retried) {
  const std::string model_content = empty_model_content();

  // Each failure happens after NewSession succeeds. Retrying the same public
  // init method checks that failure leaves the object reusable and destruction
  // checks that the final failed attempt is also safe. Leak detection itself
  // requires an LSan build; ordinary tests cannot observe released resources
  // without relying on unstable thread-count or RSS assertions.
  {
    deepmd::DeepPotTF model;
    EXPECT_THROW(model.init("unused", 0, model_content),
                 deepmd::deepmd_exception);
    EXPECT_THROW(model.init("unused", 0, model_content),
                 deepmd::deepmd_exception);
  }
  {
    deepmd::DeepSpinTF model;
    EXPECT_THROW(model.init("unused", 0, model_content),
                 deepmd::deepmd_exception);
    EXPECT_THROW(model.init("unused", 0, model_content),
                 deepmd::deepmd_exception);
  }
  {
    deepmd::DeepTensorTF model;
    EXPECT_THROW(model.init("_no_such_file.pb"), deepmd::deepmd_exception);
    EXPECT_THROW(model.init("_no_such_file.pb"), deepmd::deepmd_exception);
  }
  {
    deepmd::DipoleChargeModifierTF model;
    EXPECT_THROW(model.init("_no_such_file.pb"), deepmd::deepmd_exception);
    EXPECT_THROW(model.init("_no_such_file.pb"), deepmd::deepmd_exception);
  }
}

TEST(TestTensorFlowSessionLifecycle, successful_init_is_destroyed_normally) {
  // Directly instantiate each backend owner so LSan observes its normal
  // destructor path instead of cleanup through the public dlopen facade.
  deepmd::convert_pbtxt_to_pb("../../tests/infer/deeppot-r.pbtxt",
                              "lifecycle_deeppot.pb");
  {
    deepmd::DeepPotTF model("lifecycle_deeppot.pb");
    EXPECT_GT(model.cutoff(), 0.0);
  }
  std::remove("lifecycle_deeppot.pb");

  deepmd::convert_pbtxt_to_pb("../../tests/infer/deepspin_nlist.pbtxt",
                              "lifecycle_deepspin.pb");
  {
    deepmd::DeepSpinTF model("lifecycle_deepspin.pb");
    EXPECT_GT(model.cutoff(), 0.0);
  }
  std::remove("lifecycle_deepspin.pb");

  deepmd::convert_pbtxt_to_pb("../../tests/infer/deeppolar.pbtxt",
                              "lifecycle_deeptensor.pb");
  {
    deepmd::DeepTensorTF model("lifecycle_deeptensor.pb");
    EXPECT_GT(model.cutoff(), 0.0);
  }
  std::remove("lifecycle_deeptensor.pb");

  deepmd::convert_pbtxt_to_pb("../../tests/infer/dipolecharge_e.pbtxt",
                              "lifecycle_modifier.pb");
  {
    deepmd::DipoleChargeModifierTF model("lifecycle_modifier.pb", 0,
                                         "dipole_charge");
    EXPECT_GT(model.cutoff(), 0.0);
  }
  std::remove("lifecycle_modifier.pb");
}

#ifdef BUILD_JAX
TEST(TestTensorFlowSessionLifecycle, jax_failed_init_can_be_retried) {
  deepmd::DeepPotJAX model;
  EXPECT_THROW(model.init("_no_such_savedmodel"), deepmd::deepmd_exception);
  EXPECT_THROW(model.init("_no_such_savedmodel"), deepmd::deepmd_exception);
}

TEST(TestTensorFlowSessionLifecycle,
     jax_successful_init_is_destroyed_normally) {
  const std::string model_path = "../../tests/infer/deeppot_dpa.savedmodel";
  if (!path_exists(model_path)) {
    GTEST_SKIP() << "JAX SavedModel artifact is not available.";
  }
  deepmd::DeepPotJAX model(model_path);
  EXPECT_GT(model.cutoff(), 0.0);
}
#endif  // BUILD_JAX

}  // namespace
#endif  // BUILD_TENSORFLOW
