// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#ifdef BUILD_TENSORFLOW
#define TF_PRIVATE

#include <string>

#include "DataModifierTF.h"
#include "DeepPotTF.h"
#include "DeepSpinTF.h"
#include "DeepTensorTF.h"
#include "errors.h"

namespace {

std::string empty_model_content() {
  tensorflow::GraphDef graph;
  graph.mutable_versions()->set_producer(1);
  return graph.SerializeAsString();
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

}  // namespace
#endif  // BUILD_TENSORFLOW
