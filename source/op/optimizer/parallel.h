// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef DP_REMAPPER_H_
#define DP_REMAPPER_H_

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"

using namespace tensorflow;
using namespace tensorflow::grappler;

class DPParallel : public CustomGraphOptimizer {
 public:
  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    return Status();
  }
  std::string name() const override { return "dpparallel"; };
  bool UsesFunctionLibrary() const override { return false; }
  Status Optimize(Cluster* cluster,
                  const GrapplerItem& item,
                  GraphDef* optimized_graph) override;
#if (TF_MAJOR_VERSION >= 2 && TF_MINOR_VERSION < 6) || TF_MAJOR_VERSION < 2
  // TF 3457a2b122e50b4d44ceaaed5a663d635e5c22df
  void Feedback(Cluster* cluster,
                const GrapplerItem& item,
                const GraphDef& optimized_graph,
                double result) override {}
#endif
};

#endif  // DP_REMAPPER_H_
