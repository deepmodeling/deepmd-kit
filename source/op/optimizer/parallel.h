#ifndef DP_REMAPPER_H_
#define DP_REMAPPER_H_

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"

using namespace tensorflow;
using namespace tensorflow::grappler;

class DPParallel : public CustomGraphOptimizer {
 public:
  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    return Status::OK();
  }
  std::string name() const override { return "dpparallel"; };
  bool UsesFunctionLibrary() const override { return false; }
  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;
};

#endif  // DP_REMAPPER_H_