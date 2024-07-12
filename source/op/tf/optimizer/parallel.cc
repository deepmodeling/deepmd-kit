// SPDX-License-Identifier: LGPL-3.0-or-later
// only support v1.15 or v2
#include "tensorflow/core/public/version.h"
// skip windows
#ifndef _WIN32
#if TF_MAJOR_VERSION >= 2 || (TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 15)

#if TF_MAJOR_VERSION >= 2 && TF_MINOR_VERSION >= 7
// breaking change in tf 2.7: Renaming of tensorflow::int64 to int_64_t
#define TF_INT64 int64_t
#else
#define TF_INT64 tensorflow::int64
#endif

#include "parallel.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/util.h"

// based on tensorflow/core/grappler/optimizers/remapper.cc

struct RemapperContext {
  explicit RemapperContext(GrapplerItem *item, Status *status)
      : nodes_to_preserve(item->NodesToPreserve()),
        graph_view(&item->graph, status) {}

  std::unordered_set<std::string> nodes_to_preserve;
  utils::MutableGraphView graph_view;
};

bool IsProdForce(const NodeDef &node) { return node.op() == "ProdForceSeA"; }

bool FindProdForce(RemapperContext *ctx, int node_index) {
  const auto *node_view = ctx->graph_view.GetNode(node_index);
  const auto *node_def = node_view->node();
  return IsProdForce(*node_def);
}

TF_INT64 GetNThreads() {
  // the number of threads is based on the session...
  // For convenience, we use environment variable directly
  TF_INT64 tot = 1;
  Status status =
      ReadInt64FromEnvVar("TF_INTER_OP_PARALLELISM_THREADS", 1, &tot);
  if (!status.ok()) {
    tot = 1;
  }
  return tot;
}

Status ParallelProdForce(RemapperContext *ctx,
                         int node_index,
                         std::vector<bool> *invalidated_nodes,
                         std::vector<bool> *nodes_to_delete) {
  // skip on GPUs
  if (GetNumAvailableGPUs() > 0) {
    return Status();
  }

  const NodeDef *ori_node = ctx->graph_view.GetNode(node_index)->node();
  auto &src_attr = ori_node->attr();
  TF_INT64 tot = GetNThreads();
  if (tot <= 1) {
    return Status();
  }

  NodeDef sum_node;
  sum_node.set_name(ori_node->name());
  sum_node.set_op("AddN");
  sum_node.set_device(ori_node->device());
  auto *sum_attr = sum_node.mutable_attr();
  (*sum_attr)["N"].set_i(tot);
  (*sum_attr)["T"] = src_attr.at("T");

  utils::Mutation *mutation = ctx->graph_view.GetMutationBuilder();
  Status status;

  for (int ii = 0; ii < tot; ++ii) {
    NodeDef sub_node;
    sub_node.set_name(ori_node->name() + "/sub_" + std::to_string(ii));
    sub_node.set_op("ParallelProdForceSeA");
    sub_node.set_device(ori_node->device());
    // copy input
    for (int jj = 0; jj < 4; ++jj) {
      sub_node.add_input(ori_node->input(jj));
    }
    // set frac
    auto *sub_attr = sub_node.mutable_attr();
    (*sub_attr)["T"] = src_attr.at("T");
    (*sub_attr)["n_a_sel"] = src_attr.at("n_a_sel");
    (*sub_attr)["n_r_sel"] = src_attr.at("n_r_sel");
    (*sub_attr)["parallel"].set_b(true);
    (*sub_attr)["start_frac"].set_f((float)ii / (float)tot);
    (*sub_attr)["end_frac"].set_f((float)(ii + 1) / (float)tot);
    sum_node.add_input(sub_node.name());
    mutation->AddNode(std::move(sub_node), &status);
  }

  mutation->AddNode(std::move(sum_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());
  (*invalidated_nodes)[node_index] = true;

  return Status();
}

Status DPParallel::Optimize(Cluster *cluster,
                            const GrapplerItem &item,
                            GraphDef *optimized_graph) {
  GrapplerItem mutable_item = item;
  Status status;
  RemapperContext ctx(&mutable_item, &status);
  TF_RETURN_IF_ERROR(status);
  // Processing graph in reverse-topological sorted order allows to remap
  // longer chains of dependent ops in one pass.
  TF_RETURN_IF_ERROR(
      ctx.graph_view.SortTopologically(/*ignore_cycles=*/false, {}));

  const int num_nodes = item.graph.node_size();
  // Skip nodes that were invalidated by a remapper, e.g. do not process BiasAdd
  // and Activation nodes that were fused into a Conv2D node.
  std::vector<bool> invalidated_nodes(num_nodes);
  std::vector<bool> nodes_to_delete(num_nodes);

  for (int i = num_nodes - 1; i >= 0; --i) {
    // Check if node was invalidated by one of the previous remaps.
    if (invalidated_nodes[i] || nodes_to_delete[i]) {
      continue;
    }
    if (!item.optimization_options().is_eager_mode) {
      // Remap gelu
      std::map<std::string, int> matched_nodes_map;
      std::set<int> remove_node_indices;
      if (FindProdForce(&ctx, i)) {
        TF_RETURN_IF_ERROR(
            ParallelProdForce(&ctx, i, &invalidated_nodes, &nodes_to_delete));
        continue;
      }
    }
  }

  // Remove invalidated nodes.
  utils::Mutation *mutation = ctx.graph_view.GetMutationBuilder();
  for (int i = 0; i < num_nodes; ++i) {
    if (nodes_to_delete[i]) {
      mutation->RemoveNode(ctx.graph_view.GetNode(i));
    }
  }
  TF_RETURN_IF_ERROR(mutation->Apply());

  *optimized_graph = std::move(mutable_item.graph);

  return Status();
}

REGISTER_GRAPH_OPTIMIZER_AS(DPParallel, "dpparallel");

#endif
#endif
