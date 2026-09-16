// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
#ifndef XLA_PJRT_SIM_EXECUTION_PLAN_H_
#define XLA_PJRT_SIM_EXECUTION_PLAN_H_

#include <cstdint>
#include <string>
#include <vector>

#include "xla/hlo/ir/hlo_module.h"

namespace xla::sim {

// A topologically ordered, immutable plan from original HLO. Device numbers
// are executable-local ordinals, bound to physical devices at submission.
struct PlanNode {
  enum class Kind {
    kBarrier,
    kCompute,
    kAllReduce,
    kAllGather,
    kReduceScatter
  };
  std::string name;
  std::string framework_op;
  std::string cost_gap;
  std::string cost_source;
  Kind kind = Kind::kBarrier;
  std::vector<int> dependencies;
  double flops = 0;
  double transcendentals = 0;
  double bytes = 0;
};

struct ExecutionPlan {
  std::vector<PlanNode> nodes;
  int root = -1;
  int64_t cost_gaps = 0;
};

// Intentionally limited to uniform sharding and static calls. Unsupported work
// preserves dependencies and is explicitly marked, never divided by TP blindly.
ExecutionPlan BuildExecutionPlan(const HloModule& module, int64_t devices);

}  // namespace xla::sim
#endif  // XLA_PJRT_SIM_EXECUTION_PLAN_H_
