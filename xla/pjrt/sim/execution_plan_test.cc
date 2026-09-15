// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
#include "xla/pjrt/sim/execution_plan.h"

#include "gtest/gtest.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "tsl/platform/status_matchers.h"

namespace xla::sim {
namespace {

TEST(ExecutionPlanTest, ContractingShardingCreatesLocalWorkAndAllReduce) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
HloModule test
ENTRY main {
  x = bf16[8,16] parameter(0), sharding={devices=[1,2]0,1}
  w = bf16[16,4] parameter(1), sharding={devices=[2,1]0,1}
  ROOT y = bf16[8,4] dot(x,w), lhs_contracting_dims={1}, rhs_contracting_dims={0}, sharding={replicated}
})"));
  ExecutionPlan plan = BuildExecutionPlan(*module, 2);
  double flops = 0;
  int collectives = 0;
  for (const PlanNode& node : plan.nodes) {
    flops += node.flops;
    collectives += node.kind == PlanNode::Kind::kAllReduce;
  }
  EXPECT_EQ(flops, 2 * 8 * 8 * 4);
  EXPECT_EQ(collectives, 1);
  EXPECT_EQ(plan.cost_gaps, 0);
}

TEST(ExecutionPlanTest, UnknownShardingIsNotDividedByDeviceCount) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
HloModule test
ENTRY main {
  x = bf16[8,16] parameter(0)
  w = bf16[16,4] parameter(1)
  ROOT y = bf16[8,4] dot(x,w), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  ExecutionPlan plan = BuildExecutionPlan(*module, 2);
  EXPECT_EQ(plan.cost_gaps, 1);
  for (const PlanNode& node : plan.nodes) EXPECT_EQ(node.flops, 0);
}

TEST(ExecutionPlanTest, StaticCalleeWorkOccursAtEachCallSite) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
HloModule test
callee {
  x = f32[4] parameter(0)
  ROOT y = f32[4] add(x,x)
}
ENTRY main {
  x = f32[4] parameter(0)
  a = f32[4] call(x), to_apply=callee
  ROOT b = f32[4] call(a), to_apply=callee
})"));
  ExecutionPlan plan = BuildExecutionPlan(*module, 1);
  double bytes = 0;
  for (const PlanNode& node : plan.nodes) bytes += node.bytes;
  EXPECT_EQ(bytes, 2 * 3 * 4 * sizeof(float));
  EXPECT_EQ(plan.cost_gaps, 0);
}

}  // namespace
}  // namespace xla::sim
