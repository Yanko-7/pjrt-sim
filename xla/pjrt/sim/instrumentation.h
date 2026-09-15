/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef XLA_PJRT_SIM_INSTRUMENTATION_H_
#define XLA_PJRT_SIM_INSTRUMENTATION_H_

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/sim/hlo_model.h"
#include "xla/pjrt/sim/runtime.h"

namespace xla::sim {
// Accounts for live logical buffers and emits per-execution work to JSONL when
// PJRT_SIM_TRACE is set. Each process writes <prefix>.<pid>.jsonl.
void AddInstrumentation(PJRT_Api& api);
void RegisterRuntime(PJRT_Client* client, RuntimeConfig config);
void RegisterWork(PJRT_LoadedExecutable* executable, WorkEstimate work);
}  // namespace xla::sim
#endif  // XLA_PJRT_SIM_INSTRUMENTATION_H_
