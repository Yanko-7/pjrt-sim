# TPU simulator: SGLang-Jax execution backend

This PJRT plugin runs SGLang-Jax without TPU hardware or libtpu, including
multi-device tensor parallelism with overlap scheduling. It reports simulated
TPU7x chiplets and uses XLA's CPU runtime for buffers, transfers, collectives,
integer/control computations, donation and asynchronous functional execution. A C++
model gates public completion using original-HLO plans and timed resource reservations.
Floating-point dots and native Pallas custom calls use deterministic numerical placeholders.

**Scope:** one host and small dense models. Host memory is still physically
allocated. This is not yet a virtual-memory backend for full-size models, a TPU
compiler, or a calibrated latency predictor. No SGLang-Jax source patch is needed
for the included integration test.

## Tested source baseline

- SGLang-Jax: `7ebbbef498b9e484fdc5902986578a3504734e46`.
- Python 3.12; JAX/jaxlib 0.11.1; Flax 0.12.9.
- XLA base: `4426f247713c9f46bad35775cc032265d5795f04`; build with this checkout's
  Bazel configuration.
- Dependencies: install the pinned SGLang-Jax checkout's `python[cpu]` package.
  `constraints.txt` pins the tested dependency versions. The validation runner
  also records the complete installed environment.

For a fresh Python 3.12 environment, with the pinned framework at
`/path/to/sglang-jax`:

```sh
python3.12 -m venv /tmp/pjrt-sim-venv
/tmp/pjrt-sim-venv/bin/pip install -c xla/pjrt/sim/constraints.txt \
  '/path/to/sglang-jax/python[cpu]'
/tmp/pjrt-sim-venv/bin/pip install -r xla/pjrt/sim/profiling-requirements.txt
```

Inside the existing `xla` container, the isolated test environment used during
development is `/tmp/pjrt-sim-venv`, and the pinned framework snapshot is
`/tmp/pjrt-sim-source`. These temporary directories are not part of the repository.

## Build and validate

From the host, with the checkout mounted at `/xla` in container `xla`:

```sh
docker exec -w /xla \
  -e SIM_PYTHON=/tmp/pjrt-sim-venv/bin/python \
  xla bash xla/pjrt/sim/run_tests.sh
```

The runner builds the plugin and tests HLO substitutions, JAX runtime contracts,
native/aliased Pallas calls, and real SGLang-Jax requests:

| Devices / TP | Overlap | Framework checks |
| --- | --- | --- |
| 1 | off | 3 requests, prefix-cache reuse, flush and generate again |
| 2, 4, 8 | on | 11 requests per configuration, including two batches of four concurrent requests with unequal decode lengths, cache reuse and flush |

Multi-device runtime tests exercise all-reduce, all-gather, collective-permute,
resharding, chained asynchronous donation and per-device memory accounting.
Integration tests inspect execution traces to require model and overlap
future-token operations on **all requested devices**, preventing accidental
single-device execution or overlap fallback. Each framework test has a 180-second
timeout, adjustable with `SIM_TEST_TIMEOUT`.

The runner also tests profiler lifecycle/error handling, captures two consecutive
JAX profiles per multi-device configuration, and profiles all framework cases.
Profiles are validated with XProf's actual Trace Viewer converter. The tested
profiler is XProf 2.23.1; its dependencies share the framework environment, with
compatible protobuf and Google API Core versions pinned in the requirement files.

Tests create a tiny local Llama configuration and load dummy weights. No model
weights or tokenizer downloads are needed. The small model uses two attention
heads per device because the pinned framework/JAX combination rejects the
singleton-head explicit-sharding case before backend compilation.

Logs, per-process execution traces, reports and installed versions go into a fresh
directory under `/tmp/pjrt-sim-results` inside the container.

For an individual Python program:

```sh
export JAX_PLATFORMS=tpu
export JAX_ENABLE_COMPILATION_CACHE=false
export PJRT_NAMES_AND_LIBRARY_PATHS=tpu:/xla/bazel-bin/xla/pjrt/sim/pjrt_sim_plugin.so
export PJRT_SIM_TRACE=/tmp/my-sim-run
/tmp/pjrt-sim-venv/bin/python xla/pjrt/sim/smoke_test.py -v

# Start a new process with four simulated devices and real overlap scheduling.
export PJRT_SIM_DEVICE_COUNT=4
/tmp/pjrt-sim-venv/bin/python xla/pjrt/sim/sglang_smoke_test.py --tp-size 4 --overlap
```

Select the simulator in a fresh process. The plugin occupies JAX's `tpu` backend
name so JAX uses its native TPU/Pallas lowering rules. Persistent executable
serialization is deliberately unsupported: cached CPU binaries would bypass the
original-work capture.

`PJRT_SIM_DEVICE_COUNT` defaults to 1 and accepts 1 or an even number up to 256.
The validated configurations are 1, 2, 4 and 8. Devices form a synthetic strip
with two chiplets per chip: device `i` has `coords=(i // 2, 0, 0)` and
`core_on_chip=i % 2`. This supplies a valid JAX mesh, not a measured v7x network
layout. Execution uses the CPU runtime's in-process collectives; multi-host
distributed initialization is not supported.

## Implementation

| File | Responsibility |
| --- | --- |
| `plugin.cc` | PJRT entry point, device identity, MLIR/HLO import, CPU compilation |
| `hlo_model.cc` | Capture original logical work and apply numerical substitutions |
| `instrumentation.cc` | Bind buffer producers, gate PJRT readiness and track accepted calls |
| `execution_plan.cc`, `runtime.cc` | Original-HLO plans, resource reservations and live completion |
| `profiler.cc`, `profiler_api.cc` | Concurrent capture sessions and PJRT profiler extension |
| `profile_report.py` | XProf conversion, observed API costs and buffer dependencies |
| `report.py`, `v7x.json` | Parameterized compute/memory scenario and modeled timeline |
| `*_test.*` | Runtime, lowering, transformation and framework integration checks |

Numerical policy is explicit: floating-point dots return zeros; Pallas outputs
without aliases return zeros; aliased Pallas outputs preserve the input contents.
Other HLO operations execute normally. Thus indexing, sequence lengths, sampling
and request termination retain executable semantics. Placeholder-dependent
branches follow placeholder values, not the results a real model would produce.
Unknown custom-call targets and externally side-effecting TPU calls return an
error. Recognized Shardy annotations are passed to the CPU compiler.

`memory_stats()` reports the logical size of tracked live buffer handles **on
the queried device** and a 96 GiB device capacity. It is not a host allocator measurement and does not include
compiler temporaries or every possible external/async allocation API. Aliased
handles may count the same physical storage more than once. This diagnostic
capacity does not enforce an allocation limit; always bound model and KV-cache
sizes when using the CPU-backed baseline.

## Online timing

Timing is active even when profiling is disabled. PJRT execution completion,
buffer readiness, copies and D2H events wait for modeled completion and the CPU
functional data they require. Direct external references and raw pointer access
also wait. Host-source release still follows the actual CPU buffer ownership
contract, independently of modeled H2D readiness. Unsupported external imports,
async allocation and explicit control-donation APIs fail rather than bypassing
producer tracking; ordinary executable donation is supported.

Set parameters before initializing the JAX backend:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PJRT_SIM_COMPUTE_SCALE` | 1 | Multiply modeled compute/HBM durations |
| `PJRT_SIM_COMMUNICATION_SCALE` | 1 | Multiply modeled host/device transfer and collective durations |
| `PJRT_SIM_LAUNCH_NS` | 1000 | Device launch latency per execution |
| `PJRT_SIM_TRANSFER_NS` | 2000 | Host transfer startup latency |
| `PJRT_SIM_LINK_NS` | 1000 | Device link startup latency per ring round/copy |

Nominal per-device rates are 1.1535e15 FLOP/s, 3.69e12 logical bytes/s,
32e9 host bytes/s and 100e9 device-link bytes/s. They describe a hypothetical
scenario, with one nominal compute rate across supported dot types. They are not
calibrated v7x specifications. Scales must be finite in (0, 1e6]; latency overrides
are integer nanoseconds in [0, 1e9]. Online settings are separate from the Python
offline replay scenario JSON.

The first online model supports uniform identity sharding, static calls, ordinary
2D dots and selected memory operations. Contracting-axis TP dots infer an
all-reduce; explicit full-group partition all-reduce/all-gather/reduce-scatter use
a synthetic directed ring. Ring links are exclusive, with round barriers and
all-participant arrival; reduction arithmetic and collective HBM contention are
not modeled. Copies reserve endpoint HBM. Computation reserves compute and HBM
resources. Submissions use a conservative FIFO reservation policy; each program
waits for all its inputs and the previous execution on its participating devices.
This is not a physical v7x topology or an optimized TPU instruction schedule.

Opaque Pallas kernels, dynamic control flow, unsupported sharding and general
resharding remain explicit cost gaps with dependency-only nodes. CPU execution
may run ahead internally; public readiness is gated by both model and CPU.
Actual Host scheduling, CPU functional execution and callback overhead still
influence request time. This version advances at real time, not an accelerated
global virtual clock, and its request latency is not a calibrated TPU prediction.

## XProf profiling

The plugin exposes `PJRT_Profiler_Extension`. JAX's normal profiler lifecycle
collects simulator events into standard `*.xplane.pb` files. SGLang-Jax source
code remains unchanged. In the configured environment, a complete example is:

```sh
bazel build -c opt //xla/pjrt/sim:pjrt_sim_plugin.so
export PJRT_SIM_DEVICE_COUNT=4
/tmp/pjrt-sim-venv/bin/python xla/pjrt/sim/sglang_smoke_test.py --tp-size 4 --overlap \
  --profile-dir /tmp/sim-profile
xprof server --logdir /tmp/sim-profile
```

Use the JAX backend environment variables from the earlier example. The same
profile generated by SGLang-Jax contains framework host annotations, PJRT
host/pending observations and simulated TPU execution, HLO, DMA and communication
tracks. No Python helper, program snapshot file or manual replay is needed.
`PJRT_SIM_PROFILE_PYTHON` and `PJRT_SIM_PROFILE_HELPER` are no longer used by the
plugin. `native_profile.py` remains a legacy offline conversion utility.

The C++ runtime builds a reusable plan before numerical HLO substitution. At
execution it binds actual device IDs and buffer dependencies, reserves resources,
and schedules completion using a steady clock mapped to real elapsed time.
The timer worker dispatches completion callbacks outside its lock and on a
separate executor. Profiling records runtime reservations; collection only
serializes XSpace. Stopping clips unfinished reservations and omits future work.
Device intervals carry `clock_domain=simulated` and
`clock_alignment=runtime_realtime`. These are model reservations, not hardware
measurements. CPU readiness observation and notification lag have their own
tracks in the measured `cpu_wall` domain.

Every device has an `Executions` track with executable names such as
`jit_jitted_run_model`, `jit_jitted_sampler`, and future-token operations.
`correlation_id` joins these to host submissions; `sim_program_id` exposes the
program ID in Trace Viewer (XProf consumes the standard `program_id` internally).
HLO events retain `framework_op`
paths, including unmodeled attention operations. The framework's host annotations
remain in their original tracks.

| Events | Meaning |
| --- | --- |
| Compile, H2D/copy/Execute submit | Host API intervals, including instrumentation overhead |
| H2D/copy/D2H/Execute submit-to-ready | Time from host call entry to observation of runtime completion; includes queues and submission |
| Host source releasable | The host input may be reused; distinct from output buffer readiness |
| Buffer ready event, Event await | Readiness queries and actual blocking API calls |
| Event callback registration / callback | Registration and callback execution, correlated across threads |
| Buffer external reference | CPU-backed access that may avoid D2H copying |
| Bitcast / buffer destroy | Buffer handle lifecycle; not physical allocation/free counters |

`correlation_id` ties submissions to completion observations. `buffer_id`,
`input_buffers` and `output_buffers` identify buffer handles and dependencies,
including donated chains. IDs are process-local and remain distinct across
successive captures. They do not identify shared backing allocations.

The integration test uses the framework's native profiler request with
`host_tracer_level=1` and `python_tracer_level=0`. Calling `engine.start_profile()`
also uses this backend, but its default tracing options may collect millions of
Python/compiler events. XProf can truncate such a trace before inference events
are visible. For custom request sequences, configure the existing request:

```python
# Set SGLANG_JAX_PROFILER_DIR before creating Engine.
engine.loop.run_until_complete(
    engine.tokenizer_manager.start_profile(host_tracer_level=1, python_tracer_level=0)
)
try:
    engine.generate(input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 4})
finally:
    engine.stop_profile()
```

This starts capture in the scheduler process. An outer `jax.profiler.trace`
around `engine.generate()` alone may miss that child process. Framework host
annotations such as `forward_batch_generation` appear beside plugin events.

Each plugin session accepts at most 100,000 observed events. Modeled device
events are generated during collection and can be more numerous. It reports dropped events,
clips unfinished spans at stop with `incomplete=1`, and ignores late callbacks
from stopped sessions. Capture does not wait for pending device work. When
profiling is disabled, extra completion callbacks are not registered.

To export a report and an optional Trace Viewer JSON file:

```sh
python xla/pjrt/sim/profile_report.py /tmp/sim-profile \
  --output /tmp/profile.report.json --trace-output /tmp/profile.trace.json
```

The report includes operation counts, cumulative observed durations, per-device
unions of Execute pending intervals and buffer dependency edges. Cumulative
durations can overlap; pending intervals are not compute time or utilization.
These CPU observations also feed the standalone virtual replay below, which
uses its own virtual host submission schedule. CPU durations are never used as
TPU compute durations. Complete memory profiling and TPU hardware
counter analysis remain future work.

## Performance model

Set `PJRT_SIM_TRACE` to an output prefix. Each process writes
`<prefix>.<pid>.jsonl`, containing original dot FLOPs, logical bytes, substitution
counts, unmodeled-operation counts and participating device count per accepted
execution. New traces also record `program_id`, profiler `correlation_id`, replica
and partition counts. `<prefix>.<pid>.program<ID>.json` preserves the original
HLO protobuf as JSON (including shapes, sharding, calls and opaque kernel payloads)
and per-instruction work estimates before numerical substitutions. Program IDs
remain unique when executable addresses are reused. Enable tracing before the
process starts, create the output directory, and keep the snapshots beside the
execution JSONL. Work is captured before partitioning (`work_scope=pre_partition`). Executions used
for runtime initialization are also present; traces have no request-phase labels
yet. Static calls are counted per call site; dynamic control flow and higher-order
operations are excluded from timing and counted as unmodeled work.

```sh
python3 xla/pjrt/sim/report.py /tmp/my-sim-run.123.jsonl \
  --hardware xla/pjrt/sim/v7x.json --output /tmp/report.json
```

The initial scenario uses `launch + max(dot_flops / effective_compute,
logical_bytes / effective_bandwidth)`, and serializes these modeled components
within one process stream. Hardware units and assumptions are recorded in the
configuration and copied into every report. Utilization and launch overhead are
adjustable, uncalibrated parameters. This timing formula currently applies only
to single-device calls. Schema v2 reports `null` durations for multi-device calls,
`null` starts after the first untimed call, and a `null` total for such streams.
Pre-partition HLO mixes global shapes with shard-map local shapes, so dividing
its work by the device count would not be a valid parallel performance model.

**Reports are incomplete component estimates.** Pallas internals, dynamic loop
counts, vector compute, actual HBM traffic, communication, host scheduling and
request arrival timing are not modeled. Unknown work is explicitly reported;
the report never marks its total as a complete latency prediction. Framework
`e2e_latency` and throughput logs are CPU wall-clock measurements, not simulated
TPU results. The modeled timeline does not yet control framework completion
events or scheduling.

## Virtual replay

`replay.py` joins one process's execution log and original program snapshots to
its XProf capture using correlation IDs. It reconstructs buffer dependencies and
host thread submission order, then schedules a DAG in integer virtual nanoseconds.
CPU durations, queue waits, idle gaps and compilation times are discarded.

```sh
python xla/pjrt/sim/replay.py /tmp/sim-profile /tmp/my-sim-run.123.jsonl \
  --scenario xla/pjrt/sim/replay_scenario.json --output /tmp/replay
python xla/pjrt/sim/replay.py /tmp/sim-profile /tmp/my-sim-run.123.jsonl \
  --output /tmp/replay-serial --serial-dispatch
```

Use a newly captured profile with the matching JSONL and snapshots; old profiles
lack the program/correlation join. The first argument can also be the converted
Trace Viewer JSON exported by `profile_report.py`.

Outputs are `report.json`, `timeline.json` and a native simulated XProf capture:
`xprof/plugins/profile/simulated/pjrt-simulator.xplane.pb`. The JSON is also
viewable in Perfetto. Both simulated formats use `clock_domain=simulated`;
the native captured XSpace keeps measured host time and, when enabled, modeled
device planes aligned to those observed submissions.
The report includes resource busy time, the scheduling critical chain, inferred
collective count and per-program cost gaps. Each timeline event lists dependencies,
reserved resources and the event that last blocked it.

Build the canonical protobuf descriptor once (also included in `run_tests.sh`):

```sh
bazel build -c opt //xla/pjrt/sim:xplane_descriptor
```

Then run replay as above and open its XProf log directory:

```sh
xprof server --logdir /tmp/replay/xprof --port 8791
```

Select the `simulated` session and **Trace Viewer**. Tracks group the simulated
host, TPU devices, DMA, communication links and dependency nodes. Events carry
HLO IDs, program IDs, modeled FLOPs/bytes, dependencies, resource reservations
and blocking reasons. Unsupported HLO costs remain zero-duration markers with
`cost_status=unknown` and `null` costs represented as text.

The exporter uses the repository's `xplane.proto`, via a build-generated
`FileDescriptorSet` and the existing Python protobuf runtime. It does not need
TensorFlow. Outside this checkout, pass `--xspace-descriptor /path/to/xplane.descriptor.pb`.
A fixed synthetic epoch of 1,000,000,000 ns anchors XSpace lines; offsets and
relative durations come from integer virtual nanoseconds, encoded as picoseconds.
CPU timestamps are never mixed into this profile. Each work event appears on one primary track; other reserved resources are
listed in its stats. Additional `Executions` spans summarize existing work and
are marked `annotation_kind=executable_scope`. They do not add simulated cost.

This export supports Trace Viewer. TPU hardware-counter dashboards, compiler
Graph Viewer and native HLO Op Profile require additional metadata and are not
implemented by this custom-plane export. Displaying modeled costs in XProf does
not make the incomplete timing model a calibrated TPU predictor.

The implementation modules are:

- `virtual_clock.py`: deterministic earliest-feasible scheduling of non-preemptive
  events, dependencies, release times and exclusive resources.
- `communication.py`: explicit directed routes, alpha-plus-bytes/bandwidth transfers,
  store-and-forward hops and ring all-reduce/all-gather/reduce-scatter with round
  barriers. Every collective waits for all participants, including late arrivals.
- `workload.py`: static call expansion, local shard-map scopes, uniform identity
  sharding and ordinary 2D tensor-parallel dots. Output-dimension partitioning uses
  local compute; matching full-group contracting partitions add an all-reduce.
  Other operations receive logical memory traffic estimates when sharding is known.
- `xprof_export.py`: canonical XSpace serialization and simulated resource tracks.
- `replay.py`: capture validation, per-device execution order, conservative buffer
  readiness, H2D/D2H/copy costs and configurable host submission/launch costs.

`replay_scenario.json` is an **uncalibrated example**, with a four-chip line and
two chiplets per chip. Link bandwidths and latencies are hypothetical effective
rates; they are not inferred from public aggregate ICI bandwidth or CPU timings.
Edit the explicit links/routes to test another topology. By default communication
reserves endpoint HBM and compute reserves HBM throughout its roofline duration.
This conservative contention model can be varied with `communication_uses_hbm`;
it is not a fractional bandwidth-sharing model.

**The output is a partial fixed-workload scenario, not predicted request latency.**
Opaque Pallas attention, vector throughput, unknown/uneven sharding, general
resharding, dynamic control flow, direct HLO collective lowering and TPU compiler
fusion/scheduling are not covered. Unsupported operations stay in the dependency
graph with zero cost and a reported gap. Only inferred tensor-parallel dot
all-reduces currently connect HLO to the collective engine. The report always sets
`complete_latency_prediction=false` and names its duration `partial_makespan_ns`.

Host submission costs are scenario inputs. Cross-thread queues, callback causality,
request arrivals and waits are not fully reconstructed. Captured batches remain
fixed. `--serial-dispatch` adds a barrier before submitting the next execution;
this compares dispatch policies on the same DAG, not two native SGLang scheduler
runs. Virtual time does not yet drive PJRT completion events or alter batching.

`run_tests.sh` includes hand-computed timing tests and both replay policies for the
native TP=8 overlap capture, alongside the existing TP=1/2/4/8 regression matrix.

## Per-device HLO load

Replay reports now include `device_load`: estimated dot FLOPs by output dtype,
logical HLO read/write bytes, H2D/D2H bytes, device send/receive bytes, collective
counts, coverage gaps, and the ten largest HLO operations by FLOPs and traffic.
Repeated executions and static call sites accumulate their work. Link bytes are
reported separately: a multi-hop transfer counts once at each endpoint and once
on each traversed link.

A single original program snapshot can also be analyzed without a profile:

```sh
python xla/pjrt/sim/device_load.py /tmp/my-sim-run.123.program10.json \
  --devices 8 --output /tmp/program10.load.json
```

This describes **one invocation**, with the same supported sharding rules and
communication scenario as replay. `--scenario` selects another configuration.
The snapshot is the plugin's JSON wrapper around original HLO; raw HLO text is
not accepted. The device count must match the program's intended SPMD mesh.

Logical traffic is the sum of represented operand reads and result writes; it
is not allocated memory, peak live storage or measured HBM traffic. Fusion, cache
reuse, vector arithmetic and opaque Pallas kernels are not covered. Unknown HLO
cost fields are `null` in the virtual timeline, with explicit gap counts in the
load report. The count of covered instructions is not a percentage of covered
runtime. Balanced estimated FLOPs do not imply balanced total device latency.

## Next extensions

1. Virtual storage for large data tensors, with explicit control-value semantics.
2. Kernel-specific attention estimates using actual sequence lengths and tiling.
3. Request-phase labeling and online PJRT completion driven by virtual time.
4. General sharding/collective lowering and measured topology/bandwidth sharing.
5. Calibration against real v7x profiles when those become available.
