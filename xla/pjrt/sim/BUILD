load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_binary", "xla_cc_test")

package(default_visibility = ["//visibility:public"])

# Load the canonical XSpace schema in Python without TensorFlow or a checked-in
# generated protobuf module tied to a particular protobuf runtime version.
genrule(
    name = "xplane_descriptor",
    srcs = ["@tsl//tsl/profiler/protobuf:xplane.proto"],
    outs = ["xplane.descriptor.pb"],
    cmd = "$(execpath @com_google_protobuf//:protoc) " +
          "--proto_path=$$(dirname $(location @tsl//tsl/profiler/protobuf:xplane.proto)) " +
          "--descriptor_set_out=$@ $(location @tsl//tsl/profiler/protobuf:xplane.proto)",
    tools = ["@com_google_protobuf//:protoc"],
)

cc_library(
    name = "profiler",
    srcs = ["profiler.cc"],
    hdrs = ["profiler.h"],
    deps = [
        "//xla:future",
        "//xla/pjrt/c:pjrt_c_api_profiler_extension_hdrs",
        "//xla/tsl/profiler/utils:xplane_builder",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

xla_cc_test(
    name = "profiler_test",
    srcs = ["profiler_test.cc"],
    deps = [
        ":profiler",
        "//xla:future",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
        "@tsl//tsl/platform:status_matchers",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

xla_cc_test(
    name = "hlo_model_test",
    srcs = ["hlo_model_test.cc"],
    deps = [
        ":hlo_model",
        ":runtime",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
        "@tsl//tsl/platform:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "instrumentation_test",
    srcs = ["instrumentation_test.cc", "instrumentation.cc", "instrumentation.h"],
    deps = [
        ":hlo_model",
        ":runtime",
        ":profiler",
        "//xla/pjrt/c:pjrt_c_api_cpu_internal",
        "//xla/pjrt/c:pjrt_c_api_status_utils",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "hlo_model",
    srcs = ["hlo_model.cc"],
    hdrs = ["hlo_model.h"],
    deps = [
        ":execution_plan",
        "@com_google_protobuf//:protobuf",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_cost_analysis",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/status:status_macros",
    ],
)

xla_cc_binary(
    name = "pjrt_sim_plugin.so",
    srcs = ["plugin.cc", "instrumentation.cc", "instrumentation.h", "profiler_api.cc"],
    linkshared = True,
    deps = [
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        ":hlo_model",
        ":runtime",
        ":profiler",
        "//xla/backends/profiler/plugin:profiler_error",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt/c:pjrt_c_api_cpu_internal",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_layouts_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_status_utils",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "//xla/pjrt/cpu:cpu_client",
        "//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "//xla/pjrt/proto:compile_options_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@com_google_absl//absl/status:status_macros",
    ],
)

cc_library(
    name = "execution_plan",
    srcs = ["execution_plan.cc"],
    hdrs = ["execution_plan.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/status:status_macros",
    ],
)

cc_library(
    name = "runtime",
    srcs = ["runtime.cc"],
    hdrs = ["runtime.h"],
    deps = [
        ":execution_plan",
        ":profiler",
        "//xla:future",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:env",
    ],
)

xla_cc_test(
    name = "runtime_test",
    srcs = ["runtime_test.cc"],
    deps = [
        ":runtime",
        ":execution_plan",
        "//xla:future",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:status_matchers",
    ],
)

xla_cc_test(
    name = "execution_plan_test",
    srcs = ["execution_plan_test.cc"],
    deps = [
        ":execution_plan",
        "//xla/hlo/parser:hlo_parser",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:status_matchers",
    ],
)
