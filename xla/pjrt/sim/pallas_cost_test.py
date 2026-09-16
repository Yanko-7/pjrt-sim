"""Declared kernel costs, scope checks, and replay timing with hand-sized rates."""

import base64
import json
import unittest

from device_load import summarize_device_load
from pallas_cost import pallas_cost
from replay_test import scenario
from virtual_clock import Event, simulate, summarize
from virtual_clock_test import network
from workload import Workload


def instruction(cost=None, **call_fields):
    config = {
        "custom_call_config": {
            "cost_estimate": cost
            or {"flops": 800, "transcendentals": 40, "bytes_accessed": 160},
            **call_fields,
        }
    }
    return {
        "id": "attention",
        "name": "attention",
        "opcode": "custom-call",
        "custom_call_target": "tpu_custom_call",
        "shape": {
            "element_type": "TUPLE",
            "tuple_shapes": [
                {"element_type": "BF16", "dimensions": ["4"]},
            ],
        },
        "backend_config": base64.b64encode(json.dumps(config).encode()).decode(),
        "sharding": {"type": "REPLICATED"},
    }


class PallasCostTest(unittest.TestCase):
    def test_numeric_and_protobuf_string_counts(self):
        expected = {"flops": 800, "transcendentals": 40, "bytes_accessed": 160}
        self.assertEqual(pallas_cost(instruction(), local=True), expected)
        self.assertEqual(
            pallas_cost(
                instruction({k: str(v) for k, v in expected.items()}), local=True
            ),
            expected,
        )

    def test_unknown_scope_and_invalid_counts(self):
        with self.assertRaisesRegex(ValueError, "scope"):
            pallas_cost(instruction(), local=False)
        for bad in (-1, True, None, "NaN", "Infinity", 2**63, 1.5, [], {}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pallas_cost(
                    instruction(
                        {"flops": 800, "transcendentals": 40, "bytes_accessed": bad}
                    ),
                    local=True,
                )
        for bad in ("not base64", base64.b64encode(b"{}").decode()):
            item = instruction()
            item["backend_config"] = bad
            with self.assertRaises(ValueError):
                pallas_cost(item, local=True)

    def test_internal_communication_is_not_silently_ignored(self):
        for item in (
            instruction(has_communication=True),
            instruction(
                {
                    "flops": 1,
                    "transcendentals": 0,
                    "bytes_accessed": 1,
                    "remote_bytes_transferred": 1,
                }
            ),
        ):
            with self.assertRaisesRegex(ValueError, "communication"):
                pallas_cost(item, local=True)

    def test_three_roofline_limits_and_device_accounting(self):
        snapshot = {
            "hlo": {
                "entry_computation_id": "main",
                "computations": [
                    {
                        "id": "main",
                        "root_id": "attention",
                        "instructions": [instruction()],
                    }
                ],
            }
        }
        for rate, expected in ((1e9, 800), (1e7, 4000)):
            config = scenario()
            config["transcendentals_per_second"] = rate
            model = Workload(snapshot, (0, 1), config, network())
            events = [Event(f"start:{d}", "start") for d in (0, 1)] + model.lower()
            self.assertEqual(model.gaps, [])
            self.assertEqual(summarize(simulate(events))["makespan_ns"], expected)
            for row in summarize_device_load(events)["devices"].values():
                self.assertEqual(row["estimated_kernel_flops"], 800)
                self.assertEqual(row["estimated_kernel_bytes_accessed"], 160)
                self.assertEqual(row["estimated_kernel_transcendentals"], 40)
                self.assertEqual(row["estimated_dot_flops_by_dtype"], {})
        config["hbm_bytes_per_second"] = 1e6
        model = Workload(snapshot, (0,), config, network())
        self.assertEqual(
            summarize(simulate([Event("start:0", "start")] + model.lower()))[
                "makespan_ns"
            ],
            160000,
        )

    def test_missing_estimate_retains_dependency_and_gap(self):
        item = instruction()
        del item["backend_config"]
        snapshot = {
            "hlo": {
                "entry_computation_id": "main",
                "computations": [
                    {
                        "id": "main",
                        "root_id": "attention",
                        "instructions": [item],
                    }
                ],
            }
        }
        model = Workload(snapshot, (0,), scenario(), network())
        events = model.lower()
        self.assertEqual(len(model.gaps), 1)
        self.assertEqual(events[0].metadata["cost_status"], "unknown")
        self.assertEqual(events[0].dependencies, ("start:0",))


if __name__ == "__main__":
    unittest.main()
