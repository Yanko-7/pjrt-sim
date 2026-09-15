"""Analytic sharding and capture-to-virtual-time integration checks."""

import copy
import unittest

from replay import build_replay
from virtual_clock import Event, simulate, summarize
from virtual_clock_test import network
from workload import Workload, tile_shape


def scenario():
    comm = network()
    return {
        "bf16_flops_per_second": 1e9,
        "hbm_bytes_per_second": 1e9,
        "compute_utilization": 1.0,
        "bandwidth_utilization": 1.0,
        "host_submit_ns": 4,
        "launch_ns": 0,
        "communication": {
            "links": {name: vars(link) for name, link in comm.links.items()},
            "routes": comm.routes,
            "host_link": vars(comm.host),
            "communication_uses_hbm": False,
        },
    }


def sharding(tiles):
    return {
        "type": "OTHER",
        "tile_assignment_dimensions": list(map(str, tiles)),
        "iota_reshape_dims": ["2"],
        "iota_transpose_perm": [0],
    }


def dot_program(contract=True):
    def operand(id, shape, tiles, number):
        return {
            "id": id,
            "name": id,
            "opcode": "parameter",
            "parameter_number": str(number),
            "shape": {"element_type": "BF16", "dimensions": list(map(str, shape))},
            "sharding": sharding(tiles) if tiles else {},
        }

    instructions = [
        operand("x", (2, 4), (1, 2) if contract else None, 0),
        operand("y", (4, 6), (2, 1) if contract else (1, 2), 1),
        {
            "id": "z",
            "name": "dot",
            "opcode": "dot",
            "operand_ids": ["x", "y"],
            "shape": {"element_type": "BF16", "dimensions": ["2", "6"]},
            "sharding": {} if contract else sharding((1, 2)),
            "dot_dimension_numbers": {
                "lhs_contracting_dimensions": ["1"],
                "rhs_contracting_dimensions": ["0"],
            },
        },
    ]
    return {
        "hlo": {
            "entry_computation_id": "main",
            "computations": [
                {"id": "main", "root_id": "z", "instructions": instructions}
            ],
        }
    }


class ReplayTest(unittest.TestCase):
    def test_contracting_partition_infers_collective_and_local_work(self):
        model = Workload(dot_program(), (0, 1), scenario(), network())
        events = [Event(f"start:{d}", "start") for d in (0, 1)] + model.lower()
        self.assertEqual(model.gaps, [])
        self.assertEqual(model.inferred_collectives, 1)
        # Each device: 48 FLOPs and 56 bytes -> 56 ns. All-reduce: 2*(1+12).
        self.assertEqual(summarize(simulate(events))["makespan_ns"], 82)

    def test_output_partition_requires_no_collective(self):
        model = Workload(dot_program(False), (0, 1), scenario(), network())
        events = [Event(f"start:{d}", "start") for d in (0, 1)] + model.lower()
        self.assertEqual(model.gaps, [])
        self.assertEqual(model.inferred_collectives, 0)
        # Replicated lhs 16 bytes, local rhs 24, local output 12.
        self.assertEqual(summarize(simulate(events))["makespan_ns"], 52)

    def test_unknown_sharding_and_opaque_work_stay_explicit(self):
        snapshot = dot_program()
        del snapshot["hlo"]["computations"][0]["instructions"][-1]["sharding"]
        model = Workload(snapshot, (0, 1), scenario(), network())
        model.lower()
        self.assertEqual(model.inferred_collectives, 0)
        self.assertIn("unknown sharding", model.gaps[0]["reason"])
        bad = sharding((1, 2))
        bad["iota_transpose_perm"] = [1, 0]
        with self.assertRaises(ValueError):
            tile_shape(bad, 2, 2)

    def test_shared_callee_is_instantiated_per_call(self):
        snapshot = dot_program(False)
        main = snapshot["hlo"]["computations"][0]
        main["id"] = "callee"
        call = {
            "id": "a",
            "name": "a",
            "opcode": "call",
            "operand_ids": ["x", "y"],
            "called_computation_ids": ["callee"],
            "shape": main["instructions"][-1]["shape"],
        }
        second = {**call, "id": "b", "name": "b"}
        snapshot["hlo"]["computations"].append(
            {
                "id": "main",
                "root_id": "root",
                "instructions": [
                    *copy.deepcopy(main["instructions"][:2]),
                    call,
                    second,
                    {
                        "id": "root",
                        "name": "root",
                        "opcode": "tuple",
                        "operand_ids": ["a", "b"],
                        "shape": {"element_type": "TUPLE"},
                    },
                ],
            }
        )
        model = Workload(snapshot, (0, 1), scenario(), network())
        events = model.lower()
        self.assertEqual(len([e for e in events if e.name == "dot"]), 4)
        self.assertEqual(sum(e.duration_ns for e in events), 4 * 52)

    def test_real_capture_contract_and_buffer_dependencies(self):
        def observed(name, correlation, ts, **stats):
            return {
                "ph": "X",
                "pid": 1,
                "tid": 2,
                "name": name,
                "ts": ts,
                "dur": 99999,
                "args": {
                    "clock_domain": "cpu_wall",
                    "correlation_id": correlation,
                    "incomplete": 0,
                    **stats,
                },
            }

        trace = {
            "traceEvents": [
                observed(
                    "PJRT Execute submit",
                    1,
                    0,
                    input_buffers="1,2,3,4",
                    output_buffers="10,11",
                ),
                observed("Execute submit-to-ready", 1, 0, device_id=0),
                observed("Execute submit-to-ready", 1, 0, device_id=1),
                observed(
                    "PJRT Execute submit",
                    2,
                    100000,
                    input_buffers="10,2,11,4",
                    output_buffers="12,13",
                ),
                observed("Execute submit-to-ready", 2, 100000, device_id=0),
                observed("Execute submit-to-ready", 2, 100000, device_id=1),
            ]
        }
        rows = [
            {"correlation_id": c, "program_id": 1, "num_devices": 2} for c in (1, 2)
        ]
        snapshots = {1: dot_program(False)}
        events, report = build_replay(trace, rows, snapshots, scenario())
        schedule = simulate(events)
        self.assertEqual(summarize(schedule)["makespan_ns"], 4 + 52 * 2)
        self.assertEqual(report["executions"], 2)
        self.assertFalse(report["complete_latency_prediction"])
        self.assertEqual(
            report["buffers_assumed_ready_at_capture_start"], ["1", "2", "3", "4"]
        )
        events, _ = build_replay(
            trace, rows, snapshots, scenario(), serial_dispatch=True
        )
        self.assertEqual(summarize(simulate(events))["makespan_ns"], (4 + 52) * 2)
        # CPU durations/gaps never become virtual durations.
        trace["traceEvents"][3]["dur"] *= 1000
        events, _ = build_replay(trace, rows, snapshots, scenario())
        self.assertEqual(summarize(simulate(events))["makespan_ns"], 108)
        with self.assertRaises(ValueError):
            build_replay(trace, rows[:1], snapshots, scenario())
        trace["traceEvents"][0]["args"]["dropped_events"] = 1
        with self.assertRaises(ValueError):
            build_replay(trace, rows, snapshots, scenario())


if __name__ == "__main__":
    unittest.main()
