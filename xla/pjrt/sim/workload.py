"""Small, explicit analytic lowering of original HLO, before CPU substitutions.

Uniform identity-order sharding, ordinary 2D tensor-parallel dots, and local
author-declared Pallas costs are resolved here. Unsupported work remains a
dependency node with a coverage gap.
"""

import math

from communication import nonnegative
from pallas_cost import pallas_cost
from virtual_clock import Event

ANNOTATIONS = {
    "Sharding",
    "SPMDFullToShardShape",
    "SPMDShardToFullShape",
    "xla.sdy.FuncResultSharding",
    "xla.sdy.GlobalToLocalShape",
    "xla.sdy.LocalToGlobalShape",
}
VIEWS = {
    "parameter",
    "constant",
    "tuple",
    "get-tuple-element",
    "bitcast",
    "reshape",
    "partition-id",
    "replica-id",
}
ELEMENTWISE = {
    "abs",
    "add",
    "and",
    "broadcast",
    "clamp",
    "compare",
    "concatenate",
    "convert",
    "copy",
    "cosine",
    "divide",
    "dynamic-slice",
    "dynamic-update-slice",
    "exponential",
    "gather",
    "iota",
    "maximum",
    "minimum",
    "multiply",
    "negate",
    "not",
    "or",
    "pad",
    "power",
    "remainder",
    "reverse",
    "rsqrt",
    "select",
    "sine",
    "slice",
    "sqrt",
    "subtract",
    "tanh",
    "transpose",
    "xor",
}
WIDTHS = {
    "PRED": 1,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "S8": 1,
    "U8": 1,
    "S16": 2,
    "U16": 2,
    "S32": 4,
    "U32": 4,
    "S64": 8,
    "U64": 8,
    "C64": 8,
    "C128": 16,
}


def shape_bytes(shape, tiles=None):
    if shape["element_type"] == "TUPLE":
        return sum(shape_bytes(s) for s in shape["tuple_shapes"])
    if any(shape.get("is_dynamic_dimension", [])):
        raise ValueError("dynamic shape")
    dims = [int(d) for d in shape.get("dimensions", [])]
    tiles = tiles or (1,) * len(dims)
    if len(tiles) != len(dims) or any(d % t for d, t in zip(dims, tiles)):
        raise ValueError("uneven sharding")
    return (
        math.prod(d // t for d, t in zip(dims, tiles)) * WIDTHS[shape["element_type"]]
    )


def tile_shape(sharding, rank, devices):
    if sharding is None:
        raise ValueError("unknown sharding")
    kind = sharding.get("type", "REPLICATED")
    if kind in ("REPLICATED", "MANUAL"):
        return (1,) * rank
    tiles = tuple(int(d) for d in sharding.get("tile_assignment_dimensions", []))
    if (
        kind != "OTHER"
        or len(tiles) != rank
        or math.prod(tiles) != devices
        or sharding.get("last_tile_dims")
        or sharding.get("replicate_on_last_tile_dim")
    ):
        raise ValueError("unsupported sharding")
    explicit = sharding.get("tile_assignment_devices")
    if explicit is not None:
        identity = list(map(int, explicit)) == list(range(devices))
    else:
        identity = list(map(int, sharding.get("iota_reshape_dims", []))) == [
            devices
        ] and sharding.get("iota_transpose_perm") == [0]
    if not identity:
        raise ValueError("non-identity device assignment")
    return tiles


class Workload:
    def __init__(self, snapshot, devices, scenario, communication):
        self.hlo = snapshot["hlo"]
        self.costs = snapshot.get("costs", {})
        self.partitioned = snapshot.get("shape_scope") == "per_partition"
        self.devices = tuple(devices)
        self.scenario = scenario
        self.communication = communication
        self.computations = {c["id"]: c for c in self.hlo["computations"]}
        self.events = []
        self.gaps = []
        self.inferred_collectives = 0
        self.compute_rate = self._rate("bf16_flops_per_second", "compute_utilization")
        self.memory_rate = self._rate("hbm_bytes_per_second", "bandwidth_utilization")
        self.transcendental_rate = nonnegative(
            self.scenario.get("transcendentals_per_second", 1e12),
            "transcendentals_per_second",
        )
        if self.transcendental_rate == 0:
            raise ValueError("transcendentals_per_second must be positive")

    def _rate(self, rate, utilization):
        value = nonnegative(self.scenario[rate], rate)
        fraction = nonnegative(self.scenario[utilization], utilization)
        if not value or not 0 < fraction <= 1:
            raise ValueError(f"Invalid {rate} or {utilization}")
        return value * fraction

    def lower(self):
        roots = self._computation(
            self.hlo["entry_computation_id"], "hlo", self.partitioned, None
        )
        for d in self.devices:
            self.events.append(
                Event(
                    f"done:{d}",
                    "program complete",
                    dependencies=(roots[d],),
                    metadata={"device": d, "execution_boundary": "end"},
                )
            )
        return self.events

    def _computation(self, computation_id, path, local, arguments):
        computation = self.computations[computation_id]
        instructions = {i["id"]: i for i in computation["instructions"]}
        shardings = {id: i.get("sharding") for id, i in instructions.items()}
        # A direct Sharding annotation describes the producing value. Conflicting
        # consumers remain unresolved; they require a resharding model.
        annotations = {}
        for i in instructions.values():
            if i.get("custom_call_target") == "Sharding" and "sharding" in i:
                source = i["operand_ids"][0]
                annotations.setdefault(source, []).append(i["sharding"])
        for source, candidates in annotations.items():
            if shardings[source] is None and all(
                c == candidates[0] for c in candidates
            ):
                shardings[source] = candidates[0]

        def is_local(id):
            i = instructions[id]
            if i.get("custom_call_target") in (
                "xla.sdy.GlobalToLocalShape",
                "SPMDFullToShardShape",
            ):
                return True
            if i["opcode"] == "get-tuple-element":
                return is_local(i["operand_ids"][0])
            return i.get("sharding", {}).get("type") == "MANUAL"

        def tiles(i):
            rank = len(i["shape"].get("dimensions", []))
            if (
                local
                or len(self.devices) == 1
                or i["opcode"] == "constant"
                or is_local(i["id"])
            ):
                return (1,) * rank
            return tile_shape(shardings[i["id"]], rank, len(self.devices))

        def replicated(i):
            sharding = i.get("sharding")
            return i["opcode"] == "constant" or (
                sharding is not None
                and sharding.get("type", "REPLICATED") == "REPLICATED"
            )

        resolved = {}

        def visit(id):
            if id in resolved:
                return resolved[id]
            i = instructions[id]
            opcode = i["opcode"]
            operands = [instructions[s] for s in i.get("operand_ids", [])]
            incoming = [visit(op["id"]) for op in operands]
            incoming += [visit(s) for s in i.get("control_predecessor_ids", [])]
            deps = {d: tuple(item[d] for item in incoming) for d in self.devices}
            result = {d: f"{path}/{id}:{d}" for d in self.devices}
            if opcode == "parameter":
                number = int(i.get("parameter_number", 0))
                result = (
                    arguments[number]
                    if arguments is not None
                    else {d: f"start:{d}" for d in self.devices}
                )
            elif opcode == "call":
                child_local = local or (
                    bool(operands) and all(is_local(op["id"]) for op in operands)
                )
                child = self._computation(
                    i["called_computation_ids"][0],
                    f"{path}/{id}",
                    child_local,
                    incoming,
                )
                for d in self.devices:
                    self.events.append(
                        Event(result[d], i["name"], dependencies=deps[d] + (child[d],))
                    )
            else:
                duration, collective, size = 0, None, 0
                transfers = None
                flops, read_bytes, write_bytes = 0, 0, 0
                kernel = {}
                cost_status, reason = "structural", ""
                target = i.get("custom_call_target")
                try:
                    if opcode in VIEWS:
                        pass
                    elif target in ANNOTATIONS:
                        if (
                            target == "Sharding"
                            and operands
                            and tiles(i) != tiles(operands[0])
                        ):
                            raise ValueError("unmodeled resharding")
                    elif target == "tpu_custom_call":
                        kernel = pallas_cost(
                            i,
                            local=local
                            or len(self.devices) == 1
                            or is_local(id)
                            or (
                                replicated(i) and all(replicated(op) for op in operands)
                            ),
                        )
                        duration = math.ceil(
                            max(
                                kernel["flops"] / self.compute_rate,
                                kernel["transcendentals"] / self.transcendental_rate,
                                kernel["bytes_accessed"] / self.memory_rate,
                            )
                            * 1e9
                        )
                        cost_status = "estimated"
                    elif opcode in (
                        "all-reduce",
                        "all-gather",
                        "reduce-scatter",
                        "all-to-all",
                        "collective-permute",
                    ):
                        if "channel_id" not in i or len(operands) != 1:
                            raise ValueError("unsupported collective scope")
                        size = shape_bytes(operands[0]["shape"], tiles(operands[0]))
                        count = len(self.devices)
                        if opcode == "collective-permute":
                            pairs = [
                                (int(p.get("source", 0)), int(p.get("target", 0)))
                                for p in i.get("source_target_pairs", [])
                            ]
                            if (
                                any(
                                    s < 0 or t < 0 or s >= count or t >= count
                                    for s, t in pairs
                                )
                                or len({s for s, _ in pairs}) != len(pairs)
                                or len({t for _, t in pairs}) != len(pairs)
                            ):
                                raise ValueError("unsupported collective-permute pairs")
                            transfers = [(s, t) for s, t in pairs if s != t]
                        else:
                            groups = self.costs.get(id, {}).get("collective_groups")
                            if groups != [list(range(count))]:
                                raise ValueError("unsupported collective group")
                            if opcode == "all-to-all":
                                size = math.ceil(size / count)
                                transfers = [
                                    (s, t)
                                    for s in range(count)
                                    for t in range(count)
                                    if s != t
                                ]
                        collective = opcode
                        cost_status = "estimated"
                    elif opcode == "dot":
                        # Only lhs[M,K] @ rhs[K,N], with matching K partitions.
                        dims = i["dot_dimension_numbers"]
                        if (
                            dims.get("lhs_contracting_dimensions") != ["1"]
                            or dims.get("rhs_contracting_dimensions") != ["0"]
                            or dims.get("lhs_batch_dimensions")
                            or dims.get("rhs_batch_dimensions")
                            or any(
                                len(op["shape"].get("dimensions", [])) != 2
                                for op in operands
                            )
                        ):
                            raise ValueError("unsupported dot dimensions")
                        lhs, rhs = map(tiles, operands)
                        out = tiles(i)
                        if lhs[1] != rhs[0] or out != (lhs[0], rhs[1]):
                            raise ValueError("dot requires unsupported resharding")
                        if lhs[1] > 1:
                            if lhs[1] != len(self.devices) or out != (1, 1):
                                raise ValueError("unsupported contracting group")
                            collective = "all-reduce"
                        m, k = map(int, operands[0]["shape"]["dimensions"])
                        n = int(operands[1]["shape"]["dimensions"][1])
                        if m % lhs[0] or k % lhs[1] or n % rhs[1]:
                            raise ValueError("uneven dot sharding")
                        flops = 2 * (m // lhs[0]) * (k // lhs[1]) * (n // rhs[1])
                        size = shape_bytes(i["shape"], out)
                        read_bytes = sum(
                            shape_bytes(op["shape"], tiles(op)) for op in operands
                        )
                        write_bytes = size
                        traffic = read_bytes + write_bytes
                        cost_status = "estimated"
                        duration = math.ceil(
                            max(flops / self.compute_rate, traffic / self.memory_rate)
                            * 1e9
                        )
                    elif opcode in ELEMENTWISE:
                        read_bytes = sum(
                            shape_bytes(op["shape"], tiles(op)) for op in operands
                        )
                        write_bytes = shape_bytes(i["shape"], tiles(i))
                        traffic = read_bytes + write_bytes
                        cost_status = "estimated"
                        duration = math.ceil(traffic / self.memory_rate * 1e9)
                    else:
                        raise ValueError(f"unsupported {target or opcode}")
                except (ValueError, KeyError) as error:
                    self.gaps.append(
                        {"instruction": id, "name": i["name"], "reason": str(error)}
                    )
                    duration = 0
                    kernel = {}
                    collective = None
                    flops, read_bytes, write_bytes = None, None, None
                    cost_status, reason = "unknown", str(error)
                compute = {
                    d: result[d] + "/local" if collective else result[d]
                    for d in self.devices
                }
                for d in self.devices:
                    self.events.append(
                        Event(
                            compute[d],
                            i["name"],
                            duration,
                            deps[d] or (f"start:{d}",),
                            (f"compute:{d}", f"hbm:{d}") if duration else (),
                            metadata={
                                "hlo_id": id,
                                "framework_op": i.get("metadata", {}).get(
                                    "op_name", ""
                                ),
                                "opcode": opcode,
                                "device": d,
                                "dtype": i["shape"]["element_type"],
                                "cost_status": cost_status,
                                "cost_gap": reason,
                                "dot_flops": flops,
                                "logical_read_bytes": read_bytes,
                                "logical_write_bytes": write_bytes,
                                **(
                                    {
                                        "cost_source": "pallas_cost_estimate",
                                        "kernel_flops": kernel["flops"],
                                        "transcendentals": kernel["transcendentals"],
                                        "kernel_bytes_accessed": kernel[
                                            "bytes_accessed"
                                        ],
                                    }
                                    if kernel
                                    else {}
                                ),
                            },
                        )
                    )
                if collective:
                    self.inferred_collectives += opcode == "dot"
                    end = f"{path}/{id}/collective"
                    if transfers is None:
                        self.events += self.communication.ring(
                            end, collective, self.devices, size, compute
                        )
                    else:
                        arrivals = tuple(compute.values())
                        completed = []
                        for source, target in transfers:
                            transfer = f"{end}/{source}>{target}"
                            self.events += self.communication.transfer(
                                transfer,
                                self.devices[source],
                                self.devices[target],
                                size,
                                arrivals,
                            )
                            completed.append(transfer)
                        self.events.append(
                            Event(
                                end,
                                collective + " complete",
                                dependencies=tuple(completed) or arrivals,
                            )
                        )
                    for d in self.devices:
                        self.events.append(
                            Event(
                                result[d],
                                "collective result ready",
                                dependencies=(end,),
                            )
                        )
            resolved[id] = result
            return result

        return visit(computation["root_id"])
