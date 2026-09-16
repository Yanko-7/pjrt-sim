"""Read author-declared TPU Pallas costs from an original HLO snapshot.

Keep this contract aligned with pallas_cost.cc. No kernel-name heuristics or
division by device count: tiled/global estimates require a separate adapter.
"""

import base64
import json
import math


def pallas_cost(instruction, *, local):
    if not local:
        raise ValueError("Pallas cost requires local or replicated scope")
    if instruction.get("custom_call_has_side_effect", False):
        raise ValueError("Pallas side effects")
    try:
        # HloInstructionProto.backend_config is bytes, hence base64 in JSON.
        config = json.loads(
            base64.b64decode(instruction["backend_config"], validate=True)
        )
        call = config["custom_call_config"]
        communication = call.get("has_communication", False)
        if communication is not False:
            raise ValueError("Pallas internal communication")
        cost = call["cost_estimate"]
        result = {}
        for field in (
            "flops",
            "transcendentals",
            "bytes_accessed",
            "remote_bytes_transferred",
        ):
            value = (
                cost.get(field, 0)
                if field == "remote_bytes_transferred"
                else cost[field]
            )
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError(f"invalid Pallas {field}")
            count = float(value)
            if (
                not math.isfinite(count)
                or not 0 <= count < 2**63
                or count != int(count)
            ):
                raise ValueError(f"invalid Pallas {field}")
            result[field] = count
        if result.pop("remote_bytes_transferred"):
            raise ValueError("Pallas internal communication")
        return result
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("missing or malformed Pallas cost_estimate") from error
