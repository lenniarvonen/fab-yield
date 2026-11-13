"""JSON-schema tool definitions for LLM agent harnesses."""

from . import simulator as S

_RECIPE = {
    "type": "object",
    "description": "Recipe knobs. Omitted knobs use their defaults (or the level's fixed values).",
    "properties": {
        k: {"type": "number", "minimum": v[0], "maximum": v[1], "description": f"{v[4]} [{v[3]}]"}
        for k, v in S.KNOBS.items()
    },
    "additionalProperties": False,
}

TOOLS = [
    {
        "name": "read_docs",
        "description": "Read process/device documentation. Free. Start with topic 'index'.",
        "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]},
    },
    {
        "name": "run_wafer",
        "description": "Process one test wafer with the given recipe. Costs 1.0 budget unit. Returns wafer_id; use measure to get data.",
        "input_schema": {"type": "object", "properties": {"recipe": _RECIPE}, "required": ["recipe"]},
    },
    {
        "name": "measure",
        "description": (
            "Measure a processed wafer. tool: 'electrical' (Vt, Ion, Ioff; 0.02/die), 'ellipsometer' "
            "(oxide thickness; 0.005/site), 'four_point_probe' (sheet resistance; 0.005/site). "
            "dies: list of die ids (0..67) or an integer count."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "wafer_id": {"type": "integer"},
                "tool": {"type": "string", "enum": ["electrical", "ellipsometer", "four_point_probe"]},
                "dies": {"oneOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}}]},
            },
            "required": ["wafer_id", "tool"],
        },
    },
    {
        "name": "submit_recipe",
        "description": "Submit the final recipe for the production lot. Ends the episode and returns the score.",
        "input_schema": {"type": "object", "properties": {"recipe": _RECIPE}, "required": ["recipe"]},
    },
]
