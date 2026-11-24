from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    image: bytes
    frames: tuple
    history: tuple
    info: dict
    elapsed_s: float
    last_action: dict | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    parameters: dict


def tool_specs(width=320, height=240):
    def number(low, high):
        return {"type": "integer", "minimum": low, "maximum": high}

    point = {"x": number(0, width - 1), "y": number(0, height - 1)}
    key = {"key": {"type": "string", "minLength": 1}}
    duration = {"duration_ms": number(0, 60000)}
    def schema(properties, optional=()):
        return {"type": "object", "properties": properties,
                "required": [k for k in properties if k not in optional], "additionalProperties": False}

    schemas = {
        "click": schema({**point, "button": {"enum": ["left", "middle", "right"]}}, ("button",)),
        "move": schema(point),
        "drag": schema({**point, "to_x": point["x"], "to_y": point["y"], **duration}),
        "scroll": schema({**point, "delta_x": number(-10000, 10000), "delta_y": number(-10000, 10000)}),
        "press_key": schema({**key, **duration}),
        "hold_keys": schema({"keys": {"type": "array", "items": key["key"], "minItems": 1}, **duration}),
        "key_down": schema(key),
        "key_up": schema(key),
        "wait": schema(duration),
        "skip": schema({}),
    }
    steps = [schema({"at_ms": number(0, 60000), "tool": {"const": name}, "params": definition})
             for name, definition in schemas.items() if name != "skip"]
    schemas["sequence"] = schema({"steps": {"type": "array", "minItems": 1, "maxItems": 200,
                                            "items": {"oneOf": steps}}})
    return [ToolSpec(name, definition) for name, definition in schemas.items()]
