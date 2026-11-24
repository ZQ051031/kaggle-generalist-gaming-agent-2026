from collections.abc import Mapping


TOOLS = frozenset((
    "click", "move", "drag", "scroll", "press_key", "hold_keys", "key_down",
    "key_up", "wait", "sequence", "skip",
))
SKIP = ("skip", {})


def field(obj, name, default=None):
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _schema(spec):
    for name in ("params_schema", "param_schema", "parameters_schema", "input_schema", "json_schema",
                 "schema", "parameters", "params"):
        value = field(spec, name)
        if isinstance(value, Mapping):
            return _schema(value) or value
    if isinstance(spec, Mapping) and ("properties" in spec or "type" in spec):
        return spec
    return {}


def _valid(value, schema):
    if not isinstance(schema, Mapping):
        return schema is not False
    types = {"object": dict, "array": list, "string": str, "integer": int,
             "number": (int, float), "boolean": bool, "null": type(None)}
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    if not any(t not in types or (
        isinstance(value, types[t]) and not (type(value) is bool and t in ("integer", "number"))
    ) for t in kinds):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "const" in schema and value != schema["const"]:
        return False
    for keyword, match in (("allOf", all), ("anyOf", any)):
        if keyword in schema and not match(_valid(value, s) for s in schema[keyword]):
            return False
    if "oneOf" in schema and sum(_valid(value, s) for s in schema["oneOf"]) != 1:
        return False
    if isinstance(value, dict):
        props = schema.get("properties", {})
        if not all(k in value for k in schema.get("required", ())):
            return False
        extra = schema.get("additionalProperties", True)
        return all(_valid(v, props.get(k, extra)) for k, v in value.items())
    if isinstance(value, list):
        return (schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", float("inf"))
                and all(_valid(v, schema.get("items", {})) for v in value))
    if isinstance(value, str):
        return schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", float("inf"))
    if type(value) in (float, int):
        return (schema.get("minimum", -float("inf")) <= value <= schema.get("maximum", float("inf"))
                and schema.get("exclusiveMinimum", -float("inf")) < value
                < schema.get("exclusiveMaximum", float("inf")))
    return True


class Actions:
    def __init__(self, tools, width, height):
        self.specs = {}
        self.width, self.height = width, height
        for spec in tools:
            definition = field(spec, "function", spec)
            name = field(definition, "name") or field(definition, "tool")
            if name in TOOLS:
                self.specs[name] = _schema(definition)

    def make(self, tool, **params):
        props = self.specs.get(tool, {}).get("properties")
        if isinstance(props, Mapping):
            params = {k: v for k, v in params.items() if k in props}
        return self._validate(tool, params)

    def _validate(self, tool, params):
        if tool not in self.specs:
            return None
        spec = self.specs[tool]
        for key, value in params.items():
            bounds = {
                "x": self.width, "y": self.height, "from_x": self.width, "from_y": self.height,
                "to_x": self.width, "to_y": self.height, "start_x": self.width, "start_y": self.height,
                "end_x": self.width, "end_y": self.height,
            }
            if key in bounds and (type(value) not in (int, float) or not 0 <= value < bounds[key]):
                return None
        if tool == "sequence":
            steps = params.get("steps", params.get("actions"))
            if not isinstance(steps, list) or len(steps) > 200:
                return None
        if _valid(params, spec):
            return tool, params
        return None

    def pointer(self, tool, x, y):
        x, y = max(0, min(self.width - 1, x)), max(0, min(self.height - 1, y))
        return self.make(tool, x=x, y=y, button="left") or self.make(tool, x=x, y=y)

    def key(self, key, ms=200):
        return (self.make("press_key", key=key, duration_ms=ms)
                or self.make("press_key", key=key)
                or self.make("hold_keys", keys=[key], duration_ms=ms))

    def drag(self, start, end):
        return self.make("drag", x=start.x, y=start.y, to_x=end.x, to_y=end.y,
                         from_x=start.x, from_y=start.y, start_x=start.x, start_y=start.y,
                         end_x=end.x, end_y=end.y, duration_ms=200)

    def accepts(self, proposal):
        if not isinstance(proposal, Mapping):
            return None
        tool, params = proposal.get("tool"), proposal.get("params")
        if isinstance(tool, str) and isinstance(params, dict):
            return self._validate(tool, params)
