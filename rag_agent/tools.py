from dataclasses import dataclass
import inspect
import json
from typing import Any, Callable, Dict, List, get_type_hints


PYTHON_TO_JSON_TYPE: Dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    func: Callable[..., Any]
    parameters_schema: Dict[str, Any]


class ToolRegistry:
    """Registry that introspects Python callables into JSON-Schema tool definitions."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ) -> ToolSpec:
        tool_name = name or func.__name__
        tool_desc = (description or inspect.getdoc(func) or f"Execute {tool_name}").strip()
        sig = inspect.signature(func)
        hints = get_type_hints(func)

        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param in sig.parameters.items():
            ann = hints.get(param_name, str)
            json_type = PYTHON_TO_JSON_TYPE.get(ann, "string")
            prop_entry: Dict[str, Any] = {"type": json_type}
            if param.default is not inspect.Parameter.empty:
                prop_entry["default"] = param.default
            else:
                required.append(param_name)
            properties[param_name] = prop_entry

        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
        spec = ToolSpec(
            name=tool_name,
            description=tool_desc,
            func=func,
            parameters_schema=schema,
        )
        self._tools[tool_name] = spec
        return spec

    def list_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters_schema,
            }
            for spec in self._tools.values()
        ]

    def execute(self, name: str, arguments: Dict[str, Any] | str) -> str:
        if name not in self._tools:
            return f"ToolError: unknown tool '{name}'"

        spec = self._tools[name]
        try:
            parsed_args = (
                json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
            )
            for req in spec.parameters_schema["required"]:
                if req not in parsed_args:
                    return f"ToolError: missing required argument '{req}' for tool '{name}'"

            result = spec.func(**parsed_args)
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as exc:  # noqa: BLE001
            return f"ToolError: {type(exc).__name__}: {exc}"
