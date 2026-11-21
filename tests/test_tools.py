from rag_agent.tools import ToolRegistry


def calculate_ram_gb(num_params_billions: float, bytes_per_param: int = 2) -> dict:
    """Estimates LLM weight memory in GB."""
    return {"ram_gb": round(num_params_billions * bytes_per_param, 2)}


def test_tool_registry_schema_introspection_and_safe_execution():
    reg = ToolRegistry()
    spec = reg.register(calculate_ram_gb)

    assert spec.name == "calculate_ram_gb"
    assert spec.parameters_schema["required"] == ["num_params_billions"]
    assert spec.parameters_schema["properties"]["num_params_billions"]["type"] == "number"
    assert spec.parameters_schema["properties"]["bytes_per_param"]["default"] == 2

    out = reg.execute("calculate_ram_gb", '{"num_params_billions": 7.0, "bytes_per_param": 2}')
    assert '"ram_gb": 14.0' in out

    missing_err = reg.execute("calculate_ram_gb", "{}")
    assert "missing required argument" in missing_err

    unknown_err = reg.execute("drop_database", "{}")
    assert "unknown tool" in unknown_err
