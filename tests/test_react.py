import json
from rag_agent.react import ReActAgent
from rag_agent.tools import ToolRegistry


def test_react_agent_multi_hop_tool_execution_and_self_repair():
    reg = ToolRegistry()
    kb = {"rustvecdb_sq8_ratio": 4, "raw_index_mb": 512}

    def lookup_metric(key: str) -> str:
        return json.dumps({"key": key, "value": kb.get(key, "missing")})

    def divide_numbers(numerator: float, denominator: float) -> str:
        return json.dumps({"quotient": numerator / denominator})

    reg.register(lookup_metric)
    reg.register(divide_numbers)

    # Deterministic scripted policy testing:
    # Step 1 -> malformed output (tests self-repair!)
    # Step 2 -> calls lookup_metric
    # Step 3 -> calls divide_numbers using observation from Step 2
    # Step 4 -> emits Final Answer
    call_counter = {"n": 0}

    def scripted_policy(prompt: str, _schemas: list) -> str:
        call_counter["n"] += 1
        n = call_counter["n"]
        if n == 1:
            return "Thought: Let me think without proper action syntax."
        if n == 2:
            assert "FormatError:" in prompt
            return (
                "Thought: Fetch SQ8 compression ratio from KB.\n"
                'Action: lookup_metric\nAction Input: {"key": "rustvecdb_sq8_ratio"}'
            )
        if n == 3:
            assert '"value": 4' in prompt
            return (
                "Thought: Divide 512 MB raw footprint by 4x SQ8 compression ratio.\n"
                'Action: divide_numbers\nAction Input: {"numerator": 512, "denominator": 4}'
            )
        assert '"quotient": 128.0' in prompt
        return (
            "Thought: Both tool observations confirm 128.0 MB.\n"
            "Final Answer: Compressed SQ8 index size is 128.0 MB."
        )

    agent = ReActAgent(tools=reg, policy_fn=scripted_policy, max_steps=6)
    result = agent.run("What is the SQ8 compressed footprint of a 512 MB raw index?")

    assert not result.halted_by_max_steps
    assert "128.0 MB" in result.answer
    assert len(result.steps) == 4
    assert result.steps[0].action == "_format_repair"
    assert result.steps[1].action == "lookup_metric"
    assert result.steps[2].action == "divide_numbers"
