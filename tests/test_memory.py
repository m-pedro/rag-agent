from rag_agent.memory import ConversationMemory


def test_conversation_memory_evicts_to_summary_within_token_budget():
    mem = ConversationMemory(max_token_budget=60)
    mem.remember_entity("cluster", "prod-eu-west-1")

    for i in range(6):
        mem.add_message("user", f"Turn {i}: investigating elevated p99 latency on shard {i} with detailed logs.")
        mem.add_message("assistant", f"Turn {i}: checked metrics and Raft heartbeat queues on shard {i}.")

    assert mem.total_tokens() <= 60 or len(mem.messages) == 2
    assert len(mem.summary_lines) > 0

    rendered = mem.render_context()
    assert "cluster=prod-eu-west-1" in rendered
    assert "[Earlier Conversation Summary]:" in rendered
    assert "Turn 5" in rendered
