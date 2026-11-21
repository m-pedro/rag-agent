from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    @property
    def approx_tokens(self) -> int:
        # Fast ~4 chars/token heuristic + 4 role-framing tokens
        return max(1, len(self.content) // 4) + 4


class ConversationMemory:
    """Token-budgeted conversation buffer that compresses evicted turns + tracks key entities."""

    def __init__(self, max_token_budget: int = 512) -> None:
        if max_token_budget < 32:
            raise ValueError("max_token_budget must be >= 32")
        self.max_token_budget = max_token_budget
        self.messages: List[Message] = []
        self.summary_lines: List[str] = []
        self.entities: Dict[str, str] = {}

    def remember_entity(self, key: str, value: str) -> None:
        self.entities[key.strip()] = value.strip()

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content.strip()))
        self._enforce_budget()

    def total_tokens(self) -> int:
        return sum(m.approx_tokens for m in self.messages)

    def _enforce_budget(self) -> None:
        while len(self.messages) > 2 and self.total_tokens() > self.max_token_budget:
            evicted = self.messages.pop(0)
            snippet = evicted.content[:72].replace("\n", " ")
            ellipsis = "..." if len(evicted.content) > 72 else ""
            self.summary_lines.append(f"- {evicted.role}: {snippet}{ellipsis}")
            # Bound rolling summary buffer to the 8 most recent evicted turns
            if len(self.summary_lines) > 8:
                self.summary_lines = self.summary_lines[-8:]

    def render_context(self) -> str:
        sections: List[str] = []
        if self.entities:
            ent_str = ", ".join(f"{k}={v}" for k, v in sorted(self.entities.items()))
            sections.append(f"[Known Entities]: {ent_str}")
        if self.summary_lines:
            sections.append("[Earlier Conversation Summary]:\n" + "\n".join(self.summary_lines))
        if self.messages:
            recent = "\n".join(f"{m.role}: {m.content}" for m in self.messages)
            sections.append("[Recent Messages]:\n" + recent)
        return "\n\n".join(sections)
