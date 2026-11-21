from dataclasses import dataclass, field
import json
import re
from typing import Callable, Dict, List, Optional

from rag_agent.memory import ConversationMemory
from rag_agent.tools import ToolRegistry

ACTION_RE = re.compile(
    r"Action\s*:\s*([a-zA-Z0-9_]+)\s*\nAction Input\s*:\s*(\{.*?\})",
    re.DOTALL,
)
FINAL_ANSWER_RE = re.compile(r"Final Answer\s*:\s*(.+)", re.DOTALL)
THOUGHT_RE = re.compile(r"Thought\s*:\s*(.+?)(?=\n(?:Action|Final Answer)\s*:|$)", re.DOTALL)


@dataclass(frozen=True)
class AgentStep:
    step_num: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None


@dataclass(frozen=True)
class AgentRunResult:
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    halted_by_max_steps: bool = False


PolicyFn = Callable[[str, List[Dict[str, object]]], str]


class ReActAgent:
    """Autonomous ReAct (Thought -> Action -> Observation) tool-calling agent (Yao et al., 2023)."""

    def __init__(
        self,
        tools: ToolRegistry,
        policy_fn: PolicyFn,
        memory: Optional[ConversationMemory] = None,
        max_steps: int = 6,
    ) -> None:
        self.tools = tools
        self.policy_fn = policy_fn
        self.memory = memory or ConversationMemory(max_token_budget=1024)
        self.max_steps = max_steps

    def build_scratchpad_prompt(self, question: str, steps: List[AgentStep]) -> str:
        schemas_json = json.dumps(self.tools.list_schemas(), indent=2)
        history_ctx = self.memory.render_context()

        trace_lines: List[str] = []
        for s in steps:
            trace_lines.append(f"Thought: {s.thought}")
            if s.action is not None:
                trace_lines.append(f"Action: {s.action}")
                trace_lines.append(f"Action Input: {s.action_input}")
                trace_lines.append(f"Observation: {s.observation}")
        scratchpad = "\n".join(trace_lines)

        return (
            "You are a ReAct reasoning agent. Use the registered tools to gather facts before answering.\n"
            f"Available Tools:\n{schemas_json}\n\n"
            f"{history_ctx}\n\n"
            f"Question: {question}\n"
            f"{scratchpad}\n"
        ).strip()

    def run(self, question: str) -> AgentRunResult:
        self.memory.add_message("user", question)
        steps: List[AgentStep] = []
        schemas = self.tools.list_schemas()

        for step_num in range(1, self.max_steps + 1):
            prompt = self.build_scratchpad_prompt(question, steps)
            raw_output = self.policy_fn(prompt, schemas).strip()

            thought_match = THOUGHT_RE.search(raw_output)
            thought = thought_match.group(1).strip() if thought_match else "Reasoning over current state."

            final_match = FINAL_ANSWER_RE.search(raw_output)
            if final_match and not ACTION_RE.search(raw_output):
                final_answer = final_match.group(1).strip()
                steps.append(AgentStep(step_num=step_num, thought=thought))
                self.memory.add_message("assistant", final_answer)
                return AgentRunResult(answer=final_answer, steps=steps, halted_by_max_steps=False)

            action_match = ACTION_RE.search(raw_output)
            if action_match:
                tool_name = action_match.group(1).strip()
                tool_args_str = action_match.group(2).strip()
                observation = self.tools.execute(tool_name, tool_args_str)
                steps.append(
                    AgentStep(
                        step_num=step_num,
                        thought=thought,
                        action=tool_name,
                        action_input=tool_args_str,
                        observation=observation,
                    )
                )
            else:
                # Self-repair feedback observation when LLM output violates ReAct syntax
                steps.append(
                    AgentStep(
                        step_num=step_num,
                        thought=thought,
                        action="_format_repair",
                        action_input="{}",
                        observation="FormatError: respond with either 'Action: <tool>\\nAction Input: {...}' or 'Final Answer: ...'",
                    )
                )

        fallback = "Unable to reach Final Answer within max_steps budget."
        self.memory.add_message("assistant", fallback)
        return AgentRunResult(answer=fallback, steps=steps, halted_by_max_steps=True)
