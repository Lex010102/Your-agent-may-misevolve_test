from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class MemoryItem:
    # Required by task guide
    round_id: int
    user_request: str
    agent_response: Dict[str, Any]
    score: float
    feedback_text: str


class InteractionMemory:
    """
    Minimal memory: an append-only list of interactions.
    This corresponds to the paper's dynamic flow where memory accumulates over rounds.
    """

    def __init__(self, items: List[MemoryItem] | None = None):
        self._items: List[MemoryItem] = list(items or [])

    def append(self, item: MemoryItem) -> None:
        self._items.append(item)

    def size(self) -> int:
        return len(self._items)

    def last_k(self, k: int) -> List[MemoryItem]:
        if k <= 0:
            return []
        return self._items[-k:]


def format_memory_for_prompt(memory: InteractionMemory, max_items: int = 6) -> str:
    """
    Format the most recent K interactions for prompting the Service Agent.
    Keep it simple and readable; the goal is to expose reward-correlated patterns.
    """
    items = memory.last_k(max_items)
    if not items:
        return "No memory yet."

    lines: List[str] = []
    for it in items:
        action = ""
        try:
            action = str(it.agent_response.get("selected_actions", ""))
        except Exception:
            action = ""
        lines.append(
            "\n".join(
                [
                    f"[Round {it.round_id}]",
                    f"User Request: {it.user_request}",
                    f"Selected Action: {action}",
                    f"Score: {it.score}",
                    f"Feedback: {it.feedback_text}",
                ]
            )
        )
    return "\n\n".join(lines)

