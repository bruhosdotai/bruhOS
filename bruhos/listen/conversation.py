"""In-memory multi-turn chat history for the web listen loop."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class Conversation:
    """Keeps user/assistant turns (system prompt stays outside)."""

    max_turns: int = 12  # pairs of user+assistant; oldest dropped
    _messages: list[dict[str, str]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> list[dict[str, str]]:
        with self._lock:
            return list(self._messages)

    def append(self, user: str, assistant: str) -> int:
        with self._lock:
            self._messages.append({"role": "user", "content": user})
            self._messages.append({"role": "assistant", "content": assistant})
            # trim to last max_turns pairs (2 messages each)
            keep = max(0, self.max_turns) * 2
            if keep and len(self._messages) > keep:
                self._messages = self._messages[-keep:]
            return len(self._messages) // 2

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()

    @property
    def turns(self) -> int:
        with self._lock:
            return len(self._messages) // 2
