from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BandMessage:
    agent: str
    role: str
    content: Any
    timestamp: str = field(default_factory=lambda: time.strftime("%H:%M:%S"))


class BandRoom:
    def __init__(self, name: str = "default"):
        self.name = name
        self._messages: list[BandMessage] = []

    def send(self, agent: str, role: str, content: Any) -> BandMessage:
        msg = BandMessage(agent=agent, role=role, content=content)
        self._messages.append(msg)
        return msg

    def get_messages(self) -> list[BandMessage]:
        return list(self._messages)

    def to_dict(self) -> list[dict]:
        return [
            {
                "timestamp": m.timestamp,
                "agent": m.agent,
                "role": m.role,
                "content": m.content,
            }
            for m in self._messages
        ]

    def clear(self) -> None:
        self._messages.clear()
