from __future__ import annotations

import json
import re

from groq import Groq


class BaseAgent:
    MODEL = "llama-3.3-70b-versatile"
    TEMPERATURE = 0.3

    def __init__(self, client: Groq):
        self.client = client

    def _call(self, prompt: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.TEMPERATURE,
        )
        raw = response.choices[0].message.content.strip()
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        clean = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Agent returned non-JSON output.\n\nRaw output:\n{raw}"
            ) from exc
