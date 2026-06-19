from __future__ import annotations

import json

from groq import Groq

from .base import BaseAgent


SYSTEM_PROMPT = """You are a rigorous academic peer reviewer.

You will receive a structured summary of a research paper.
Your job is to critically evaluate the work and identify:

- strengths          : List of genuine strengths (list of strings, 2-5 items)
- weaknesses         : List of methodological or conceptual weaknesses (list of strings, 2-5 items)
- missing_experiments: Experiments or ablations the authors should have run (list of strings, 2-4 items)
- limitations        : Limitations the authors may not have fully acknowledged (list of strings, 2-4 items)

Be specific and constructive. Reference the actual content from the summary.
Return ONLY a valid JSON object with these exact keys. No markdown, no preamble, no explanation."""


class CriticAgent(BaseAgent):
    TEMPERATURE = 0.5

    def __init__(self, client: Groq):
        super().__init__(client)

    def run(self, summary: dict) -> dict:
        summary_str = json.dumps(summary, indent=2)
        prompt = f"{SYSTEM_PROMPT}\n\n---\nPAPER SUMMARY:\n{summary_str}"
        return self._call(prompt)
