from __future__ import annotations

import json

from groq import Groq

from .base import BaseAgent


SYSTEM_PROMPT = """You are a senior program committee chair at a top-tier AI conference (e.g. NeurIPS, ICML, ACL).

You have received:
1. A structured summary of a research paper.
2. A peer reviewer's critique of that paper.

Your task is to produce a final editorial decision with these fields:

- score         : A numeric score from 1.0 to 10.0 (one decimal, e.g. 7.5)
- decision      : Exactly one of: "Accept", "Accept with Minor Revisions", "Major Revisions Required", "Reject"
- justification : 2-4 sentence explanation of the decision (string)
- future_work   : List of concrete suggestions for future research directions (list of strings, 3-5 items)

Score guide: 9-10 = Accept, 7-8 = Accept with Minor Revisions, 5-6 = Major Revisions Required, 1-4 = Reject

Return ONLY a valid JSON object with these exact keys. No markdown, no preamble, no explanation."""


class RecommenderAgent(BaseAgent):
    TEMPERATURE = 0.4

    def __init__(self, client: Groq):
        super().__init__(client)

    def run(self, summary: dict, critique: dict) -> dict:
        payload = {"summary": summary, "critique": critique}
        prompt = f"{SYSTEM_PROMPT}\n\n---\nINPUT:\n{json.dumps(payload, indent=2)}"
        return self._call(prompt)
