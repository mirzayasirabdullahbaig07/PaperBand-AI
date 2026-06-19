from __future__ import annotations

from groq import Groq

from .base import BaseAgent


SYSTEM_PROMPT = """You are a research paper summarization expert.

Read the paper text below and extract the following fields:

- title            : The paper's title (string)
- authors          : List of author names (list of strings, or ["Unknown"] if not found)
- research_problem : What problem is the paper solving? (1-3 sentences)
- methodology      : How did the authors approach the problem? (2-4 sentences)
- key_results      : What were the main findings? (2-4 sentences)
- conclusion       : What do the authors conclude and claim? (1-3 sentences)

Return ONLY a valid JSON object with these exact keys. No markdown, no preamble, no explanation."""


class SummarizerAgent(BaseAgent):
    TEMPERATURE = 0.2

    def __init__(self, client: Groq):
        super().__init__(client)

    def run(self, paper_text: str) -> dict:
        prompt = f"{SYSTEM_PROMPT}\n\n---\nPAPER TEXT:\n{paper_text}"
        return self._call(prompt)
