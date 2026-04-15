from __future__ import annotations

import os
from typing import List
from openai import OpenAI

from app.reranker import RerankedItem


class CitationLLM:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def build_context(self, items: List[RerankedItem]) -> str:
        blocks = []
        for i, item in enumerate(items, start=1):
            blocks.append(
                f"[{i}] Source: {item.source}\n{item.content}"
            )
        return "\n\n".join(blocks)

    def answer_with_citations(self, question: str, items: List[RerankedItem]) -> str:
        context = self.build_context(items)

        system_prompt = (
            "You are a precise QA system. "
            "Answer ONLY using the provided context. "
            "Cite sources using [1], [2]. "
            "If unsure, say you do not know."
        )

        user_prompt = f"""
Question:
{question}

Context:
{context}

Answer with citations.
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.choices[0].message.content.strip()