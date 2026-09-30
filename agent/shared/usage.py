# agent/shared/usage.py  |  shared helpers (not a block)
"""
COST AND TOKEN ACCOUNTING

Sums `usage_metadata` from the AI messages a turn produced. Prices are optional env vars
(USD per 1M tokens); without them you get token counts only.
"""

import os
from dataclasses import dataclass

from langchain_core.messages import BaseMessage


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0

    @property
    def cost_usd(self) -> float | None:
        p_in, p_out = os.getenv("PRICE_IN_PER_M"), os.getenv("PRICE_OUT_PER_M")
        if not p_in or not p_out:
            return None
        return (self.input_tokens * float(p_in) + self.output_tokens * float(p_out)) / 1_000_000

    def __str__(self) -> str:
        line = (
            f"tokens in={self.input_tokens} out={self.output_tokens} "
            f"model_calls={self.model_calls}"
        )
        cost = self.cost_usd
        return f"{line} cost=${cost:.5f}" if cost is not None else line


def usage_from_messages(messages: list[BaseMessage]) -> Usage:
    usage = Usage()
    for m in messages:
        meta = getattr(m, "usage_metadata", None)
        if m.type == "ai" and meta:
            usage.input_tokens += meta.get("input_tokens", 0)
            usage.output_tokens += meta.get("output_tokens", 0)
            usage.model_calls += 1
    return usage
