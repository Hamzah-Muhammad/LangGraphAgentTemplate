# agent/shared/usage.py  |  shared helpers (not a block)
"""
COST AND TOKEN ACCOUNTING

UsageCounter is a callback handler. Attach it to a run's config and it sees EVERY model
call in that run: nested workers, specialists, graders, summaries and structured-output
calls. Reading usage off the final messages misses all nested calls, which is how the
planner once reported 0 tokens.

    counter = UsageCounter()
    await graph.ainvoke(payload, config={**config, "callbacks": [counter]})
    print(counter)

LangChain's own get_usage_metadata_callback skips any call whose response has no
model_name, which some OpenAI-compatible hosts omit. This counter does not.
Prices are optional env vars (USD per 1M tokens); without them you get token counts only.
"""

import os
import threading

from langchain_core.callbacks import BaseCallbackHandler


class UsageCounter(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_calls = 0
        self._lock = threading.Lock()  # parallel workers finish on different threads

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                meta = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if meta:
                    tokens_in += meta.get("input_tokens", 0)
                    tokens_out += meta.get("output_tokens", 0)
        with self._lock:
            self.model_calls += 1
            self.input_tokens += tokens_in
            self.output_tokens += tokens_out

    @property
    def cost_usd(self) -> float | None:
        p_in, p_out = os.getenv("PRICE_IN_PER_M"), os.getenv("PRICE_OUT_PER_M")
        if not p_in or not p_out:
            return None
        return (self.input_tokens * float(p_in) + self.output_tokens * float(p_out)) / 1e6

    def __str__(self) -> str:
        line = (f"tokens in={self.input_tokens} out={self.output_tokens} "
                f"model_calls={self.model_calls}")
        cost = self.cost_usd
        return f"{line} cost=${cost:.5f}" if cost is not None else line


def with_counter(config: dict, counter: UsageCounter) -> dict:
    """Copy of a run config with the counter attached (callbacks are never checkpointed)."""
    return {**config, "callbacks": [*config.get("callbacks", []), counter]}
