"""Task-level latency cap via cumulative latency_sum (turns), aligned with social LLM.

Each emitted word records:
  - latency: turns for that word (n_pass_before + 1, same as latency_turns)
  - latency_sum: cumulative sum of latency across the list so far

Stop when latency_sum would exceed latency_cap (no per-word time limit).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ModelParams


@dataclass
class LatencyTracker:
    """Cumulative latency_sum in discrete turns (social MH semantics)."""

    cap: int
    latency_sum: int = 0

    @property
    def enabled(self) -> bool:
        return self.cap > 0

    def exhausted(self) -> bool:
        return self.enabled and self.latency_sum >= self.cap

    def would_exceed(self, word_latency: int) -> bool:
        if not self.enabled:
            return False
        return self.latency_sum + int(word_latency) > self.cap

    def commit(self, word_latency: int) -> dict[str, int | float]:
        wl = int(word_latency)
        self.latency_sum += wl
        return {
            "latency": wl,
            "latency_sum": self.latency_sum,
            "latency_cap": self.cap,
        }


def tracker_from_params(params: ModelParams) -> LatencyTracker:
    cap = int(params.latency_cap) if params.use_latency_cap else 0
    return LatencyTracker(cap=cap)


def apply_latency_cap_defaults(params: ModelParams) -> ModelParams:
    """Enable task-level cap; default 1000 turns (social nested sim)."""
    if not params.use_latency_cap:
        return params
    if params.latency_cap <= 0:
        params.latency_cap = 1000
    return params


def inner_search_limit(params: ModelParams) -> int:
    """Safety bound on proposals for one word (not a per-word time limit)."""
    if params.use_latency_cap and params.latency_cap > 0:
        return max(1, int(params.latency_cap))
    return max(10_000, int(params.max_steps) * 5000)
