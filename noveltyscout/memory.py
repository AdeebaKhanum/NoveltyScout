"""Conversation memory.

The assignment asks the system to "retain context" between turns.  We keep two kinds:
  * Short-term chat history  -> last few turns are inserted into the router prompt.
  * Working state            -> the stored idea, its features, queries, evidence, comparison,
                               gaps and report, so a later message like "compare it" or
                               "give me the report" needs no repetition.

In Streamlit this object lives in st.session_state (it survives UI re-runs).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionMemory:
    idea: str = ""
    summary: str = ""
    features: list[str] = field(default_factory=list)
    queries_done: list[str] = field(default_factory=list)
    pending_queries: list[str] = field(default_factory=list)
    evidence: dict[str, dict] = field(default_factory=dict)      # id -> doc dict (all seen so far)
    comparisons: list[dict] = field(default_factory=list)
    flags: dict = field(default_factory=dict)
    score: float = 0.0
    gaps: dict = field(default_factory=dict)
    report_md: str = ""
    history: list[tuple[str, str]] = field(default_factory=list)  # (role, text)

    # ---- history helpers
    def add_turn(self, role: str, text: str) -> None:
        self.history.append((role, text))

    def recent_history(self, turns: int = 4) -> str:
        items = self.history[-turns:]
        if not items:
            return "(no earlier messages)"
        return "\n".join(f"{role}: {text[:200]}" for role, text in items)

    # ---- state helpers
    def set_idea(self, idea: str) -> None:
        """A NEW idea replaces everything derived from the old one."""
        self.idea = idea.strip()
        self.summary = ""
        self.features = []
        self.queries_done = []
        self.pending_queries = []
        self.evidence = {}
        self.comparisons = []
        self.flags = {}
        self.score = 0.0
        self.gaps = {}
        self.report_md = ""

    @property
    def has_idea(self) -> bool:
        return bool(self.idea)

    @property
    def has_comparison(self) -> bool:
        return bool(self.comparisons)
