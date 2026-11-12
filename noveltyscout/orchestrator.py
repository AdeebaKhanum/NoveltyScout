"""Orchestrator = the "manager" of the agent team (acceptance criterion 6: intent routing).

handle(message):
   1. Router agent classifies the message  -> intent
   2. ROUTES table maps intent -> handler function   (the actual "routing")
   3. Handler calls the needed agents, updates memory, returns text for the chat window
"""
from dataclasses import dataclass, field

from . import config, scoring
from .agents import AgentError, PatentAgents
from .memory import SessionMemory
from . import sources


@dataclass
class Reply:
    text: str
    intent: str
    method: str = ""
    trace: list[str] = field(default_factory=list)


class NoveltyScout:
    def __init__(self, agents: PatentAgents, memory: SessionMemory):
        self.agents = agents
        self.mem = memory
        # intent -> function.  Adding a new capability = add one line here + one handler.
        self.routes = {
            "prior_art_search": self._handle_search,
            "novelty_comparison": self._handle_compare,
            "report": self._handle_report,
            "help": self._handle_help,
        }

    # -------------------------------------------------------------- entry point
    def handle(self, message: str) -> Reply:
        mem, trace = self.mem, []
        decision = self.agents.route(message, mem)
        intent = decision["intent"]
        trace.append(f"Router -> intent='{intent}' new_idea={decision['is_new_idea']} "
                     f"({decision['method']}: {decision['reason']})")
        mem.add_turn("user", message)
        try:
            text = self.routes[intent](message, decision, trace)
        except AgentError as exc:
            text = f"Sorry - {exc}\n\nIf it says servers are busy (503), just wait a minute and try again. Otherwise check the key in .env, the model name in the sidebar, your internet, or that Ollama is running."
        mem.add_turn("assistant", text)
        return Reply(text=text, intent=intent, method=decision["method"], trace=trace)

    # -------------------------------------------------------------- shared steps
    def _need_idea(self) -> str:
        return ("Please describe your invention first - what it is, how it works and what problem it "
                "solves (3 or more sentences works best).")

    def _ensure_searched(self, trace):
        mem = self.mem
        if not mem.features:
            self.agents.plan(mem, trace)
        if not mem.evidence:
            self.agents.search(mem, trace)

    def _ensure_compared(self, trace):
        mem = self.mem
        self._ensure_searched(trace)
        if not mem.comparisons:
            self.agents.compare(mem, trace)
        if not mem.gaps:
            self.agents.find_gaps(mem, trace)

    # ----------------------------------------------------------------- handlers
    def _handle_search(self, message, decision, trace) -> str:
        mem = self.mem
        if decision["is_new_idea"] or not mem.has_idea:
            if len(message.split()) < 6 and not mem.has_idea:
                return self._need_idea()
            mem.set_idea(message)
            self.agents.plan(mem, trace)
            self.agents.search(mem, trace)
        else:                                            # "search more about ..."
            mem.comparisons, mem.gaps, mem.report_md = [], {}, ""   # old results are now stale
            self.agents.search(mem, trace, extra_queries=[message])
        return self._format_search(trace)

    def _handle_compare(self, message, decision, trace) -> str:
        if not self.mem.has_idea:
            return self._need_idea()
        self._ensure_searched(trace)
        self.agents.compare(self.mem, trace)
        self.agents.find_gaps(self.mem, trace)
        return self._format_comparison()

    def _handle_report(self, message, decision, trace) -> str:
        if not self.mem.has_idea:
            return self._need_idea()
        self._ensure_compared(trace)
        self.agents.write_report(self.mem, trace)
        return "Your report is ready - open the **Report** tab to read and download it.\n\n" + \
               f"Quick result: overlap index **{self.mem.score}/100** " \
               f"({scoring.overlap_band(self.mem.score)})."

    def _handle_help(self, message, decision, trace) -> str:
        return self.agents._invoke(self.agents.helper, {"message": message}).strip()

    # --------------------------------------------------------------- formatting
    def _format_search(self, trace) -> str:
        mem = self.mem
        lines = [f"**Idea understood:** {mem.summary}", "", "**Features I will check:**"]
        lines += [f"{i}. {f}" for i, f in enumerate(mem.features, 1)]
        lines += ["", f"**Queries run:** {len(mem.queries_done)} | **Evidence collected:** "
                      f"{len(mem.evidence)} documents (vector DB now holds {self.agents.store.count()})",
                  "", "**Closest references so far:**"]
        for d in sorted(mem.evidence.values(), key=lambda x: x["similarity"], reverse=True)[:5]:
            lines.append(f"- [{d['id']}] {d['title']} - similarity {d['similarity']:.2f}")
        lines += ["", "Next: ask me to *compare for novelty*, or *search more about ...*."]
        manual = sources.google_patents_link(" ".join(mem.queries_done[:1]))
        lines.append(f"\nPatents are not searched live (no free official API). Search them manually "
                     f"[on Google Patents]({manual}) and paste abstracts into the **Evidence** tab.")
        return "\n".join(lines)

    def _format_comparison(self) -> str:
        mem = self.mem
        lines = [f"**Overlap index: {mem.score}/100** - {scoring.overlap_band(mem.score)}",
                 f"_{mem.flags.get('note', '')}_", "",
                 "| # | Feature | Verdict | References |", "|---|---|---|---|"]
        for i, c in enumerate(mem.comparisons, 1):
            lines.append(f"| {i} | {c['feature']} | {c['verdict']} | {', '.join(c['evidence_ids']) or '-'} |")
        lines += ["", "**Possibly novel:**"]
        lines += [f"- {g.get('aspect', '')} - {g.get('why', '')}" for g in mem.gaps.get("novel_aspects", [])] \
            or ["- none identified"]
        lines += ["", "**Under-explored:**"]
        lines += [f"- {g.get('aspect', '')} - {g.get('why', '')}" for g in mem.gaps.get("underexplored", [])] \
            or ["- none identified"]
        lines += ["", "Ask for the **report** to get the full note."]
        return "\n".join(lines)