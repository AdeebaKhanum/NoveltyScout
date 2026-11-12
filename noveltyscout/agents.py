"""The agent team.

Each "agent" is a LangChain LCEL chain   prompt | llm | to_text | parse_json   plus plain-Python
logic around it.  Agents in this file:

  route()      Intent Router        (few-shot, JSON)           -> which handler runs
  plan()       Search Planner       (zero-shot CoT)            -> features + search queries
  search()     Search Agent         (ReAct loop + tools)       -> evidence collected in ChromaDB
  compare()    Novelty Comparison   (Agentic RAG + CoT)        -> verdict per feature
  find_gaps()  Gap Agent            (draft -> Reflection)      -> novel / under-explored aspects
  write_report() Report Agent       (grounded generation)      -> final Markdown note

"Agentic RAG" = the agent DECIDES when to retrieve and from where:
  * the router decides whether retrieval is needed at all (help/report do not search);
  * the search agent decides after each round whether more searching is needed (ReAct);
  * the comparison agent triggers a targeted live search when a feature has weak evidence.
"""
import json
import logging
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool

from . import config, prompts, scoring, sources
from .llm import parse_json, to_text
from .memory import SessionMemory
from .vectorstore import PriorArtStore

log = logging.getLogger(__name__)
INTENTS = ("prior_art_search", "novelty_comparison", "report", "help")


class AgentError(Exception):
    """A user-presentable failure (e.g. the LLM returned unusable output)."""


TRANSIENT_MARKERS = ("503", "unavailable", "429", "resource_exhausted", "overloaded",
                     "high demand", "timed out", "timeout", "temporarily")


def is_transient(exc: Exception) -> bool:
    """Temporary server-side problems worth waiting and retrying."""
    text = str(exc).lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


class ModelUnavailable(AgentError):
    """The model/API itself failed (bad key, retired model name, quota, no internet)."""


# ----------------------------------------------------------------------- chain factory
def build_chain(llm, system: str, user: str, as_json: bool = True):
    """LCEL pipeline:  ChatPromptTemplate | chat model | text normaliser [| JSON parser]."""
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
    chain = prompt | llm | to_text
    return (chain | parse_json) if as_json else chain


# -------------------------------------------------------------------------------- tools
def _arxiv_search(query: str) -> list:
    """Search arXiv for research papers related to the query."""
    return sources.search_arxiv(query)


def _openalex_search(query: str) -> list:
    """Search OpenAlex (scholarly works, abstracts included) for papers related to the query."""
    return sources.search_openalex(query)


def build_tools() -> dict:
    """LangChain tools the Search Agent can call (more can be added the same way)."""
    return {
        "arXiv": StructuredTool.from_function(
            func=_arxiv_search, name="search_arxiv", description=_arxiv_search.__doc__),
        "OpenAlex": StructuredTool.from_function(
            func=_openalex_search, name="search_openalex", description=_openalex_search.__doc__),
    }


# ----------------------------------------------------------------------- text formatting
def format_verdicts(comparisons: list[dict]) -> str:
    if not comparisons:
        return "(no comparison yet)"
    lines = []
    for i, c in enumerate(comparisons, 1):
        ids = ", ".join(c["evidence_ids"]) or "-"
        lines.append(f"{i}. {c['feature']} | {c['verdict']} | {ids} | {c.get('reasoning', '')}")
    return "\n".join(lines)


def keyword_route(message: str, mem: SessionMemory) -> dict:
    """Rule-based router used only if the LLM router fails (safety net)."""
    text = message.lower()
    words = text.split()
    if any(w in text for w in ("report", "final note", "prior-art note", "prior art note")):
        intent, new = "report", False
    elif any(w in text for w in ("compare", "novel", "overlap", "similar", "patentable")):
        intent, new = "novelty_comparison", False
    elif len(words) >= 15:
        intent, new = "prior_art_search", True
    elif any(w in text for w in ("search", "find", "look for", "prior art")):
        intent, new = "prior_art_search", False
    else:
        intent, new = "help", False
    return {"intent": intent, "is_new_idea": new, "reason": "keyword fallback", "method": "keyword"}


# ------------------------------------------------------------------------------- agents
class PatentAgents:
    def __init__(self, llm, store: PriorArtStore, live_search: bool = True):
        self.store = store
        self.live = live_search
        self.tools = build_tools()
        self.router = build_chain(llm, prompts.ROUTER_SYSTEM, prompts.ROUTER_USER)
        self.planner = build_chain(llm, prompts.PLANNER_SYSTEM, prompts.PLANNER_USER)
        self.sufficiency = build_chain(llm, prompts.SUFFICIENCY_SYSTEM, prompts.SUFFICIENCY_USER)
        self.novelty = build_chain(llm, prompts.NOVELTY_SYSTEM, prompts.NOVELTY_USER)
        self.gap_draft = build_chain(llm, prompts.GAP_DRAFT_SYSTEM, prompts.GAP_DRAFT_USER)
        self.gap_reflect = build_chain(llm, prompts.GAP_REFLECT_SYSTEM, prompts.GAP_REFLECT_USER)
        self.reporter = build_chain(llm, prompts.REPORT_SYSTEM, prompts.REPORT_USER, as_json=False)
        self.helper = build_chain(llm, prompts.HELP_SYSTEM, prompts.HELP_USER, as_json=False)

    # ------------------------------------------------------------ safe invocation
    def _invoke(self, chain, inputs: dict, retries: int = 1, transient_retries: int = 3):
        """Call a chain.

        * malformed JSON from the model   -> retry `retries` times
        * temporary server trouble (503/429, overload) -> wait 2s, 5s, 10s and retry
        * anything else (bad key, retired model, no internet) -> raise ModelUnavailable
        """
        json_failures, waits = 0, [2, 5, 10]
        transient_used = 0
        last = None
        while True:
            try:
                return chain.invoke(inputs)
            except ValueError as exc:
                if "valid JSON" not in str(exc) and not isinstance(exc, json.JSONDecodeError):
                    raise ModelUnavailable(f"The language model call failed: {exc}") from exc
                last = exc
                json_failures += 1
                log.warning("bad JSON from model: %s", exc)
                if json_failures > retries:
                    raise AgentError(f"The model kept returning malformed output: {last}") from exc
            except Exception as exc:  # noqa: BLE001 - API/network/key/model-name problems
                if is_transient(exc) and transient_used < transient_retries:
                    wait = waits[min(transient_used, len(waits) - 1)]
                    transient_used += 1
                    log.warning("temporary model error (%s); retrying in %ss", exc, wait)
                    time.sleep(wait)
                    continue
                if is_transient(exc):
                    raise ModelUnavailable(
                        "Google's servers are busy right now (temporary overload). "
                        f"Wait a minute and press Enter again. Details: {exc}") from exc
                raise ModelUnavailable(f"The language model call failed: {exc}") from exc

    # ---------------------------------------------------------------- 1. router
    def route(self, message: str, mem: SessionMemory) -> dict:
        try:
            data = self._invoke(self.router, {
                "history": mem.recent_history(), "message": message,
                "has_idea": mem.has_idea, "has_comparison": mem.has_comparison,
            }, retries=0, transient_retries=1)
            intent = str(data.get("intent", "")).strip()
            if intent not in INTENTS:
                raise ValueError(f"bad intent {intent!r}")
            return {"intent": intent, "is_new_idea": bool(data.get("is_new_idea")),
                    "reason": str(data.get("reason", "")), "method": "llm"}
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM router failed (%s); using keyword router", exc)
            return keyword_route(message, mem)

    # --------------------------------------------------------------- 2. planner
    def plan(self, mem: SessionMemory, trace: list[str]) -> None:
        try:
            data = self._invoke(self.planner, {"idea": mem.idea})
            features = [str(f).strip() for f in data.get("features", []) if str(f).strip()][:6]
            queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()][:5]
            mem.summary = str(data.get("summary", "")).strip()
            trace.append(f"Planner reasoning: {data.get('reasoning', '')}")
        except ModelUnavailable:
            raise                                # API/model problem: show it to the user
        except AgentError:
            features, queries = [], []
            trace.append("Planner output was unusable - using a simple keyword plan instead.")
        if not features:
            features = [mem.idea]
        if not queries:
            queries = [" ".join(sources.keywords(mem.idea, 6)) or mem.idea]
        if not mem.summary:
            mem.summary = mem.idea[:200]
        mem.features = features
        mem.pending_queries = queries            # picked up by search()
        trace.append(f"Planner -> {len(features)} features, {len(queries)} queries")

    # ------------------------------------------------------------ 3. search agent
    def _live_search(self, queries: list[str], trace: list[str]) -> int:
        added = 0
        for q in queries:
            for name, tool in self.tools.items():
                try:
                    docs = tool.invoke({"query": q})
                except Exception as exc:  # noqa: BLE001 - network problems must not crash the agent
                    trace.append(f"Tool {name} failed for '{q}': {exc}")
                    continue
                added += self.store.add_documents(docs)
                trace.append(f"Tool {name}('{q}') -> {len(docs)} documents stored in ChromaDB")
        return added

    def _retrieve(self, mem: SessionMemory) -> list[list[dict]]:
        """Similarity search in ChromaDB for every feature; remember what was seen."""
        per_feature = []
        for feature in mem.features:
            hits = self.store.query(feature, k=config.TOP_K)
            per_feature.append(hits)
            for h in hits:
                old = mem.evidence.get(h["id"])
                if old is None or h["similarity"] > old["similarity"]:
                    mem.evidence[h["id"]] = h
        return per_feature

    def search(self, mem: SessionMemory, trace: list[str], extra_queries: list[str] | None = None) -> None:
        pending = list(extra_queries or mem.pending_queries)
        for round_no in range(1, config.MAX_SEARCH_ROUNDS + 1):
            new_q = [q for q in pending if q not in mem.queries_done]
            if self.live and new_q:
                trace.append(f"Round {round_no}: running {len(new_q)} live queries")
                self._live_search(new_q, trace)
            mem.queries_done.extend(new_q)
            per_feature = self._retrieve(mem)
            if not self.live or round_no == config.MAX_SEARCH_ROUNDS:
                break
            # ReAct decision: is the evidence broad enough, or should we search again?
            summary_lines = "\n".join(
                f"{i}. {f}" for i, f in enumerate(mem.features, 1))
            evidence_lines = "\n".join(
                f"{h['id']} | {h['similarity']:.2f} | {h['title']}"
                for hits in per_feature for h in hits[:2])
            try:
                decision = self._invoke(self.sufficiency, {
                    "features": summary_lines,
                    "queries_done": "\n".join(mem.queries_done),
                    "evidence": evidence_lines or "(nothing yet)",
                }, retries=0)
            except AgentError:
                break
            trace.append(f"Thought: {decision.get('thought', '')} | Action: {decision.get('action')}")
            action = str(decision.get("action", "FINISH")).upper()
            pending = [str(q) for q in decision.get("new_queries", [])][:3]
            if action != "SEARCH_MORE" or not pending:
                break

    # ------------------------------------------------- 4. novelty comparison agent
    def compare(self, mem: SessionMemory, trace: list[str]) -> None:
        per_feature = self._retrieve(mem)
        if self.live:                                   # Agentic RAG: weak evidence -> search again
            for i, hits in enumerate(per_feature):
                best = hits[0]["similarity"] if hits else 0.0
                if best < config.LOW_SIMILARITY and mem.features[i] not in mem.queries_done:
                    trace.append(f"Feature {i + 1} has weak evidence (best similarity {best:.2f}) "
                                 "- running a targeted search")
                    self._live_search([mem.features[i]], trace)
                    mem.queries_done.append(mem.features[i])
            per_feature = self._retrieve(mem)

        blocks, allowed = [], {}
        for i, (feature, hits) in enumerate(zip(mem.features, per_feature)):
            blocks.append(f'Feature {i + 1}: "{feature}"')
            if not hits:
                blocks.append("  (no excerpts retrieved)")
            for h in hits:
                blocks.append(f"  Excerpt [{h['id']}] ({h['kind']}, {h['year'] or 'n.d.'}): "
                              f"{h['text'][:600]}")
            allowed[i] = {h["id"] for h in hits}
        raw = self._invoke(self.novelty, {"summary": mem.summary or mem.idea,
                                          "blocks": "\n".join(blocks)})
        mem.comparisons = scoring.validate_comparisons(raw, mem.features, allowed)
        for comp, hits in zip(mem.comparisons, per_feature):
            comp["top_similarity"] = hits[0]["similarity"] if hits else 0.0
        mem.score = scoring.overlap_index(mem.comparisons)
        mem.flags = scoring.risk_flags(mem.comparisons)
        trace.append(f"Overlap index {mem.score} ({scoring.overlap_band(mem.score)}). {mem.flags['note']}")

    # ----------------------------------------------------------------- 5. gap agent
    def find_gaps(self, mem: SessionMemory, trace: list[str]) -> None:
        table = format_verdicts(mem.comparisons)
        draft = self._invoke(self.gap_draft, {"idea": mem.idea, "verdicts": table})
        trace.append("Gap agent: first draft written")
        final = self._invoke(self.gap_reflect, {
            "idea": mem.idea, "verdicts": table, "draft": json.dumps(draft)})
        trace.append(f"Reflection: {final.get('critique', '')}")
        mem.gaps = {
            "critique": final.get("critique", ""),
            "novel_aspects": final.get("novel_aspects", []) or [],
            "underexplored": final.get("underexplored", []) or [],
        }

    # --------------------------------------------------------------- 6. report agent
    def closest_references(self, mem: SessionMemory, n: int = 5) -> list[dict]:
        cited = {e for c in mem.comparisons for e in c["evidence_ids"]}
        ranked = sorted(mem.evidence.values(), key=lambda d: d["similarity"], reverse=True)
        chosen = [d for d in ranked if d["id"] in cited]
        chosen += [d for d in ranked if d["id"] not in cited][: max(0, n - len(chosen))]
        return chosen

    def write_report(self, mem: SessionMemory, trace: list[str]) -> str:
        refs = self.closest_references(mem)
        ref_lines = "\n".join(f"{d['id']} | {d['title']} | {d['similarity']:.2f}" for d in refs)
        score_summary = (f"Overlap index: {mem.score}/100 ({scoring.overlap_band(mem.score)}). "
                         f"{mem.flags.get('note', '')} "
                         f"Anticipation risk: {mem.flags.get('anticipation_risk')}. "
                         f"Obviousness risk: {mem.flags.get('obviousness_risk')}.")
        gaps_text = json.dumps(mem.gaps, indent=1)
        narrative = self._invoke(self.reporter, {
            "idea": mem.idea, "score_summary": score_summary,
            "verdicts": format_verdicts(mem.comparisons), "gaps": gaps_text,
            "references": ref_lines or "(none)"}).strip()
        trace.append("Report agent: narrative written, tables assembled by code")
        mem.report_md = assemble_report(mem, refs, narrative)
        return mem.report_md


# ------------------------------------------------------------ deterministic report parts
def assemble_report(mem: SessionMemory, refs: list[dict], narrative: str) -> str:
    """Tables, scores, references and the disclaimer come from CODE, not from the LLM."""
    out = ["# Preliminary Prior-Art & Novelty Note", "",
           f"> {config.DISCLAIMER}", "",
           "## 1. Invention", mem.idea, ""]
    out += ["## 2. Screening result", "",
            f"- **Overlap index:** {mem.score}/100 - {scoring.overlap_band(mem.score)}",
            f"- **Closest reference:** {mem.flags.get('closest_reference') or 'none'} "
            f"(covers {mem.flags.get('closest_coverage', '0/0')} features)",
            f"- **Possible lack of novelty (one reference has everything):** "
            f"{'YES' if mem.flags.get('anticipation_risk') else 'no'}",
            f"- **Possible lack of inventive step (all known, spread across references):** "
            f"{'YES' if mem.flags.get('obviousness_risk') else 'no'}",
            f"- {mem.flags.get('note', '')}", ""]
    out += ["## 3. Feature-by-feature comparison", "",
            "| # | Feature | Verdict | References |", "|---|---|---|---|"]
    for i, c in enumerate(mem.comparisons, 1):
        out.append(f"| {i} | {c['feature']} | {c['verdict']} | {', '.join(c['evidence_ids']) or '-'} |")
    out += ["", "## 4. Novelty gaps", ""]
    for item in mem.gaps.get("novel_aspects", []):
        out.append(f"- **Possibly novel:** {item.get('aspect', '')} - {item.get('why', '')} "
                   f"(risk: {item.get('risk', 'n/a')})")
    for item in mem.gaps.get("underexplored", []):
        out.append(f"- **Under-explored:** {item.get('aspect', '')} - {item.get('why', '')}")
    if not mem.gaps.get("novel_aspects") and not mem.gaps.get("underexplored"):
        out.append("- No clear gaps identified.")
    out += ["", "## 5. Analysis", "", narrative, "", "## 6. References", ""]
    for d in refs:
        link = f" - {d['url']}" if d.get("url") else ""
        year = d["year"] or "n.d."
        out.append(f"- **[{d['id']}]** {d['title']} ({d['kind']}, {year}; source: {d['source']}; "
                   f"similarity {d['similarity']:.2f}){link}")
    out += ["", "---", f"*{config.DISCLAIMER}*"]
    return "\n".join(out)