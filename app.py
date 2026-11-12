"""NoveltyScout - Prior-Art & Patentability Screener (Streamlit front end).

Run with:   streamlit run app.py
The Gemini key is read from the .env file (GOOGLE_API_KEY=...), never typed into the page.
"""
import html
import os

import pandas as pd
import streamlit as st

from noveltyscout import config, prompts, sources
from noveltyscout.agents import PatentAgents
from noveltyscout.llm import get_llm
from noveltyscout.memory import SessionMemory
from noveltyscout.orchestrator import NoveltyScout
from noveltyscout import scoring
from noveltyscout.vectorstore import PriorArtStore

st.set_page_config(page_title="NoveltyScout", page_icon="🔎", layout="wide")

EXAMPLE_IDEA = (
    "A child-resistant medicine bottle with no lock and no cap to twist. The bottle neck has a "
    "flow-restricting insert with a narrow zig-zag channel, and a tilt sensor that only lets a single "
    "tablet pass when the bottle is tilted to a specific angle and tapped once. This stops a toddler "
    "from pouring out many tablets at once."
)

CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg, #123a6b 0%, #1b6f8a 55%, #14b8a6 100%);
       border-radius: 18px; padding: 26px 32px; margin-bottom: 14px;
       box-shadow: 0 10px 30px rgba(20,184,166,.18);}
.hero h1 {margin: 0; font-size: 2.3rem; color: #fff; letter-spacing: .3px;}
.hero p {margin: 6px 0 0 0; color: #dff6f3; font-size: 1.02rem;}
.hero .tag {display:inline-block; margin-top:12px; margin-right:6px; padding:3px 11px; border-radius:999px;
            background: rgba(255,255,255,.16); color:#fff; font-size:.78rem;}
.steps {display:flex; gap:8px; margin: 6px 0 16px 0; flex-wrap:wrap;}
.step {flex:1; min-width:120px; padding:10px 12px; border-radius:12px; font-size:.92rem; font-weight:600;
       border:1px solid rgba(255,255,255,.10); background: rgba(255,255,255,.04); color:#9fb0c8;}
.step span {display:inline-block; width:22px; height:22px; line-height:22px; text-align:center; border-radius:50%;
            margin-right:8px; font-size:.78rem; background: rgba(255,255,255,.10);}
.step.done {background: rgba(45,212,191,.14); border-color: rgba(45,212,191,.55); color:#7ff0e0;}
.step.done span {background:#2dd4bf; color:#06231f;}
.step.next {border-color:#f5b942; color:#ffd88a;}
.step.next span {background:#f5b942; color:#2b1d00;}
.card {border-radius:14px; padding:14px 16px; background: rgba(255,255,255,.045);
       border:1px solid rgba(255,255,255,.09); height:100%;}
.card .lbl {font-size:.74rem; text-transform:uppercase; letter-spacing:.8px; color:#8fa3bf;}
.card .val {font-size:1.9rem; font-weight:700; color:#eaf2ff; line-height:1.2;}
.card .sub {font-size:.8rem; color:#9fb0c8;}
.bar {height:12px; border-radius:99px; background: rgba(255,255,255,.10); overflow:hidden; margin:8px 0 4px 0;}
.bar > div {height:100%; border-radius:99px;}
.badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:700;}
.badge.disclosed {background: rgba(248,113,113,.18); color:#ff9b9b; border:1px solid rgba(248,113,113,.5);}
.badge.partial {background: rgba(245,185,66,.16); color:#ffd88a; border:1px solid rgba(245,185,66,.5);}
.badge.not_found {background: rgba(52,211,153,.16); color:#7ff0bf; border:1px solid rgba(52,211,153,.5);}
table.cmp {width:100%; border-collapse: collapse; font-size:.92rem;}
table.cmp th {text-align:left; padding:8px 10px; color:#8fa3bf; font-weight:600; font-size:.78rem;
              text-transform:uppercase; letter-spacing:.6px; border-bottom:1px solid rgba(255,255,255,.14);}
table.cmp td {padding:10px; vertical-align:top; border-bottom:1px solid rgba(255,255,255,.07);}
.note {border-left:4px solid #2dd4bf; background: rgba(45,212,191,.08); padding:10px 14px; border-radius:6px;
       margin:10px 0; font-size:.93rem;}
.warn {border-left:4px solid #f5b942; background: rgba(245,185,66,.08); padding:10px 14px; border-radius:6px;
       margin:10px 0; font-size:.88rem;}
.hw {display:flex; gap:12px; flex-wrap:wrap; margin: 8px 0;}
.hw .box {flex:1; min-width:210px; padding:14px 16px; border-radius:14px; background: rgba(255,255,255,.045);
          border:1px solid rgba(255,255,255,.09);}
.hw .box h4 {margin:0 0 6px 0; color:#7ff0e0; font-size:1rem;}
.hw .box p {margin:0; font-size:.88rem; color:#c5d2e6;}
div[data-baseweb="tab-list"] {gap: 6px;}
button[data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: 8px 16px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Opening vector database (first run downloads a small model)...")
def get_store() -> PriorArtStore:
    store = PriorArtStore()
    if store.count_by_origin().get("sample", 0) == 0:       # first run: load demo corpus
        store.add_documents(sources.load_sample_corpus())
    return store


def build_engine(provider: str, model: str, live: bool) -> NoveltyScout:
    llm = get_llm(provider, model)
    agents = PatentAgents(llm, get_store(), live_search=live)
    return NoveltyScout(agents, st.session_state.memory)


def card(label, value, sub=""):
    return (f'<div class="card"><div class="lbl">{html.escape(str(label))}</div>'
            f'<div class="val">{html.escape(str(value))}</div>'
            f'<div class="sub">{html.escape(str(sub))}</div></div>')


def stepper(mem: SessionMemory) -> str:
    steps = [("Describe", mem.has_idea), ("Plan", bool(mem.features)), ("Search", bool(mem.evidence)),
             ("Compare", bool(mem.comparisons)), ("Report", bool(mem.report_md))]
    out, marked_next = ['<div class="steps">'], False
    for i, (name, done) in enumerate(steps, 1):
        cls = "done" if done else ("todo" if marked_next else "next")
        marked_next = marked_next or not done
        out.append(f'<div class="step {cls}"><span>{i}</span>{name}</div>')
    out.append("</div>")
    return "".join(out)


# ---------------------------------------------------------------------- session state
if "memory" not in st.session_state:
    st.session_state.memory = SessionMemory()
    st.session_state.messages = []          # [(role, text, trace)]
    st.session_state.engine_key = None
    st.session_state.engine = None
    st.session_state.queued = None

store = get_store()
mem: SessionMemory = st.session_state.memory

# ---------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🔎 NoveltyScout")
    st.caption("Prior-art & patentability screener")
    provider = st.selectbox("LLM provider", ["gemini", "ollama"],
                            index=0 if config.DEFAULT_PROVIDER == "gemini" else 1)
    default_model = config.GEMINI_MODEL if provider == "gemini" else config.OLLAMA_MODEL
    model = st.text_input("Model name", value=default_model)
    if provider == "gemini":
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            st.success("API key loaded from .env", icon="✅")
        else:
            st.error("No API key found. Add GOOGLE_API_KEY=... to the .env file, then restart the app.")
    live = st.toggle("Live search (arXiv + OpenAlex)", value=True,
                     help="Off = use only what is already in the vector database.")
    st.divider()
    counts = store.count_by_origin()
    st.caption(f"Vector DB  |  sample {counts.get('sample', 0)}  ·  live {counts.get('live', 0)}  ·  "
               f"uploaded {counts.get('upload', 0)}")
    if st.button("Forget live results", width="stretch"):
        store.delete_origin("live")
        mem.evidence = {key: value for key, value in mem.evidence.items()
                        if value.get("origin") != "live"}
        mem.comparisons, mem.gaps, mem.report_md = [], {}, ""
        mem.score, mem.flags = 0.0, {}
        st.rerun()
    if st.button("New session", width="stretch"):
        st.session_state.memory = SessionMemory()
        st.session_state.messages = []
        st.session_state.engine_key = None
        st.rerun()
    st.divider()
    st.caption(config.DISCLAIMER)

# Rebuild the engine only when settings change (engine holds the LLM + agent chains).
engine_key = (provider, model, live)
if st.session_state.engine_key != engine_key:
    try:
        st.session_state.engine = build_engine(provider, model, live)
        st.session_state.engine_key = engine_key
    except Exception as exc:  # noqa: BLE001 - show setup problems in the UI
        st.session_state.engine = None
        st.error(f"Could not start the language model: {exc}")
engine: NoveltyScout | None = st.session_state.engine
mem = st.session_state.memory

# ------------------------------------------------------------------------------ header
st.markdown(
    '<div class="hero"><h1>🔎 NoveltyScout</h1>'
    '<p>Describe your invention. A team of AI agents searches available evidence, compares it feature by feature, '
    'and highlights matching material and potential evidence gaps.</p>'
    '<span class="tag">Agentic RAG</span><span class="tag">LangChain LCEL</span>'
    '<span class="tag">ChromaDB</span><span class="tag">ReAct + Reflection</span></div>',
    unsafe_allow_html=True)
st.markdown(stepper(mem), unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.markdown(card("Documents in vector DB", store.count(), "searchable by meaning"), unsafe_allow_html=True)
m2.markdown(card("Features checked", len(mem.features), "from your idea"), unsafe_allow_html=True)
m3.markdown(card("Evidence found", len(mem.evidence), "references retrieved"), unsafe_allow_html=True)
m4.markdown(card("Evidence overlap", f"{mem.score:.0f}/100" if mem.comparisons else "-",
                 scoring.overlap_band(mem.score).split(" - ")[0] if mem.comparisons else "run a comparison"),
            unsafe_allow_html=True)
st.write("")

tab_chat, tab_evidence, tab_analysis, tab_report, tab_how, tab_prompts = st.tabs(
    ["💬 Chat", "📚 Evidence", "🧪 Analysis", "📄 Report", "🧭 How it works", "🧠 Prompts & trace"])

# ------------------------------------------------------------------------------ chat
with tab_chat:
    c1, c2, c3 = st.columns(3)
    c1.button("✨ Try the example idea", width="stretch",
              on_click=lambda: st.session_state.update(queued=EXAMPLE_IDEA))
    c2.button("⚖️ Compare for novelty", width="stretch", on_click=lambda: st.session_state.update(
        queued="How does my idea compare with the prior art? Is it novel?"))
    c3.button("📄 Give me the report", width="stretch", on_click=lambda: st.session_state.update(
        queued="Please give me the final prior-art note."))

    if not st.session_state.messages:
        st.markdown('<div class="note">Start by typing your invention below (what it is, how it works, what '
                    'problem it solves) or click <b>Try the example idea</b>.</div>', unsafe_allow_html=True)

    for role, text, trace in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(text)
            if trace:
                with st.expander("Agent steps"):
                    for line in trace:
                        st.write("• " + line)

    typed = st.chat_input("Describe your invention or ask a question...")
    user_text = typed or st.session_state.queued
    st.session_state.queued = None
    if user_text:
        if engine is None:
            st.error("The language model is not ready - check the sidebar.")
        else:
            st.session_state.messages.append(("user", user_text, None))
            with st.chat_message("user"):
                st.markdown(user_text)
            with st.chat_message("assistant"):
                with st.spinner("Agents are working..."):
                    reply = engine.handle(user_text)
                st.markdown(reply.text)
                with st.expander("Agent steps"):
                    for line in reply.trace:
                        st.write("• " + line)
            st.session_state.messages.append(("assistant", reply.text, reply.trace))
            st.rerun()

# -------------------------------------------------------------------------- evidence
with tab_evidence:
    st.subheader("Evidence collected for the current idea")
    if mem.evidence:
        df = pd.DataFrame(mem.evidence.values())[
            ["id", "title", "kind", "year", "source", "origin", "similarity"]
        ].sort_values("similarity", ascending=False)
        st.dataframe(df, width="stretch", hide_index=True,
                     column_config={"similarity": st.column_config.ProgressColumn(
                         "similarity", min_value=0.0, max_value=1.0, format="%.2f")})
    else:
        st.info("Nothing yet - describe an idea in the Chat tab.")

    if mem.queries_done:
        st.markdown("**Search patents manually (free):**")
        for q in mem.queries_done[:4]:
            st.markdown(f"- [{q}]({sources.google_patents_link(q)})")

    st.divider()
    st.subheader("Add your own evidence")
    st.caption("Live search currently covers scholarly papers, not patent databases. Copy a patent abstract/claims from Google Patents, Espacenet or the Indian patent portal, or upload "
               "a PDF/TXT. It is split into chunks, embedded and stored in ChromaDB.")
    u_title = st.text_input("Title / patent number")
    u_kind = st.selectbox("Type", ["patent", "paper"])
    u_text = st.text_area("Paste text", height=140)
    u_file = st.file_uploader("...or upload a file", type=["txt", "pdf"])
    if st.button("Add to vector database", type="primary"):
        text = u_text
        try:
            if u_file is not None:
                text = (sources.text_from_pdf(u_file) if u_file.name.lower().endswith(".pdf")
                        else u_file.read().decode("utf-8", errors="ignore"))
            if not text.strip():
                st.warning("No selectable text was found. If this is a scanned PDF, run OCR first or paste the abstract/claims manually.")
            else:
                docs = sources.docs_from_text(u_title or (u_file.name if u_file else ""), text, u_kind)
                n = store.add_documents(docs)
                if n:
                    st.success(f"Stored {n} chunk(s). Ask for the comparison again to use them.")
                    mem.comparisons, mem.gaps, mem.report_md = [], {}, ""
                else:
                    st.warning("No evidence chunks were created. Please check the supplied text.")
        except Exception as exc:  # noqa: BLE001 - display upload/parser errors in the UI
            st.error(f"Could not add this evidence: {exc}")

# -------------------------------------------------------------------------- analysis
with tab_analysis:
    if mem.comparisons:
        score = mem.score
        color = "#34d399" if score < 25 else ("#f5b942" if score < 55 else "#f87171")
        st.markdown(
            f'<div class="card"><div class="lbl">Evidence-overlap indicator (heuristic)</div>'
            f'<div class="val">{score:.0f} / 100 &nbsp;<span style="font-size:1rem;color:{color}">'
            f'{html.escape(scoring.overlap_band(score))}</span></div>'
            f'<div class="bar"><div style="width:{max(2, min(100, score)):.0f}%;background:{color}"></div></div>'
            f'<div class="sub">Closest reference: {html.escape(str(mem.flags.get("closest_reference") or "none"))} '
            f'(covers {html.escape(str(mem.flags.get("closest_coverage", "0/0")))} features) · '
            f'Single-reference coverage signal: {"YES" if mem.flags.get("anticipation_risk") else "no"} · '
            f'Multi-reference combination review: {"YES" if mem.flags.get("combination_review") else "no"}</div></div>',
            unsafe_allow_html=True)
        st.markdown(f'<div class="note">{html.escape(mem.flags.get("note", ""))}</div>', unsafe_allow_html=True)

        rows = ["<table class='cmp'><tr><th>#</th><th>Feature</th><th>Verdict</th><th>References</th>"
                "<th>Why</th></tr>"]
        for i, c in enumerate(mem.comparisons, 1):
            rows.append(
                f"<tr><td>{i}</td><td>{html.escape(c['feature'])}</td>"
                f"<td><span class='badge {c['verdict']}'>{c['verdict'].replace('_', ' ')}</span></td>"
                f"<td>{html.escape(', '.join(c['evidence_ids']) or '-')}</td>"
                f"<td>{html.escape(c.get('reasoning', ''))}</td></tr>")
        rows.append("</table>")
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.caption("Red = a retrieved excerpt was judged to disclose the feature · amber = partial match · green = no clear match was found in the retrieved evidence set")

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### 🔎 Potential evidence gaps")
            for g in mem.gaps.get("novel_aspects", []) or []:
                st.markdown(f"- **{g.get('aspect', '')}** - {g.get('why', '')} _(risk: {g.get('risk', 'n/a')})_")
        with g2:
            st.markdown("#### 🔍 Under-explored")
            for g in mem.gaps.get("underexplored", []) or []:
                st.markdown(f"- **{g.get('aspect', '')}** - {g.get('why', '')}")
        if mem.gaps.get("critique"):
            st.caption(f"Reflection step (the reviewer agent's note): {mem.gaps['critique']}")
    else:
        st.info("Ask for a novelty comparison in the Chat tab to see the analysis here.")

# ---------------------------------------------------------------------------- report
with tab_report:
    if mem.report_md:
        st.download_button("⬇️ Download report (.md)", mem.report_md,
                           file_name="prior_art_note.md", mime="text/markdown", type="primary")
        st.markdown(mem.report_md)
    else:
        st.info("Ask for the report in the Chat tab.")

# ----------------------------------------------------------------------- how it works
with tab_how:
    st.subheader("How NoveltyScout works")
    st.markdown(
        '<div class="hw">'
        '<div class="box"><h4>1. Understand</h4><p>The <b>router</b> reads your message and decides what you want: '
        'search, compare, report or help. The <b>planner</b> splits your invention into 3-6 checkable features '
        'and writes search queries.</p></div>'
        '<div class="box"><h4>2. Collect evidence</h4><p>The <b>search agent</b> calls arXiv and OpenAlex, '
        'converts every abstract into an embedding (numbers capturing meaning) and stores it in <b>ChromaDB</b>. '
        'It then decides whether to search again (ReAct).</p></div>'
        '<div class="box"><h4>3. Compare</h4><p>For each feature the closest documents are fetched by similarity '
        'search. The <b>novelty agent</b> labels each feature disclosed, partial or not found. Weak evidence '
        'triggers a targeted extra search (Agentic RAG).</p></div>'
        '<div class="box"><h4>4. Judge and report</h4><p>Python code computes a heuristic evidence-overlap indicator and review signals. '
        'The <b>gap agent</b> drafts what looks new and then criticises its own draft (Reflection). '
        'The <b>report agent</b> writes the cited note.</p></div></div>',
        unsafe_allow_html=True)
    st.markdown(
        "**Key ideas**\n"
        "- **Embeddings + vector DB:** text becomes numbers; similar meaning = nearby numbers, so 'closure' matches 'cap'.\n"
        "- **Agentic RAG:** the system decides when to retrieve more, instead of always retrieving once.\n"
        "- **LCEL:** every agent is a chain `prompt | model | text | JSON` built with LangChain.\n"
        "- **Guardrails:** invented references are dropped, numbers are computed in code, tables are built by code.")
    st.markdown('<div class="warn"><b>Important:</b> this is a screening aid, not a legal patentability opinion. '
                'Absence of a match does not prove an idea is new. Consult a registered patent agent before '
                'filing.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------- prompts
with tab_prompts:
    st.subheader("System and user prompts used by each agent")
    st.caption("Words in curly braces such as {idea} are LangChain template variables filled at run time.")
    for name, (system_p, user_p) in prompts.ALL_PROMPTS.items():
        with st.expander(name):
            st.markdown("**System prompt**")
            st.code(system_p, language="text")
            st.markdown("**User prompt template**")
            st.code(user_p, language="text")