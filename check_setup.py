"""Run this ONCE before the app:   python check_setup.py

It tests every outside piece the app depends on (packages, API key, Gemini models, JSON parsing of
real Gemini output, embeddings + ChromaDB, arXiv, OpenAlex) and prints PASS/FAIL for each, so all
problems show up together instead of one at a time. Send me the full output if anything fails.
"""
import importlib.metadata as md
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
results = []


def step(name):
    def deco(fn):
        started = time.time()
        try:
            msg = fn() or "ok"
            results.append((name, True, msg))
            print(f"[PASS] {name}: {msg}  ({time.time() - started:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            text = f"{type(exc).__name__}: {exc}"[:400]
            results.append((name, False, text))
            print(f"[FAIL] {name}: {text}", flush=True)
        return fn
    return deco


@step("Python version")
def _py():
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 13):
        return f"{version} (newer than the recommended 3.10-3.12 range; verify dependency compatibility)"
    return version


@step("Packages installed")
def _pkgs():
    names = ["streamlit", "pandas", "langchain-core", "langchain-google-genai",
             "langchain-ollama", "chromadb", "requests", "pypdf", "python-dotenv", "pytest"]
    return ", ".join(f"{n} {md.version(n)}" for n in names)


from noveltyscout import config  # noqa: E402  (loads .env)


@step("API key found in .env")
def _key():
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No GOOGLE_API_KEY. The file must be named exactly .env (not .env.example).")
    return "API key is present (value hidden)."


@step("Config")
def _cfg():
    return f"model={config.GEMINI_MODEL}, fallbacks={config.GEMINI_FALLBACKS}"


def _try_model(name):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from noveltyscout.llm import _message_to_text

    llm = ChatGoogleGenerativeAI(model=name, max_retries=1)
    text = _message_to_text(llm.invoke("Reply with exactly the word: OK"))
    return f"replied {text.strip()[:30]!r}"


for _name in [config.GEMINI_MODEL] + [m for m in config.GEMINI_FALLBACKS if m != config.GEMINI_MODEL]:
    step(f"Gemini model {_name}")(lambda n=_name: _try_model(n))


@step("LCEL chain + JSON parsing with real Gemini output")
def _chain():
    from noveltyscout.agents import build_chain
    from noveltyscout.llm import get_llm

    chain = build_chain(get_llm(), "You reply with JSON only.",
                        'Return {{"intent": "help", "ok": true}} for the message: {message}')
    data = chain.invoke({"message": "hello"})
    if not isinstance(data, dict):
        raise RuntimeError(f"expected a dict, got {type(data)}")
    return f"parsed {data}"


@step("Embeddings + ChromaDB (first run downloads ~80 MB)")
def _chroma():
    from noveltyscout import sources
    from noveltyscout.vectorstore import PriorArtStore

    store = PriorArtStore(path=tempfile.mkdtemp(prefix="ns_check_"))
    store.add_documents(sources.load_sample_corpus())
    hits = store.query("child resistant medicine bottle cap", k=2)
    if not hits:
        raise RuntimeError("query returned nothing")
    return f"{store.count()} docs stored; best match {hits[0]['id']} similarity {hits[0]['similarity']}"


@step("arXiv search")
def _arxiv():
    from noveltyscout import sources
    return f"{len(sources.search_arxiv('child resistant medicine bottle'))} results"


@step("OpenAlex search")
def _openalex():
    from noveltyscout import sources
    return f"{len(sources.search_openalex('child resistant medicine bottle'))} results"


print("\n" + "=" * 60)
failed = [r for r in results if not r[1]]
if not failed:
    print("ALL CHECKS PASSED - run:  streamlit run app.py")
else:
    print(f"{len(failed)} check(s) failed:")
    for name, _, msg in failed:
        print(f" - {name}: {msg}")
    print("\nCopy this whole output and send it to me.")