"""LLM integration (acceptance criterion 1) + small LCEL helper steps.

Two providers are supported so the app works with an API model OR a local model:
  * gemini : Google Gemini through the API (needs GOOGLE_API_KEY)
  * ollama : a model running locally through Ollama (no key, no internet needed)
"""
import json
import re

from langchain_core.runnables import RunnableLambda

from . import config


def get_llm(provider: str | None = None, model: str | None = None,
            temperature: float = config.LLM_TEMPERATURE):
    """Return a LangChain chat model. Imports are lazy so only the chosen provider is needed."""
    provider = (provider or config.DEFAULT_PROVIDER).lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Gemini 3.x: Google's migration guide says to NOT send temperature/top_p/top_k,
        # so only the model name and retries are set here.
        primary_name = model or config.GEMINI_MODEL
        primary = ChatGoogleGenerativeAI(model=primary_name, max_retries=0, timeout=25)
        backups = [ChatGoogleGenerativeAI(model=name, max_retries=0, timeout=25)
                   for name in config.GEMINI_FALLBACKS if name != primary_name]
        # LangChain fallbacks: if the primary model errors (e.g. 503 overload), try the next one.
        return primary.with_fallbacks(backups) if backups else primary
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model or config.OLLAMA_MODEL, temperature=temperature)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


# --------------------------------------------------------------------------
# Output-processing steps used inside LCEL chains:  prompt | llm | to_text | parse_json
# --------------------------------------------------------------------------
def _message_to_text(message) -> str:
    """Turn a chat message into plain text.

    Newer Gemini versions may return `content` as a list of blocks instead of a string,
    so we normalise both shapes.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def extract_json(text: str):
    """Pull the first JSON object/array out of an LLM reply (handles ```json fences)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fall back: find the outermost {...} or [...]
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("LLM did not return valid JSON")


to_text = RunnableLambda(_message_to_text)
parse_json = RunnableLambda(extract_json)