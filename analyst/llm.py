"""Chat model factory."""
from langchain_openai import ChatOpenAI

from analyst import config


def make_llm(api_key: str, model: str = config.LLM_MODEL) -> ChatOpenAI:
    """Deterministic chat model (temperature 0) with timeout and automatic retries."""
    return ChatOpenAI(model=model, temperature=0, api_key=api_key, timeout=40, max_retries=2)
