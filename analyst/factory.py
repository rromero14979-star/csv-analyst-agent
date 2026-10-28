"""Builds a ready-to-use graph from API keys and a DataFrame."""
from dataclasses import dataclass

import pandas as pd
from langchain_openai import OpenAIEmbeddings

from analyst import config
from analyst.builder import build_graph
from analyst.dictionary import DictionarySearch, draft_dictionary
from analyst.llm import make_llm
from analyst.nodes import Deps, make_nodes
from analyst.profiling import profile_dataframe
from analyst.search import make_web_search_fn


class MissingKeyError(RuntimeError):
    """Raised with a friendly message when an API key is missing."""


@dataclass
class DatasetContext:
    df: pd.DataFrame
    dictionary: list
    search: DictionarySearch


def prepare_dataset(openai_key, df, dictionary=None, use_embeddings=True) -> DatasetContext:
    """Draft the dictionary and index it ONCE per dataset (cache this in the app)."""
    if not openai_key:
        raise MissingKeyError("OPENAI_API_KEY is missing. Add it to .env or to Streamlit secrets.")
    llm = make_llm(openai_key)
    dictionary = dictionary or draft_dictionary(llm, profile_dataframe(df))
    embeddings = OpenAIEmbeddings(model=config.EMBED_MODEL, api_key=openai_key) if use_embeddings else None
    return DatasetContext(df=df, dictionary=dictionary, search=DictionarySearch(dictionary, embeddings))


def create_graph(openai_key, tavily_key, ctx: DatasetContext, web_enabled=True, runner=None, **flags):
    if not openai_key:
        raise MissingKeyError("OPENAI_API_KEY is missing. Add it to .env or to Streamlit secrets.")
    if web_enabled and not tavily_key:
        raise MissingKeyError("TAVILY_API_KEY is missing. Add it, or disable web search.")
    extra = {"runner": runner} if runner else {}
    deps = Deps(
        llm=make_llm(openai_key), df=ctx.df, dictionary=ctx.dictionary, dictionary_search=ctx.search,
        web_search=make_web_search_fn(tavily_key) if web_enabled else None,
        web_enabled=web_enabled, **extra, **flags,
    )
    return build_graph(make_nodes(deps))
