"""Data dictionary: drafted from the profile (or loaded from JSON) and searched by dictionary_rag."""
import json
import re
from pathlib import Path

from langchain_community.vectorstores import FAISS

from analyst.prompts import DICTIONARY_SYSTEM
from analyst.schemas import DictionaryDraft


def rule_based_dictionary(profile: dict) -> list:
    entries = []
    for c in profile["columns"]:
        desc = f"{c['kind']} column"
        if "min" in c:
            desc += f", values from {c['min']} to {c['max']}"
        if "top_values" in c:
            desc += ", categories: " + ", ".join(c["top_values"])
        if c["missing"]:
            desc += f", {c['missing']} missing values"
        entries.append({"column": c["name"], "dtype": c["kind"], "description": desc})
    return entries


def draft_dictionary(llm, profile: dict) -> list:
    """Ask the LLM for a description per column; never raises (falls back to rules)."""
    base = rule_based_dictionary(profile)
    try:
        from analyst.profiling import profile_to_text

        draft = llm.with_structured_output(DictionaryDraft).invoke(
            [("system", DICTIONARY_SYSTEM), ("human", profile_to_text(profile))]
        )
        described = {e.column: e.description for e in draft.entries}
        for entry in base:
            if entry["column"] in described:
                entry["description"] = described[entry["column"]]
    except Exception:
        pass
    return base


def load_dictionary_json(path, profile: dict | None = None) -> list:
    """Load [{'column','description'}]; dtype is taken from the profile when available."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    kinds = {c["name"]: c["kind"] for c in profile["columns"]} if profile else {}
    return [
        {"column": e["column"], "dtype": kinds.get(e["column"], e.get("dtype", "")), "description": e["description"]}
        for e in raw
    ]


def _entry_text(e: dict) -> str:
    return f"{e['column']}: {e['description']}"


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3}


class DictionarySearch:
    """Top-k dictionary entries for a question: FAISS if embeddings are given, keywords otherwise."""

    def __init__(self, entries: list, embeddings=None, k: int = 4):
        self.entries, self.k, self.store = entries, k, None
        if embeddings is not None and entries:
            try:
                self.store = FAISS.from_texts(
                    [_entry_text(e) for e in entries], embeddings, metadatas=[dict(e) for e in entries]
                )
            except Exception:
                self.store = None  # fall back to keyword search

    def __call__(self, query: str) -> list:
        if self.store is not None:
            try:
                return [dict(d.metadata) for d in self.store.similarity_search(query, k=self.k)]
            except Exception:
                pass
        q = _words(query)
        scored = sorted(self.entries, key=lambda e: -len(q & _words(_entry_text(e))))
        return [dict(e) for e in scored[: self.k]]
