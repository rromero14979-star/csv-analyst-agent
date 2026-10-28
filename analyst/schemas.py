"""Pydantic schemas for structured LLM outputs."""
from typing import List, Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    route: Literal["numeric", "chart", "concept", "external", "out_of_scope"]
    reason: str = Field(description="One short sentence")


class CodeOutput(BaseModel):
    code: str = Field(description="Python code only, no markdown fences")


class DictEntry(BaseModel):
    column: str
    description: str


class DictionaryDraft(BaseModel):
    entries: List[DictEntry]


class ValidationResult(BaseModel):
    passed: bool
    issue: Literal["none", "answer", "code"]
    feedback: str
