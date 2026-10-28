"""Automatic grading of one answer against the ground truth of eval/questions.csv."""
import pandas as pd

from analyst.checks import extract_numbers, number_supported

DECLINE_WORDS = ("cannot", "can't", "could not", "unable", "not available", "no column", "does not exist",
                 "doesn't exist", "not in the", "not possible", "no information")


def _s(value) -> str:
    return "" if pd.isna(value) else str(value)


def grade(row, state: dict) -> bool:
    """True if the system behaved as expected for this question."""
    kind = row["expected_type"]
    answer = (state.get("answer") or "").lower()
    trace = state.get("trace") or []
    fell_back = bool(trace) and trace[-1] == "fallback_response"

    if kind == "decline":
        return fell_back or any(w in answer for w in DECLINE_WORDS)
    if fell_back:
        return False
    if kind == "number":
        target = float(row["expected_value"])
        return any(number_supported(v, d, [target]) for v, d, _ in extract_numbers(state.get("answer") or ""))
    if kind == "text":
        return _s(row["expected_text"]).lower() in answer
    if kind == "chart":
        cols = [c for c in _s(row["expected_columns"]).split("|") if c]
        code = state.get("code") or ""
        return state.get("figure") is not None and all(c in code for c in cols)
    if kind == "keywords":
        return any(k.lower() in answer for k in _s(row["expected_keywords"]).split("|") if k)
    if kind == "web":
        return state.get("route") == "external" and bool(state.get("evidence")) and bool(answer)
    return False
