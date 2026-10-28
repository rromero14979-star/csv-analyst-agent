"""Deterministic number checks used by the answer validator and by the evaluation.

An answer is 'hallucinated' when it contains numbers that do not appear (within rounding
or as a percentage form) in the executed result or evidence it is supposed to be based on.
"""
import re

# '%' in the lookbehind: in a range such as "2%-6%" the hyphen is a dash, not a minus sign
_NUM = re.compile(r"(?<![\w.%])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.%])-?\d+(?:\.\d+)?")


def extract_numbers(text: str) -> list:
    """Return [(value, decimals, raw_token)] for every number in the text."""
    found = []
    for match in _NUM.finditer(text or ""):
        raw = match.group(0)
        clean = raw.replace(",", "")
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        try:
            found.append((float(clean), decimals, raw))
        except ValueError:
            continue
    return found


def number_supported(value: float, decimals: int, source_values: list) -> bool:
    """True if `value` matches a source number within rounding (also as % of or x100)."""
    for m in source_values:
        rel = 0.005 * abs(m)
        if abs(value - m) <= max(0.51 * 10 ** (-decimals), rel):
            return True
        if abs(value / 100 - m) <= max(0.51 * 10 ** (-decimals - 2), rel):
            return True
        if abs(value * 100 - m) <= max(0.51 * 10 ** (-decimals + 2), rel):
            return True
    return False


def unsupported_numbers(answer: str, source_text: str, question: str = "") -> list:
    """Numbers in `answer` that are neither in `source_text` nor in the question."""
    source_values = [v for v, _, _ in extract_numbers(source_text)]
    question_values = {v for v, _, _ in extract_numbers(question)} | {100.0}  # "x 100" in percentage wording
    bad = []
    for value, decimals, raw in extract_numbers(answer):
        # a number from the question, also restated as a fraction or percentage ("20%" -> 0.2)
        if any(abs(value - q) < 1e-9 or abs(value * 100 - q) < 1e-6 or abs(value / 100 - q) < 1e-9
               for q in question_values):
            continue
        if not number_supported(value, decimals, source_values):
            bad.append(raw)
    return bad
