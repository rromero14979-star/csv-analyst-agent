"""Input validation and per-session usage limit (protects the public demo)."""
from analyst.config import MAX_INPUT_CHARS, MAX_QUERIES_PER_SESSION


def validate_input(text):
    """Return (ok, clean_text_or_error_message)."""
    clean = (text or "").strip()
    if not clean:
        return False, "Please type a question."
    if len(clean) > MAX_INPUT_CHARS:
        return False, f"The question is too long ({len(clean)} characters). Maximum: {MAX_INPUT_CHARS}."
    return True, clean


class QuotaLimiter:
    """Counts questions asked in a session."""

    def __init__(self, max_queries: int = MAX_QUERIES_PER_SESSION):
        self.max_queries = max_queries
        self.used = 0

    @property
    def remaining(self) -> int:
        return max(self.max_queries - self.used, 0)

    def allow(self) -> bool:
        return self.used < self.max_queries

    def register(self) -> None:
        self.used += 1
