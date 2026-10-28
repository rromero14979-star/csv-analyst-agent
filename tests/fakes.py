"""Test doubles: a scripted fake LLM and fake node factories. No API calls are made."""
from types import SimpleNamespace


def _pop(seq):
    return seq.pop(0) if len(seq) > 1 else seq[0]


class FakeLLM:
    """Mimics ChatOpenAI: with_structured_output(schema).invoke(...) and .invoke(...)."""

    def __init__(self, structured=None, texts=("ok",), error=None, error_for=None):
        self.structured = structured or {}   # schema class -> object or list of objects (consumed in order)
        self.texts = list(texts)
        self.error = error                   # raise on every call
        self.error_for = error_for           # or only for this schema class
        self.calls = []

    def with_structured_output(self, schema):
        outer = self

        class Runner:
            def invoke(self, messages):
                outer.calls.append((schema.__name__, messages))
                if outer.error or outer.error_for is schema:
                    raise RuntimeError("fake LLM failure")
                resp = outer.structured[schema]
                return _pop(resp) if isinstance(resp, list) else resp

        return Runner()

    def invoke(self, messages):
        self.calls.append(("plain", messages))
        if self.error:
            raise RuntimeError("fake LLM failure")
        return SimpleNamespace(content=_pop(self.texts))


def fake_nodes(route="numeric", execs=(True,), validations=((True, "none"),)):
    """Eleven fake nodes whose decisions are scripted, to test routing in isolation."""
    execs, validations = list(execs), list(validations)
    n = {"exec": 0, "val": 0}

    def regen(s):  # same semantics as the real nodes that consume validation feedback
        if s.get("validation_passed") is False:
            return {"validation_passed": None, "regen_count": s.get("regen_count", 0) + 1, "attempts": 0}
        return {}

    def schema_profiler(s):
        return {"max_attempts": 3, "trace": ["schema_profiler"]}

    def question_router(s):
        return {"route": route, "trace": ["question_router"]}

    def pandas_agent(s):
        return {**regen(s), "trace": ["pandas_agent"]}

    def chart_agent(s):
        return {**regen(s), "trace": ["chart_agent"]}

    def code_executor(s):
        ok = execs[min(n["exec"], len(execs) - 1)]
        n["exec"] += 1
        if ok:
            return {"exec_ok": True, "exec_error": "", "trace": ["code_executor"]}
        return {"exec_ok": False, "exec_error": "boom", "attempts": s.get("attempts", 0) + 1, "trace": ["code_executor"]}

    def dictionary_rag(s):
        return {"trace": ["dictionary_rag"]}

    def web_search(s):
        return {"trace": ["web_search"]}

    def answer_composer(s):
        return {**regen(s), "trace": ["answer_composer"]}

    def answer_validator(s):
        passed, issue = validations[min(n["val"], len(validations) - 1)]
        n["val"] += 1
        return {"validation_passed": passed, "issue": issue, "trace": ["answer_validator"]}

    def report_builder(s):
        return {"final_report": "ok", "trace": ["report_builder"]}

    def fallback_response(s):
        return {"answer": "fallback", "final_report": "fallback", "trace": ["fallback_response"]}

    return {
        "schema_profiler": schema_profiler, "question_router": question_router, "pandas_agent": pandas_agent,
        "chart_agent": chart_agent, "code_executor": code_executor, "dictionary_rag": dictionary_rag,
        "web_search": web_search, "answer_composer": answer_composer, "answer_validator": answer_validator,
        "report_builder": report_builder, "fallback_response": fallback_response,
    }
