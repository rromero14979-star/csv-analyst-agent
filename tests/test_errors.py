"""E1-E8: failures must degrade gracefully, never crash the app."""
import pytest

from analyst import config
from analyst.builder import build_graph, run_question
from analyst.factory import MissingKeyError, create_graph, prepare_dataset
from analyst.nodes import Deps, make_nodes
from analyst.profiling import load_dataframe
from analyst.sandbox import run_code
from analyst.schemas import CodeOutput, RouteDecision, ValidationResult
from analyst.validation import QuotaLimiter, validate_input
from tests.fakes import FakeLLM

DF = load_dataframe(config.DEMO_CSV)


def test_E1_missing_keys_raise_friendly_error():
    with pytest.raises(MissingKeyError):
        prepare_dataset(None, DF)
    with pytest.raises(MissingKeyError):
        create_graph(None, "tvly", ctx=None)
    with pytest.raises(MissingKeyError):
        create_graph("sk", None, ctx=None, web_enabled=True)


def test_E2_web_failure_does_not_crash():
    def boom(q):
        raise RuntimeError("network down")
    llm = FakeLLM({RouteDecision: RouteDecision(route="external", reason="r"),
                   ValidationResult: ValidationResult(passed=True, issue="none", feedback="")},
                  texts=["I could not find information."])
    g = build_graph(make_nodes(Deps(llm=llm, df=DF, dictionary=[], web_search=boom)))
    s = run_question(g, "What is the current inflation rate?")
    assert "web_search failed" in s["error"] and s["trace"][-1] in ("report_builder", "fallback_response")


def test_E3_llm_failure_in_router_ends_in_fallback():
    g = build_graph(make_nodes(Deps(llm=FakeLLM(error=RuntimeError("x")), df=DF, dictionary=[])))
    s = run_question(g, "anything")
    assert s["trace"] == ["schema_profiler", "question_router", "fallback_response"]


def test_E4_empty_code_from_llm_is_counted_as_a_failed_attempt():
    llm = FakeLLM({RouteDecision: RouteDecision(route="numeric", reason="r"), CodeOutput: CodeOutput(code="")})
    g = build_graph(make_nodes(Deps(llm=llm, df=DF, dictionary=[])))
    s = run_question(g, "total?")
    assert s["attempts"] == 3 and s["trace"][-1] == "fallback_response"


def test_E5_loops_are_bounded_even_with_a_stubborn_validator():
    fake_ok = lambda code, df, kind: {"ok": True, "text": "30.0", "table_json": None, "figure_png": None}
    llm = FakeLLM(
        {RouteDecision: RouteDecision(route="numeric", reason="r"), CodeOutput: CodeOutput(code="result = 1"),
         ValidationResult: ValidationResult(passed=False, issue="code", feedback="bad")},
        texts=["Revenue is 30."],
    )
    g = build_graph(make_nodes(Deps(llm=llm, df=DF, dictionary=[], runner=fake_ok)))
    s = run_question(g, "total?")
    assert len(s["trace"]) < config.RECURSION_LIMIT and s["trace"][-1] == "fallback_response"


def test_E6_input_validation():
    assert validate_input("   ")[0] is False
    assert validate_input("x" * 5000)[0] is False
    assert validate_input("  total revenue? ") == (True, "total revenue?")


def test_E7_session_quota():
    q = QuotaLimiter(max_queries=2)
    q.register(); q.register()
    assert q.allow() is False and q.remaining == 0


def test_E8_secrets_are_not_visible_to_generated_code(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-123")
    res = run_code("result = 'ok'", DF)
    assert res["ok"]  # the worker env is scrubbed; os/environ are also blocked by the AST whitelist
    assert not run_code("result = os.environ", DF)["ok"]


def test_E9_sandbox_worker_does_not_load_dotenv(monkeypatch):
    import importlib

    import dotenv

    calls = []
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: calls.append(1))
    monkeypatch.setenv("ANALYST_SANDBOX", "1")
    importlib.reload(config)
    assert calls == []
    monkeypatch.delenv("ANALYST_SANDBOX")
    importlib.reload(config)
    assert calls == [1]
