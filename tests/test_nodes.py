"""Node-level tests with a fake LLM and a fake code runner (real node code, no network, no subprocess)."""
import pandas as pd

from analyst.nodes import Deps, make_nodes, strip_fences
from analyst.schemas import CodeOutput, RouteDecision, ValidationResult
from tests.fakes import FakeLLM

DF = pd.DataFrame({"Region": ["N", "S"], "Revenue": [10.0, 20.0]})
DICT = [{"column": "Revenue", "dtype": "number", "description": "Order revenue in dollars."}]


def nodes(llm=None, runner=None, **kw):
    extra = {"runner": runner} if runner else {}
    return make_nodes(Deps(llm=llm or FakeLLM(), df=DF, dictionary=DICT, **extra, **kw))


def base_state(**kw):
    return {"question": "total revenue?", "df_profile": "profile", "attempts": 0, "regen_count": 0, **kw}


def test_strip_fences():
    assert strip_fences("```python\nresult = 1\n```") == "result = 1"
    assert strip_fences("result = 1") == "result = 1"


def test_profiler_uses_prebuilt_dictionary_and_sets_max_attempts():
    out = nodes(max_attempts=2)["schema_profiler"]({})
    assert out["data_dictionary"] == DICT and out["max_attempts"] == 2 and "Revenue" in out["df_profile"]


def test_router_returns_route_and_blocks_external_when_web_disabled():
    llm = FakeLLM({RouteDecision: RouteDecision(route="external", reason="needs web")})
    assert nodes(llm)["question_router"](base_state())["route"] == "external"
    assert nodes(llm, web_enabled=False)["question_router"](base_state())["route"] == "out_of_scope"


def test_router_ablation_forces_numeric():
    llm = FakeLLM(error=RuntimeError("must not be called"))
    assert nodes(llm, skip_router=True)["question_router"](base_state())["route"] == "numeric"


def test_pandas_agent_first_draft_repair_and_recode_prompts():
    llm = FakeLLM({CodeOutput: CodeOutput(code="```python\nresult = df['Revenue'].sum()\n```")})
    n = nodes(llm)
    first = n["pandas_agent"](base_state())
    assert first["code"] == "result = df['Revenue'].sum()" and "failed" not in llm.calls[-1][1][1][1]
    n["pandas_agent"](base_state(code="x", exec_error="KeyError: 'Sale Date'"))
    assert "KeyError: 'Sale Date'" in llm.calls[-1][1][1][1]
    out = n["pandas_agent"](base_state(code="x", validation_passed=False, issue="code",
                                       validation_feedback="wrong year", regen_count=0, attempts=2))
    assert "wrong year" in llm.calls[-1][1][1][1]
    assert out["regen_count"] == 1 and out["attempts"] == 0 and out["validation_passed"] is None


def test_executor_success_and_failure():
    ok = lambda code, df, kind: {"ok": True, "text": "30.0", "table_json": '{"columns":["a"],"index":[0],"data":[[1]]}', "figure_png": None}
    bad = lambda code, df, kind: {"ok": False, "error": "KeyError: 'x'"}
    s = nodes(runner=ok)["code_executor"](base_state(code="result=1", route="numeric"))
    assert s["exec_ok"] and s["exec_result"] == "30.0" and list(s["result_table"].columns) == ["a"]
    f = nodes(runner=bad)["code_executor"](base_state(code="result=1", route="numeric", attempts=1))
    assert f["exec_ok"] is False and f["attempts"] == 2 and "KeyError" in f["exec_error"]


def test_executor_passes_chart_kind_for_chart_route():
    seen = {}
    def runner(code, df, kind):
        seen["kind"] = kind
        return {"ok": True, "text": "", "table_json": None, "figure_png": b"png"}
    nodes(runner=runner)["code_executor"](base_state(code="x", route="chart"))
    assert seen["kind"] == "chart"


def test_dictionary_rag_builds_numbered_evidence():
    out = nodes()["dictionary_rag"](base_state(data_dictionary=DICT))
    assert out["evidence"][0]["id"] == 1 and out["evidence"][0]["source"] == "Revenue"


def test_web_search_disabled_and_enabled():
    assert nodes(web_enabled=False)["web_search"](base_state())["evidence"] == []
    deps_web = lambda q: [{"text": "t", "source": "s", "locator": "http://x", "kind": "web"}]
    out = make_nodes(Deps(llm=FakeLLM(), df=DF, web_search=deps_web))["web_search"](base_state())
    assert out["evidence"][0]["id"] == 1


def test_validator_deterministic_check_rejects_invented_number_without_calling_llm():
    llm = FakeLLM()
    out = nodes(llm)["answer_validator"](base_state(route="numeric", answer="Revenue was 999,999.", exec_result="30.0"))
    assert out["validation_passed"] is False and out["issue"] == "answer" and llm.calls == []


def test_validator_uses_judge_after_numbers_pass():
    llm = FakeLLM({ValidationResult: ValidationResult(passed=False, issue="code", feedback="wrong filter")})
    out = nodes(llm)["answer_validator"](base_state(route="numeric", answer="Revenue was 30.", exec_result="30.0", code="c"))
    assert out["validation_passed"] is False and out["issue"] == "code" and out["validation_feedback"] == "wrong filter"


def test_validator_judge_failure_does_not_discard_a_verified_answer():
    llm = FakeLLM(error_for=ValidationResult)
    out = nodes(llm)["answer_validator"](base_state(route="numeric", answer="Revenue was 30.", exec_result="30.0", code="c"))
    assert out["validation_passed"] is True and "judge" in out["error"]


def test_validator_ablation_always_passes():
    out = nodes(skip_validator=True)["answer_validator"](base_state(route="numeric", answer="999", exec_result="1"))
    assert out["validation_passed"] is True


def test_composer_redraft_counts_regeneration_and_includes_feedback():
    llm = FakeLLM(texts=["Revenue was 30."])
    out = nodes(llm)["answer_composer"](base_state(route="numeric", exec_result="30.0", code="c",
                                                   validation_passed=False, validation_feedback="fix it"))
    assert out["answer"] == "Revenue was 30." and out["regen_count"] == 1 and out["validation_passed"] is None
    assert "fix it" in llm.calls[-1][1][0][1]


def test_report_and_fallback_texts():
    n = nodes()
    report = n["report_builder"](base_state(answer="Total 30.", code="result = 1", route="numeric", trace=["a"]))
    assert "Total 30." in report["final_report"] and "result = 1" in report["final_report"]
    out = n["fallback_response"](base_state(route="out_of_scope"))
    assert "only help with this table" in out["answer"]
    out = n["fallback_response"](base_state(route="numeric", exec_ok=False, exec_error="KeyError: 'margin'", attempts=3))
    assert "'Region'" in out["answer"] and "'Revenue'" in out["answer"] and "3 attempt" in out["answer"]


def test_fallback_explains_that_web_search_is_disabled():
    llm = FakeLLM({RouteDecision: RouteDecision(route="external", reason="needs web")})
    n = nodes(llm, web_enabled=False)
    routed = n["question_router"](base_state())
    out = n["fallback_response"](base_state(**routed))
    assert "web search is disabled" in out["answer"]


def test_strip_fences_removes_indentation_shared_by_all_statements():
    code = "# convert\n df['x'] = 1\n\n# total\n result = df['x'].sum()"
    assert strip_fences(code) == "# convert\ndf['x'] = 1\n\n# total\nresult = df['x'].sum()"
    loop = "for c in df.columns:\n    print(c)\nresult = 1"
    assert strip_fences(loop) == loop  # real blocks keep their indentation


def test_strip_fences_removes_a_stray_space_after_a_comment():
    code = "fig, ax = plt.subplots()\n# convert\n df['x'] = 1\n# plot\n result = df['x']\nfor c in df.columns:\n     ax.plot(df[c])"
    assert strip_fences(code) == "fig, ax = plt.subplots()\n# convert\ndf['x'] = 1\n# plot\nresult = df['x']\nfor c in df.columns:\n    ax.plot(df[c])"
