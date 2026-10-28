"""Real nodes + real graph + REAL sandbox + scripted fake LLM: full end-to-end paths without API calls."""
import pytest

from analyst import config
from analyst.builder import build_graph, run_question
from analyst.dictionary import load_dictionary_json
from analyst.nodes import Deps, make_nodes
from analyst.profiling import load_dataframe, profile_dataframe
from analyst.schemas import CodeOutput, RouteDecision, ValidationResult
from tests.fakes import FakeLLM

DF = load_dataframe(config.DEMO_CSV)
DICT = load_dictionary_json(config.DEMO_DICTIONARY, profile_dataframe(DF))
TOTAL = round(float(DF["Revenue"].sum()), 2)
PASS = ValidationResult(passed=True, issue="none", feedback="")


def run(route, codes=("result = df['Revenue'].sum().round(2)",), texts=(f"Total revenue is {TOTAL}.",), judge=PASS, **kw):
    llm = FakeLLM(
        {RouteDecision: RouteDecision(route=route, reason="r"), CodeOutput: [CodeOutput(code=c) for c in codes],
         ValidationResult: judge},
        texts=list(texts),
    )
    graph = build_graph(make_nodes(Deps(llm=llm, df=DF, dictionary=DICT, **kw)))
    return run_question(graph, "What is the total revenue?"), llm


def test_numeric_happy_path_with_real_sandbox():
    s, _ = run("numeric")
    assert s["trace"] == ["schema_profiler", "question_router", "pandas_agent", "code_executor",
                          "answer_composer", "answer_validator", "report_builder"]
    assert str(TOTAL) in s["answer"] and "result = df" in s["final_report"]


def test_repair_loop_fixes_a_wrong_column_name():
    s, llm = run("numeric", codes=("result = df['Sale Revenue'].sum()", "result = df['Revenue'].sum().round(2)"))
    assert s["attempts"] == 1 and s["trace"].count("code_executor") == 2 and s["trace"][-1] == "report_builder"
    assert "KeyError" in llm.calls[-1][1][1][1] or any("KeyError" in str(c[1]) for c in llm.calls)


def test_sandbox_policy_violation_triggers_repair():
    s, _ = run("numeric", codes=("import os\nresult = 1", "result = df['Revenue'].sum().round(2)"))
    assert s["attempts"] == 1 and s["trace"][-1] == "report_builder"


def test_gives_up_after_three_failed_executions_and_lists_columns():
    s, _ = run("numeric", codes=("result = df['Profit'].sum()",))
    assert s["attempts"] == 3 and s["trace"][-1] == "fallback_response"
    assert "'Revenue'" in s["answer"] and "Profit" in s["answer"]


def test_chart_path_returns_png_and_data():
    code = ("g = df.groupby('Region')['Revenue'].sum()\nfig, ax = plt.subplots()\n"
            "ax.bar(g.index, g.values)\nax.set_title('Revenue')\nresult = g")
    s, _ = run("chart", codes=(code,), texts=("Revenue by region is shown.",))
    assert s["figure"][:4] == b"\x89PNG" and len(s["result_table"]) == 5 and s["trace"][-1] == "report_builder"


def test_concept_path_uses_dictionary():
    s, _ = run("concept", texts=("Discount is a fraction between 0 and 1.",))
    assert s["trace"] == ["schema_profiler", "question_router", "dictionary_rag", "answer_composer",
                          "answer_validator", "report_builder"]
    assert s["evidence"][0]["kind"] == "dictionary"


def test_external_path_with_web_evidence():
    web = lambda q: [{"text": "Typical gross margin is about 40 percent.", "source": "site", "locator": "http://x", "kind": "web"}]
    s, _ = run("external", texts=("Typical gross margin is about 40 percent [1].",), web_search=web)
    assert s["trace"][2] == "web_search" and s["trace"][-1] == "report_builder"


def test_out_of_scope_goes_straight_to_fallback():
    s, _ = run("out_of_scope")
    assert s["trace"][-1] == "fallback_response" and "only help with this table" in s["answer"]


def test_validator_rejects_hallucinated_number_then_redraft_passes():
    s, _ = run("numeric", texts=("Total revenue is 999999.", f"Total revenue is {TOTAL}."))
    assert s["regen_count"] == 1 and s["trace"].count("answer_composer") == 2
    assert s["trace"][-1] == "report_builder" and "999999" not in s["answer"]


def test_judge_says_code_is_wrong_then_recode_passes():
    llm_judge = [ValidationResult(passed=False, issue="code", feedback="Use the year 2024 only."), PASS]
    s, _ = run("numeric", judge=llm_judge, codes=("result = df['Revenue'].sum().round(2)",))
    assert s["regen_count"] == 1 and s["trace"].count("pandas_agent") == 2 and s["trace"][-1] == "report_builder"


def test_ablation_no_repair_gives_up_immediately():
    s, _ = run("numeric", codes=("result = df['Profit'].sum()",), max_attempts=1)
    assert s["trace"].count("code_executor") == 1 and s["trace"][-1] == "fallback_response"


def test_ablation_no_validator_lets_hallucination_through():
    s, _ = run("numeric", texts=("Total revenue is 999999.",), skip_validator=True)
    assert "999999" in s["answer"] and s["trace"][-1] == "report_builder"
