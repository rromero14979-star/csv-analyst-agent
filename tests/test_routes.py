"""P0-P14: every node and every branch of the 3 conditional edges is executed (no API, no sandbox)."""
from analyst.builder import build_graph, run_question
from tests.fakes import fake_nodes


def run(**kw):
    return run_question(build_graph(fake_nodes(**kw)), "any question")


def test_P0_eleven_nodes_excluding_start_end():
    names = set(build_graph(fake_nodes()).get_graph().nodes) - {"__start__", "__end__"}
    assert len(names) == 11


# ---- C1: question_router (5 destinations)
def test_P1_c1_numeric():
    assert "pandas_agent" in run(route="numeric")["trace"]


def test_P2_c1_chart():
    assert "chart_agent" in run(route="chart")["trace"]


def test_P3_c1_concept():
    t = run(route="concept")["trace"]
    assert "dictionary_rag" in t and "code_executor" not in t


def test_P4_c1_external():
    t = run(route="external")["trace"]
    assert "web_search" in t and "code_executor" not in t


def test_P5_c1_out_of_scope():
    assert run(route="out_of_scope")["trace"] == ["schema_profiler", "question_router", "fallback_response"]


# ---- C2: code_executor (self-repair loop)
def test_P6_c2_ok_first_time():
    t = run(execs=(True,))["trace"]
    assert t.count("code_executor") == 1 and "answer_composer" in t


def test_P7_c2_repair_pandas_then_ok():
    s = run(route="numeric", execs=(False, True))
    assert s["attempts"] == 1 and s["trace"].count("pandas_agent") == 2 and "answer_composer" in s["trace"]


def test_P8_c2_repair_chart_then_ok():
    s = run(route="chart", execs=(False, True))
    assert s["trace"].count("chart_agent") == 2 and "pandas_agent" not in s["trace"]


def test_P9_c2_give_up_after_max_attempts():
    s = run(execs=(False,))
    assert s["attempts"] == 3 and s["trace"].count("code_executor") == 3 and s["trace"][-1] == "fallback_response"


# ---- C3: answer_validator (5 destinations)
def test_P10_c3_report():
    assert run(validations=((True, "none"),))["trace"][-1] == "report_builder"


def test_P11_c3_redraft_then_pass():
    s = run(validations=((False, "answer"), (True, "none")))
    assert s["regen_count"] == 1 and s["trace"].count("answer_composer") == 2 and s["trace"][-1] == "report_builder"


def test_P12_c3_recode_pandas_then_pass():
    s = run(route="numeric", validations=((False, "code"), (True, "none")))
    assert s["regen_count"] == 1 and s["trace"].count("pandas_agent") == 2 and s["trace"].count("code_executor") == 2


def test_P13_c3_recode_chart_then_pass():
    s = run(route="chart", validations=((False, "code"), (True, "none")))
    assert s["trace"].count("chart_agent") == 2


def test_P14_c3_give_up_after_regeneration():
    s = run(validations=((False, "answer"),))
    assert s["regen_count"] == 1 and s["trace"][-1] == "fallback_response"


def test_evidence_route_with_code_issue_is_redrafted_not_recoded():
    s = run(route="concept", validations=((False, "code"), (True, "none")))
    assert "pandas_agent" not in s["trace"] and s["trace"].count("answer_composer") == 2
