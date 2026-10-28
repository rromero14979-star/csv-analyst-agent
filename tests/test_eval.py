"""The ground truth and the grader must be trustworthy before any experiment is run."""
import pandas as pd

from analyst import config
from analyst.profiling import load_dataframe
from eval.build_questions import build
from eval.grading import grade
from eval.metrics import compute_metrics

Q = build()


def row(qid):
    return Q[Q["id"] == qid].iloc[0]


def test_thirty_questions_in_six_groups():
    assert len(Q) == 30
    assert Q["group"].value_counts().to_dict() == {"numeric": 12, "tricky": 3, "chart": 5, "concept": 4, "external": 3, "decline": 3}


def test_text_ground_truth_has_no_ties_and_is_a_real_value():
    assert row(2)["expected_text"] in {"North", "South", "East", "West", "Central"}
    assert row(8)["expected_text"] in ["January", "February", "March", "April", "May", "June", "July", "August",
                                       "September", "October", "November", "December"]


def test_ground_truth_matches_independent_computation():
    df = load_dataframe(config.DEMO_CSV)
    assert float(row(1)["expected_value"]) == round(df["Revenue"].sum(), 2)
    assert float(row(13)["expected_value"]) == round(df["Discount"].mean(), 4)


def test_grader_number_text_chart_keywords_web_decline():
    total = float(row(1)["expected_value"])
    assert grade(row(1), {"answer": f"Total revenue is {total:,.2f}.", "trace": ["report_builder"]})
    assert not grade(row(1), {"answer": "Total revenue is 5.", "trace": ["report_builder"]})
    assert grade(row(13), {"answer": f"The average discount is {float(row(13)['expected_value']) * 100:.1f}%.", "trace": ["x"]})
    assert grade(row(2), {"answer": f"{row(2)['expected_text']} led.", "trace": ["x"]})
    assert grade(row(16), {"figure": b"png", "code": "ax.bar(df['Region'], df['Revenue'])", "trace": ["x"]})
    assert not grade(row(16), {"figure": None, "code": "", "trace": ["x"]})
    assert grade(row(21), {"answer": "It is a fraction between 0 and 1.", "trace": ["x"]})
    assert grade(row(25), {"route": "external", "evidence": [{"text": "t"}], "answer": "About 40% [1].", "trace": ["report_builder"]})
    assert grade(row(28), {"answer": "I can only help with this table.", "trace": ["fallback_response"]})
    assert not grade(row(28), {"answer": "Roses are red", "trace": ["report_builder"]})


def test_fallback_is_never_correct_for_answerable_questions():
    assert not grade(row(1), {"answer": f"{row(1)['expected_value']}", "trace": ["fallback_response"]})


def test_compute_metrics_shape():
    df = pd.DataFrame({
        "config": ["full"] * 2, "group": ["numeric", "chart"], "correct": [True, False], "is_graph": [True, True],
        "expected_route": ["numeric", "chart"], "route_actual": ["numeric", "numeric"], "first_try_ok": [True, False],
        "exec_ok": [True, True], "hallucinated": [False, True], "fallback": [False, False], "latency_s": [1.0, 3.0],
    })
    m = compute_metrics(df)
    assert m["accuracy"] == 0.5 and m["router_accuracy"] == 0.5 and m["avg_latency_s"] == 2.0
    assert m["exec_first_try"] == 0.5 and m["hallucinated_number_rate"] == 0.5 and m["acc_chart"] == 0.0
