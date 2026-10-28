"""Dry run of eval/run_eval.py with a fake LLM: the experiment pipeline must work before spending API credit."""
import pandas as pd

from analyst import config
from analyst.builder import build_graph
from analyst.dictionary import load_dictionary_json
from analyst.nodes import Deps, make_nodes
from analyst.profiling import load_dataframe, profile_dataframe
from analyst.schemas import CodeOutput, RouteDecision, ValidationResult
from eval import run_eval
from tests.fakes import FakeLLM

DF = load_dataframe(config.DEMO_CSV)
DICT = load_dictionary_json(config.DEMO_DICTIONARY, profile_dataframe(DF))
TOTAL = round(float(DF["Revenue"].sum()), 2)


def fake_llm():
    return FakeLLM(
        {RouteDecision: RouteDecision(route="numeric", reason="r"),
         CodeOutput: CodeOutput(code="result = df['Revenue'].sum().round(2)"),
         ValidationResult: ValidationResult(passed=True, issue="none", feedback="")},
        texts=[f"Total revenue is {TOTAL}."],
    )


def questions():
    q = pd.read_csv(config.BASE_DIR / "eval" / "questions.csv")
    return q[q["id"].isin([1, 2])]


def test_evaluate_graph_config(monkeypatch):
    llm = fake_llm()
    monkeypatch.setattr(run_eval, "make_llm", lambda key: llm)
    monkeypatch.setattr(run_eval, "create_graph", lambda *a, **k: build_graph(make_nodes(Deps(llm=llm, df=DF, dictionary=DICT))))
    res = run_eval.evaluate("full", questions(), DF, None, "k", "t")
    assert list(res["correct"]) == [True, False]          # Q1 correct; Q2 (text answer expected) is not
    assert res["is_graph"].all() and res["first_try_ok"].all() and not res["hallucinated"].any()


def test_evaluate_baselines_and_metrics(monkeypatch):
    llm = fake_llm()
    monkeypatch.setattr(run_eval, "make_llm", lambda key: llm)
    for cfg in ("b0", "b1"):
        res = run_eval.evaluate(cfg, questions(), DF, None, "k", "t")
        assert len(res) == 2 and not res["is_graph"].any() and res["correct"].iloc[0]
        m = run_eval.compute_metrics(res)
        assert m["accuracy"] == 0.5 and m["router_accuracy"] is None


def test_citation_markers_are_not_counted_as_hallucinated_numbers():
    state = {"route": "external", "evidence": [{"id": 1, "source": "site", "text": "Margins are 40 percent."}]}
    assert run_eval.unsupported_numbers("Margins are about 40% [1].", run_eval.evidence_source(state)) == []
