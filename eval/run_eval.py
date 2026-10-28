"""Run the experiments over eval/questions.csv (needs OPENAI_API_KEY and TAVILY_API_KEY).

Usage (from the project root):
    python eval/build_questions.py                          # once: computes the ground truth
    python eval/run_eval.py --configs b0,b1,full,a1,a2,a3
    python eval/run_eval.py --configs full --limit 5        # quick trial

Configurations:
    b0   one LLM call seeing only schema + first rows   a1  full graph, no repair loop (1 attempt)
    b1   one-shot code, no router/repair/validator      a2  full graph, no answer validator
    full complete CSV Analyst Agent                     a3  full graph, no router (everything goes to pandas_agent)
Outputs: eval/results/<config>.csv (one row per question) and eval/results/summary.csv / summary.md
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyst import config  # noqa: E402
from analyst.baselines import run_b0, run_b1  # noqa: E402
from analyst.builder import run_question  # noqa: E402
from analyst.checks import unsupported_numbers  # noqa: E402
from analyst.config import get_secret  # noqa: E402
from analyst.dictionary import load_dictionary_json  # noqa: E402
from analyst.factory import create_graph, prepare_dataset  # noqa: E402
from analyst.llm import make_llm  # noqa: E402
from analyst.nodes import evidence_text  # noqa: E402
from analyst.profiling import load_dataframe, profile_dataframe  # noqa: E402
from eval.grading import grade  # noqa: E402
from eval.metrics import compute_metrics  # noqa: E402

GRAPH_CONFIGS = {
    "full": {},
    "a1": {"max_attempts": 1},
    "a2": {"skip_validator": True},
    "a3": {"skip_router": True},
}
RESULTS = ROOT / "eval" / "results"


def evidence_source(state: dict) -> str:
    """The same source text the answer validator checks against (with the [n] ids that citations use)."""
    if state.get("route") in ("concept", "external"):
        return evidence_text(state.get("evidence", [])) if state.get("evidence") else ""
    return state.get("exec_result", "")


def evaluate(cfg, questions, df, ctx, openai_key, tavily_key) -> pd.DataFrame:
    llm = make_llm(openai_key)
    graph = create_graph(openai_key, tavily_key, ctx, **GRAPH_CONFIGS[cfg]) if cfg in GRAPH_CONFIGS else None
    rows = []
    for _, q in questions.iterrows():
        start = time.time()
        try:
            if cfg == "b0":
                state = run_b0(llm, df, q["question"])
            elif cfg == "b1":
                state = run_b1(llm, df, q["question"])
            else:
                state = run_question(graph, q["question"])
        except Exception as exc:  # one failure must not lose the whole run
            state = {"answer": f"ERROR: {exc}", "route": "none", "trace": [], "error": str(exc)}
        trace = state.get("trace", [])
        fell_back = bool(trace) and trace[-1] == "fallback_response"
        source = evidence_source(state)
        rows.append({
            "config": cfg, "id": q["id"], "group": q["group"], "question": q["question"],
            "expected_route": q["expected_route"], "route_actual": state.get("route", "none"),
            "correct": grade(q, state), "answer": state.get("answer", ""),
            "is_graph": cfg in GRAPH_CONFIGS, "fallback": fell_back,
            "exec_ok": bool(state.get("exec_ok")), "attempts": state.get("attempts", 0),
            "first_try_ok": bool(state.get("exec_ok")) and state.get("attempts", 0) == 0,
            "hallucinated": bool(unsupported_numbers(state.get("answer", ""), source, q["question"])) if source else False,
            "latency_s": round(time.time() - start, 2), "n_steps": len(trace), "trace": " > ".join(trace),
            "error": state.get("error", ""),
        })
        print(f"[{cfg}] {q['id']:>2} {q['group']:<9} correct={rows[-1]['correct']}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="b0,b1,full,a1,a2,a3")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N questions")
    parser.add_argument("--csv", default=str(config.DEMO_CSV), help="dataset to evaluate on")
    parser.add_argument("--questions", default=str(ROOT / "eval" / "questions.csv"))
    parser.add_argument("--results", default=str(RESULTS), help="output folder")
    parser.add_argument("--dictionary", default="",
                        help="dictionary JSON for --csv; drafted by the LLM and saved here if it does not exist yet")
    args = parser.parse_args()

    openai_key, tavily_key = get_secret("OPENAI_API_KEY"), get_secret("TAVILY_API_KEY")
    if not openai_key or not tavily_key:
        raise SystemExit("Set OPENAI_API_KEY and TAVILY_API_KEY in .env first.")
    qfile = Path(args.questions)
    if not qfile.exists():
        raise SystemExit(f"{qfile} not found. Run the matching eval/build_questions*.py first.")
    questions = pd.read_csv(qfile)
    if args.limit:
        questions = questions.head(args.limit)

    df = load_dataframe(args.csv)
    dict_path = Path(args.dictionary) if args.dictionary else None
    if dict_path and dict_path.exists():
        dictionary = load_dictionary_json(dict_path, profile_dataframe(df))
    elif Path(args.csv).resolve() == config.DEMO_CSV.resolve():
        dictionary = load_dictionary_json(config.DEMO_DICTIONARY, profile_dataframe(df))
    else:
        dictionary = None  # drafted by prepare_dataset
    ctx = prepare_dataset(openai_key, df, dictionary)
    if dict_path and not dict_path.exists():  # every configuration then shares the same draft
        dict_path.parent.mkdir(parents=True, exist_ok=True)
        dict_path.write_text(json.dumps(ctx.dictionary, indent=1, ensure_ascii=False), encoding="utf-8")
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)

    summary = []
    for cfg in [c.strip() for c in args.configs.split(",") if c.strip()]:
        res = evaluate(cfg, questions, df, ctx, openai_key, tavily_key)
        res.to_csv(results / f"{cfg}.csv", index=False)
        summary.append({"config": cfg, **compute_metrics(res)})

    table = pd.DataFrame(summary)
    table.to_csv(results / "summary.csv", index=False)
    header = "| " + " | ".join(table.columns) + " |\n|" + "---|" * len(table.columns) + "\n"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False))
    (results / "summary.md").write_text(header + body + "\n", encoding="utf-8")
    print("\n", table.to_string(index=False))


if __name__ == "__main__":
    main()
