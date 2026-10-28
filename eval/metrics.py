"""Metrics for the experiments."""
import pandas as pd


def compute_metrics(df: pd.DataFrame) -> dict:
    graph = df[df["is_graph"]]
    code_rows = df[df["route_actual"].isin(["numeric", "chart"]) | (df["config"] == "b1")]
    answered = graph[~graph["fallback"]]
    out = {
        "n": len(df),
        "accuracy": round(float(df["correct"].mean()), 3),
        "router_accuracy": round(float((graph["expected_route"] == graph["route_actual"]).mean()), 3) if len(graph) else None,
        "exec_first_try": round(float(code_rows["first_try_ok"].mean()), 3) if len(code_rows) else None,
        "exec_final": round(float(code_rows["exec_ok"].mean()), 3) if len(code_rows) else None,
        "hallucinated_number_rate": round(float(answered["hallucinated"].mean()), 3) if len(answered) else None,
        "fallback_rate": round(float(graph["fallback"].mean()), 3) if len(graph) else None,
        "avg_latency_s": round(float(df["latency_s"].mean()), 2),
    }
    for group, sub in df.groupby("group"):
        out[f"acc_{group}"] = round(float(sub["correct"].mean()), 3)
    return out
