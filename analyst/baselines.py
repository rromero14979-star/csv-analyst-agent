"""Baselines for the experiments (no graph).

B0: one LLM call that sees only the schema and the first rows (no code execution).
B1: one-shot code generation + one execution + one answer (no router, no repair, no validation).
Both return a dict shaped like the graph State so the evaluation code is shared.
"""
from analyst.nodes import strip_fences
from analyst.profiling import profile_dataframe, profile_to_text
from analyst.prompts import B0_SYSTEM, B1_SYSTEM, PANDAS_SYSTEM
from analyst.sandbox import run_code
from analyst.schemas import CodeOutput


def run_b0(llm, df, question: str) -> dict:
    material = f"{profile_to_text(profile_dataframe(df))}\n\nFirst rows:\n{df.head(20).to_string()}"
    reply = llm.invoke([("system", B0_SYSTEM), ("human", f"{material}\n\nQuestion: {question}")])
    return {"answer": (reply.content or "").strip(), "route": "none", "trace": ["llm_only"], "exec_ok": None, "attempts": 0}


def run_b1(llm, df, question: str, runner=run_code) -> dict:
    profile = profile_to_text(profile_dataframe(df))
    out = llm.with_structured_output(CodeOutput).invoke(
        [("system", PANDAS_SYSTEM), ("human", f"Dataset profile:\n{profile}\n\nQuestion: {question}")]
    )
    code = strip_fences(out.code)
    res = runner(code, df, "result")
    if not res.get("ok"):
        return {"answer": f"Execution failed: {res.get('error', '')}", "route": "none", "trace": ["codegen", "execute"],
                "exec_ok": False, "attempts": 1, "exec_result": ""}
    reply = llm.invoke([("system", B1_SYSTEM), ("human", f"Question: {question}\n\nResult:\n{res.get('text', '')}")])
    return {"answer": (reply.content or "").strip(), "route": "none", "trace": ["codegen", "execute", "answer"],
            "exec_ok": True, "attempts": 0, "exec_result": res.get("text", "")}
