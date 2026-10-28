"""The 11 nodes (agents) of the graph.

Every node receives the shared State and returns ONLY the fields it modifies, plus its own name in 'trace'.
Dependencies (LLM, DataFrame, dictionary search, web search, code runner) are injected through `Deps`,
which makes the nodes easy to test with fakes.
"""
import ast
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pandas as pd

from analyst import config
from analyst.checks import unsupported_numbers
from analyst.dictionary import draft_dictionary
from analyst.profiling import profile_dataframe, profile_to_text
from analyst.prompts import (
    CHART_SYSTEM, COMPOSER_SYSTEM, JUDGE_SYSTEM, PANDAS_SYSTEM, ROUTER_SYSTEM,
)
from analyst.sandbox import run_code, table_from_json
from analyst.schemas import CodeOutput, RouteDecision, ValidationResult

EVIDENCE_ROUTES = ("concept", "external")
WEB_DISABLED_REASON = "Web search is disabled, so outside information is unavailable."


@dataclass
class Deps:
    llm: Any                                         # chat model (ChatOpenAI or a fake in tests)
    df: pd.DataFrame                                 # the loaded table (never sent to the LLM)
    dictionary: Optional[list] = None                # pre-built data dictionary
    dictionary_search: Optional[Callable] = None     # question -> dictionary entries
    web_search: Optional[Callable] = None            # query -> web results
    runner: Callable = run_code                      # code runner (sandbox); replaced by fakes in tests
    web_enabled: bool = True
    # Ablation switches (used only by eval/run_eval.py)
    max_attempts: int = config.MAX_ATTEMPTS
    skip_validator: bool = False
    skip_router: bool = False


def strip_fences(code: str) -> str:
    code = (code or "").strip()
    match = re.match(r"^```(?:python)?\s*\n(.*?)\n?```\s*$", code, re.S)
    return _dedent_code(match.group(1).strip() if match else code)


def _dedent_code(code: str) -> str:
    """Fix a habit of the model: one stray space after a newline ('# comment\\n df = ...'). Python rejects it as an
    'unexpected indent' and the repair loop repeats the same habit. Only applied when the code does not compile."""
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        pass
    lines = code.splitlines()
    statements = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    pad = min((len(line) - len(line.lstrip()) for line in statements), default=0)
    if pad:  # indentation shared by every statement
        lines = [line[pad:] if not line[:pad].strip() else line.lstrip() for line in lines]
    # single stray spaces: an indentation of 4k + 1 becomes 4k
    fixed = "\n".join(line[1:] if (len(line) - len(line.lstrip())) % 4 == 1 else line for line in lines)
    try:
        ast.parse(fixed)
        return fixed
    except SyntaxError:
        return code  # leave it unchanged: the executor reports the real error for the repair loop


def evidence_text(evidence: list) -> str:
    return "\n".join(f"[{e['id']}] {e['source']}: {e['text']}" for e in evidence) or "(none)"


def make_nodes(deps: Deps) -> dict:
    router_llm = deps.llm.with_structured_output(RouteDecision)
    code_llm = deps.llm.with_structured_output(CodeOutput)
    judge_llm = deps.llm.with_structured_output(ValidationResult)

    # 1 ------------------------------------------------------------------ schema_profiler
    def schema_profiler(state):
        profile = profile_dataframe(deps.df)
        dictionary = deps.dictionary or draft_dictionary(deps.llm, profile)
        return {
            "df_profile": profile_to_text(profile),
            "data_dictionary": dictionary,
            "max_attempts": deps.max_attempts,
            "trace": ["schema_profiler"],
        }

    # 2 ------------------------------------------------------------------ question_router
    def question_router(state):
        if deps.skip_router:
            return {"route": "numeric", "route_reason": "router disabled", "trace": ["question_router"]}
        try:
            out = router_llm.invoke(
                [("system", ROUTER_SYSTEM.format(profile=state["df_profile"])), ("human", state["question"])]
            )
            route, reason = out.route, out.reason
            if route == "external" and not deps.web_enabled:
                route, reason = "out_of_scope", WEB_DISABLED_REASON
            return {"route": route, "route_reason": reason, "trace": ["question_router"]}
        except Exception as exc:
            return {
                "route": "out_of_scope", "route_reason": "The question could not be classified.",
                "error": f"question_router failed: {exc}", "trace": ["question_router"],
            }

    def _write_code(state, system, writer):
        """Shared logic of pandas_agent and chart_agent (first draft, repair or recode)."""
        update = {"writer": writer, "trace": [writer]}
        human = f"Dataset profile:\n{state['df_profile']}\n\nQuestion: {state['question']}"
        if state.get("exec_error"):  # repair after an execution error
            human += (
                f"\n\nYour previous code:\n{state.get('code', '')}\n\nIt failed with:\n{state['exec_error']}\n"
                "Fix the code."
            )
        elif state.get("validation_passed") is False and state.get("issue") == "code":  # recode after validation
            human += (
                f"\n\nYour previous code:\n{state.get('code', '')}\n\nIt ran, but a reviewer said:\n"
                f"{state.get('validation_feedback', '')}\nWrite corrected code."
            )
            update.update(validation_passed=None, regen_count=state.get("regen_count", 0) + 1, attempts=0)
        try:
            out = code_llm.invoke([("system", system), ("human", human)])
            update["code"] = strip_fences(out.code)
        except Exception as exc:
            update.update(code="", error=f"{writer} failed: {exc}")
        return update

    # 3 ------------------------------------------------------------------ pandas_agent
    def pandas_agent(state):
        return _write_code(state, PANDAS_SYSTEM, "pandas_agent")

    # 4 ------------------------------------------------------------------ chart_agent
    def chart_agent(state):
        return _write_code(state, CHART_SYSTEM, "chart_agent")

    # 5 ------------------------------------------------------------------ code_executor
    def code_executor(state):
        kind = "chart" if state.get("route") == "chart" else "result"
        res = deps.runner(state.get("code", ""), deps.df, kind)
        if not res.get("ok"):
            return {
                "exec_ok": False, "exec_error": res.get("error", "Unknown error"),
                "attempts": state.get("attempts", 0) + 1, "trace": ["code_executor"],
            }
        return {
            "exec_ok": True, "exec_error": "", "exec_result": res.get("text", ""),
            "result_table": table_from_json(res.get("table_json")), "figure": res.get("figure_png"),
            "trace": ["code_executor"],
        }

    # 6 ------------------------------------------------------------------ dictionary_rag
    def dictionary_rag(state):
        try:
            if deps.dictionary_search:
                found = deps.dictionary_search(state["question"])
            else:
                found = (state.get("data_dictionary") or [])[:4]
            evidence = [
                {"id": i, "kind": "dictionary", "source": e["column"], "locator": e.get("dtype", ""), "text": e["description"]}
                for i, e in enumerate(found, start=1)
            ]
            return {"evidence": evidence, "trace": ["dictionary_rag"]}
        except Exception as exc:
            return {"evidence": [], "error": f"dictionary_rag failed: {exc}", "trace": ["dictionary_rag"]}

    # 7 ------------------------------------------------------------------ web_search
    def web_search(state):
        if not deps.web_enabled or deps.web_search is None:
            return {"evidence": [], "trace": ["web_search"]}
        try:
            found = deps.web_search(state["question"])
            evidence = [dict(e, id=i) for i, e in enumerate(found, start=1)]
            return {"evidence": evidence, "trace": ["web_search"]}
        except Exception as exc:
            return {"evidence": [], "error": f"web_search failed: {exc}", "trace": ["web_search"]}

    # 8 ------------------------------------------------------------------ answer_composer
    def answer_composer(state):
        update = {"trace": ["answer_composer"]}
        feedback = ""
        if state.get("validation_passed") is False:  # redraft after a failed validation
            feedback = "A reviewer rejected the previous draft. Fix this: " + state.get("validation_feedback", "")
            update.update(validation_passed=None, regen_count=state.get("regen_count", 0) + 1)
        route = state.get("route")
        if route in EVIDENCE_ROUTES:
            material = "Evidence:\n" + evidence_text(state.get("evidence", []))
        elif route == "chart":
            material = (
                "A chart was generated. Underlying data (may be empty):\n"
                f"{state.get('exec_result', '') or '(none)'}\nCode:\n{state.get('code', '')}"
            )
        else:
            material = f"Executed result:\n{state.get('exec_result', '')}\nCode used:\n{state.get('code', '')}"
        try:
            reply = deps.llm.invoke(
                [
                    ("system", COMPOSER_SYSTEM.format(feedback=feedback)),
                    ("human", f"Question: {state['question']}\n\n{material}"),
                ]
            )
            update["answer"] = (reply.content or "").strip()
        except Exception as exc:
            update.update(answer="", error=f"answer_composer failed: {exc}")
        return update

    # 9 ------------------------------------------------------------------ answer_validator
    def answer_validator(state):
        base = {"trace": ["answer_validator"]}
        if deps.skip_validator:
            return dict(base, validation_passed=True, validation_feedback="", issue="none")
        answer, route = state.get("answer", ""), state.get("route")
        if not answer:
            return dict(base, validation_passed=False, issue="answer", validation_feedback="The answer is empty.")

        # (a) deterministic check: every number must come from the result / evidence
        source = state.get("exec_result", "") if route not in EVIDENCE_ROUTES else evidence_text(state.get("evidence", []))
        bad = unsupported_numbers(answer, source, state["question"])
        if bad:
            return dict(
                base, validation_passed=False, issue="answer",
                validation_feedback="These numbers are not in the executed result/evidence: " + ", ".join(bad[:6]),
            )
        # (b) semantic check by an LLM judge
        try:
            material = (
                evidence_text(state.get("evidence", []))
                if route in EVIDENCE_ROUTES
                else f"Code:\n{state.get('code', '')}\n\nExecuted result:\n{state.get('exec_result', '')}"
            )
            out = judge_llm.invoke(
                [
                    ("system", JUDGE_SYSTEM),
                    ("human", f"Question: {state['question']}\n\n{material}\n\nAnswer to validate:\n{answer}"),
                ]
            )
            return dict(base, validation_passed=out.passed, issue=out.issue, validation_feedback=out.feedback)
        except Exception as exc:  # the deterministic check already passed: do not lose a good answer
            return dict(
                base, validation_passed=True, issue="none", validation_feedback="",
                error=f"answer_validator (judge) failed: {exc}",
            )

    # 10 ----------------------------------------------------------------- report_builder
    def report_builder(state):
        parts = [state.get("answer", "")]
        if state.get("code") and state.get("route") in ("numeric", "chart"):
            parts += ["", "```python", state["code"], "```"]
        path = " → ".join(state.get("trace", []) + ["report_builder"])
        parts += ["", f"**Path:** {path}"]
        return {"final_report": "\n".join(parts), "trace": ["report_builder"]}

    # 11 ----------------------------------------------------------------- fallback_response
    def fallback_response(state):
        columns = ", ".join(repr(c) for c in deps.df.columns)
        if state.get("route_reason") == WEB_DISABLED_REASON:
            msg = (
                "This question needs outside information, but web search is disabled. Enable it in the sidebar, "
                "or ask about the table itself."
            )
        elif state.get("route") == "out_of_scope":
            msg = (
                "I can only help with this table: calculations, charts, what the columns mean, or general "
                "context from the web. I cannot modify or export data, or answer unrelated requests."
            )
            if state.get("error"):
                msg += f" ({state['error']})"
        elif state.get("exec_ok") is False:
            msg = (
                f"I could not run a valid analysis after {state.get('attempts', 0)} attempt(s). Last error: "
                f"{state.get('exec_error', 'unknown')}. The question may refer to data that does not exist. "
                f"Available columns: {columns}."
            )
        elif state.get("validation_passed") is False:
            msg = (
                "I produced an answer but it did not pass verification, so I am not showing it. "
                f"Reason: {state.get('validation_feedback', 'n/a')}"
            )
        else:
            msg = "I could not complete this request. " + state.get("error", "")
        path = " → ".join(state.get("trace", []) + ["fallback_response"])
        return {"answer": msg.strip(), "final_report": f"{msg.strip()}\n\n**Path:** {path}", "trace": ["fallback_response"]}

    return {
        "schema_profiler": schema_profiler, "question_router": question_router,
        "pandas_agent": pandas_agent, "chart_agent": chart_agent, "code_executor": code_executor,
        "dictionary_rag": dictionary_rag, "web_search": web_search, "answer_composer": answer_composer,
        "answer_validator": answer_validator, "report_builder": report_builder,
        "fallback_response": fallback_response,
    }
