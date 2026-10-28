"""Graph assembly (StateGraph) and a safe runner."""
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

from analyst import config
from analyst.routing import route_after_exec, route_after_question, route_after_validation
from analyst.state import State


def build_graph(nodes: dict):
    """Compile the graph. `nodes` maps each of the 11 node names to a function."""
    g = StateGraph(State)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.set_entry_point("schema_profiler")
    g.add_edge("schema_profiler", "question_router")

    # C1: route by question type
    g.add_conditional_edges(
        "question_router", route_after_question,
        {
            "numeric": "pandas_agent", "chart": "chart_agent", "concept": "dictionary_rag",
            "external": "web_search", "out_of_scope": "fallback_response",
        },
    )
    g.add_edge("pandas_agent", "code_executor")
    g.add_edge("chart_agent", "code_executor")

    # C2: did the code run? (self-repair loop)
    g.add_conditional_edges(
        "code_executor", route_after_exec,
        {"ok": "answer_composer", "repair_pandas": "pandas_agent", "repair_chart": "chart_agent", "give_up": "fallback_response"},
    )
    g.add_edge("dictionary_rag", "answer_composer")
    g.add_edge("web_search", "answer_composer")
    g.add_edge("answer_composer", "answer_validator")

    # C3: is the answer supported by the result / evidence?
    g.add_conditional_edges(
        "answer_validator", route_after_validation,
        {
            "report": "report_builder", "redraft": "answer_composer", "recode_pandas": "pandas_agent",
            "recode_chart": "chart_agent", "give_up": "fallback_response",
        },
    )
    g.add_edge("report_builder", END)
    g.add_edge("fallback_response", END)
    return g.compile()


def run_question(graph, question: str) -> dict:
    """Run the graph on one question. Never raises: loops are bounded and errors are reported."""
    initial = {"question": question, "attempts": 0, "regen_count": 0, "validation_passed": None, "evidence": [], "trace": []}
    try:
        return graph.invoke(initial, config={"recursion_limit": config.RECURSION_LIMIT})
    except GraphRecursionError:
        return {**initial, "answer": "The analysis stopped because it exceeded the step limit.", "error": "recursion limit reached"}
    except Exception as exc:
        return {**initial, "answer": f"Unexpected error: {exc}", "error": str(exc)}
