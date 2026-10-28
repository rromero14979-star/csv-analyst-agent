"""Routing functions of the 3 conditional edges."""
from analyst import config


def route_after_question(state) -> str:
    """C1: numeric | chart | concept | external | out_of_scope."""
    return state.get("route", "out_of_scope")


def route_after_exec(state) -> str:
    """C2: ok | repair_pandas | repair_chart | give_up."""
    if state.get("exec_ok"):
        return "ok"
    if state.get("attempts", 0) < state.get("max_attempts", config.MAX_ATTEMPTS):
        return "repair_chart" if state.get("route") == "chart" else "repair_pandas"
    return "give_up"


def route_after_validation(state) -> str:
    """C3: report | redraft | recode_pandas | recode_chart | give_up."""
    if state.get("validation_passed"):
        return "report"
    if state.get("regen_count", 0) >= config.MAX_REGEN:
        return "give_up"
    if state.get("issue") == "code" and state.get("route") in ("numeric", "chart"):
        return "recode_chart" if state["route"] == "chart" else "recode_pandas"
    return "redraft"
