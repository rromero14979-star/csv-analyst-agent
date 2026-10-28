"""Shared State of the graph: every agent reads and writes fields of this dictionary."""
import operator
from typing import Annotated, TypedDict


class State(TypedDict, total=False):
    question: str
    df_profile: str            # schema + aggregate statistics (never raw rows)
    data_dictionary: list      # [{'column','dtype','description'}]
    max_attempts: int          # copied from the dependencies so routing functions can read it
    route: str                 # numeric | chart | concept | external | out_of_scope
    route_reason: str
    code: str                  # code written by pandas_agent / chart_agent
    writer: str                # which agent wrote the code
    attempts: int              # failed executions of the current code
    exec_ok: bool
    exec_error: str
    exec_result: str           # text rendering of the executed result
    result_table: object       # pandas DataFrame for the UI (or None)
    figure: object             # PNG bytes (or None)
    evidence: list             # dictionary entries or web results
    answer: str
    validation_passed: object  # None | True | False
    validation_feedback: str
    issue: str                 # none | answer | code
    regen_count: int
    final_report: str
    error: str
    trace: Annotated[list, operator.add]   # every node appends its own name
