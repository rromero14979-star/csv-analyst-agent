"""Sandbox policy (static) and real execution in the worker process."""
import pandas as pd
import pytest

from analyst.sandbox import CodeRejected, run_code, validate_code

DF = pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 2.0, 10.0]})


@pytest.mark.parametrize("code", [
    "import os\nresult = 1",
    "from os import path\nresult = 1",
    "result = open('/etc/passwd').read()",
    "result = __import__('os').listdir('.')",
    "result = df.__class__",
    "result = ().__class__.__mro__",
    "result = getattr(df, 'shape')",
    "result = pd.read_csv('x.csv')",
    "df.to_csv('out.csv')\nresult = 1",
    "result = eval('1+1')",
    "result = '{0.__class__}'.format(df)",
    "result = pd.io.common.os.listdir('.')",
    "class A: pass\nresult = 1",
    "result = (",
    "",
])
def test_validate_code_rejects(code):
    with pytest.raises(CodeRejected):
        validate_code(code)


def test_validate_code_accepts_normal_pandas():
    validate_code("result = df.groupby('g')['v'].sum().sort_values(ascending=False)")


def test_run_scalar():
    res = run_code("result = df['v'].sum()", DF)
    assert res["ok"] and res["text"].strip() == "13.0"


def test_run_dataframe_returns_text_and_table():
    res = run_code("result = df.groupby('g')['v'].sum()", DF)
    assert res["ok"] and "10.0000" in res["text"] and res["table_json"]


def test_run_chart_returns_png():
    code = "fig, ax = plt.subplots()\nax.bar(df['g'], df['v'])\nresult = df"
    res = run_code(code, DF, kind="chart")
    assert res["ok"] and res["figure_png"][:8] == b"\x89PNG\r\n\x1a\n"


def test_chart_without_figure_is_an_error():
    res = run_code("result = 1", DF, kind="chart")
    assert not res["ok"] and "figure" in res["error"].lower()


def test_missing_result_variable_is_an_error():
    res = run_code("x = 1", DF)
    assert not res["ok"] and "result" in res["error"]


def test_runtime_error_message_is_returned_for_repair():
    res = run_code("result = df['Sale Date']", DF)
    assert not res["ok"] and "KeyError" in res["error"] and "Sale Date" in res["error"]


def test_policy_violation_is_reported_without_running():
    res = run_code("import os\nresult = 1", DF)
    assert not res["ok"] and res["error"].startswith("CodeRejected")


def test_infinite_loop_times_out():
    res = run_code("while True:\n    pass\nresult = 1", DF, timeout=3)
    assert not res["ok"] and "Timeout" in res["error"]


def test_print_inside_pandas_does_not_corrupt_output():
    res = run_code("df.info()\nresult = len(df)", DF)
    assert res["ok"] and res["text"].strip() == "3"


def test_original_dataframe_is_not_modified():
    run_code("df['v'] = 0\nresult = 1", DF)
    assert DF["v"].sum() == 13.0


def test_series_named_like_its_index_is_serialized():
    res = run_code("result = df.groupby('g')['g'].count()", DF)
    assert res["ok"] and res["table_json"]
