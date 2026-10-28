"""Restricted execution of LLM-generated pandas / matplotlib code.

Defense in depth (this REDUCES risk; it is not a perfect security boundary):
  1. AST whitelist: no imports, no dunder/private attributes, no file/OS/eval-style names.
  2. Restricted builtins inside the worker (no open, __import__, getattr, ...).
  3. A separate Python process with a scrubbed environment (API keys are NOT passed),
     a temporary working directory, a memory limit and a wall-clock timeout.
  4. The worker answers in JSON (never pickle) so nothing is deserialized unsafely by the parent.
"""
import ast
import base64
import json
import os
import pickle
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path

import pandas as pd

from analyst import config

ROOT = Path(__file__).resolve().parent.parent
MARKER = b"\n<<ANALYST_RESULT>>"

FORBIDDEN_NAMES = {
    "__import__", "open", "exec", "eval", "compile", "input", "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "hasattr", "breakpoint", "exit", "quit", "help", "memoryview",
    "os", "sys", "subprocess", "builtins", "importlib", "shutil", "pathlib", "socket", "pickle", "ctypes",
}
FORBIDDEN_ATTRS = {
    "system", "popen", "spawn", "fork", "exec", "eval", "environ", "getenv", "listdir", "walk", "remove",
    "unlink", "rmdir", "mkdir", "makedirs", "rename", "load", "loads", "dump", "dumps", "fromfile", "tofile",
    "loadtxt", "genfromtxt", "savetxt", "savez", "save", "savefig", "to_csv", "to_excel", "to_json",
    "to_pickle", "to_parquet", "to_sql", "to_hdf", "to_feather", "to_clipboard", "to_stata", "to_html",
    "to_latex", "to_markdown", "to_xml", "os", "sys", "io", "subprocess", "compat", "testing", "core",
    "util", "ctypes", "builtins", "importlib", "shutil", "git", "show",
}

SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "callable", "dict", "divmod", "enumerate", "filter", "float", "frozenset",
    "int", "isinstance", "iter", "len", "list", "map", "max", "min", "next", "pow", "range", "reversed",
    "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
]


class CodeRejected(ValueError):
    """The code violates the sandbox policy."""


def validate_code(code: str) -> None:
    """Static check. Raises CodeRejected with a clear reason."""
    if not code or not code.strip():
        raise CodeRejected("No code was generated.")
    if len(code) > config.MAX_CODE_CHARS:
        raise CodeRejected("The code is too long.")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodeRejected(f"Syntax error: {exc.msg} (line {exc.lineno})") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise CodeRejected("Imports are not allowed: pd, np and plt are already available.")
        if isinstance(node, (ast.ClassDef, ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
            raise CodeRejected(f"{type(node).__name__} is not allowed.")
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            raise CodeRejected(f"Use of '{node.id}' is not allowed.")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRS or node.attr.startswith("read_"):
                raise CodeRejected(f"Access to attribute '{node.attr}' is not allowed.")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise CodeRejected("Strings containing '__' are not allowed.")


def run_code(code: str, df: pd.DataFrame, kind: str = "result", timeout: float = config.EXEC_TIMEOUT_S) -> dict:
    """Execute code in a worker process.

    kind='result': the code must assign the final value to `result`.
    kind='chart' : the code must assign a matplotlib Figure to `fig` (and may also set `result`).
    Returns {'ok','error','text','table_json','figure_png'}.
    """
    try:
        validate_code(code)
    except CodeRejected as exc:
        return {"ok": False, "error": f"CodeRejected: {exc}"}

    payload = pickle.dumps({"code": code, "df": df, "kind": kind})
    with tempfile.TemporaryDirectory(prefix="analyst_", ignore_cleanup_errors=True) as tmp:
        env = {
            "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg", "MPLCONFIGDIR": tmp, "HOME": tmp, "TMPDIR": tmp,
            "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONIOENCODING": "utf-8",
            "ANALYST_SANDBOX": "1",  # tells analyst.config not to load .env inside the worker
        }
        for key in ("SYSTEMROOT", "TEMP", "TMP"):  # needed by Python on Windows
            if key in os.environ:
                env[key] = os.environ[key]
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "analyst.sandbox_worker"],
                input=payload, capture_output=True, timeout=timeout, env=env, cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"Timeout: the code ran longer than {timeout:.0f} seconds."}
        except Exception as exc:  # could not start the worker
            return {"ok": False, "error": f"Execution failed to start: {exc}"}

    if MARKER not in proc.stdout:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["no output"]
        return {"ok": False, "error": f"Execution failed (exit code {proc.returncode}): {tail[0][:300]}"}
    try:
        out = json.loads(proc.stdout.rsplit(MARKER, 1)[1].decode("utf-8"))
    except Exception:
        return {"ok": False, "error": "The execution result could not be read."}
    if out.get("figure_png"):
        out["figure_png"] = base64.b64decode(out["figure_png"])
    return out


def table_from_json(table_json):
    """Rebuild the DataFrame for the UI from the worker's JSON (or None)."""
    if not table_json:
        return None
    try:
        return pd.read_json(StringIO(table_json), orient="split")
    except Exception:
        return None
