"""Worker process for sandbox.run_code. Reads a pickled payload from stdin (trusted: it comes from
our own parent process) and writes the result as JSON after a marker. Never import this from the app."""
import base64
import builtins
import io
import json
import pickle
import sys

MARKER = b"\n<<ANALYST_RESULT>>"


def _limit_resources():
    try:
        import resource

        from analyst import config

        mem = config.MEMORY_LIMIT_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        cpu = int(config.EXEC_TIMEOUT_S) + 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    except Exception:
        pass  # not available on this platform (e.g. Windows)


def _serialize_result(result):
    import numpy as np
    import pandas as pd

    from analyst import config

    if result is None:
        raise ValueError("`result` is None: assign the final value to `result`.")
    if isinstance(result, pd.Series):
        result = result.to_frame()
    if isinstance(result, pd.DataFrame):
        if not isinstance(result.index, pd.RangeIndex):
            names = [f"{n} (index)" if n in result.columns else n for n in result.index.names]
            result = result.rename_axis(names).reset_index()  # renamed so it cannot clash with a column
        total = len(result)
        shown = result.head(config.RESULT_MAX_ROWS)
        text = shown.to_string(float_format=lambda x: f"{x:.4f}")
        if total > len(shown):
            text += f"\n... showing the first {len(shown)} of {total} rows"
        return text[: config.RESULT_MAX_CHARS], shown.to_json(orient="split", date_format="iso", default_handler=str)
    if isinstance(result, (np.generic,)):
        result = result.item()
    if isinstance(result, (list, tuple, set, dict)):
        return repr(result)[: config.RESULT_MAX_CHARS], None
    return str(result)[: config.RESULT_MAX_CHARS], None


def execute(code, df, kind):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from analyst.sandbox import SAFE_BUILTIN_NAMES, validate_code

    validate_code(code)  # defense in depth: validate again inside the worker
    safe = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
    ns = {"__builtins__": safe, "pd": pd, "np": np, "plt": plt, "df": df.copy()}
    exec(compile(code, "<analysis>", "exec"), ns)  # noqa: S102 (sandboxed by design)

    out = {"ok": True, "error": "", "text": "", "table_json": None, "figure_png": None}
    if kind == "chart":
        fig = ns.get("fig")
        if fig is None and plt.get_fignums():
            fig = plt.gcf()
        if not isinstance(fig, matplotlib.figure.Figure) or not any(ax.has_data() for ax in fig.axes):
            raise ValueError("No figure with data was produced: assign a matplotlib Figure to `fig`.")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        out["figure_png"] = base64.b64encode(buf.getvalue()).decode("ascii")
        if ns.get("result") is not None:
            out["text"], out["table_json"] = _serialize_result(ns["result"])
    else:
        if "result" not in ns:
            raise ValueError("The code must assign the final value to a variable named `result`.")
        out["text"], out["table_json"] = _serialize_result(ns["result"])
    return out


def main():
    _limit_resources()
    real_stdout = sys.stdout.buffer
    sys.stdout = io.StringIO()  # anything printed by libraries must not corrupt the JSON
    try:
        payload = pickle.loads(sys.stdin.buffer.read())
        out = execute(payload["code"], payload["df"], payload["kind"])
    except BaseException as exc:  # noqa: BLE001
        msg = str(exc).strip().splitlines()[-1:] or [""]
        out = {"ok": False, "error": f"{type(exc).__name__}: {msg[0][:300]}"}
    real_stdout.write(MARKER + json.dumps(out).encode("utf-8"))
    real_stdout.flush()


if __name__ == "__main__":
    main()
