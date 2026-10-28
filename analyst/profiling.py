"""Loading and profiling of the user's CSV. Prompts only ever see this profile, never raw rows."""
import csv
import io
import re
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

from analyst import config

_DATE_LIKE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")
_SEPARATORS = (",", ";", "\t", "|")
_MONEY = re.compile(r"^[-+]?[$€£¥₩]?[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")  # $1,234.56  -$12.30  1,234


class DataLoadError(ValueError):
    """Raised with a friendly message when a CSV cannot be used."""


def _read_bytes(source) -> bytes:
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if isinstance(source, bytes):
        return source
    return source.read()


def _maybe_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)):
            continue
        sample = s.dropna().astype(str).head(100)
        if len(sample) == 0 or sample.map(lambda v: bool(_DATE_LIKE.match(v))).mean() < 0.9:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        if parsed.notna().sum() >= 0.9 * s.notna().sum():
            df[col] = parsed
    return df


def _maybe_parse_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Turn text columns of amounts such as '$1,234.56' into numbers (pd.to_numeric alone makes them NaN)."""
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)):
            continue
        sample = s.dropna().astype(str).str.replace(" ", "", regex=False)
        if len(sample) == 0 or not sample.str.contains(r"[$€£¥₩]|\d,\d{3}", regex=True).any():
            continue
        if sample.map(lambda v: bool(_MONEY.match(v))).mean() >= 0.95:
            df[col] = pd.to_numeric(s.astype(str).str.replace(r"[$€£¥₩,\s]", "", regex=True), errors="coerce")
    return df


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")  # never fails


def _layout(text: str):
    """Guess (separator, lines above the header, decimal mark) from the first lines of the file."""
    lines = text.splitlines()[:60]
    filled = [(i, line) for i, line in enumerate(lines) if line.strip()]
    best = (False, 1, ",")  # (consistent, number of fields, separator)
    for sep in _SEPARATORS:
        counts = [len(next(csv.reader([line], delimiter=sep))) for _, line in filled]
        tail = counts[len(counts) // 2:] or counts
        fields, hits = Counter(tail).most_common(1)[0]
        best = max(best, (hits / len(tail) >= 0.9 and fields > 1, fields, sep), key=lambda b: (b[0], b[1]))
    _, fields, sep = best
    skip = next((i for i, line in filled if len(next(csv.reader([line], delimiter=sep))) == fields), 0)
    body = "\n".join(lines[skip + 1:])
    decimal = "," if sep != "," and re.search(r"(?:^|[;\t|])-?\d+,\d+(?:[;\t|]|$)", body, re.M) else "."
    return sep, skip, decimal


def load_dataframe(source, max_bytes: int = config.MAX_UPLOAD_BYTES) -> pd.DataFrame:
    """Read a CSV (path, bytes or file-like) with size and shape limits."""
    raw = _read_bytes(source)
    if len(raw) > max_bytes:
        raise DataLoadError(f"The file is too large ({len(raw) / 1e6:.1f} MB). Maximum: {max_bytes / 1e6:.0f} MB.")
    if not raw.strip():
        raise DataLoadError("The file is empty.")
    if b"\x00" in raw[:4096]:  # binary content (image, xlsx, zip...) renamed to .csv
        raise DataLoadError("The file is not a text CSV.")
    text = _decode(raw)
    sep, skip, decimal = _layout(text)  # ';' files, decimal commas and report lines above the header
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, skiprows=skip, decimal=decimal)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DataLoadError(f"The file is not a valid CSV: {exc}") from exc
    if df.shape[1] == 0 or len(df) == 0:
        raise DataLoadError("The CSV has no data rows or no columns.")
    if len(df) > config.MAX_ROWS:
        raise DataLoadError(f"Too many rows ({len(df):,}). Maximum: {config.MAX_ROWS:,}.")
    if df.shape[1] > config.MAX_COLS:
        raise DataLoadError(f"Too many columns ({df.shape[1]}). Maximum: {config.MAX_COLS}.")
    df.columns = [str(c).strip() for c in df.columns]
    return _maybe_parse_numbers(_maybe_parse_dates(df))


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "number"
    return "text"


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Schema + aggregate statistics per column."""
    columns = []
    for name in df.columns:
        s = df[name]
        kind = _kind(s)
        info = {
            "name": name, "kind": kind, "pandas_dtype": str(s.dtype),
            "missing": int(s.isna().sum()), "unique": int(s.nunique(dropna=True)),
        }
        if kind == "number" and s.notna().any():
            info.update(min=round(float(s.min()), 4), max=round(float(s.max()), 4), mean=round(float(s.mean()), 4))
        elif kind == "datetime" and s.notna().any():
            info.update(min=str(s.min().date()), max=str(s.max().date()))
        elif kind == "text" and 0 < info["unique"] <= 12:
            info["top_values"] = [str(v) for v in s.value_counts().head(12).index]
        if kind == "text" and s.notna().any():  # numbers stored as text next to placeholders such as 'ERROR'
            numeric = pd.to_numeric(s, errors="coerce")
            share = numeric.notna().sum() / s.notna().sum()
            if share >= 0.8:
                info["numeric_share"] = round(float(share), 3)
                info["non_numeric"] = [str(v) for v in s[numeric.isna() & s.notna()].value_counts().head(5).index]
        columns.append(info)
    return {"n_rows": int(len(df)), "n_cols": int(df.shape[1]), "columns": columns}


def profile_to_text(profile: dict) -> str:
    """Compact text for prompts. Column names are quoted so spaces and typos are visible."""
    lines = [f"DataFrame `df`: {profile['n_rows']} rows, {profile['n_cols']} columns."]
    for c in profile["columns"]:
        extra = ""
        if c["kind"] == "number" and "min" in c:
            extra = f"range {c['min']} to {c['max']}, mean {c['mean']}"
        elif c["kind"] == "datetime" and "min" in c:
            extra = f"from {c['min']} to {c['max']}"
        elif "top_values" in c:
            extra = "values: " + ", ".join(repr(v) for v in c["top_values"])
        else:
            extra = f"{c['unique']} distinct values"
        if "numeric_share" in c:
            extra += (f"; stored as TEXT: {c['numeric_share']:.0%} of the values are numbers, the rest are "
                      + ", ".join(repr(v) for v in c["non_numeric"]) + " (convert with pd.to_numeric(errors='coerce'))")
        lines.append(f"- {c['name']!r} ({c['kind']}): {extra}; missing {c['missing']}")
    return "\n".join(lines)
