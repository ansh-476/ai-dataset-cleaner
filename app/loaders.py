"""Load uploaded files (CSV / TSV / TXT / Excel / JSON) into a pandas DataFrame.

The loader is deliberately forgiving: it sniffs encoding and delimiter, keeps
every column as text when the file is a CSV/Excel so that the *profiler* (not
pandas) decides what the real type is.  That way "N/A", "$1,200", "12%" and
" 42 " survive into the analysis instead of being silently coerced.
"""
from __future__ import annotations

import csv
import io
import json
import os
from typing import Any

import pandas as pd

try:
    import chardet
except ImportError:  # pragma: no cover - optional
    chardet = None

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls", ".json"}


class LoadError(Exception):
    pass


def detect_encoding(raw: bytes) -> str:
    """Best-effort text encoding detection with sane fallbacks."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return "utf-16"
    if chardet is not None:
        guess = chardet.detect(raw[:200_000]) or {}
        enc = (guess.get("encoding") or "").lower()
        conf = guess.get("confidence") or 0
        if enc and conf > 0.6:
            if enc == "ascii":
                return "utf-8"
            return enc
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def detect_delimiter(text: str) -> str:
    sample = text[:50_000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        # fall back to the most common candidate on the first lines
        first_lines = sample.splitlines()[:20]
        counts = {d: sum(line.count(d) for line in first_lines) for d in [",", ";", "\t", "|"]}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","


def _read_csv_text(text: str, delimiter: str) -> pd.DataFrame:
    return pd.read_csv(
        io.StringIO(text),
        sep=delimiter,
        dtype=str,              # keep raw text; the profiler infers real types
        keep_default_na=False,  # keep "NA", "null" etc. visible as text
        na_filter=False,
        skip_blank_lines=False,
        engine="python",
        on_bad_lines="warn",
    )


def load_dataframe(path: str, sheet: str | int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a file from disk. Returns (DataFrame, metadata)."""
    ext = os.path.splitext(path)[1].lower()
    meta: dict[str, Any] = {"file_name": os.path.basename(path), "extension": ext}
    if ext not in SUPPORTED_EXTENSIONS:
        raise LoadError(f"Unsupported file type '{ext}'. Upload CSV, TSV, TXT, XLSX, XLS or JSON.")

    if ext in (".csv", ".tsv", ".txt"):
        with open(path, "rb") as fh:
            raw = fh.read()
        if not raw.strip():
            raise LoadError("The file is empty.")
        encoding = detect_encoding(raw)
        try:
            text = raw.decode(encoding, errors="replace")
        except LookupError:
            encoding = "utf-8"
            text = raw.decode(encoding, errors="replace")
        delimiter = "\t" if ext == ".tsv" else detect_delimiter(text)
        df = _read_csv_text(text, delimiter)
        meta.update({"encoding": encoding, "delimiter": delimiter})

    elif ext in (".xlsx", ".xlsm", ".xls"):
        try:
            xls = pd.ExcelFile(path)
        except Exception as exc:  # noqa: BLE001
            raise LoadError(f"Could not open Excel file: {exc}") from exc
        sheets = xls.sheet_names
        target = sheet if sheet is not None else sheets[0]
        if isinstance(target, str) and target not in sheets:
            raise LoadError(f"Sheet '{target}' not found. Available: {sheets}")
        df = xls.parse(target, dtype=str, keep_default_na=False, na_filter=False)
        meta.update({"sheets": sheets, "sheet": target})

    else:  # json
        with open(path, "rb") as fh:
            raw = fh.read()
        encoding = detect_encoding(raw)
        text = raw.decode(encoding, errors="replace")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # maybe JSON lines
            try:
                df = pd.read_json(io.StringIO(text), lines=True, dtype=False)
                meta.update({"encoding": encoding, "json_format": "lines"})
                return _finalize(df), meta
            except ValueError as exc:
                raise LoadError(f"Invalid JSON: {exc}") from exc
        if isinstance(data, dict):
            # common shapes: {"data": [...]}, {"records": [...]}, or a dict of columns
            for key in ("data", "records", "rows", "items", "results"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            raise LoadError("JSON must be an array of records or an object containing one.")
        meta.update({"encoding": encoding, "json_format": "records"})

    return _finalize(df), meta


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    # Make column names strings and unique-ish (pandas already suffixes .1, .2).
    df.columns = [str(c) for c in df.columns]
    # Drop rows that are completely empty strings (artifacts of trailing newlines)
    if len(df) and df.shape[1]:
        as_text = df.astype(str)
        blank_rows = (as_text.apply(lambda s: s.str.strip() == "")).all(axis=1)
        # Keep them - the profiler reports them as "empty rows" so the AI can decide.
        df.attrs["blank_row_count"] = int(blank_rows.sum())
    df = df.reset_index(drop=True)
    return df


def save_dataframe(df: pd.DataFrame, path: str, fmt: str = "csv") -> str:
    if fmt == "xlsx":
        df.to_excel(path, index=False)
    elif fmt == "json":
        df.to_json(path, orient="records", indent=2, date_format="iso", force_ascii=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    return path
