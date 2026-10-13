"""The cleaning toolbox: every action the AI (or the user) can apply.

Each action is a small, well-defined, *auditable* transformation.  The LLM
never writes code - it only chooses actions and parameters from this registry,
which keeps the pipeline safe and reproducible.  Every applied action produces
a log entry with before/after counts and a few example changes.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .utils import (
    CONTROL_CHAR_RE, DISGUISED_NULL_TOKENS, EMAIL_RE, FALSE_TOKENS, TRUE_TOKENS, display_value,
    is_bool_series, is_datetime_series, is_missing_scalar, is_numeric_series, is_text_series, jsonable,
    norm_key, parse_number, to_snake_case,
)

try:
    import ftfy
except ImportError:  # pragma: no cover
    ftfy = None


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

@dataclass
class ActionSpec:
    name: str
    scope: str                   # "column" | "dataset"
    description: str             # shown to the LLM and in the UI
    params: dict[str, str]       # param name -> short description (for the LLM)
    priority: int                # execution order (lower runs first)
    fn: Callable[..., tuple[pd.DataFrame, str]] = field(repr=False, default=None)
    risk: str = "low"            # low | medium | high  (high = deletes data)


REGISTRY: dict[str, ActionSpec] = {}


def register(name, scope, description, params, priority, risk="low"):
    def deco(fn):
        REGISTRY[name] = ActionSpec(name, scope, description, params, priority, fn, risk)
        return fn
    return deco


def toolbox_for_prompt(scope: str | None = None) -> str:
    """Compact, LLM-friendly description of the available actions."""
    lines = []
    for spec in sorted(REGISTRY.values(), key=lambda s: s.priority):
        if scope and spec.scope != scope:
            continue
        p = ", ".join(f"{k}: {v}" for k, v in spec.params.items()) or "no params"
        lines.append(f"- {spec.name}: {spec.description} (params: {p})")
    return "\n".join(lines)


def toolbox_json() -> list[dict[str, Any]]:
    return [{"name": s.name, "scope": s.scope, "description": s.description, "params": s.params,
             "priority": s.priority, "risk": s.risk} for s in sorted(REGISTRY.values(), key=lambda s: s.priority)]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _missing_mask_series(s: pd.Series) -> pd.Series:
    if is_text_series(s):
        return s.isna() | s.map(lambda v: isinstance(v, str) and v.strip().casefold() in DISGUISED_NULL_TOKENS)
    return s.isna()


def _missing_mask(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or list(df.columns)
    return pd.DataFrame({c: _missing_mask_series(df[c]) for c in cols}, index=df.index)


def _as_text(s: pd.Series) -> pd.Series:
    """Object series of python str (None for missing) regardless of the input dtype.

    Built from a python list on purpose: Series.map silently turns None into NaN,
    which breaks `v is None` checks downstream.
    """
    return pd.Series([None if is_missing_scalar(v) else str(v) for v in s.tolist()], index=s.index, dtype=object)


def _apply_text(s: pd.Series, fn: Callable[[str], Any]) -> pd.Series:
    vals = [None if is_missing_scalar(v) else fn(str(v)) for v in s.tolist()]
    return pd.Series(vals, index=s.index, dtype=object)


def _to_numeric(s: pd.Series) -> pd.Series:
    if is_numeric_series(s):
        return s.astype(float)
    if is_bool_series(s):
        return s.astype("float")
    return _as_text(s).map(lambda v: None if is_missing_scalar(v) else parse_number(v)).astype(float)


def _finish_numeric(nums: pd.Series) -> pd.Series:
    """float series -> Int64 when every value is a whole number."""
    valid = nums.dropna()
    if len(valid) and np.allclose(valid, np.round(valid)) and valid.abs().max() < 9e15:
        return nums.round().astype("Int64")
    return nums


def _parse_datetime(s: pd.Series, dayfirst: bool = False, fmt: str | None = None) -> pd.Series:
    if is_datetime_series(s):
        return s
    text = _as_text(s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fmt:
            try:
                return pd.to_datetime(text, errors="coerce", format=fmt)
            except (TypeError, ValueError):
                pass
        try:
            return pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=dayfirst)
        except (TypeError, ValueError):
            return pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)


def _condition_mask(s: pd.Series, params: dict[str, Any]) -> pd.Series:
    """Boolean mask of values matching a condition (lt/gt/eq/ne/between/outside/regex/in)."""
    cond = str(params.get("condition", "")).lower()
    value = params.get("value")
    lower, upper = params.get("lower"), params.get("upper")
    if cond == "regex":
        return _as_text(s).map(lambda v: bool(re.search(str(value), v)) if v is not None else False)
    if cond == "in":
        vals = params.get("values") or ([value] if value is not None else [])
        keys = {norm_key(v) for v in vals}
        return _as_text(s).map(lambda v: v is not None and norm_key(v) in keys)
    if is_datetime_series(s) or (cond and isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}", value)):
        col = _parse_datetime(s)
        def ts(v):
            try:
                return pd.Timestamp(v)
            except Exception:  # noqa: BLE001
                return None
        v_ts = ts(value) if value is not None else None
        lo_ts, hi_ts = ts(lower) if lower is not None else None, ts(upper) if upper is not None else None
        # a bare year like "2027" -> Jan 1st of that year
        if v_ts is None and value is not None and re.fullmatch(r"\d{4}", str(value).strip()):
            v_ts = pd.Timestamp(int(str(value).strip()), 1, 1)
        if cond == "lt" and v_ts is not None:
            return col < v_ts
        if cond == "gt" and v_ts is not None:
            return col > v_ts
        if cond == "outside" and (lo_ts is not None or hi_ts is not None):
            m = pd.Series(False, index=s.index)
            if lo_ts is not None:
                m |= col < lo_ts
            if hi_ts is not None:
                m |= col > hi_ts
            return m
        if cond == "between" and lo_ts is not None and hi_ts is not None:
            return (col >= lo_ts) & (col <= hi_ts)
    nums = _to_numeric(s)
    fv = parse_number(value) if value is not None else None
    flo = parse_number(lower) if lower is not None else None
    fhi = parse_number(upper) if upper is not None else None
    if cond == "lt" and fv is not None:
        return nums < fv
    if cond == "le" and fv is not None:
        return nums <= fv
    if cond == "gt" and fv is not None:
        return nums > fv
    if cond == "ge" and fv is not None:
        return nums >= fv
    if cond == "eq":
        if fv is not None:
            return nums == fv
        return _as_text(s).map(lambda v: v is not None and norm_key(v) == norm_key(value))
    if cond == "ne":
        if fv is not None:
            return nums != fv
        return _as_text(s).map(lambda v: v is not None and norm_key(v) != norm_key(value))
    if cond == "outside":
        m = pd.Series(False, index=s.index)
        if flo is not None:
            m |= nums < flo
        if fhi is not None:
            m |= nums > fhi
        return m
    if cond == "between" and flo is not None and fhi is not None:
        return (nums >= flo) & (nums <= fhi)
    raise ValueError(f"Unsupported or incomplete condition: {params}")


# --------------------------------------------------------------------------------------
# Dataset-level actions
# --------------------------------------------------------------------------------------

@register("drop_columns", "dataset", "Remove whole columns (empty, constant, duplicated or useless).",
          {"columns": "list of column names to drop"}, priority=10, risk="high")
def act_drop_columns(df, params, col=None):
    cols = params.get("columns") or ([col] if col else [])
    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise ValueError("None of the requested columns exist.")
    return df.drop(columns=cols), f"Dropped {len(cols)} column(s): {cols}"


@register("drop_empty_rows", "dataset", "Drop rows where at least `threshold` fraction of the cells are missing (placeholders count as missing).",
          {"threshold": "fraction 0-1, default 0.6 (1.0 = only completely empty rows)"}, priority=15, risk="high")
def act_drop_empty_rows(df, params, col=None):
    thr = float(params.get("threshold", 0.6) or 0.6)
    thr = min(max(thr, 0.05), 1.0)
    frac = _missing_mask(df).mean(axis=1)
    keep = frac < thr
    return df[keep].reset_index(drop=True), f"Dropped {int((~keep).sum())} rows with ≥{int(thr * 100)}% missing cells"


@register("fix_encoding", "column", "Repair mojibake / broken UTF-8 text such as 'JosÃ©' -> 'José'.",
          {}, priority=20)
def act_fix_encoding(df, params, col=None):
    def fix(v: str) -> str:
        if ftfy is not None:
            return ftfy.fix_text(v)
        try:
            return v.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return v
    df[col] = _apply_text(df[col], fix)
    return df, "Repaired text encoding"


@register("strip_whitespace", "column", "Trim leading/trailing spaces and collapse repeated internal whitespace.",
          {}, priority=21)
def act_strip_whitespace(df, params, col=None):
    df[col] = _apply_text(df[col], lambda v: re.sub(r"\s+", " ", v).strip())
    return df, "Trimmed and collapsed whitespace"


@register("replace_disguised_nulls", "column", "Turn placeholder strings ('N/A', '-', '?', 'null', 'unknown', '' ...) into real missing values.",
          {"tokens": "optional list of extra placeholder strings to treat as missing"}, priority=22)
def act_replace_disguised_nulls(df, params, col=None):
    extra = {str(t).strip().casefold() for t in (params.get("tokens") or [])}
    tokens = DISGUISED_NULL_TOKENS | extra
    s = df[col]
    if is_text_series(s) or s.dtype == object:
        mask = s.map(lambda v: isinstance(v, str) and v.strip().casefold() in tokens)
        df[col] = s.mask(mask, np.nan)
        return df, f"Replaced {int(mask.sum())} placeholder values with null"
    return df, "Column is not text; nothing to replace"


@register("replace_values_with_null", "column", "Blank out specific values or every value matching a condition (sentinels like -999, impossible values).",
          {"values": "list of exact values to blank", "condition": "lt|le|gt|ge|eq|ne|outside|between|regex|in",
           "value": "comparison value", "lower": "for outside/between", "upper": "for outside/between"}, priority=23)
def act_replace_values_with_null(df, params, col=None):
    s = df[col]
    if params.get("values") and not params.get("condition"):
        vals = params["values"]
        keys = {norm_key(v) for v in vals}
        nums = {parse_number(v) for v in vals} - {None}
        if is_numeric_series(s):
            mask = s.isin(list(nums))
        else:
            mask = _as_text(s).map(lambda v: v is not None and (norm_key(v) in keys or parse_number(v) in nums))
    else:
        mask = _condition_mask(s, params).fillna(False).astype(bool)
    if is_numeric_series(s):
        df[col] = s.astype(float).mask(mask, np.nan)
    else:
        df[col] = s.mask(mask, np.nan)
    return df, f"Set {int(mask.sum())} values to null"


@register("remove_special_characters", "column", "Remove unwanted characters: control chars, punctuation, or a custom regex.",
          {"mode": "control|punctuation|non_alphanumeric|custom", "pattern": "regex for custom mode"}, priority=24)
def act_remove_special_characters(df, params, col=None):
    mode = str(params.get("mode", "control")).lower()
    if mode == "punctuation":
        pat = re.compile(r"[^\w\s]")
    elif mode == "non_alphanumeric":
        pat = re.compile(r"[^0-9a-zA-Z\s]")
    elif mode == "custom" and params.get("pattern"):
        pat = re.compile(str(params["pattern"]))
    else:
        pat = CONTROL_CHAR_RE
    df[col] = _apply_text(df[col], lambda v: pat.sub("", v))
    return df, f"Removed characters ({mode})"


@register("regex_replace", "column", "Search-and-replace with a regular expression.",
          {"pattern": "regex to find", "replacement": "replacement text"}, priority=25)
def act_regex_replace(df, params, col=None):
    pat = re.compile(str(params.get("pattern", "")))
    rep = str(params.get("replacement", ""))
    if not pat.pattern:
        raise ValueError("pattern is required")
    df[col] = _apply_text(df[col], lambda v: pat.sub(rep, v))
    return df, f"Replaced /{pat.pattern}/ with '{rep}'"


@register("normalize_case", "column", "Make text case consistent.",
          {"case": "lower|upper|title|sentence"}, priority=30)
def act_normalize_case(df, params, col=None):
    case = str(params.get("case", "lower")).lower()
    fn = {"lower": str.lower, "upper": str.upper, "title": str.title,
          "sentence": lambda v: v[:1].upper() + v[1:].lower()}.get(case, str.lower)
    df[col] = _apply_text(df[col], fn)
    return df, f"Converted text to {case} case"


@register("normalize_categories", "column", "Merge variant spellings of the same category into one canonical value (case/space variants, typos, synonyms).",
          {"mapping": "list of {from, to} pairs; if omitted, detected variant groups are merged automatically"}, priority=31)
def act_normalize_categories(df, params, col=None):
    mapping_pairs = params.get("mapping") or []
    lookup: dict[str, str] = {}
    for pair in mapping_pairs:
        if isinstance(pair, dict) and pair.get("from") is not None and pair.get("to") is not None:
            lookup[norm_key(pair["from"])] = str(pair["to"])
        elif isinstance(pair, (list, tuple)) and len(pair) == 2:
            lookup[norm_key(pair[0])] = str(pair[1])
    if not lookup:
        from .profiler import _fuzzy_groups, _variant_groups  # lazy import to avoid cycle
        vals = _as_text(df[col]).dropna().map(str.strip)
        vals = vals[vals != ""]
        groups = _variant_groups(vals)
        groups += _fuzzy_groups(vals, groups)
        for g in groups:
            for v in g["variants"]:
                lookup[norm_key(v)] = g["canonical"]
    if not lookup:
        return df, "No variant groups found; nothing merged"
    before = _as_text(df[col])
    after = before.map(lambda v: None if is_missing_scalar(v) else lookup.get(norm_key(v), v))
    # also tidy whitespace on untouched values so that keys match
    after = after.map(lambda v: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else v).astype(object)
    df[col] = after
    changed = int(((before.fillna("\0") != after.fillna("\0"))).sum())
    return df, f"Merged {len(set(lookup.values()))} canonical categories; {changed} values rewritten"


@register("normalize_boolean", "column", "Convert yes/no, y/n, true/false, 1/0 text into a real boolean column.",
          {}, priority=32)
def act_normalize_boolean(df, params, col=None):
    def conv(v):
        if is_missing_scalar(v):
            return pd.NA
        k = str(v).strip().casefold()
        if k in TRUE_TOKENS:
            return True
        if k in FALSE_TOKENS:
            return False
        return pd.NA
    out = _as_text(df[col]).map(conv)
    df[col] = pd.array(out.tolist(), dtype="boolean")
    return df, "Converted to boolean (unrecognised values -> null)"


@register("standardize_email", "column", "Lowercase and trim email addresses; blank invalid ones.",
          {}, priority=33)
def act_standardize_email(df, params, col=None):
    def conv(v):
        v = v.strip().lower()
        return v if EMAIL_RE.match(v) else None
    df[col] = _apply_text(df[col], conv)
    return df, "Standardised emails (invalid -> null)"


@register("standardize_phone", "column", "Keep only digits (and a leading +) in phone numbers; blank numbers with fewer than 7 digits.",
          {}, priority=34)
def act_standardize_phone(df, params, col=None):
    def conv(v):
        v = v.strip()
        plus = v.startswith("+")
        digits = re.sub(r"\D", "", v)
        if len(digits) < 7:
            return None
        return ("+" if plus else "") + digits
    df[col] = _apply_text(df[col], conv)
    return df, "Standardised phone numbers"


@register("parse_numeric", "column", "Convert numbers stored as text ('$1,200', '45 kg', '12%', ' 42 ') into a numeric column; unparseable values become null.",
          {}, priority=40)
def act_parse_numeric(df, params, col=None):
    nums = _to_numeric(df[col])
    before_valid = int(_as_text(df[col]).notna().sum())
    df[col] = _finish_numeric(nums)
    lost = before_valid - int(nums.notna().sum())
    return df, f"Parsed to numeric ({str(df[col].dtype)}); {lost} unparseable values became null"


@register("parse_dates", "column", "Parse dates written as text (any mix of formats) into a real datetime column.",
          {"dayfirst": "true if days come before months (e.g. 04/03/2021 = 4 March)", "format": "optional explicit strptime format"},
          priority=41)
def act_parse_dates(df, params, col=None):
    dayfirst = bool(params.get("dayfirst", False))
    fmt = params.get("format") or None
    before_valid = int(_as_text(df[col]).notna().sum())
    parsed = _parse_datetime(df[col], dayfirst=dayfirst, fmt=fmt)
    df[col] = parsed
    lost = before_valid - int(parsed.notna().sum())
    return df, f"Parsed to datetime (dayfirst={dayfirst}); {lost} unparseable values became null"


@register("convert_type", "column", "Cast a column to a specific type.",
          {"to": "int|float|str|bool|datetime|category"}, priority=42)
def act_convert_type(df, params, col=None):
    to = str(params.get("to", "str")).lower()
    s = df[col]
    if to in ("int", "integer"):
        df[col] = _to_numeric(s).round().astype("Int64")
    elif to in ("float", "number", "numeric"):
        df[col] = _to_numeric(s)
    elif to in ("bool", "boolean"):
        return act_normalize_boolean(df, params, col)
    elif to in ("datetime", "date"):
        return act_parse_dates(df, params, col)
    elif to == "category":
        df[col] = _as_text(s).astype("category")
    else:
        df[col] = _as_text(s)
    return df, f"Converted to {to}"


@register("round_numeric", "column", "Round numeric values to a number of decimals.",
          {"decimals": "integer, default 2"}, priority=45)
def act_round_numeric(df, params, col=None):
    d = int(params.get("decimals", 2) or 0)
    df[col] = _finish_numeric(_to_numeric(df[col]).round(d)) if d == 0 else _to_numeric(df[col]).round(d)
    return df, f"Rounded to {d} decimals"


@register("remove_rows_where", "column", "Delete rows whose value in this column matches a condition (impossible or invalid records).",
          {"condition": "lt|le|gt|ge|eq|ne|outside|between|regex|in", "value": "comparison value",
           "lower": "for outside/between", "upper": "for outside/between", "values": "list for 'in'"}, priority=50, risk="high")
def act_remove_rows_where(df, params, col=None):
    mask = _condition_mask(df[col], params).fillna(False).astype(bool)
    n = int(mask.sum())
    return df[~mask].reset_index(drop=True), f"Removed {n} rows where {col} {params.get('condition')} {params.get('value', params.get('lower', ''))}{('-' + str(params.get('upper'))) if params.get('upper') is not None else ''}"


@register("clip_values", "column", "Clamp numeric values into a valid range (e.g. percentages to 0-100).",
          {"lower": "minimum allowed value", "upper": "maximum allowed value"}, priority=51)
def act_clip_values(df, params, col=None):
    nums = _to_numeric(df[col])
    lo = parse_number(params.get("lower")) if params.get("lower") is not None else None
    hi = parse_number(params.get("upper")) if params.get("upper") is not None else None
    if lo is None and hi is None:
        raise ValueError("clip_values needs lower and/or upper")
    clipped = nums.clip(lower=lo, upper=hi)
    n = int(((clipped != nums) & nums.notna()).sum())
    df[col] = _finish_numeric(clipped)
    return df, f"Clipped {n} values into [{lo}, {hi}]"


@register("handle_outliers", "column", "Treat statistical outliers in a numeric column.",
          {"method": "iqr|zscore|percentile", "treatment": "clip|remove|null", "factor": "IQR multiplier (1.5 normal, 3 extreme) or z threshold (3) or percentile tail (0.01)"},
          priority=52, risk="medium")
def act_handle_outliers(df, params, col=None):
    method = str(params.get("method", "iqr")).lower()
    treatment = str(params.get("treatment", params.get("action", "clip"))).lower()
    factor = params.get("factor")
    nums = _to_numeric(df[col])
    valid = nums.dropna()
    if len(valid) < 5:
        return df, "Too few numeric values to assess outliers"
    if method == "zscore":
        f = float(factor or 3.0)
        mean, std = valid.mean(), valid.std()
        if not std:
            return df, "Zero variance; no outliers"
        lo, hi = mean - f * std, mean + f * std
    elif method == "percentile":
        f = float(factor or 0.01)
        lo, hi = valid.quantile(f), valid.quantile(1 - f)
    else:
        f = float(factor or 1.5)
        q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - f * iqr, q3 + f * iqr
    mask = (nums < lo) | (nums > hi)
    n = int(mask.sum())
    if treatment == "remove":
        return df[~mask.fillna(False)].reset_index(drop=True), f"Removed {n} outlier rows outside [{round(lo, 3)}, {round(hi, 3)}] ({method})"
    if treatment in ("null", "nan", "missing"):
        df[col] = _finish_numeric(nums.mask(mask, np.nan))
        return df, f"Blanked {n} outliers outside [{round(lo, 3)}, {round(hi, 3)}] ({method})"
    df[col] = _finish_numeric(nums.clip(lower=lo, upper=hi))
    return df, f"Clipped {n} outliers into [{round(lo, 3)}, {round(hi, 3)}] ({method})"


@register("fill_missing", "column", "Impute missing values.",
          {"strategy": "mean|median|mode|constant|zero|ffill|bfill|interpolate", "value": "value for constant strategy"},
          priority=60)
def act_fill_missing(df, params, col=None):
    strategy = str(params.get("strategy", "median")).lower()
    s = df[col]
    # treat placeholders as missing first
    if is_text_series(s) or s.dtype == object:
        s = s.mask(_missing_mask_series(s), np.nan)
    missing_before = int(s.isna().sum())
    if missing_before == 0:
        df[col] = s
        return df, "No missing values to fill"
    if strategy in ("mean", "median", "interpolate", "zero") and not is_numeric_series(s):
        nums = _to_numeric(s)
        if nums.notna().sum() >= 0.8 * s.notna().sum() and nums.notna().any():
            s = nums
        elif strategy != "zero":
            strategy = "mode"
    if is_numeric_series(s):
        s = s.astype(float)  # nullable Int64 cannot hold a float mean; _finish_numeric restores ints
    if strategy in ("mean", "median"):
        fill = float(s.mean() if strategy == "mean" else s.median())
        # round to the precision the column already uses (whole numbers stay whole)
        valid = s.dropna()
        if len(valid) and np.allclose(valid, np.round(valid)):
            fill = float(round(fill))
        else:
            decimals = max((len(str(v).split(".")[1].rstrip("0")) for v in valid.head(500).tolist() if "." in str(v)), default=2)
            fill = round(fill, min(max(decimals, 1), 4))
        s = s.fillna(fill)
    elif strategy == "zero":
        s = s.fillna(0) if is_numeric_series(s) else s.fillna("0")
        fill = 0
    elif strategy == "mode":
        m = s.mode(dropna=True)
        if m.empty:
            return df, "No mode available; nothing filled"
        fill = m.iloc[0]
        s = s.fillna(fill)
    elif strategy == "constant":
        fill = params.get("value", "unknown")
        if is_numeric_series(s):
            fv = parse_number(fill)
            if fv is None:
                s = _as_text(s).fillna(str(fill))
            else:
                s = s.fillna(fv)
        else:
            s = s.fillna(str(fill))
    elif strategy == "ffill":
        s = s.ffill()
        fill = "previous value"
    elif strategy == "bfill":
        s = s.bfill()
        fill = "next value"
    elif strategy == "interpolate":
        s = s.interpolate(limit_direction="both")
        fill = "interpolated"
    else:
        raise ValueError(f"Unknown strategy {strategy}")
    if is_numeric_series(s):
        s = _finish_numeric(s.astype(float))
    df[col] = s
    filled = missing_before - int(df[col].isna().sum())
    return df, f"Filled {filled} missing values with {strategy} ({display_value(fill)})"


@register("drop_rows_with_missing", "dataset", "Drop rows that have a missing value in any of the given key columns (e.g. ID or target).",
          {"columns": "list of key column names"}, priority=58, risk="high")
def act_drop_rows_with_missing(df, params, col=None):
    cols = params.get("columns") or ([col] if col else [])
    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise ValueError("No valid columns given")
    mask = _missing_mask(df, cols).any(axis=1)
    return df[~mask].reset_index(drop=True), f"Dropped {int(mask.sum())} rows missing {cols}"


@register("drop_duplicate_rows", "dataset", "Remove duplicated records (optionally judged on a subset of columns, ignoring case/whitespace in text).",
          {"subset": "optional list of columns that define a duplicate", "keep": "first|last"}, priority=72, risk="high")
def act_drop_duplicate_rows(df, params, col=None):
    subset = params.get("subset") or None
    if subset:
        subset = [c for c in subset if c in df.columns] or None
    keep = params.get("keep", "first")
    keep = keep if keep in ("first", "last") else "first"
    key = df.copy()
    for c in key.columns:
        if is_text_series(key[c]) or key[c].dtype == object:
            key[c] = _as_text(key[c]).map(lambda v: norm_key(v) if v is not None else None)
    dup = key.duplicated(subset=subset, keep=keep)
    return df[~dup].reset_index(drop=True), f"Dropped {int(dup.sum())} duplicate rows" + (f" (by {subset})" if subset else "")


@register("rename_column", "dataset", "Rename a single column.",
          {"column": "current name", "new_name": "new name"}, priority=90)
def act_rename_column(df, params, col=None):
    src = params.get("column") or col
    dst = params.get("new_name")
    if src not in df.columns or not dst:
        raise ValueError("rename_column needs an existing column and new_name")
    return df.rename(columns={src: str(dst)}), f"Renamed '{src}' -> '{dst}'"


@register("standardize_column_names", "dataset", "Convert all column names to clean snake_case identifiers (unique, no spaces or symbols).",
          {}, priority=91)
def act_standardize_column_names(df, params, col=None):
    mapping, used = {}, {}
    for c in df.columns:
        base = to_snake_case(c)
        name = base
        while name in used:
            used[base] = used.get(base, 1) + 1
            name = f"{base}_{used[base]}"
        used[name] = 1
        mapping[c] = name
    changed = {k: v for k, v in mapping.items() if k != v}
    return df.rename(columns=mapping), f"Renamed {len(changed)} columns: " + ", ".join(f"'{k}'→'{v}'" for k, v in list(changed.items())[:8]) + (" …" if len(changed) > 8 else "")


# --------------------------------------------------------------------------------------
# Plan execution
# --------------------------------------------------------------------------------------

def normalize_action(a: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults so that every action dict has the same shape."""
    spec = REGISTRY.get(a.get("action", ""))
    return {
        "id": a.get("id"),
        "action": a.get("action"),
        "column": a.get("column") if spec and spec.scope == "column" else a.get("column"),
        "params": a.get("params") or {},
        "reason": a.get("reason") or "",
        "confidence": a.get("confidence"),
        "source": a.get("source", "ai"),
        "enabled": a.get("enabled", True),
        "scope": spec.scope if spec else "unknown",
        "risk": spec.risk if spec else "unknown",
        "priority": spec.priority if spec else 999,
    }


def validate_action(a: dict[str, Any], df: pd.DataFrame | None = None) -> list[str]:
    problems = []
    spec = REGISTRY.get(a.get("action", ""))
    if not spec:
        return [f"Unknown action '{a.get('action')}'"]
    params = a.get("params") or {}
    if spec.scope == "column":
        if not a.get("column"):
            problems.append("column is required")
        elif df is not None and a["column"] not in df.columns:
            problems.append(f"column '{a['column']}' no longer exists (dropped or renamed earlier)")
    if spec.name == "drop_columns":
        cols = params.get("columns") or ([a["column"]] if a.get("column") else [])
        if not cols:
            problems.append("columns list is required")
        elif df is not None and not any(c in df.columns for c in cols):
            problems.append("column(s) already removed by an earlier action")
    if spec.name == "drop_rows_with_missing" and df is not None:
        cols = params.get("columns") or ([a["column"]] if a.get("column") else [])
        if cols and not any(c in df.columns for c in cols):
            problems.append("column(s) no longer exist")
    if spec.name == "fill_missing" and str(params.get("strategy", "median")).lower() not in (
            "mean", "median", "mode", "constant", "zero", "ffill", "bfill", "interpolate"):
        problems.append(f"unknown strategy '{params.get('strategy')}'")
    if spec.name in ("remove_rows_where", "replace_values_with_null"):
        if not params.get("condition") and not params.get("values"):
            problems.append("condition or values is required")
    if spec.name == "regex_replace" and not params.get("pattern"):
        problems.append("pattern is required")
    if spec.name == "clip_values" and params.get("lower") is None and params.get("upper") is None:
        problems.append("lower or upper is required")
    if spec.name == "rename_column" and not params.get("new_name"):
        problems.append("new_name is required")
    return problems


def apply_plan(df: pd.DataFrame, actions: list[dict[str, Any]], progress=None) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply enabled actions in canonical order. Returns (cleaned df, log)."""
    log: list[dict[str, Any]] = []
    work = df.copy()
    # stable sort by priority while keeping original order inside a priority bucket
    ordered = [normalize_action(a) for a in actions if a.get("enabled", True)]
    ordered.sort(key=lambda a: a["priority"])
    for i, a in enumerate(ordered):
        spec = REGISTRY.get(a["action"])
        entry: dict[str, Any] = {"action_id": a["id"], "action": a["action"], "column": a["column"],
                                 "params": a["params"], "reason": a["reason"], "source": a["source"],
                                 "rows_before": int(len(work)), "cols_before": int(work.shape[1])}
        if progress:
            progress(i, len(ordered), a)
        if not spec:
            entry.update(status="skipped", message=f"Unknown action '{a['action']}'")
            log.append(entry)
            continue
        problems = validate_action(a, work)
        if problems:
            entry.update(status="skipped", message="; ".join(problems))
            log.append(entry)
            continue
        col = a["column"] if spec.scope == "column" else a.get("column")
        before_col = work[col].copy() if (col and col in work.columns) else None
        try:
            new_df, message = spec.fn(work.copy(), dict(a["params"]), col)
        except Exception as exc:  # noqa: BLE001
            entry.update(status="error", message=f"{type(exc).__name__}: {exc}")
            log.append(entry)
            continue
        entry.update(status="applied", message=message, rows_after=int(len(new_df)), cols_after=int(new_df.shape[1]))
        if before_col is not None and col in new_df.columns and len(new_df) == len(work):
            after_col = new_df[col]
            b, af = before_col.astype(object), after_col.astype(object)
            b_na, a_na = before_col.isna(), after_col.isna()
            changed = ((b.astype(str) != af.astype(str)) & ~(b_na & a_na))
            entry.update(cells_changed=int(changed.sum()), nulls_before=int(b_na.sum()), nulls_after=int(a_na.sum()),
                         dtype_before=str(before_col.dtype), dtype_after=str(after_col.dtype))
            ex = []
            seen = set()
            for idx in changed[changed].index[:200]:
                pair = (display_value(before_col.loc[idx]), display_value(after_col.loc[idx]))
                if pair in seen:
                    continue
                seen.add(pair)
                ex.append(list(pair))
                if len(ex) >= 5:
                    break
            entry["examples"] = ex
        work = new_df
        log.append(entry)
    return work, jsonable(log)
