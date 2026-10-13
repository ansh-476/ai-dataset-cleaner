"""The AI planner: turns profiler evidence into a cleaning plan with Ollama.

Flow
----
1. Dataset-level call  - structure problems (duplicates, useless columns, names).
2. One call per column - the model reads that column's profile + detected issues
   + rule-based candidate actions, *thinks*, and returns the actions it wants.
3. Optional summary call - a short human-readable explanation of the plan.

Every model reply is constrained to a JSON schema (Ollama structured outputs)
and then validated against the real toolbox and the real column values, so a
hallucinated action or category can never reach the data.
"""
from __future__ import annotations

import difflib
import json
import re
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Callable


from .cleaner import REGISTRY, normalize_action, toolbox_for_prompt, validate_action
from .config import Settings
from .llm import LLMError, LLMUnavailable, make_client
from .utils import jsonable, norm_key

COLUMN_ACTIONS = [n for n, s in REGISTRY.items() if s.scope == "column"] + [
    "drop_columns", "drop_rows_with_missing", "drop_duplicate_rows"]
DATASET_ACTIONS = [n for n, s in REGISTRY.items() if s.scope == "dataset"]

PARAM_SCHEMA = {
    "type": "object",
    "properties": {
        "strategy": {"type": "string"},
        "value": {"type": "string"},
        "case": {"type": "string"},
        "method": {"type": "string"},
        "treatment": {"type": "string"},
        "factor": {"type": "number"},
        "lower": {"type": "string"},
        "upper": {"type": "string"},
        "condition": {"type": "string"},
        "to": {"type": "string"},
        "dayfirst": {"type": "boolean"},
        "format": {"type": "string"},
        "pattern": {"type": "string"},
        "replacement": {"type": "string"},
        "mode": {"type": "string"},
        "keep": {"type": "string"},
        "threshold": {"type": "number"},
        "decimals": {"type": "integer"},
        "new_name": {"type": "string"},
        "tokens": {"type": "array", "items": {"type": "string"}},
        "values": {"type": "array", "items": {"type": "string"}},
        "columns": {"type": "array", "items": {"type": "string"}},
        "subset": {"type": "array", "items": {"type": "string"}},
        "mapping": {"type": "array", "items": {"type": "object",
                                                "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
                                                "required": ["from", "to"]}},
    },
}


def _plan_schema(action_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "thinking": {"type": "string"},
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": action_names},
                        "column": {"type": "string"},
                        "params": PARAM_SCHEMA,
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["action", "params", "reason"],
                },
            },
        },
        "required": ["thinking", "actions"],
    }


SUMMARY_SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}


# --------------------------------------------------------------------------------------
# Prompt building
# --------------------------------------------------------------------------------------

SYSTEM_COLUMN = """You are a meticulous senior data engineer doing DEEP data cleaning on ONE column of a tabular dataset.
You receive: the column's profile (statistics, samples), the issues an automatic profiler detected, and rule-based candidate actions.
Your job: THINK about what this column really represents, decide what is wrong, then choose the cleaning actions that fix it properly.

Available actions (choose only from these; the system runs them in the correct order automatically):
{toolbox}

Principles:
- Think first: what does the column mean? what type SHOULD it be? which values are impossible for that meaning?
- Fix root causes: placeholders -> real nulls; numbers/dates/booleans stored as text -> proper types; inconsistent category spellings -> one canonical spelling; broken encoding -> repaired; whitespace -> trimmed.
- Impute thoughtfully: median for skewed or outlier-prone numerics, mean for symmetric ones, mode for categories, ffill for ordered time series; never impute identifiers or free text with statistics.
- Impossible values (negative ages, ages > 120, percentages > 100, sentinel codes like -999/9999, far-future dates) must be blanked or the rows removed - do not leave them in.
- Outliers: clip (winsorise) when they are plausible but extreme; remove rows only when clearly erroneous; keep them when the quantity is naturally heavy-tailed and values are plausible.
- Drop a column only when it is empty, constant, an exact duplicate, or more than ~60% missing. Keep identifiers.
- Candidate actions are good defaults: keep the right ones, fix their parameters, discard wrong ones, add missing ones.
- For normalize_categories, give an explicit mapping only when you know the correct canonical spellings (e.g. synonyms like 'Bengaluru' -> 'Bangalore'); otherwise omit mapping and the detected variant groups will be merged.
- Return an empty actions list if the column is already clean.
- Keep each reason short, specific and tied to the evidence. Confidence is 0-1.
Respond ONLY with JSON: {{"thinking": "...", "actions": [{{"action": "...", "column": "...", "params": {{...}}, "reason": "...", "confidence": 0.9}}]}}"""

SYSTEM_DATASET = """You are a meticulous senior data engineer planning the STRUCTURAL cleaning of a tabular dataset (whole-table decisions only).
Column-level fixes (types, nulls, categories, outliers) are handled separately - do NOT propose them here.

Available dataset-level actions:
{toolbox}

Decide, with reasoning, whether to: drop empty/constant/duplicate columns, drop empty or mostly-empty rows, remove duplicate rows (and on which key), drop rows missing a critical identifier, and standardise column names.
Keep identifiers and informative columns. Only drop columns that are empty, constant, exact duplicates or useless.
Respond ONLY with JSON: {{"thinking": "...", "actions": [{{"action": "...", "params": {{...}}, "reason": "...", "confidence": 0.9}}]}}"""

SYSTEM_SUMMARY = """You are a data engineer writing a concise report for a non-technical stakeholder.
Summarise the cleaning plan below in 4-7 plain sentences: what was wrong with the data, what will be done, and what the main judgement calls are. No bullet points, no JSON keys, no column-by-column list.
Respond ONLY with JSON: {"summary": "..."}"""


def _fmt(v: Any) -> str:
    return json.dumps(jsonable(v), ensure_ascii=False, default=str)


def build_column_prompt(profile: dict[str, Any], col: dict[str, Any], issues: list[dict[str, Any]],
                        with_header: bool = True) -> str:
    ds = profile["dataset"]
    lines = []
    if with_header:
        lines += [f"DATASET: {ds['rows']} rows × {ds['columns']} columns. Columns: {_fmt([c['name'] for c in profile['columns']])}", ""]
    lines += [
        f"COLUMN: {col['name']!r} (position {col['position'] + 1})",
        f"Inferred type: {col['inferred_type']}   | stored dtype: {col['pandas_dtype']}",
        f"Missing: {col['missing']} of {col['rows']} ({col['missing_pct']}%) -> {col['missing_real']} truly empty, "
        f"{col['missing_disguised']} placeholder text {_fmt(col.get('disguised_tokens', {}))}",
        f"Unique values: {col['unique']} ({col['unique_pct']}% of valid values)",
        f"Sample values: {_fmt(col.get('samples', []))}",
        f"Most frequent: {_fmt(col.get('top_values', [])[:8])}",
    ]
    if col.get("numeric"):
        n = dict(col["numeric"])
        n.pop("outlier_examples", None)
        lines.append(f"Numeric summary: {_fmt(n)}")
        if col["numeric"].get("outlier_examples"):
            lines.append(f"Outlier examples: {_fmt(col['numeric']['outlier_examples'])}")
    if col.get("datetime"):
        lines.append(f"Date summary: {_fmt(col['datetime'])}")
    if col.get("boolean"):
        lines.append(f"Boolean-like values: {_fmt(col['boolean']['values'])}")
    if col.get("email"):
        lines.append(f"Email check: {_fmt(col['email'])}")
    t = col.get("text") or {}
    tb = {k: v for k, v in t.items() if k in ("leading_trailing_whitespace", "internal_double_spaces", "mojibake",
                                                "mojibake_examples", "control_chars", "min_len", "max_len", "avg_len",
                                                "non_numeric_count", "non_numeric_examples", "number_format_hints")}
    if tb:
        lines.append(f"Text summary: {_fmt(tb)}")
    if t.get("case_variant_groups"):
        g = [{"canonical": x["canonical"], "variants": x["variants"][:6]} for x in t["case_variant_groups"][:8]]
        lines.append(f"Spelling variants of the same value: {_fmt(g)}")
    if t.get("fuzzy_groups"):
        g = [{"canonical": x["canonical"], "variants": x["variants"][:6]} for x in t["fuzzy_groups"][:8]]
        lines.append(f"Possible typos (near-identical values): {_fmt(g)}")
    lines.append("")
    if issues:
        lines.append("DETECTED ISSUES:")
        for i, it in enumerate(issues, 1):
            lines.append(f"{i}. [{it['severity']}] {it['title']} - {it['detail']}")
        lines.append("")
        lines.append("CANDIDATE ACTIONS (rule-based suggestions you may keep, adjust or reject; lines marked ALTERNATIVE are mutually exclusive options for the same issue - pick at most one):")
        for it in issues:
            for k, sg in enumerate(it.get("suggestions", [])):
                params = dict(sg.get("params") or {})
                if "mapping" in params:
                    params["mapping"] = f"<{len(params['mapping'])} detected variant pairs - omit to use them>"
                tag = "  [ALTERNATIVE]" if k > 0 else ""
                lines.append(f"- {sg['action']} {_fmt(params)}  # {sg.get('reason', '')}{tag}")
    else:
        lines.append("DETECTED ISSUES: none by the automatic profiler. Decide whether anything still needs cleaning.")
    lines.append("")
    if with_header:
        lines.append("Think about this column, then output the actions for it. Use the exact column name "
                     f"{col['name']!r} in the 'column' field.")
    return "\n".join(lines)


def build_batch_prompt(profile: dict[str, Any], cols: list[dict[str, Any]], issues_by_col: dict[str, list]) -> str:
    ds = profile["dataset"]
    parts = [f"DATASET: {ds['rows']} rows × {ds['columns']} columns. Columns: {_fmt([c['name'] for c in profile['columns']])}",
             f"You are given {len(cols)} columns. Think about each one, then output actions for all of them, "
             "setting the 'column' field to the exact column name each action applies to.", ""]
    for i, c in enumerate(cols, 1):
        parts.append(f"=============== COLUMN {i} of {len(cols)} ===============")
        parts.append(build_column_prompt(profile, c, issues_by_col.get(c["name"], []), with_header=False))
    parts.append("Output the actions for every column above (empty list for columns that are clean).")
    return "\n".join(parts)


def build_dataset_prompt(profile: dict[str, Any], file_name: str) -> str:
    ds = profile["dataset"]
    lines = [f"FILE: {file_name}", f"Shape: {ds['rows']} rows × {ds['columns']} columns; {ds['missing_pct']}% of cells missing.",
             f"Exact duplicate rows: {ds.get('duplicate_rows', 0)}; duplicates ignoring case/spacing: {ds.get('duplicate_rows_normalized', 0)}",
             f"Completely empty rows: {ds.get('empty_rows', 0)}; rows ≥60% empty: {ds.get('mostly_empty_rows', 0)}",
             f"Empty columns: {_fmt(ds.get('empty_columns', []))}; constant columns: {_fmt(ds.get('constant_columns', []))}; "
             f">60% missing columns: {_fmt(ds.get('mostly_missing_columns', []))}",
             f"Identical column pairs: {_fmt(ds.get('duplicate_column_pairs', []))}",
             f"Column name problems: {_fmt([(p['name'], p['proposed']) for p in ds.get('column_name_issues', [])][:20])}",
             "", "COLUMNS:"]
    for c in profile["columns"]:
        extra = ""
        if c["inferred_type"] == "identifier":
            extra = f" | identifier, {c['unique']} unique of {c['valid']} valid"
        lines.append(f"- {c['name']!r}: {c['inferred_type']}, {c['missing_pct']}% missing, {c['unique']} unique, "
                     f"{len(c.get('issue_ids', []))} issues{extra}; samples {_fmt(c.get('samples', [])[:4])}")
    lines.append("")
    lines.append("DATASET-LEVEL ISSUES:")
    ds_issues = [i for i in profile["issues"] if i["scope"] == "dataset"]
    for i, it in enumerate(ds_issues, 1):
        lines.append(f"{i}. [{it['severity']}] {it['title']} - {it['detail']}")
    if not ds_issues:
        lines.append("(none)")
    lines.append("")
    lines.append("CANDIDATE ACTIONS:")
    for it in ds_issues:
        for k, sg in enumerate(it.get("suggestions", [])):
            tag = "  [ALTERNATIVE]" if k > 0 else ""
            lines.append(f"- {sg['action']} {_fmt(sg.get('params') or {})}  # {sg.get('reason', '')}{tag}")
    id_cols = [c["name"] for c in profile["columns"] if c["inferred_type"] == "identifier"]
    if id_cols:
        lines.append(f"- drop_rows_with_missing {_fmt({'columns': id_cols})}  # rows without an identifier are unusable")
        lines.append(f"- drop_duplicate_rows {_fmt({'subset': id_cols, 'keep': 'first'})}  # if the identifier must be unique")
    lines.append("")
    lines.append("Think about the table as a whole and output the structural actions.")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Validation of model output
# --------------------------------------------------------------------------------------

def _closest_action(name: str, allowed: list[str]) -> str | None:
    if name in allowed:
        return name
    aliases = {"fillna": "fill_missing", "fill_na": "fill_missing", "impute": "fill_missing", "impute_missing": "fill_missing",
               "drop_column": "drop_columns", "remove_column": "drop_columns", "remove_columns": "drop_columns",
               "drop_duplicates": "drop_duplicate_rows", "remove_duplicates": "drop_duplicate_rows", "dedupe": "drop_duplicate_rows",
               "to_numeric": "parse_numeric", "convert_to_numeric": "parse_numeric", "to_datetime": "parse_dates",
               "convert_to_datetime": "parse_dates", "parse_date": "parse_dates", "trim": "strip_whitespace",
               "trim_whitespace": "strip_whitespace", "lowercase": "normalize_case", "to_lower": "normalize_case",
               "standardize_categories": "normalize_categories", "merge_categories": "normalize_categories",
               "map_values": "normalize_categories", "to_boolean": "normalize_boolean", "convert_to_boolean": "normalize_boolean",
               "remove_outliers": "handle_outliers", "clip_outliers": "handle_outliers", "winsorize": "handle_outliers",
               "replace_nulls": "replace_disguised_nulls", "replace_placeholders": "replace_disguised_nulls",
               "drop_rows": "remove_rows_where", "filter_rows": "remove_rows_where", "drop_missing_rows": "drop_rows_with_missing",
               "dropna": "drop_rows_with_missing", "rename_columns": "standardize_column_names", "snake_case": "standardize_column_names"}
    key = re.sub(r"[^a-z_]", "", name.lower().replace("-", "_").replace(" ", "_"))
    if key in aliases and aliases[key] in allowed:
        return aliases[key]
    m = difflib.get_close_matches(key, allowed, n=1, cutoff=0.75)
    return m[0] if m else None


def _clean_params(action: str, params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (params or {}).items():
        if v is None or v == "" or v == [] or v == {}:
            continue
        out[k] = v
    # normalise some values the model likes to vary
    if action == "fill_missing" and "strategy" in out:
        s = str(out["strategy"]).lower().strip()
        s = {"average": "mean", "most_frequent": "mode", "most frequent": "mode", "forward_fill": "ffill",
             "backward_fill": "bfill", "forward": "ffill", "backward": "bfill", "constant_value": "constant",
             "fill_value": "constant", "0": "zero", "zeros": "zero"}.get(s, s)
        out["strategy"] = s
        if s == "constant" and "value" not in out:
            out["value"] = "unknown"
    if action == "handle_outliers":
        out["method"] = str(out.get("method", "iqr")).lower()
        out["treatment"] = str(out.get("treatment", out.get("action", "clip"))).lower()
        if out["treatment"] in ("winsorize", "winsorise", "cap"):
            out["treatment"] = "clip"
        if out["treatment"] in ("drop", "delete"):
            out["treatment"] = "remove"
        if out["treatment"] in ("nan", "missing", "blank"):
            out["treatment"] = "null"
        try:
            out["factor"] = float(out.get("factor", 1.5 if out["method"] == "iqr" else 3.0))
        except (TypeError, ValueError):
            out.pop("factor", None)
    if action == "normalize_case":
        c = str(out.get("case", "lower")).lower()
        out["case"] = c if c in ("lower", "upper", "title", "sentence") else "lower"
    if action in ("remove_rows_where", "replace_values_with_null") and "condition" in out:
        c = str(out["condition"]).lower().strip()
        c = {"<": "lt", ">": "gt", "<=": "le", ">=": "ge", "==": "eq", "=": "eq", "!=": "ne", "less_than": "lt",
             "greater_than": "gt", "equals": "eq", "not_equals": "ne", "not_in_range": "outside", "out_of_range": "outside",
             "range": "between", "in_range": "between", "matches": "regex", "isin": "in"}.get(c, c)
        out["condition"] = c
    if action == "convert_type" and "to" in out:
        out["to"] = str(out["to"]).lower()
    if action == "drop_duplicate_rows" and "keep" in out:
        out["keep"] = str(out["keep"]).lower() if str(out["keep"]).lower() in ("first", "last") else "first"
    if action == "drop_empty_rows" and "threshold" in out:
        try:
            thr = float(out["threshold"])
            out["threshold"] = thr / 100 if thr > 1 else thr
        except (TypeError, ValueError):
            out.pop("threshold")
    return out


def validate_model_actions(raw_actions: list[dict[str, Any]], *, allowed: list[str], df_columns: list[str],
                           column: str | None, profile_col: dict[str, Any] | None, warnings: list[str],
                           batch_columns: list[str] | None = None,
                           profile_cols: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Turn raw model output into safe, executable actions.

    column        - when set, every column-scoped action is forced onto this column
    batch_columns - when set (multi-column prompt), column-scoped actions must target one of these
    """
    result: list[dict[str, Any]] = []
    seen = set()
    for raw in raw_actions or []:
        if not isinstance(raw, dict):
            continue
        name = _closest_action(str(raw.get("action", "")), allowed)
        if not name:
            warnings.append(f"Ignored unknown action '{raw.get('action')}'" + (f" for column '{column}'" if column else ""))
            continue
        spec = REGISTRY[name]
        params = _clean_params(name, raw.get("params") or {})
        col = raw.get("column") or column
        if spec.scope == "column":
            if column is not None:
                col = column                     # never let a column prompt touch another column
            elif batch_columns:
                if col not in batch_columns:
                    match = difflib.get_close_matches(str(col), batch_columns, n=1, cutoff=0.7)
                    if not match:
                        warnings.append(f"Ignored {name}: column '{col}' is not one of the columns in this batch")
                        continue
                    col = match[0]
            if col not in df_columns:
                match = difflib.get_close_matches(str(col), df_columns, n=1, cutoff=0.8)
                if not match:
                    warnings.append(f"Ignored {name}: column '{col}' does not exist")
                    continue
                col = match[0]
            if profile_col is None and profile_cols:
                profile_col_for_mapping = profile_cols.get(col)
            else:
                profile_col_for_mapping = profile_col
        else:
            profile_col_for_mapping = None
            if name == "drop_columns":
                cols = params.get("columns") or ([col] if col else [])
                cols = [c for c in cols if c in df_columns] or ([column] if column in df_columns and column else [])
                if not cols:
                    warnings.append("Ignored drop_columns with no valid columns")
                    continue
                params["columns"] = cols
            if name == "drop_rows_with_missing":
                cols = [c for c in (params.get("columns") or ([col] if col else [])) if c in df_columns]
                if not cols:
                    warnings.append("Ignored drop_rows_with_missing with no valid columns")
                    continue
                params["columns"] = cols
            if name == "drop_duplicate_rows" and params.get("subset"):
                params["subset"] = [c for c in params["subset"] if c in df_columns] or None
                if params["subset"] is None:
                    params.pop("subset")
            if name == "rename_column":
                params.setdefault("column", col)
            col = col if name in ("rename_column",) else (col if spec.scope == "column" else None)
        # mapping sanity for normalize_categories
        if name == "normalize_categories" and profile_col_for_mapping is not None:
            params = _sanitize_mapping(params, profile_col_for_mapping, warnings)
        action = {"action": name, "column": col, "params": params, "reason": str(raw.get("reason") or "").strip(),
                  "confidence": _conf(raw.get("confidence")), "source": "ai", "enabled": True}
        probs = validate_action(action, None)
        if probs:
            warnings.append(f"Ignored {name} on '{col}': " + "; ".join(probs))
            continue
        key = (name, col, json.dumps(params, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        result.append(action)
    return result


def _conf(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f > 1:
        f = f / 100 if f <= 100 else 1.0
    return round(max(0.0, min(1.0, f)), 2)


def _sanitize_mapping(params: dict[str, Any], profile_col: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    text = profile_col.get("text") or {}
    known_values = set()
    for g in (text.get("case_variant_groups") or []) + (text.get("fuzzy_groups") or []):
        known_values.update(norm_key(v) for v in g["variants"])
    for v, _ in profile_col.get("top_values") or []:
        known_values.add(norm_key(v))
    for v in profile_col.get("samples") or []:
        known_values.add(norm_key(v))
    mapping = params.get("mapping")
    if not mapping:
        params.pop("mapping", None)
        return params
    cleaned = []
    for pair in mapping:
        if isinstance(pair, dict) and "from" in pair and "to" in pair:
            src, dst = str(pair["from"]), str(pair["to"])
        elif isinstance(pair, (list, tuple)) and len(pair) == 2:
            src, dst = str(pair[0]), str(pair[1])
        else:
            continue
        if norm_key(src) == norm_key(dst):
            continue
        if known_values and norm_key(src) not in known_values and profile_col.get("unique", 0) <= 500:
            warnings.append(f"Mapping '{src}' -> '{dst}' dropped: '{src}' is not a value of '{profile_col['name']}'")
            continue
        cleaned.append({"from": src, "to": dst})
    if cleaned:
        params["mapping"] = cleaned
    else:
        params.pop("mapping", None)
    return params


# --------------------------------------------------------------------------------------
# Rule-based fallback planner
# --------------------------------------------------------------------------------------

def rule_based_plan(profile: dict[str, Any]) -> dict[str, Any]:
    """A plan built purely from the profiler's primary suggestions (no LLM)."""
    actions: list[dict[str, Any]] = []
    seen = set()
    for it in profile["issues"]:
        sg = (it.get("suggestions") or [None])[0]
        if not sg:
            continue
        a = {"action": sg["action"], "column": it["column"] if REGISTRY[sg["action"]].scope == "column" else
             (it["column"] if sg["action"] in ("drop_rows_with_missing",) else None),
             "params": dict(sg.get("params") or {}), "reason": sg.get("reason", ""), "confidence": None,
             "source": "rules", "enabled": True, "issue_id": it["id"]}
        if a["action"] == "drop_rows_with_missing" and it["column"]:
            a["params"]["columns"] = [it["column"]]
            a["column"] = None
        key = (a["action"], a["column"], json.dumps(a["params"], sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        actions.append(a)
    # add a dedupe step after text normalisation if duplicates exist anywhere
    thinking = ("Rule-based plan: one default action per detected issue. Placeholders become nulls, text is trimmed and "
                "normalised, numbers/dates/booleans stored as text are converted, impossible values are blanked, "
                "outliers are clipped, missing values are imputed, and empty/duplicate rows & columns are dropped.")
    return _finish_plan(actions, source="rules", model=None, dataset_thinking=thinking, column_thoughts={},
                        summary=thinking, warnings=[], calls=[])


def _finish_plan(actions, *, source, model, dataset_thinking, column_thoughts, summary, warnings, calls) -> dict[str, Any]:
    # global de-duplication (dataset-level and column-level calls can propose the same thing)
    seen, unique = set(), []
    for a in actions:
        key = (a["action"], a.get("column"), json.dumps(a.get("params") or {}, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        unique.append(a)
    actions = unique
    for i, a in enumerate(actions, 1):
        a["id"] = f"act_{i}"
    normalized = [normalize_action(a) | {k: v for k, v in a.items() if k in ("issue_id",)} for a in actions]
    return jsonable({
        "id": uuid.uuid4().hex[:10],
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "model": model,
        "summary": summary,
        "dataset_thinking": dataset_thinking,
        "column_thoughts": column_thoughts,
        "actions": normalized,
        "warnings": warnings,
        "calls": calls,
    })


# --------------------------------------------------------------------------------------
# The planning job (runs in a background thread so the UI can poll progress)
# --------------------------------------------------------------------------------------

class PlanJob:
    def __init__(self, dataset_id: str, profile: dict[str, Any], df_columns: list[str], file_name: str, settings: Settings):
        self.id = uuid.uuid4().hex[:10]
        self.dataset_id = dataset_id
        self.profile = profile
        self.df_columns = df_columns
        self.file_name = file_name
        self.settings = settings
        self.status = "queued"
        self.error: str | None = None
        self.log: list[dict[str, Any]] = []
        self.progress = {"done": 0, "total": 0, "current": None, "stage": "queued"}
        self.partial_actions: list[dict[str, Any]] = []
        self.column_thoughts: dict[str, str] = {}
        self.dataset_thinking = ""
        self.plan: dict[str, Any] | None = None
        self.started = time.time()
        self.cancel = False
        self.session_token: str | None = None
        self.on_done = None           # callback(job) once a plan exists
        self._lock = threading.Lock()

    def _log(self, msg: str, level: str = "info"):
        with self._lock:
            self.log.append({"t": round(time.time() - self.started, 1), "level": level, "msg": msg})

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return jsonable({
                "id": self.id, "dataset_id": self.dataset_id, "status": self.status, "error": self.error,
                "log": list(self.log), "progress": dict(self.progress),
                "partial_actions": list(self.partial_actions), "column_thoughts": dict(self.column_thoughts),
                "dataset_thinking": self.dataset_thinking, "plan": self.plan,
                "elapsed": round(time.time() - self.started, 1),
            })

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    # ------------------------------------------------------------------
    def _run(self):
        self.status = "running"
        try:
            self._run_inner()
        except LLMUnavailable as exc:
            if self.settings.allow_rule_fallback:
                self._log(f"{exc} - falling back to the rule-based planner.", "warn")
                self.plan = rule_based_plan(self.profile)
                self.plan["warnings"].insert(0, str(exc))
                self.status = "done"
            else:
                self.status, self.error = "error", str(exc)
        except Exception as exc:  # noqa: BLE001
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"
            self._log(self.error, "error")
        if self.plan and self.on_done:
            try:
                self.on_done(self)
            except Exception:  # noqa: BLE001
                pass

    def _run_inner(self):
        s = self.settings
        client = make_client(s)
        st = client.status()
        if not st["reachable"]:
            raise LLMUnavailable(st.get("error") or f"Cannot reach {s.provider}")
        if not st["model_available"]:
            if s.provider == "ollama":
                raise LLMError(f"Model '{s.resolved_model()}' is not pulled. Run:  ollama pull {s.resolved_model()}")
            raise LLMUnavailable(st.get("error") or f"Model '{s.resolved_model()}' is not available at {s.provider}")
        self._log(f"Connected to {s.provider}{(' ' + st['version']) if st.get('version') else ''} - using model {s.resolved_model()}")

        profile = self.profile
        warnings: list[str] = []
        calls: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []

        issues_by_col: dict[str, list[dict]] = {}
        for it in profile["issues"]:
            if it["scope"] == "column":
                issues_by_col.setdefault(it["column"], []).append(it)
        columns = [c for c in profile["columns"] if c["inferred_type"] != "empty"
                   and (s.include_clean_columns or issues_by_col.get(c["name"]))]
        batches = [columns[i:i + max(1, int(s.columns_per_call))] for i in range(0, len(columns), max(1, int(s.columns_per_call)))]
        total = 1 + len(batches) + (1 if s.summarize_plan else 0)
        self.progress.update(total=total, done=0, stage="dataset", current="dataset structure")

        # ---- 1. dataset level
        self._log("Thinking about the dataset structure (duplicates, useless columns, names)...")
        sys_prompt = SYSTEM_DATASET.format(toolbox=toolbox_for_prompt("dataset"))
        user = build_dataset_prompt(profile, self.file_name)
        data, meta = client.chat_json(sys_prompt, user, _plan_schema(DATASET_ACTIONS))
        calls.append({"stage": "dataset", "seconds": meta["seconds"], "raw": meta["raw"], "prompt": user})
        self.dataset_thinking = str(data.get("thinking", "")).strip()
        got = validate_model_actions(data.get("actions", []), allowed=DATASET_ACTIONS, df_columns=self.df_columns,
                                     column=None, profile_col=None, warnings=warnings)
        actions.extend(got)
        # columns that are completely empty are never sent to the model - drop them by rule
        empty_cols = [c["name"] for c in profile["columns"] if c["inferred_type"] == "empty"]
        already = {c for a in actions if a["action"] == "drop_columns" for c in a["params"].get("columns", [])}
        missing_empty = [c for c in empty_cols if c not in already]
        if missing_empty:
            actions.append({"action": "drop_columns", "column": None, "params": {"columns": missing_empty},
                            "reason": "Column is completely empty.", "confidence": 1.0, "source": "rules", "enabled": True})
        with self._lock:
            self.partial_actions = list(actions)
        self._log(f"Dataset level: {len(got)} action(s) in {meta['seconds']}s")
        self.progress["done"] = 1

        # ---- 2. columns (one call per column by default; several per call when columns_per_call > 1)
        col_prompt_sys = SYSTEM_COLUMN.format(toolbox=toolbox_for_prompt())
        profile_cols = {c["name"]: c for c in profile["columns"]}
        for batch in batches:
            if self.cancel:
                raise LLMError("Cancelled")
            names = [c["name"] for c in batch]
            label = f"column '{names[0]}'" if len(batch) == 1 else f"columns {names}"
            self.progress.update(stage="column", current=names[0] if len(batch) == 1 else ", ".join(names))
            n_issues = sum(len(issues_by_col.get(n, [])) for n in names)
            self._log(f"Thinking about {label} ({batch[0]['inferred_type'] + ', ' if len(batch) == 1 else ''}{n_issues} issues)...")
            if len(batch) == 1:
                user = build_column_prompt(profile, batch[0], issues_by_col.get(names[0], []))
            else:
                user = build_batch_prompt(profile, batch, issues_by_col)
            try:
                data, meta = client.chat_json(col_prompt_sys, user, _plan_schema(COLUMN_ACTIONS),
                                              num_predict=1800 if len(batch) == 1 else 1200 + 900 * len(batch))
            except LLMError as exc:
                if isinstance(exc, LLMUnavailable):
                    raise
                warnings.append(f"{label}: {exc}; used rule-based suggestions instead")
                self._log(f"Model failed on {label} ({exc}); using rule-based suggestions", "warn")
                data, meta = {"thinking": "(model failed - rule-based fallback)", "actions": [
                    {"action": sg["action"], "column": c["name"], "params": sg.get("params"), "reason": sg.get("reason")}
                    for c in batch for it in issues_by_col.get(c["name"], []) for sg in (it.get("suggestions") or [])[:1]]}, {"seconds": 0, "raw": ""}
            calls.append({"stage": f"column:{','.join(names)}", "seconds": meta["seconds"], "raw": meta["raw"], "prompt": user})
            thought = str(data.get("thinking", "")).strip()
            if len(batch) == 1:
                got = validate_model_actions(data.get("actions", []), allowed=COLUMN_ACTIONS, df_columns=self.df_columns,
                                             column=names[0], profile_col=batch[0], warnings=warnings)
            else:
                got = validate_model_actions(data.get("actions", []), allowed=COLUMN_ACTIONS, df_columns=self.df_columns,
                                             column=None, profile_col=None, warnings=warnings,
                                             batch_columns=names, profile_cols=profile_cols)
            got = got[:12 * len(batch)]
            with self._lock:
                for n in names:
                    self.column_thoughts[n] = thought
                actions.extend(got)
                self.partial_actions = list(actions)
            self._log(f"{label}: {len(got)} action(s) in {meta['seconds']}s")
            self.progress["done"] += 1

        # ---- 3. summary
        summary = ""
        if s.summarize_plan:
            self.progress.update(stage="summary", current="summary")
            self._log("Writing the plan summary...")
            listing = "\n".join(f"- {a['action']} on {a.get('column') or a['params'].get('columns') or 'dataset'}: {a['reason']}" for a in actions) or "- (no actions)"
            try:
                data, meta = client.chat_json(SYSTEM_SUMMARY, f"DATASET: {self.file_name}, {profile['dataset']['rows']} rows.\n"
                                              f"STRUCTURAL REASONING: {self.dataset_thinking}\n\nPLANNED ACTIONS:\n{listing}",
                                              SUMMARY_SCHEMA, num_predict=600)
                summary = str(data.get("summary", "")).strip()
                calls.append({"stage": "summary", "seconds": meta["seconds"], "raw": meta["raw"]})
            except LLMError as exc:
                warnings.append(f"Summary failed: {exc}")
            self.progress["done"] += 1

        self.plan = _finish_plan(actions, source=s.provider, model=s.resolved_model(), dataset_thinking=self.dataset_thinking,
                                 column_thoughts=self.column_thoughts, summary=summary, warnings=warnings, calls=calls)
        self.progress.update(stage="done", current=None)
        self._log(f"Plan ready: {len(actions)} actions, {len(warnings)} warnings, {round(time.time() - self.started, 1)}s total")
        self.status = "done"
