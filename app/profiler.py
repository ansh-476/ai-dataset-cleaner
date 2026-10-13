"""Deep dataset profiling and issue detection.

The profiler is the "evidence engine".  It never changes the data; it produces
(1) a per-column + dataset profile, (2) a list of concrete issues with
severity and evidence, and (3) rule-based candidate actions for every issue.
The LLM planner receives all three and decides what to actually do.
"""
from __future__ import annotations

import difflib
import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .utils import (
    BOOL_TOKENS, TRUE_TOKENS, FALSE_TOKENS, CONTROL_CHAR_RE, DATE_NAME_HINTS, DISGUISED_NULL_TOKENS, EMAIL_RE, IDENTIFIER_NAME_HINTS, MOJIBAKE_RE,
    NON_NEGATIVE_NAME_HINTS, PERCENT_NAME_HINTS, PHONE_NAME_HINTS, PHONE_RE, POSTAL_NAME_HINTS,
    URL_RE, YEAR_NAME_HINTS, display_value, is_bool_series, is_datetime_series, is_disguised_null,
    is_missing_scalar, is_numeric_series, is_text_series, jsonable, name_has_hint, norm_key, number_format_hints,
    parse_number, safe_float, to_snake_case,
)

try:
    import ftfy
except ImportError:  # pragma: no cover
    ftfy = None

SAMPLE_LIMIT = 2000          # values used for expensive text checks
FUZZY_MAX_UNIQUE = 80        # fuzzy typo search only below this cardinality
DATE_PATTERNS = {
    "ISO (YYYY-MM-DD)": re.compile(r"^\d{4}-\d{2}-\d{2}"),
    "slash (D/M/Y or M/D/Y)": re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),
    "slash (Y/M/D)": re.compile(r"^\d{4}/\d{1,2}/\d{1,2}$"),
    "dash (D-M-Y or M-D-Y)": re.compile(r"^\d{1,2}-\d{1,2}-\d{2,4}$"),
    "dotted (D.M.Y)": re.compile(r"^\d{1,2}\.\d{1,2}\.\d{2,4}$"),
    "text month (Mar 4, 2021)": re.compile(r"^[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4}$"),
    "text month (4 Mar 2021)": re.compile(r"^\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{2,4}$"),
    "month-year (Mar-21 / 2021-03)": re.compile(r"^([A-Za-z]{3,9}[- ]\d{2,4}|\d{4}-\d{2})$"),
    "with time": re.compile(r"\d[ T]\d{1,2}:\d{2}"),
}
SENTINELS = (-1, -99, -999, -9999, -99999, 99, 999, 9999, 99999, 999999)


class _Issue:
    """Small helper to build issue dicts with stable ids."""

    def __init__(self):
        self.items: list[dict[str, Any]] = []
        self._n = 0

    def add(self, *, scope: str, type_: str, severity: str, title: str, detail: str,
            column: str | None = None, evidence: dict | None = None,
            suggestions: list[dict] | None = None) -> dict:
        self._n += 1
        item = {
            "id": f"issue_{self._n}",
            "scope": scope,
            "column": column,
            "type": type_,
            "severity": severity,
            "title": title,
            "detail": detail,
            "evidence": jsonable(evidence or {}),
            "suggestions": jsonable(suggestions or []),
        }
        self.items.append(item)
        return item


def _severity_from_pct(pct: float, high: float = 50, medium: float = 15) -> str:
    if pct >= high:
        return "high"
    if pct >= medium:
        return "medium"
    return "low"


# --------------------------------------------------------------------------------------
# Column profiling
# --------------------------------------------------------------------------------------

def _text_values(s: pd.Series) -> pd.Series:
    """Non-null values as python strings (not stripped)."""
    v = s.dropna()
    return v.astype(str)


def _infer_text_type(name: str, stripped: pd.Series, n_non_missing: int) -> dict[str, Any]:
    """Classify the (non-missing, stripped) text values of a column."""
    info: dict[str, Any] = {"numeric_ratio": 0.0, "date_ratio": 0.0, "bool_ratio": 0.0,
                            "email_ratio": 0.0, "url_ratio": 0.0, "phone_ratio": 0.0}
    if n_non_missing == 0:
        info["inferred_type"] = "empty"
        return info

    sample = stripped if len(stripped) <= SAMPLE_LIMIT else stripped.sample(SAMPLE_LIMIT, random_state=0)
    n = len(sample)
    parsed = sample.map(parse_number)
    numeric_mask = parsed.notna()
    info["numeric_ratio"] = float(numeric_mask.mean())
    lower = sample.str.casefold()
    info["bool_ratio"] = float(lower.isin(BOOL_TOKENS).mean())
    info["email_ratio"] = float(sample.str.match(EMAIL_RE).mean())
    info["url_ratio"] = float(sample.str.match(URL_RE).mean())
    info["phone_ratio"] = float(sample.str.match(PHONE_RE).mean())
    nunique = stripped.nunique()
    unique_ratio = nunique / max(n_non_missing, 1)
    avg_len = float(sample.str.len().mean())
    info.update({"nunique": int(nunique), "unique_ratio": float(unique_ratio), "avg_len": avg_len})

    # ---- dates: only try when it does not look numeric and lengths are plausible
    date_ratio, date_meta = 0.0, {}
    if info["numeric_ratio"] < 0.6 and 5 <= avg_len <= 35:
        date_ratio, date_meta = _try_dates(sample)
    info["date_ratio"] = date_ratio
    info["date_meta"] = date_meta

    name_is_id = name_has_hint(name, IDENTIFIER_NAME_HINTS)
    name_is_phone = name_has_hint(name, PHONE_NAME_HINTS)
    name_is_postal = name_has_hint(name, POSTAL_NAME_HINTS)
    name_is_year = name_has_hint(name, YEAR_NAME_HINTS)

    uniq_lower = set(lower.unique().tolist())
    if len(uniq_lower) >= 2 and uniq_lower <= BOOL_TOKENS and (uniq_lower & TRUE_TOKENS) \
            and (uniq_lower & FALSE_TOKENS) and info["bool_ratio"] >= 0.95 \
            and not (uniq_lower <= {"0", "1", "0.0", "1.0"} and name_is_id):
        t = "boolean_text"
    elif name_is_id and unique_ratio >= 0.7 and not info["numeric_ratio"] >= 0.85:
        t = "identifier"
    elif info["email_ratio"] >= 0.7:
        t = "email"
    elif info["url_ratio"] >= 0.7:
        t = "url"
    elif name_is_phone and info["phone_ratio"] >= 0.6:
        t = "phone"
    elif name_is_postal and info["numeric_ratio"] >= 0.6:
        t = "postal_code"
    elif date_ratio >= 0.8:
        t = "datetime_text"
    elif info["numeric_ratio"] >= 0.85:
        if name_is_id and unique_ratio >= 0.7:
            t = "identifier"
        elif name_is_year:
            t = "year"
        else:
            t = "numeric_text"
    elif 0.3 <= info["numeric_ratio"] < 0.85:
        t = "mixed"
    elif unique_ratio >= 0.97 and n_non_missing >= 20 and (name_is_id or avg_len <= 24):
        t = "identifier"
    elif nunique <= max(25, 0.05 * n_non_missing) and avg_len <= 50:
        t = "categorical"
    elif avg_len > 40 or unique_ratio > 0.6:
        t = "free_text"
    else:
        t = "categorical"
    info["inferred_type"] = t
    return info


def _try_dates(sample: pd.Series) -> tuple[float, dict[str, Any]]:
    """Return (parse success ratio, metadata) for a text sample."""
    best_ratio, best_meta = 0.0, {}
    # Avoid the pandas "could not infer format" spam.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dayfirst in (False, True):
            try:
                try:
                    parsed = pd.to_datetime(sample, errors="coerce", format="mixed", dayfirst=dayfirst)
                except (TypeError, ValueError):
                    parsed = pd.to_datetime(sample, errors="coerce", dayfirst=dayfirst)
            except Exception:  # noqa: BLE001
                continue
            ratio = float(parsed.notna().mean())
            if ratio > best_ratio + 1e-9:
                best_ratio = ratio
                valid = parsed.dropna()
                best_meta = {
                    "dayfirst": dayfirst,
                    "min": valid.min().isoformat() if len(valid) else None,
                    "max": valid.max().isoformat() if len(valid) else None,
                }
                if len(valid):
                    now = pd.Timestamp(datetime.now())
                    best_meta["future_count"] = int((valid > now + pd.Timedelta(days=366)).sum())
                    best_meta["before_1900_count"] = int((valid < pd.Timestamp("1900-01-01")).sum())
            elif abs(ratio - best_ratio) < 1e-9 and ratio > 0 and dayfirst:
                best_meta["dayfirst_ambiguous"] = True
    if best_ratio > 0:
        fam = Counter()
        for v in sample.tolist():
            for label, pat in DATE_PATTERNS.items():
                if label == "with time":
                    continue
                if pat.search(v):
                    fam[label] += 1
                    break
        best_meta["formats"] = dict(fam)
        best_meta["with_time_count"] = int(sample.str.contains(DATE_PATTERNS["with time"]).sum())
    return best_ratio, best_meta


def _clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def pick_canonical(members: list[tuple[str, int]]) -> str:
    """Choose the best spelling for a group of variants.

    Prefers the most frequent variant, but breaks near-ties in favour of a
    tidy form (no stray whitespace, Title Case / Capitalised over all-lower or
    all-upper).  The returned value always has whitespace collapsed.
    """
    if not members:
        return ""
    top = max(c for _, c in members)

    def score(m):
        raw, cnt = m
        s = cnt / top
        tidy = raw == _clean_spaces(raw)
        s += 0.15 if tidy else 0
        if raw.istitle() or (raw[:1].isupper() and not raw.isupper()):
            s += 0.1
        elif raw.isupper() and len(raw) > 3:
            s -= 0.05
        return s

    best = max(members, key=score)[0]
    return _clean_spaces(best)


def _variant_groups(values: pd.Series) -> list[dict[str, Any]]:
    """Groups of raw values that collapse to the same normalised key (case/space/punct)."""
    counts = values.value_counts()
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for raw, cnt in counts.items():
        groups[norm_key(raw)].append((str(raw), int(cnt)))
    out = []
    for key, members in groups.items():
        if len(members) > 1 and key:
            members.sort(key=lambda m: -m[1])
            out.append({
                "canonical": pick_canonical(members),
                "variants": [m[0] for m in members],
                "counts": {m[0]: m[1] for m in members},
                "kind": "case_or_whitespace",
            })
    out.sort(key=lambda g: -sum(g["counts"].values()))
    return out[:40]


def _fuzzy_groups(values: pd.Series, already: list[dict]) -> list[dict[str, Any]]:
    """Possible typos: distinct normalised keys that are very similar (e.g. 'New Yrok')."""
    counts = values.value_counts()
    key_members: dict[str, list[tuple[str, int]]] = defaultdict(list)
    key_counts: Counter = Counter()
    for raw, cnt in counts.items():
        k = norm_key(raw)
        if not k:
            continue
        key_counts[k] += int(cnt)
        key_members[k].append((str(raw), int(cnt)))
    key_to_best = {k: (pick_canonical(m), key_counts[k]) for k, m in key_members.items()}
    keys = list(key_counts)
    if len(keys) < 2 or len(keys) > FUZZY_MAX_UNIQUE:
        return []
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def similar(a: str, b: str) -> bool:
        if len(a) < 4 or len(b) < 4:
            return False
        if a.isdigit() or b.isdigit():
            return False
        r = difflib.SequenceMatcher(None, a, b).ratio()
        if r < 0.85:
            return False
        ta, tb = a.split(), b.split()
        if len(ta) == len(tb):
            for x, y in zip(ta, tb):
                if x != y:
                    # "product a" vs "product b" -> differing tokens are too different
                    if difflib.SequenceMatcher(None, x, y).ratio() < 0.6:
                        return False
                    if x.isdigit() or y.isdigit():
                        return False
            return True
        return r >= 0.9

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if similar(keys[i], keys[j]):
                parent[find(keys[i])] = find(keys[j])
    clusters: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        clusters[find(k)].append(k)
    out = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda k: -key_counts[k])
        canonical = key_to_best[members[0]][0]
        variants = [key_to_best[k][0] for k in members]
        out.append({
            "canonical": canonical,
            "variants": variants,
            "counts": {key_to_best[k][0]: key_counts[k] for k in members},
            "kind": "possible_typo",
        })
    out.sort(key=lambda g: -sum(g["counts"].values()))
    return out[:25]


def _numeric_stats(name: str, nums: pd.Series, is_integer_like: bool) -> dict[str, Any]:
    nums = nums.dropna().astype(float)
    if nums.empty:
        return {}
    q1, q3 = nums.quantile(0.25), nums.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    xlo, xhi = q1 - 3 * iqr, q3 + 3 * iqr
    std = float(nums.std()) if len(nums) > 1 else 0.0
    mean = float(nums.mean())
    z = (nums - mean) / std if std > 0 else pd.Series(np.zeros(len(nums)), index=nums.index)
    out_iqr = nums[(nums < lo) | (nums > hi)] if iqr > 0 else nums.iloc[0:0]
    out_ext = nums[(nums < xlo) | (nums > xhi)] if iqr > 0 else nums.iloc[0:0]
    out_z = nums[z.abs() > 3]
    stats = {
        "min": safe_float(nums.min()), "max": safe_float(nums.max()), "mean": safe_float(mean),
        "median": safe_float(nums.median()), "std": safe_float(std),
        "skew": safe_float(nums.skew()) if len(nums) > 2 else 0.0,
        "q1": safe_float(q1), "q3": safe_float(q3),
        "zeros": int((nums == 0).sum()), "negatives": int((nums < 0).sum()),
        "is_integer": bool(is_integer_like),
        "iqr_bounds": [safe_float(lo), safe_float(hi)],
        "outliers_iqr": int(len(out_iqr)), "outliers_extreme": int(len(out_ext)),
        "outliers_z": int(len(out_z)),
        "outlier_examples": [safe_float(v) for v in pd.concat([out_iqr.nsmallest(3), out_iqr.nlargest(3)]).unique().tolist()][:6],
    }
    sent = {}
    for sv in SENTINELS:
        c = int((nums == sv).sum())
        if c:
            others = nums[nums != sv]
            if others.empty:
                continue
            far = (sv < others.min() - 2 * max(iqr, 1)) or (sv > others.max() + 2 * max(iqr, 1))
            if far or (sv < 0 and (others >= 0).all() and name_has_hint(name, NON_NEGATIVE_NAME_HINTS)):
                sent[str(sv)] = c
    if sent:
        stats["sentinel_values"] = sent
    return stats


def profile_column(df: pd.DataFrame, col: str, issues: _Issue) -> tuple[dict[str, Any], pd.Series]:
    """Profile one column. Returns (profile dict, missing mask incl. disguised nulls)."""
    s = df[col]
    n = len(s)
    prof: dict[str, Any] = {"name": col, "position": int(df.columns.get_loc(col)),
                            "pandas_dtype": str(s.dtype), "rows": n}
    real_missing = s.isna()
    disguised_counter: Counter = Counter()
    text_mode = is_text_series(s)

    if text_mode:
        as_str = s.map(lambda v: None if is_missing_scalar(v) else str(v)).astype(object)
        stripped = as_str.map(lambda v: None if is_missing_scalar(v) else v.strip()).astype(object)
        disguised_mask = stripped.map(lambda v: (not is_missing_scalar(v)) and v.casefold() in DISGUISED_NULL_TOKENS).astype(bool)
        for v in stripped[disguised_mask].tolist():
            disguised_counter[v] += 1
        missing_mask = real_missing | disguised_mask
        values = stripped[~missing_mask].astype(str)
    else:
        missing_mask = real_missing.copy()
        values = s[~missing_mask]

    n_missing_real = int(real_missing.sum())
    n_disguised = int(sum(disguised_counter.values()))
    n_missing = int(missing_mask.sum())
    n_valid = n - n_missing
    prof.update({
        "missing": n_missing, "missing_real": n_missing_real, "missing_disguised": n_disguised,
        "missing_pct": round(100.0 * n_missing / max(n, 1), 2),
        "disguised_tokens": dict(disguised_counter.most_common(10)),
        "valid": n_valid,
    })

    # unique / top values
    if n_valid:
        vc = values.value_counts()
        prof["unique"] = int(vc.shape[0])
        prof["unique_pct"] = round(100.0 * vc.shape[0] / n_valid, 2)
        prof["top_values"] = [[display_value(k), int(v)] for k, v in vc.head(10).items()]
        prof["samples"] = [display_value(v) for v in values.drop_duplicates().head(8).tolist()]
    else:
        prof.update({"unique": 0, "unique_pct": 0.0, "top_values": [], "samples": []})

    # ---------------- typed branches ----------------
    inferred = "empty"
    numeric_series: pd.Series | None = None
    if n_valid == 0:
        inferred = "empty"
    elif is_numeric_series(s):
        inferred = "integer" if pd.api.types.is_integer_dtype(s) else "float"
        numeric_series = values.astype(float)
        if name_has_hint(col, IDENTIFIER_NAME_HINTS) and prof["unique_pct"] >= 95:
            inferred = "identifier"
        elif name_has_hint(col, YEAR_NAME_HINTS):
            inferred = "year"
    elif is_datetime_series(s):
        inferred = "datetime"
        valid = pd.to_datetime(values, errors="coerce")
        now = pd.Timestamp(datetime.now())
        prof["datetime"] = {
            "min": valid.min().isoformat() if valid.notna().any() else None,
            "max": valid.max().isoformat() if valid.notna().any() else None,
            "future_count": int((valid > now + pd.Timedelta(days=366)).sum()),
            "before_1900_count": int((valid < pd.Timestamp("1900-01-01")).sum()),
        }
    elif is_bool_series(s):
        inferred = "boolean"
    else:
        info = _infer_text_type(col, values, n_valid)
        inferred = info["inferred_type"]
        prof["type_scores"] = {k: round(v, 3) for k, v in info.items()
                               if k.endswith("_ratio") and isinstance(v, float)}
        text_block: dict[str, Any] = {}
        raw_nonmissing = as_str[~missing_mask].astype(str)
        ws = int((raw_nonmissing != raw_nonmissing.str.strip()).sum())
        dbl = int(raw_nonmissing.str.contains(r"\s{2,}", regex=True).sum())
        text_block["leading_trailing_whitespace"] = ws
        text_block["internal_double_spaces"] = dbl
        text_block["control_chars"] = int(raw_nonmissing.str.contains(CONTROL_CHAR_RE, regex=True).sum())
        moj = int(raw_nonmissing.str.contains(MOJIBAKE_RE, regex=True).sum())
        if moj == 0 and ftfy is not None and len(raw_nonmissing):
            sample = raw_nonmissing.head(300)
            moj_ftfy = int(sum(1 for v in sample.tolist() if ftfy.fix_text(v) != v and any(ord(ch) > 127 for ch in v)))
            if moj_ftfy:
                moj = moj_ftfy
                text_block["mojibake_estimated"] = True
        text_block["mojibake"] = moj
        if moj:
            text_block["mojibake_examples"] = [display_value(v) for v in raw_nonmissing[raw_nonmissing.str.contains(MOJIBAKE_RE, regex=True)].head(4).tolist()]
        lens = values.str.len()
        text_block.update({"min_len": int(lens.min()), "max_len": int(lens.max()), "avg_len": round(float(lens.mean()), 1)})
        if inferred in ("categorical", "boolean_text", "mixed", "free_text", "identifier", "postal_code", "url"):
            if prof["unique"] <= 500:
                groups = _variant_groups(values)
                text_block["case_variant_groups"] = groups
                if inferred in ("categorical", "mixed") and prof["unique"] <= FUZZY_MAX_UNIQUE:
                    text_block["fuzzy_groups"] = _fuzzy_groups(values, groups)
        if inferred in ("numeric_text", "mixed", "year", "postal_code", "identifier"):
            parsed = values.map(parse_number)
            non_num = values[parsed.isna()]
            text_block["non_numeric_count"] = int(len(non_num))
            text_block["non_numeric_examples"] = [display_value(v) for v in non_num.drop_duplicates().head(8).tolist()]
            hints = Counter()
            for v in values.head(SAMPLE_LIMIT).tolist():
                for k, flag in number_format_hints(v).items():
                    if flag:
                        hints[k] += 1
            text_block["number_format_hints"] = dict(hints)
            if inferred in ("numeric_text", "year") and parsed.notna().any():
                numeric_series = parsed.dropna().astype(float)
        if inferred == "datetime_text":
            prof["datetime"] = info.get("date_meta", {})
            prof["datetime"]["parse_rate"] = round(info["date_ratio"], 3)
            unparsed = None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    p = pd.to_datetime(values, errors="coerce", format="mixed",
                                       dayfirst=bool(info.get("date_meta", {}).get("dayfirst", False)))
                except (TypeError, ValueError):
                    p = pd.to_datetime(values, errors="coerce")
                unparsed = values[p.isna()]
            prof["datetime"]["unparseable_count"] = int(len(unparsed))
            prof["datetime"]["unparseable_examples"] = [display_value(v) for v in unparsed.drop_duplicates().head(6).tolist()]
        if inferred == "boolean_text":
            prof["boolean"] = {"values": sorted(values.str.casefold().unique().tolist())}
        if inferred == "phone":
            text_block["formatted_count"] = int(values.str.contains(r"[^\d+]", regex=True).sum())
        if inferred == "email":
            bad = values[~values.str.match(EMAIL_RE)]
            prof["email"] = {"invalid_count": int(len(bad)),
                             "invalid_examples": [display_value(v) for v in bad.drop_duplicates().head(5).tolist()],
                             "uppercase_count": int((values != values.str.lower()).sum())}
        prof["text"] = text_block

    prof["inferred_type"] = inferred

    if numeric_series is not None and len(numeric_series):
        int_like = bool(np.allclose(numeric_series, np.round(numeric_series)))
        prof["numeric"] = _numeric_stats(col, numeric_series, int_like)

    _column_issues(df, col, prof, issues)
    return prof, missing_mask


# --------------------------------------------------------------------------------------
# Issue detection per column
# --------------------------------------------------------------------------------------

def _column_issues(df: pd.DataFrame, col: str, p: dict[str, Any], issues: _Issue) -> None:
    n = p["rows"]
    t = p["inferred_type"]
    text = p.get("text", {})
    num = p.get("numeric", {})
    col_issue_ids: list[str] = []

    def add(**kw):
        kw.setdefault("scope", "column")
        kw["column"] = col
        item = issues.add(**kw)
        col_issue_ids.append(item["id"])
        return item

    # --- empty / mostly empty column
    if p["valid"] == 0:
        add(type_="empty_column", severity="high", title="Column is completely empty",
            detail=f"All {n} values are missing or placeholders.",
            suggestions=[{"action": "drop_columns", "params": {"columns": [col]}, "reason": "Column has no data."}])
        p["issue_ids"] = col_issue_ids
        return
    if p["missing_pct"] >= 60:
        add(type_="mostly_missing", severity="high",
            title=f"{p['missing_pct']}% of values are missing",
            detail=f"{p['missing']} of {n} values are missing ({p['missing_disguised']} are placeholder text).",
            evidence={"missing_pct": p["missing_pct"]},
            suggestions=[{"action": "drop_columns", "params": {"columns": [col]},
                          "reason": "More than 60% missing - usually too sparse to impute reliably."}])
    elif p["missing"] > 0:
        pct = p["missing_pct"]
        strategy = "median" if num else ("mode" if t in ("categorical", "boolean_text", "boolean") else "mode")
        if num and abs(num.get("skew") or 0) < 0.5:
            strategy = "mean"
        sugg = [{"action": "fill_missing", "params": {"strategy": strategy},
                 "reason": f"Fill {pct}% missing values using {strategy}."}]
        if t == "identifier":
            sugg = [{"action": "drop_rows_with_missing", "params": {"columns": [col]},
                     "reason": "Rows without an identifier are usually unusable."}]
        elif t == "free_text":
            sugg = [{"action": "fill_missing", "params": {"strategy": "constant", "value": "not provided"},
                     "reason": "Free text cannot be statistically imputed; mark as 'not provided' or leave as null."}]
        elif t in ("email", "url", "phone"):
            sugg = []  # contact details cannot be imputed - leave missing
        elif t == "datetime_text" or t == "datetime":
            sugg = []  # dates are usually best left missing unless the rows form an ordered time series (then ffill)
        add(type_="missing_values", severity=_severity_from_pct(pct, 40, 10),
            title=f"{pct}% missing values ({p['missing']} rows)",
            detail=f"{p['missing_real']} truly empty and {p['missing_disguised']} placeholder values.",
            evidence={"missing_pct": pct, "missing": p["missing"]}, suggestions=sugg)

    # --- disguised nulls
    if p["missing_disguised"] > 0:
        add(type_="disguised_nulls", severity="medium" if p["missing_disguised"] > 0.02 * n else "low",
            title=f"{p['missing_disguised']} placeholder values that mean 'missing'",
            detail="Values like " + ", ".join(f"'{k}' ({v})" for k, v in list(p["disguised_tokens"].items())[:5]) +
                   " should be converted to real nulls before imputation.",
            evidence={"tokens": p["disguised_tokens"]},
            suggestions=[{"action": "replace_disguised_nulls", "params": {"tokens": list(p["disguised_tokens"].keys())},
                          "reason": "Convert placeholder strings to proper missing values."}])

    # --- constant / near constant
    if p["unique"] == 1 and p["valid"] >= 2:
        add(type_="constant_column", severity="medium", title="Column has a single constant value",
            detail=f"Every non-missing value is '{p['top_values'][0][0]}'. It carries no information.",
            suggestions=[{"action": "drop_columns", "params": {"columns": [col]}, "reason": "Constant column adds no information."}])
    elif p["top_values"] and p["valid"] >= 50 and p["top_values"][0][1] / p["valid"] >= 0.99 and p["unique"] > 1:
        add(type_="near_constant_column", severity="low", title="Column is almost constant",
            detail=f"'{p['top_values'][0][0]}' makes up {round(100 * p['top_values'][0][1] / p['valid'], 1)}% of values.",
            suggestions=[{"action": "drop_columns", "params": {"columns": [col]}, "reason": "Nearly constant; consider dropping."}])

    # --- text hygiene
    if text:
        if text.get("leading_trailing_whitespace", 0) or text.get("internal_double_spaces", 0):
            c = text.get("leading_trailing_whitespace", 0) + text.get("internal_double_spaces", 0)
            add(type_="whitespace", severity="low" if c < 0.05 * n else "medium",
                title=f"{c} values with stray whitespace",
                detail=f"{text.get('leading_trailing_whitespace', 0)} values have leading/trailing spaces and "
                       f"{text.get('internal_double_spaces', 0)} contain double spaces.",
                suggestions=[{"action": "strip_whitespace", "params": {}, "reason": "Trim and collapse whitespace."}])
        if text.get("control_chars", 0):
            add(type_="control_characters", severity="low", title=f"{text['control_chars']} values contain control characters",
                detail="Non-printable characters (tabs, NULs...) found inside values.",
                suggestions=[{"action": "remove_special_characters", "params": {"mode": "control"}, "reason": "Remove non-printable characters."}])
        if text.get("mojibake", 0):
            add(type_="encoding_damage", severity="medium",
                title=f"{text['mojibake']} values look like broken encoding (mojibake)",
                detail="Sequences such as 'Ã©' or 'â€™' indicate UTF-8 text decoded as Latin-1. Examples: " +
                       ", ".join(str(x) for x in text.get("mojibake_examples", [])[:3]),
                evidence={"examples": text.get("mojibake_examples", [])},
                suggestions=[{"action": "fix_encoding", "params": {}, "reason": "Repair mojibake with ftfy-style fixing."}])
        groups = text.get("case_variant_groups") or []
        if groups:
            total = sum(sum(g["counts"].values()) - max(g["counts"].values()) for g in groups)
            ex = "; ".join(" / ".join(repr(v) for v in g["variants"][:4]) for g in groups[:3])
            if t == "free_text":
                sugg = [{"action": "normalize_case", "params": {"case": "title"},
                         "reason": "Names/free text: make capitalisation consistent (Title Case)."}]
            else:
                sugg = [{"action": "normalize_categories", "params": {
                    "mapping": [{"from": v, "to": g["canonical"]} for g in groups[:40] for v in g["variants"] if v != g["canonical"]]},
                    "reason": "Map variants to the most frequent spelling."}]
            add(type_="inconsistent_categories", severity="medium" if total >= 0.02 * n else "low",
                title=f"{len(groups)} {'values' if t == 'free_text' else 'categories'} written in several ways",
                detail=f"Same value with different case/spacing/punctuation, e.g. {ex}.",
                evidence={"groups": groups[:10]},
                suggestions=sugg)
        fuzzy = text.get("fuzzy_groups") or []
        if fuzzy:
            ex = "; ".join(" / ".join(repr(v) for v in g["variants"][:3]) for g in fuzzy[:3])
            add(type_="possible_typos", severity="low",
                title=f"{len(fuzzy)} groups of near-identical categories (possible typos)",
                detail=f"Values that differ by one or two characters, e.g. {ex}. Review before merging.",
                evidence={"groups": fuzzy[:10]},
                suggestions=[{"action": "normalize_categories", "params": {
                    "mapping": [{"from": v, "to": g["canonical"]} for g in fuzzy[:25] for v in g["variants"] if v != g["canonical"]]},
                    "reason": "Merge probable typos into the most frequent spelling."}])

    # --- type problems
    if t == "numeric_text":
        hints = text.get("number_format_hints", {})
        extras = [k for k, v in hints.items() if v]
        add(type_="numeric_stored_as_text", severity="high",
            title="Numbers stored as text" + (f" (with {', '.join(extras)})" if extras else ""),
            detail=f"{round(100 * p['type_scores'].get('numeric_ratio', 0), 1)}% of values parse as numbers. "
                   + (f"{text.get('non_numeric_count', 0)} values do not, e.g. {text.get('non_numeric_examples', [])[:4]}." if text.get("non_numeric_count") else ""),
            evidence={"format_hints": hints, "non_numeric_examples": text.get("non_numeric_examples", [])},
            suggestions=[{"action": "parse_numeric", "params": {}, "reason": "Strip symbols/separators and convert to a numeric type."}])
    elif t == "mixed":
        add(type_="mixed_types", severity="high",
            title="Column mixes numbers and text",
            detail=f"{round(100 * p['type_scores'].get('numeric_ratio', 0), 1)}% numeric; non-numeric examples: {text.get('non_numeric_examples', [])[:5]}.",
            evidence={"non_numeric_examples": text.get("non_numeric_examples", [])},
            suggestions=[{"action": "parse_numeric", "params": {}, "reason": "Convert to numeric; unparseable values become missing."},
                         {"action": "normalize_case", "params": {"case": "lower"}, "reason": "If it is really categorical, standardise the text instead."}])
    elif t == "datetime_text":
        dm = p.get("datetime", {})
        fam = dm.get("formats", {})
        sev = "high" if len(fam) > 1 or dm.get("unparseable_count") else "medium"
        add(type_="date_stored_as_text", severity=sev,
            title="Dates stored as text" + (f" in {len(fam)} different formats" if len(fam) > 1 else ""),
            detail=f"Formats seen: {fam}. {dm.get('unparseable_count', 0)} values cannot be parsed"
                   + (f", e.g. {dm.get('unparseable_examples', [])[:3]}" if dm.get("unparseable_count") else "") + ".",
            evidence={"formats": fam, "dayfirst": dm.get("dayfirst"), "dayfirst_ambiguous": dm.get("dayfirst_ambiguous", False)},
            suggestions=[{"action": "parse_dates", "params": {"dayfirst": bool(dm.get("dayfirst", False))},
                          "reason": "Parse to a real datetime type (ISO format)."}])
    elif t == "boolean_text":
        add(type_="boolean_stored_as_text", severity="medium",
            title="Yes/No style values stored as text",
            detail=f"Values: {p.get('boolean', {}).get('values')}. Convert to a true boolean column.",
            suggestions=[{"action": "normalize_boolean", "params": {}, "reason": "Map yes/no/true/false/1/0 to booleans."}])
    elif t == "email":
        em = p.get("email", {})
        if em.get("invalid_count") or em.get("uppercase_count"):
            add(type_="email_format", severity="low",
                title=f"{em.get('invalid_count', 0)} invalid and {em.get('uppercase_count', 0)} non-lowercase emails",
                detail=f"Invalid examples: {em.get('invalid_examples', [])[:3]}",
                suggestions=[{"action": "standardize_email", "params": {}, "reason": "Lowercase, trim and blank invalid addresses."}])
    elif t == "phone":
        formatted = int(text.get("formatted_count", 0))
        if formatted:
            add(type_="phone_format", severity="low", title=f"{formatted} phone numbers contain spaces, dots or brackets",
                detail="Keep digits only (and leading +) so numbers compare equal.",
                suggestions=[{"action": "standardize_phone", "params": {}, "reason": "Normalise phone number formatting."}])

    # --- date plausibility (text dates and real datetime columns alike)
    dm = p.get("datetime") or {}
    if t in ("datetime", "datetime_text"):
        if dm.get("future_count"):
            add(type_="implausible_dates", severity="medium", title=f"{dm['future_count']} dates are more than a year in the future",
                detail="Future-dated records are often typos (e.g. wrong year).",
                suggestions=[{"action": "remove_rows_where", "params": {"condition": "gt", "value": f"{datetime.now().year + 1}-12-31"}, "reason": "Drop rows with impossible future dates."},
                             {"action": "replace_values_with_null", "params": {"condition": "gt", "value": f"{datetime.now().year + 1}-12-31"}, "reason": "Or blank the impossible dates and keep the rows."}])
        if dm.get("before_1900_count"):
            add(type_="implausible_dates", severity="medium", title=f"{dm['before_1900_count']} dates are before 1900",
                detail="Very old dates are usually parsing artefacts or placeholders (e.g. 01/01/1900).",
                suggestions=[{"action": "replace_values_with_null", "params": {"condition": "lt", "value": "1900-01-02"}, "reason": "Blank placeholder dates."},
                             {"action": "remove_rows_where", "params": {"condition": "lt", "value": "1900-01-02"}, "reason": "Or drop those rows."}])

    # --- numeric plausibility
    if num:
        if num.get("sentinel_values"):
            add(type_="sentinel_values", severity="medium",
                title=f"Suspicious sentinel values: {num['sentinel_values']}",
                detail="Values like -999 or 9999 are usually codes for 'missing', not real measurements.",
                evidence={"sentinels": num["sentinel_values"]},
                suggestions=[{"action": "replace_values_with_null", "params": {"values": list(num["sentinel_values"].keys())},
                              "reason": "Treat sentinel codes as missing."}])
        if num.get("negatives", 0) and name_has_hint(col, NON_NEGATIVE_NAME_HINTS):
            add(type_="negative_values", severity="medium",
                title=f"{num['negatives']} negative values in a column that should be non-negative",
                detail=f"Minimum is {num['min']}. Column name suggests a non-negative quantity.",
                suggestions=[{"action": "remove_rows_where", "params": {"condition": "lt", "value": 0}, "reason": "Drop impossible negative records."},
                             {"action": "replace_values_with_null", "params": {"condition": "lt", "value": 0}, "reason": "Or blank them and impute."}])
        if name_has_hint(col, ("age",)) and (num.get("max", 0) or 0) > 120:
            add(type_="implausible_values", severity="medium", title=f"Ages up to {num['max']} found",
                detail="Human ages above 120 are not plausible.",
                suggestions=[{"action": "replace_values_with_null", "params": {"condition": "gt", "value": 120}, "reason": "Blank impossible ages, then impute."}])
        if ("%" in col or name_has_hint(col, PERCENT_NAME_HINTS) or (text.get("number_format_hints", {}).get("percent", 0) > 0.5 * p["valid"])) \
                and (num.get("max", 0) or 0) > 100 and (num.get("min", 0) or 0) >= 0 and (num.get("q3") or 0) <= 100:
            add(type_="implausible_values", severity="low", title=f"Percentage column has values above 100 (max {num['max']})",
                detail="Most values are within 0-100 but some exceed it.",
                suggestions=[{"action": "clip_values", "params": {"lower": 0, "upper": 100}, "reason": "Clip to the valid 0-100 range."}])
        if name_has_hint(col, YEAR_NAME_HINTS) and ((num.get("min") or 1900) < 1800 or (num.get("max") or 2000) > datetime.now().year + 5):
            add(type_="implausible_values", severity="medium", title="Year values outside a plausible range",
                detail=f"Range {num.get('min')} - {num.get('max')}.",
                suggestions=[{"action": "replace_values_with_null", "params": {"condition": "outside", "lower": 1800, "upper": datetime.now().year + 5}, "reason": "Blank impossible years."}])
        skip_outliers = t in ("identifier", "year", "postal_code", "phone") or name_has_hint(col, IDENTIFIER_NAME_HINTS)
        if not skip_outliers and p["valid"] >= 20 and p["unique"] > 5 and num.get("outliers_iqr", 0):
            cnt = num["outliers_iqr"]
            extreme = num.get("outliers_extreme", 0)
            pct = 100 * cnt / p["valid"]
            if extreme:
                sugg = [{"action": "handle_outliers", "params": {"method": "iqr", "treatment": "clip", "factor": 3.0},
                         "reason": "Winsorise (clip) the extreme values (beyond 3×IQR) to the fence."},
                        {"action": "handle_outliers", "params": {"method": "iqr", "treatment": "remove", "factor": 3.0},
                         "reason": "Or remove the extreme rows if they are clearly errors."},
                        {"action": "handle_outliers", "params": {"method": "iqr", "treatment": "clip", "factor": 1.5},
                         "reason": "Or clip more aggressively at the usual 1.5×IQR fence."}]
            else:
                sugg = []  # mild outliers in a plausible range are usually real data - leave them
            add(type_="outliers", severity="medium" if extreme else "low",
                title=f"{cnt} statistical outliers ({pct:.1f}%)" + (f", {extreme} extreme" if extreme else ""),
                detail=f"Outside IQR fence {num['iqr_bounds']}; {extreme} are extreme (3×IQR). "
                       f"Examples: {num.get('outlier_examples', [])}. Mean {num['mean']} vs median {num['median']}."
                       + ("" if extreme else " Mild outliers only - probably genuine values."),
                evidence={"iqr_bounds": num["iqr_bounds"], "examples": num.get("outlier_examples", []),
                          "extreme": extreme, "skew": num.get("skew")},
                suggestions=sugg)
        if num.get("is_integer") and p["pandas_dtype"].startswith("float") and t in ("float",):
            add(type_="float_should_be_int", severity="low", title="Whole numbers stored as float",
                detail="All values are integers; stored as float (probably because of missing values).",
                suggestions=[{"action": "convert_type", "params": {"to": "int"}, "reason": "Convert to a nullable integer type."}])

    # --- identifier problems
    if t == "identifier" and p["unique_pct"] < 100 and p["valid"] >= 2:
        dup = p["valid"] - p["unique"]
        add(type_="duplicate_ids", severity="high", title=f"{dup} duplicated identifier values",
            detail="An ID column should be unique. Duplicate IDs usually mean duplicated records.",
            suggestions=[{"action": "drop_duplicate_rows", "params": {"subset": [col], "keep": "first"}, "reason": "Keep the first record per ID."}])

    if t == "free_text" and p["unique_pct"] > 95 and text.get("avg_len", 0) > 80:
        pass  # long free text: nothing to clean automatically

    p["issue_ids"] = col_issue_ids


# --------------------------------------------------------------------------------------
# Dataset level
# --------------------------------------------------------------------------------------

def _dataset_issues(df: pd.DataFrame, cols: list[dict], missing: pd.DataFrame, issues: _Issue) -> dict[str, Any]:
    n, k = df.shape
    ds: dict[str, Any] = {"rows": n, "columns": k, "total_cells": int(n * k)}
    ds["missing_cells"] = int(missing.values.sum()) if n and k else 0
    ds["missing_pct"] = round(100.0 * ds["missing_cells"] / max(n * k, 1), 2)
    ds["memory_mb"] = round(float(df.memory_usage(deep=True).sum()) / 1e6, 3)

    # column names
    names = list(df.columns)
    name_problems = []
    seen = Counter(names)
    for c in names:
        probs = []
        if c != c.strip():
            probs.append("surrounding whitespace")
        if c.startswith("Unnamed:") or c == "":
            probs.append("unnamed")
        if re.search(r"[^0-9a-zA-Z_]", c):
            probs.append("special characters/spaces")
        if c != c.lower() and not probs:
            probs.append("not lowercase")
        if seen[c] > 1:
            probs.append("duplicate name")
        if probs:
            name_problems.append({"name": c, "problems": probs, "proposed": to_snake_case(c)})
    ds["column_name_issues"] = name_problems
    if name_problems:
        mapping = {p["name"]: p["proposed"] for p in name_problems}
        # make proposed names unique
        used = Counter()
        for src, dst in list(mapping.items()):
            used[dst] += 1
            if used[dst] > 1:
                mapping[src] = f"{dst}_{used[dst]}"
        issues.add(scope="dataset", type_="column_names", severity="low",
                   title=f"{len(name_problems)} column names are not clean identifiers",
                   detail="Names contain spaces, symbols, mixed case or are unnamed. Example: " +
                          ", ".join(f"'{p['name']}' → '{p['proposed']}'" for p in name_problems[:4]),
                   evidence={"mapping": mapping},
                   suggestions=[{"action": "standardize_column_names", "params": {}, "reason": "Convert to snake_case for consistent access."}])

    # duplicate rows
    if n:
        dup_exact = int(df.duplicated().sum())
        norm = df.copy()
        for c in norm.columns:
            if is_text_series(norm[c]):
                norm[c] = norm[c].map(lambda v: None if is_missing_scalar(v) else norm_key(v)).astype(object)
        dup_norm = int(norm.duplicated().sum())
        ds["duplicate_rows"] = dup_exact
        ds["duplicate_rows_normalized"] = dup_norm
        if dup_exact or dup_norm:
            pct = 100 * max(dup_exact, dup_norm) / n
            issues.add(scope="dataset", type_="duplicate_rows", severity=_severity_from_pct(pct, 10, 2),
                       title=f"{dup_exact} exact duplicate rows ({dup_norm} after ignoring case/spacing)",
                       detail="Duplicate records inflate counts and bias statistics.",
                       evidence={"exact": dup_exact, "normalized": dup_norm},
                       suggestions=[{"action": "drop_duplicate_rows", "params": {"keep": "first"}, "reason": "Remove duplicate records (run after text normalisation)."}])

    # empty & mostly-empty rows
    if n and k:
        row_missing = missing.mean(axis=1)
        empty_rows = int((row_missing >= 1.0).sum())
        mostly = int(((row_missing >= 0.6) & (row_missing < 1.0)).sum())
        ds["empty_rows"] = empty_rows
        ds["mostly_empty_rows"] = mostly
        if empty_rows or mostly:
            issues.add(scope="dataset", type_="empty_rows", severity="medium" if (empty_rows + mostly) > 0.01 * n else "low",
                       title=f"{empty_rows} empty rows and {mostly} rows that are mostly empty (≥60% missing)",
                       detail="Such rows rarely carry usable information.",
                       evidence={"empty": empty_rows, "mostly_empty": mostly},
                       suggestions=[{"action": "drop_empty_rows", "params": {"threshold": 0.6}, "reason": "Drop rows with ≥60% missing values."}])

    # duplicate columns
    dup_pairs = []
    if k > 1 and n:
        hashes: dict[int, list[str]] = defaultdict(list)
        for c in df.columns:
            try:
                h = int(pd.util.hash_pandas_object(df[c].astype(str), index=False).sum())
            except Exception:  # noqa: BLE001
                continue
            hashes[h].append(c)
        for group in hashes.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if df[group[i]].astype(str).equals(df[group[j]].astype(str)):
                        dup_pairs.append([group[i], group[j]])
    ds["duplicate_column_pairs"] = dup_pairs
    if dup_pairs:
        issues.add(scope="dataset", type_="duplicate_columns", severity="medium",
                   title=f"{len(dup_pairs)} pairs of identical columns",
                   detail="; ".join(f"'{a}' == '{b}'" for a, b in dup_pairs[:5]),
                   evidence={"pairs": dup_pairs},
                   suggestions=[{"action": "drop_columns", "params": {"columns": [b for _, b in dup_pairs]}, "reason": "Keep one copy of each duplicated column."}])

    ds["constant_columns"] = [c["name"] for c in cols if c.get("unique") == 1 and c.get("valid", 0) >= 2]
    ds["empty_columns"] = [c["name"] for c in cols if c.get("valid", 0) == 0]
    ds["mostly_missing_columns"] = [c["name"] for c in cols if c.get("missing_pct", 0) >= 60 and c.get("valid", 0) > 0]
    useless = ds["empty_columns"] + ds["constant_columns"]
    if useless:
        issues.add(scope="dataset", type_="useless_columns", severity="medium",
                   title=f"{len(useless)} columns carry no information (empty or constant)",
                   detail=f"Empty: {ds['empty_columns']}; constant: {ds['constant_columns']}.",
                   evidence={"empty": ds["empty_columns"], "constant": ds["constant_columns"]},
                   suggestions=[{"action": "drop_columns", "params": {"columns": useless}, "reason": "Empty/constant columns add nothing to analysis."}])
    type_counts = Counter(c["inferred_type"] for c in cols)
    ds["type_counts"] = dict(type_counts)
    return ds


# --------------------------------------------------------------------------------------
# Quality score
# --------------------------------------------------------------------------------------

def quality_score(ds: dict[str, Any], issues: list[dict]) -> dict[str, Any]:
    penalties: dict[str, float] = {}
    penalties["missing_cells"] = min(25.0, ds.get("missing_pct", 0) * 0.6)
    n = max(ds.get("rows", 1), 1)
    dup = max(ds.get("duplicate_rows", 0), ds.get("duplicate_rows_normalized", 0))
    penalties["duplicate_rows"] = min(15.0, (100 * dup / n) * 0.8 + (2 if dup else 0))
    per_type_weight = {
        "numeric_stored_as_text": 4, "date_stored_as_text": 4, "mixed_types": 4, "boolean_stored_as_text": 2,
        "disguised_nulls": 2, "inconsistent_categories": 3, "possible_typos": 1.5, "whitespace": 1,
        "encoding_damage": 2, "outliers": 1, "sentinel_values": 2, "negative_values": 2, "implausible_values": 2,
        "implausible_dates": 1.5, "constant_column": 2, "near_constant_column": 1, "empty_column": 3,
        "mostly_missing": 2, "duplicate_columns": 2, "duplicate_ids": 4, "column_names": 2, "empty_rows": 2, "useless_columns": 0,
        "control_characters": 1, "email_format": 1, "phone_format": 0.5, "float_should_be_int": 0.5,
    }
    caps = {"outliers": 6, "whitespace": 5, "inconsistent_categories": 12, "possible_typos": 5,
            "numeric_stored_as_text": 16, "date_stored_as_text": 12, "disguised_nulls": 8}
    acc: dict[str, float] = defaultdict(float)
    for it in issues:
        if it["type"] in ("missing_values", "duplicate_rows"):
            continue
        w = per_type_weight.get(it["type"], 1)
        acc[it["type"]] += w
    for t, v in acc.items():
        penalties[t] = min(v, caps.get(t, 10))
    total = sum(penalties.values())
    score = max(0.0, min(100.0, 100.0 - total))
    grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 40 else "F"
    return {"score": round(score, 1), "grade": grade,
            "penalties": {k: round(v, 1) for k, v in sorted(penalties.items(), key=lambda kv: -kv[1]) if v > 0}}


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    issues = _Issue()
    col_profiles: list[dict] = []
    missing_frames = {}
    for col in df.columns:
        prof, miss = profile_column(df, col, issues)
        col_profiles.append(prof)
        missing_frames[col] = miss
    missing = pd.DataFrame(missing_frames, index=df.index) if len(df.columns) else pd.DataFrame(index=df.index)
    ds = _dataset_issues(df, col_profiles, missing, issues)
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    items = sorted(issues.items, key=lambda i: (sev_rank[i["severity"]], i["scope"] != "dataset"))
    ds["issue_counts"] = dict(Counter(i["severity"] for i in items))
    score = quality_score(ds, items)
    return jsonable({"dataset": ds, "columns": col_profiles, "issues": items, "score": score})


def preview_rows(df: pd.DataFrame, limit: int = 25, raw: bool = False) -> dict[str, Any]:
    head = df.head(limit)
    rows = []
    for _, r in head.iterrows():
        rows.append([display_value(v, max_len=80) for v in r.tolist()])
    return {"columns": [str(c) for c in df.columns], "rows": rows, "total_rows": int(len(df)),
            "dtypes": {str(c): str(df[c].dtype) for c in df.columns}}
