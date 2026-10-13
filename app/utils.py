"""Shared helpers used by the profiler, the planner and the cleaner."""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

# Strings that usually *mean* missing but are not recognised by pandas by default.
DISGUISED_NULL_TOKENS = {
    "", "na", "n/a", "n.a.", "n.a", "nan", "null", "none", "nil", "-", "--", "---", "?", "??", "???",
    "missing", "unknown", "undefined", "not available", "not applicable", "#n/a", "#na", "#null!",
    "(blank)", "blank", ".", "n/a.", "na.", "empty", "void", "tbd", "tba", "<na>", "[null]", "[none]",
    "notavailable", "no data", "nodata", "n\\a", "−", "—", "–",
}

TRUE_TOKENS = {"true", "t", "yes", "y", "1", "1.0", "on", "si", "oui", "ja"}
FALSE_TOKENS = {"false", "f", "no", "n", "0", "0.0", "off", "nein", "non"}
BOOL_TOKENS = TRUE_TOKENS | FALSE_TOKENS

CURRENCY_RE = r"(?:[$€£₹¥]|rs\.?|inr|usd|eur|gbp|aud|cad|rupees?|dollars?)"
_NUM_CORE = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+"
NUMBER_RE = re.compile(
    rf"^\s*(?P<cur>{CURRENCY_RE})?\s*(?P<sign>[-+−])?\s*(?P<cur2>{CURRENCY_RE})?\s*"
    rf"(?P<num>{_NUM_CORE})\s*(?P<pct>%)?\s*(?P<unit>{CURRENCY_RE}|[a-zA-Z]{{1,6}})?\s*$",
    re.IGNORECASE,
)
# European style 1.234,56
EURO_NUMBER_RE = re.compile(r"^\s*[-+]?\d{1,3}(?:\.\d{3})+,\d+\s*$")
PLAIN_NUMBER_RE = re.compile(r"^\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*$")
EMAIL_RE = re.compile(r"^[\w.%+'-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}$")
URL_RE = re.compile(r"^(https?://|www\.)\S+$", re.IGNORECASE)
PHONE_RE = re.compile(r"^\+?[\d\s().-]{7,20}$")
MOJIBAKE_RE = re.compile(
    r"(?:Ã[\x80-\xBF¡-¿]|â[\x80-\x9F€][\x80-\xBF™œ\"\x9d“”–—¦]|Â[\s\x80-\xBF]|Ã©|Ã¨|Ã¡|Ã³|Ã±|Ã¼|Ã¶|Ã¤|Ã‰|Ã¢|Ãª|Ã´|Ã§|â€™|â€œ|â€“|â€”|Â£|Â°)"
)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

NON_NEGATIVE_NAME_HINTS = (
    "age", "price", "cost", "amount", "qty", "quantity", "count", "salary", "income", "revenue",
    "height", "weight", "distance", "duration", "length", "width", "size", "fee", "charge", "total",
    "units", "stock", "population", "score", "sales", "balance", "hours", "years", "experience",
)
IDENTIFIER_NAME_HINTS = ("id", "code", "key", "uuid", "guid", "sku", "serial", "ref", "number", "no")
PHONE_NAME_HINTS = ("phone", "mobile", "tel", "contact", "cell")
POSTAL_NAME_HINTS = ("zip", "pin", "postal", "postcode")
DATE_NAME_HINTS = ("date", "time", "dob", "birth", "joined", "created", "updated", "timestamp", "day")
YEAR_NAME_HINTS = ("year", "yr")
PERCENT_NAME_HINTS = ("percent", "pct", "percentage", "rate", "ratio")


def is_text_series(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def is_datetime_series(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def is_bool_series(s: pd.Series) -> bool:
    return pd.api.types.is_bool_dtype(s)


def norm_key(value: Any) -> str:
    """Normalisation key used to detect case / whitespace / punctuation variants."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"[\s_\-\.]+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_missing_scalar(v: Any) -> bool:
    """True for None, NaN, pd.NA, pd.NaT - the scalar 'missing' zoo."""
    if v is None or v is pd.NA or v is pd.NaT:
        return True
    if isinstance(v, (float, np.floating)):
        return math.isnan(float(v))
    return False


def is_disguised_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if not isinstance(value, str):
        return False
    return value.strip().casefold() in DISGUISED_NULL_TOKENS


def parse_number(value: Any) -> float | None:
    """Parse '$1,200.50', '12%', '45 kg', '1.234,56', ' 42 ' -> float. None if not numeric."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        f = float(value)
        return None if math.isnan(f) else f
    text = str(value).strip()
    if not text:
        return None
    if PLAIN_NUMBER_RE.match(text):
        try:
            return float(text)
        except ValueError:
            return None
    if EURO_NUMBER_RE.match(text):
        return float(text.replace(".", "").replace(",", "."))
    m = NUMBER_RE.match(text)
    if not m:
        # accounting negative "(1,200)"
        m2 = re.match(r"^\s*\((.+)\)\s*$", text)
        if m2:
            inner = parse_number(m2.group(1))
            return -inner if inner is not None else None
        return None
    num = m.group("num").replace(",", "")
    try:
        f = float(num)
    except ValueError:
        return None
    sign = m.group("sign")
    if sign in ("-", "−"):
        f = -f
    return f


def number_format_hints(value: Any) -> dict[str, bool]:
    text = str(value).strip() if value is not None else ""
    m = NUMBER_RE.match(text)
    hints = {"currency": False, "percent": False, "thousands": False, "unit": False, "euro": False}
    if EURO_NUMBER_RE.match(text):
        hints["euro"] = True
        return hints
    if not m:
        return hints
    if m.group("cur") or m.group("cur2") or (m.group("unit") and re.fullmatch(CURRENCY_RE, m.group("unit"), re.I)):
        hints["currency"] = True
    if m.group("pct"):
        hints["percent"] = True
    if "," in (m.group("num") or ""):
        hints["thousands"] = True
    if m.group("unit") and not hints["currency"]:
        hints["unit"] = True
    return hints


def to_snake_case(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = text.strip()
    text = re.sub(r"[%]", " pct ", text)
    text = re.sub(r"[#]", " num ", text)
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    if not text:
        text = "column"
    if text[0].isdigit():
        text = "col_" + text
    return text


def safe_float(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 6)


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars so json.dumps works."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta,)):
        return str(obj)
    if obj is pd.NaT:
        return None
    if obj is pd.NA:
        return None
    if isinstance(obj, (np.ndarray,)):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [jsonable(v) for v in obj.tolist()]
    return obj


def display_value(v: Any, max_len: int = 60) -> Any:
    """Make a cell value safe & short for JSON / the UI."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if math.isnan(f) else round(f, 6)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        if v.hour == 0 and v.minute == 0 and v.second == 0 and v.microsecond == 0:
            return v.strftime("%Y-%m-%d")
        return v.isoformat(sep=" ")
    text = str(v)
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def name_has_hint(name: str, hints: tuple[str, ...]) -> bool:
    key = norm_key(name)
    tokens = set(key.split())
    joined = key.replace(" ", "")
    for h in hints:
        if h in tokens:
            return True
        if len(h) > 3 and h in joined:
            return True
    return False
