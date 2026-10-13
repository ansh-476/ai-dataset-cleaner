"""Human-readable cleaning report (Markdown) built from the session history."""
from __future__ import annotations

from datetime import datetime
from typing import Any


def _row(cells: list[Any]) -> str:
    return "| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |"


def build_markdown_report(ds: dict[str, Any]) -> str:
    meta = ds["meta"]
    history = ds.get("history", [])
    first_profile = ds.get("profile_original") or ds.get("profile")
    last_profile = ds.get("profile")
    lines = [f"# Data Cleaning Report - {meta.get('file_name')}", "",
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
             f"Original shape: {ds['original'].shape[0]} rows × {ds['original'].shape[1]} columns  ",
             f"Current shape: {ds['current'].shape[0]} rows × {ds['current'].shape[1]} columns", ""]

    if first_profile and last_profile:
        lines += ["## Quality score", "", _row(["", "Before", "After"]), _row(["---", "---", "---"]),
                  _row(["Score (0-100)", first_profile["score"]["score"], last_profile["score"]["score"]]),
                  _row(["Grade", first_profile["score"]["grade"], last_profile["score"]["grade"]]),
                  _row(["Issues detected", len(first_profile["issues"]), len(last_profile["issues"])]),
                  _row(["Missing cells", f"{first_profile['dataset']['missing_pct']}%", f"{last_profile['dataset']['missing_pct']}%"]),
                  _row(["Duplicate rows", first_profile["dataset"].get("duplicate_rows", 0), last_profile["dataset"].get("duplicate_rows", 0)]), ""]

    if first_profile:
        lines += ["## Issues found in the original data", ""]
        for sev in ("high", "medium", "low"):
            items = [i for i in first_profile["issues"] if i["severity"] == sev]
            if not items:
                continue
            lines.append(f"### {sev.title()} severity ({len(items)})")
            lines.append("")
            for it in items:
                where = it["column"] if it["column"] else "dataset"
                lines.append(f"- **{where}** - {it['title']}. {it['detail']}")
            lines.append("")

    for n, h in enumerate(history, 1):
        plan = h["plan"]
        lines += [f"## Cleaning pass {n} ({plan.get('source')}{' / ' + plan['model'] if plan.get('model') else ''})", ""]
        if plan.get("summary"):
            lines += ["### AI summary", "", plan["summary"], ""]
        if plan.get("dataset_thinking"):
            lines += ["### Reasoning about the dataset structure", "", plan["dataset_thinking"], ""]
        if plan.get("column_thoughts"):
            lines += ["### Reasoning per column", ""]
            for col, th in plan["column_thoughts"].items():
                if th:
                    lines.append(f"- **{col}**: {th}")
            lines.append("")
        lines += ["### Actions applied", "", _row(["#", "Action", "Column", "Parameters", "Reason", "Result"]),
                  _row(["---"] * 6)]
        for i, e in enumerate(h["log"], 1):
            params = {k: v for k, v in (e.get("params") or {}).items() if k != "mapping"}
            if (e.get("params") or {}).get("mapping"):
                params["mapping"] = f"{len(e['params']['mapping'])} pairs"
            result = e.get("message", "")
            if e.get("status") != "applied":
                result = f"{e.get('status').upper()}: {result}"
            lines.append(_row([i, e["action"], e.get("column") or "-", params or "-", e.get("reason") or "-", result]))
        lines.append("")
        if plan.get("warnings"):
            lines += ["### Warnings", ""] + [f"- {w}" for w in plan["warnings"]] + [""]
        lines += [f"Rows: {h['rows_before']} → {h['rows_after']}; columns: {h['cols_before']} → {h['cols_after']}; "
                  f"score {h['score_before']} → {h['score_after']}", ""]

    if last_profile:
        lines += ["## Final column types", "", _row(["Column", "Type", "Missing %", "Unique"]), _row(["---"] * 4)]
        for c in last_profile["columns"]:
            lines.append(_row([c["name"], c["inferred_type"], c["missing_pct"], c["unique"]]))
        lines.append("")
    return "\n".join(lines)
