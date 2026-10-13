"""Render ranking reports and the audit export.

Every rendering starts with the report's watermark line. Columns (CSV/table/markdown):
rank, candidate (name), opportunity_id, profile_url, score, tier, status, top strengths
(first 2, "; "-joined), top gaps (first 2), flags (codes, comma-joined), review_reasons,
evaluation_id, writeback ("sent"/"planned"/"unchanged"/"failed"/"-"; see
``writeback_label``). Markdown escapes "|" in cells. CSV via
the csv module (no injection: cells starting with = + - @ are prefixed with "'").
JSON is ``report.model_dump(mode="json")`` with the watermark included.

Also: ranked rows come first (by rank), then unranked rows in their original order.
Cells are single-line with control/format characters (terminal escapes, bidi overrides)
removed. Strengths/gaps of an evaluation whose rationale failed validation are shown as
withheld (the JSON keeps everything, for audit). A write-back with an error (failed or
refused send, or a result that could not be recorded) shows "failed".
"""

from __future__ import annotations

import csv
import io
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from applicant_ranking.domain import RankingReport, RankingRow, WritebackResult
from applicant_ranking.writeback import NOTHING_TO_WRITE, REASON_TRANSIENT, shorten

COLUMNS: tuple[str, ...] = (
    "rank", "candidate", "opportunity_id", "profile_url", "score", "tier", "status",
    "top_strengths", "top_gaps", "flags", "review_reasons", "evaluation_id", "writeback",
)
LABELS: dict[str, str] = {
    "rank": "Rank", "candidate": "Candidate", "opportunity_id": "Opportunity",
    "profile_url": "Profile", "score": "Score", "tier": "Tier", "status": "Status",
    "top_strengths": "Top strengths", "top_gaps": "Top gaps", "flags": "Flags",
    "review_reasons": "Review reasons", "evaluation_id": "Evaluation",
    "writeback": "Write-back",
}
AUDIT_COLUMNS: tuple[str, ...] = (
    "evaluation_id", "created_at", "mode", "posting_id", "opportunity_id", "application_id",
    "criteria_role_id", "criteria_version", "prompt_version", "model_id", "model_version",
    "status", "score", "tier", "rank_in_posting", "above_posting_median", "flag_codes",
)
WITHHELD = "(withheld: failed validation)"
_FORMULA_PREFIXES = ("=", "+", "-", "@")
_CONSOLE_CELL_WIDTH = 40
_MAX_CELL_CHARS = 2000


# ------------------------------------------------------------------------------- helpers


def _cell(value: Any) -> str:
    """One-line, control-character-free text for a table cell."""
    if value is None:
        return ""
    return shorten(str(value), _MAX_CELL_CHARS)


def csv_safe(value: Any) -> Any:
    """Neutralise spreadsheet formula injection: text starting with = + - @ gets a
    leading "'". Numbers and one-character cells (e.g. the "-" placeholder) pass."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return value
    text = _cell(value)
    if len(text) > 1 and text.startswith(_FORMULA_PREFIXES):
        return "'" + text
    return text


def writeback_label(result: WritebackResult | None) -> str:
    """"-" (nothing to write), "unchanged" (a transient error left the existing tags and
    note alone, so nothing was planned), "failed" (any error), "sent" or "planned"."""
    if result is None or (not result.executed and result.dry_run_reason == NOTHING_TO_WRITE):
        return "-"
    if result.error:
        return "failed"
    if not result.executed and result.dry_run_reason == REASON_TRANSIENT:
        return "unchanged"
    return "sent" if result.executed else "planned"


def ordered_rows(report: RankingReport) -> list[RankingRow]:
    """Ranked rows by rank, then unranked rows in their original order."""
    ranked = sorted((r for r in report.rows if r.rank is not None), key=lambda r: r.rank or 0)
    return ranked + [r for r in report.rows if r.rank is None]


def _row_values(row: RankingRow) -> dict[str, Any]:
    ev = row.evaluation
    if ev.rationale_publishable:
        strengths = "; ".join(_cell(s) for s in ev.strengths[:2])
        gaps = "; ".join(_cell(g) for g in ev.gaps[:2])
    else:
        strengths = gaps = WITHHELD
    return {
        "rank": row.rank,
        "candidate": ev.candidate_name,
        "opportunity_id": ev.opportunity_id,
        "profile_url": ev.profile_url,
        "score": ev.score,
        "tier": ev.tier.value if ev.tier is not None else None,
        "status": ev.status.value,
        "top_strengths": strengths,
        "top_gaps": gaps,
        "flags": ",".join(dict.fromkeys(f.code.value for f in ev.flags)),
        "review_reasons": "; ".join(_cell(r) for r in ev.review_reasons),
        "evaluation_id": ev.evaluation_id,
        "writeback": writeback_label(row.writeback),
    }


def _header_lines(report: RankingReport) -> list[str]:
    title = _cell(report.posting_title)
    posting = f"{title} ({_cell(report.posting_id)})" if title else _cell(report.posting_id)
    criteria = "-"
    if report.criteria_role_id:
        criteria = _cell(report.criteria_role_id)
        if report.criteria_version is not None:
            criteria += f" v{report.criteria_version}"
    ranked = sum(1 for r in report.rows if r.rank is not None)
    lines = [
        f"Posting: {posting}",
        f"Criteria: {criteria}",
        f"Mode: {report.mode.value}",
        f"Generated: {report.generated_at.isoformat(timespec='seconds')}",
        f"Applications: {len(report.rows)} ({ranked} ranked, {len(report.rows) - ranked} "
        "not ranked; nobody is excluded)",
    ]
    if report.policy_regimes:
        lines.append("Policy regimes: " + ", ".join(_cell(r) for r in report.policy_regimes))
    lines.extend(f"Policy note: {_cell(n)}" for n in report.policy_notes)
    lines.extend(f"Note: {_cell(n)}" for n in report.notes)
    return lines


def _watermark(report: RankingReport) -> list[str]:
    return [report.watermark] if report.watermark else []


# ----------------------------------------------------------------------------- renderers


def render_console(report: RankingReport) -> str:
    rows = [[_cell(v) for v in _row_values(r).values()] for r in ordered_rows(report)]
    header = [LABELS[c] for c in COLUMNS]
    table = [header, *rows]
    cells = [[shorten(c, _CONSOLE_CELL_WIDTH) for c in line] for line in table]
    widths = [max(len(line[i]) for line in cells) for i in range(len(COLUMNS))]

    def fmt(line: Sequence[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(line, widths, strict=True)).rstrip()

    out = [*_watermark(report), *_header_lines(report), ""]
    out.append(fmt(cells[0]))
    out.append(fmt(["-" * w for w in widths]))
    out.extend(fmt(line) for line in cells[1:])
    if not rows:
        out.append("(no applications)")
    return "\n".join(out) + "\n"


def report_to_csv(report: RankingReport) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    if report.watermark:
        writer.writerow([report.watermark])
    writer.writerow(COLUMNS)
    for row in ordered_rows(report):
        writer.writerow([csv_safe(v) for v in _row_values(row).values()])
    return buffer.getvalue()


def _md(value: Any) -> str:
    text = _cell(value)
    for char, escaped in (("\\", "\\\\"), ("|", "\\|"), ("<", "&lt;"), (">", "&gt;")):
        text = text.replace(char, escaped)
    return text


def report_to_markdown(report: RankingReport) -> str:
    title = _cell(report.posting_title) or _cell(report.posting_id)
    out = [*_watermark(report), "", f"# Ranking: {_md(title)}", ""]
    out.extend(f"- {_md(line)}" for line in _header_lines(report))
    out.append("")
    out.append("| " + " | ".join(LABELS[c] for c in COLUMNS) + " |")
    out.append("|" + "|".join("---" for _ in COLUMNS) + "|")
    for row in ordered_rows(report):
        out.append("| " + " | ".join(_md(v) for v in _row_values(row).values()) + " |")
    return "\n".join(out) + "\n"


def report_to_json(report: RankingReport) -> str:
    payload = {"watermark": report.watermark, **report.model_dump(mode="json")}
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------- audit export


def _flag_codes(row: dict[str, Any]) -> str:
    raw = row.get("flag_codes_json", row.get("flag_codes"))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return raw
    if isinstance(raw, Iterable) and not isinstance(raw, str | bytes | dict):
        return ",".join(str(code) for code in raw)
    return ""


def _score(row: dict[str, Any]) -> float | None:
    value = row.get("score")
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _posting_stats(rows: list[dict[str, Any]]) -> dict[int, tuple[int, bool]]:
    """Row index -> (rank_in_posting, above_posting_median).

    Groups are (posting_id, mode), so offline EVALUATION runs never mix with real use.
    Within a group only the latest row per opportunity counts (re-scoring must not count
    a candidate twice); superseded and unscored rows get no rank. Ties share the best
    rank (1, 2, 2, 4).

    The latest row is picked before scores are looked at: an opportunity whose newest
    row is unscored (an error, say) is left out entirely, even if an older row scored."""

    def stamp(index: int) -> tuple[str, str]:
        row = rows[index]
        return (str(row.get("created_at") or ""), str(row.get("evaluation_id") or ""))

    groups: dict[tuple[Any, Any], dict[Any, int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        key = (row.get("posting_id"), row.get("mode"))
        opp = row.get("opportunity_id")
        current = groups[key].get(opp)
        if current is None or stamp(index) >= stamp(current):
            groups[key][opp] = index

    stats: dict[int, tuple[int, bool]] = {}
    for members in groups.values():
        scored = [(index, score) for index in members.values()
                  if (score := _score(rows[index])) is not None]
        if not scored:
            continue
        median = statistics.median(score for _, score in scored)
        first_rank: dict[float, int] = {}
        for position, score in enumerate(sorted((s for _, s in scored), reverse=True), 1):
            first_rank.setdefault(score, position)
        for index, score in scored:
            stats[index] = (first_rank[score], score > median)
    return stats


def audit_export_csv(rows: list[dict[str, Any]], *,
                     stage_names: dict[str, str] | None = None) -> str:
    """CSV of pseudonymous audit rows for a separate adverse-impact analysis (the service
    itself never touches demographics). Adds ``above_posting_median`` (score > median of
    that posting's current scored rows) and ``rank_in_posting``. Columns: evaluation_id,
    created_at, mode, posting_id, opportunity_id, application_id, criteria_role_id,
    criteria_version, prompt_version, model_id, model_version, status, score, tier,
    rank_in_posting, above_posting_median, flag_codes.

    See ``_posting_stats`` for how ranks and medians are grouped. With ``stage_names``
    (Lever stage id -> name), a trailing ``stage`` column names the row's ``stage_id``
    (or ``stage``) value, when the caller has added one."""
    stats = _posting_stats(rows)
    columns = list(AUDIT_COLUMNS)
    if stage_names is not None:
        columns.append("stage")

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for index, row in enumerate(rows):
        rank, above = stats.get(index, (None, None))
        values: dict[str, Any] = {column: row.get(column) for column in AUDIT_COLUMNS}
        values["rank_in_posting"] = rank
        values["above_posting_median"] = above
        values["flag_codes"] = _flag_codes(row)
        if stage_names is not None:
            stage_id = row.get("stage_id", row.get("stage"))
            values["stage"] = stage_names.get(stage_id, stage_id) if stage_id else None
        writer.writerow([csv_safe(values[column]) for column in columns])
    return buffer.getvalue()
