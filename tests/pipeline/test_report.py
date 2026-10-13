from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from typing import Any

import pytest

from applicant_ranking.domain import (
    Evaluation,
    Flag,
    FlagCode,
    RankingReport,
    RankingRow,
    RunMode,
    Status,
    Tier,
    WritebackResult,
)
from applicant_ranking.report import (
    AUDIT_COLUMNS,
    COLUMNS,
    WITHHELD,
    audit_export_csv,
    csv_safe,
    ordered_rows,
    render_console,
    report_to_csv,
    report_to_json,
    report_to_markdown,
    writeback_label,
)
from applicant_ranking.writeback import NOTHING_TO_WRITE, REASON_TRANSIENT

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
ADVISORY = "Advisory ranking - not a decision. Review every application."
EVAL_ONLY = "EVALUATION ONLY - not for hiring decisions"


def make_evaluation(**overrides: Any) -> Evaluation:
    base: dict[str, Any] = {
        "evaluation_id": "ev-1",
        "created_at": NOW,
        "mode": RunMode.DRY_RUN,
        "posting_id": "posting-1",
        "opportunity_id": "opp-1",
        "candidate_name": "Jane Doe",
        "profile_url": "https://hire.lever.co/candidates/opp-1",
        "status": Status.SCORED,
        "score": 88,
        "tier": Tier.A,
        "strengths": ["GCP pipelines", "Python", "Workshops"],
        "gaps": ["No consulting", "No Kafka", "Third gap"],
    }
    base.update(overrides)
    return Evaluation(**base)


def sent(executed: bool = True, **kw: Any) -> WritebackResult:
    return WritebackResult(opportunity_id="x", evaluation_id="x", executed=executed, **kw)


def make_report(mode: RunMode = RunMode.DRY_RUN) -> RankingReport:
    unranked = RankingRow(rank=None, evaluation=make_evaluation(
        evaluation_id="ev-3", opportunity_id="opp-3", candidate_name="Error Person",
        status=Status.ERROR, score=None, tier=None, strengths=[], gaps=[],
        flags=[Flag(code=FlagCode.NO_RESUME)]))
    second = RankingRow(rank=2, evaluation=make_evaluation(
        evaluation_id="ev-2", opportunity_id="opp-2", candidate_name="Bob | Smith",
        status=Status.NEEDS_REVIEW, score=61, tier=Tier.C,
        flags=[Flag(code=FlagCode.DUPLICATE_RESUME, severity="review"),
               Flag(code=FlagCode.MODEL_RED_FLAG, source="model"),
               Flag(code=FlagCode.MODEL_RED_FLAG, source="model")],
        review_reasons=["Duplicate resume", "Second reason"]),
        writeback=sent(False, dry_run_reason="dry run"))
    first = RankingRow(rank=1, evaluation=make_evaluation(), writeback=sent())
    return RankingReport(posting_id="posting-1", posting_title="Senior Data Engineer",
                         criteria_role_id="senior-data-engineer", criteria_version=2,
                         mode=mode, generated_at=NOW, policy_regimes=["IL_HRA"],
                         policy_notes=["AI notice published"],
                         rows=[unranked, second, first])


def parse_csv(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text)))


# --------------------------------------------------------------------------- ordering


def test_ranked_rows_first_then_unranked() -> None:
    assert [r.rank for r in ordered_rows(make_report())] == [1, 2, None]


@pytest.mark.parametrize(
    ("result", "label"),
    [
        (None, "-"),
        (sent(False, dry_run_reason=NOTHING_TO_WRITE), "-"),
        (sent(False, dry_run_reason="dry run"), "planned"),
        (sent(False, dry_run_reason=REASON_TRANSIENT), "unchanged"),
        (sent(False, dry_run_reason=REASON_TRANSIENT, error="result not recorded"), "failed"),
        (sent(True), "sent"),
        (sent(True, error="POST failed"), "failed"),
        (sent(False, error="refused: 1 tag(s) outside the namespace"), "failed"),
    ],
)
def test_writeback_label(result: WritebackResult | None, label: str) -> None:
    assert writeback_label(result) == label


# ------------------------------------------------------------------------------- csv


def test_csv_layout() -> None:
    rows = parse_csv(report_to_csv(make_report()))
    assert rows[0] == [ADVISORY]
    assert rows[1] == list(COLUMNS)
    body = [dict(zip(COLUMNS, r, strict=True)) for r in rows[2:]]
    assert [r["rank"] for r in body] == ["1", "2", ""]
    first, second, third = body
    assert first == {
        "rank": "1", "candidate": "Jane Doe", "opportunity_id": "opp-1",
        "profile_url": "https://hire.lever.co/candidates/opp-1", "score": "88", "tier": "A",
        "status": "scored", "top_strengths": "GCP pipelines; Python",
        "top_gaps": "No consulting; No Kafka", "flags": "", "review_reasons": "",
        "evaluation_id": "ev-1", "writeback": "sent",
    }
    assert second["flags"] == "duplicate_resume,model_red_flag"
    assert second["review_reasons"] == "Duplicate resume; Second reason"
    assert second["writeback"] == "planned"
    assert third["status"] == "error"
    assert third["score"] == ""
    assert third["writeback"] == "-"


@pytest.mark.parametrize("payload", ["=HYPERLINK(\"http://x\")", "+1+1", "-2+3", "@SUM(A1)",
                                     " =1+1", "\t=1+1", "\r=1"])
def test_csv_formula_injection_guard(payload: str) -> None:
    report = RankingReport(posting_id="p", mode=RunMode.DRY_RUN, rows=[RankingRow(
        rank=1, evaluation=make_evaluation(candidate_name=payload,
                                           strengths=[payload, "ok"]))])
    row = parse_csv(report_to_csv(report))[2]
    candidate = row[COLUMNS.index("candidate")]
    assert candidate.startswith("'")
    assert candidate.lstrip("'").lstrip()[0] in "=+-@"
    assert row[COLUMNS.index("top_strengths")].startswith("'")


def test_csv_safe_values() -> None:
    assert csv_safe(42) == 42
    assert csv_safe(None) == ""
    assert csv_safe(True) == "true"
    assert csv_safe("-") == "-"
    assert csv_safe("plain") == "plain"
    assert csv_safe("multi\nline") == "multi line"


def test_csv_evaluation_watermark() -> None:
    assert parse_csv(report_to_csv(make_report(RunMode.EVALUATION)))[0] == [EVAL_ONLY]


def test_unpublishable_rationale_is_withheld_in_tables() -> None:
    report = RankingReport(posting_id="p", mode=RunMode.DRY_RUN, rows=[RankingRow(
        rank=1, evaluation=make_evaluation(status=Status.NEEDS_REVIEW,
                                           rationale_publishable=False,
                                           strengths=["young and energetic"]))])
    for text in (report_to_csv(report), report_to_markdown(report), render_console(report)):
        assert "young" not in text
        assert WITHHELD.split(":")[0] in text
    assert "young and energetic" in report_to_json(report)  # full audit record


# -------------------------------------------------------------------------- markdown


def test_markdown() -> None:
    text = report_to_markdown(make_report())
    lines = text.splitlines()
    assert lines[0] == ADVISORY
    assert "# Ranking: Senior Data Engineer" in lines
    table = [line for line in lines if line.startswith("|")]
    assert len(table) == 2 + 3
    assert table[0].count(" | ") == len(COLUMNS) - 1
    assert "Bob \\| Smith" in text
    for row in table[2:]:
        assert row.replace("\\|", "").count("|") == len(COLUMNS) + 1
    assert table[2].startswith("| 1 | Jane Doe")
    assert table[4].startswith("|  | Error Person")


def test_markdown_escapes_html_and_newlines() -> None:
    report = RankingReport(posting_id="p", mode=RunMode.EVALUATION, rows=[RankingRow(
        rank=1, evaluation=make_evaluation(candidate_name="<img src=x onerror=alert(1)>\nX"))])
    text = report_to_markdown(report)
    assert text.splitlines()[0] == EVAL_ONLY
    assert "<img" not in text
    assert "&lt;img src=x onerror=alert(1)&gt; X" in text


# --------------------------------------------------------------------------- console


def test_console() -> None:
    text = render_console(make_report())
    lines = text.splitlines()
    assert lines[0] == ADVISORY
    assert "Posting: Senior Data Engineer (posting-1)" in lines
    assert "Criteria: senior-data-engineer v2" in lines
    assert "Policy regimes: IL_HRA" in lines
    assert "Policy note: AI notice published" in lines
    assert "Applications: 3 (2 ranked, 1 not ranked; nobody is excluded)" in lines
    header = next(i for i, line in enumerate(lines) if line.startswith("Rank"))
    data = lines[header + 2:]
    assert [line.split()[0] for line in data[:2]] == ["1", "2"]
    assert "Error Person" in data[2]


def test_run_notes_are_listed_in_every_header() -> None:
    report = make_report().model_copy(update={"notes": ["Lever returned no applicants."]})
    assert "Note: Lever returned no applicants." in render_console(report).splitlines()
    assert "- Note: Lever returned no applicants." in report_to_markdown(report)
    assert json.loads(report_to_json(report))["notes"] == ["Lever returned no applicants."]
    assert "Note:" not in render_console(make_report())


def test_console_strips_terminal_escapes() -> None:
    report = RankingReport(posting_id="p", mode=RunMode.DRY_RUN, rows=[RankingRow(
        rank=1, evaluation=make_evaluation(candidate_name="\x1b]0;pwned\x07Eve\u202e"))])
    text = render_console(report)
    assert "\x1b" not in text and "\x07" not in text and "\u202e" not in text
    assert "Eve" in text


def test_console_empty_report() -> None:
    text = render_console(RankingReport(posting_id="p", mode=RunMode.EVALUATION))
    assert text.splitlines()[0] == EVAL_ONLY
    assert "(no applications)" in text


# ------------------------------------------------------------------------------ json


def test_json_round_trip() -> None:
    report = make_report()
    text = report_to_json(report)
    assert text.splitlines()[1].strip() == f'"watermark": "{ADVISORY}",'
    payload = json.loads(text)
    assert payload.pop("watermark") == ADVISORY
    assert RankingReport.model_validate(payload) == report


def test_json_evaluation_watermark() -> None:
    assert json.loads(report_to_json(make_report(RunMode.EVALUATION)))["watermark"] == EVAL_ONLY


# ---------------------------------------------------------------------------- audit


def audit_row(evaluation_id: str, opp: str, score: int | None, *, posting: str = "p1",
              mode: str = "dry_run", created_at: str = "2026-10-08T10:00:00+00:00",
              flags: Any = "[]", **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "evaluation_id": evaluation_id, "created_at": created_at, "mode": mode,
        "posting_id": posting, "opportunity_id": opp, "application_id": f"app-{opp}",
        "resume_id": "r", "criteria_role_id": "senior-data-engineer", "criteria_version": 2,
        "criteria_hash": "h", "prompt_version": "prompt-v3", "model_id": "gemini",
        "model_version": "gemini-001", "status": "scored" if score is not None else "error",
        "score": score, "tier": None, "flag_codes_json": flags, "file_hmac": "secret-hmac",
    }
    row.update(extra)
    return row


def test_audit_export_rank_and_median() -> None:
    rows = [
        audit_row("e-a-old", "a", 10, created_at="2026-10-01T10:00:00+00:00"),
        audit_row("e-a", "a", 90, flags='["duplicate_resume", "low_confidence"]'),
        audit_row("e-b", "b", 70),
        audit_row("e-c", "c", 70, flags=["jd_copy"]),
        audit_row("e-d", "d", 50),
        audit_row("e-e", "e", None),
        audit_row("e-f", "f", 60, posting="p2"),
        audit_row("e-a-eval", "a", 20, mode="evaluation"),
    ]
    parsed = parse_csv(audit_export_csv(rows))
    assert parsed[0] == list(AUDIT_COLUMNS)
    out = {r[0]: dict(zip(AUDIT_COLUMNS, r, strict=True)) for r in parsed[1:]}
    assert len(out) == len(rows)

    # p1 / dry_run, latest per opportunity: 90, 70, 70, 50 -> median 70; ties share a rank.
    assert [(out[e]["rank_in_posting"], out[e]["above_posting_median"])
            for e in ("e-a", "e-b", "e-c", "e-d")] == [
        ("1", "true"), ("2", "false"), ("2", "false"), ("4", "false")]
    assert (out["e-a-old"]["rank_in_posting"], out["e-a-old"]["above_posting_median"]) == (
        "", "")
    assert (out["e-e"]["rank_in_posting"], out["e-e"]["above_posting_median"]) == ("", "")
    assert (out["e-f"]["rank_in_posting"], out["e-f"]["above_posting_median"]) == ("1", "false")
    assert (out["e-a-eval"]["rank_in_posting"],
            out["e-a-eval"]["above_posting_median"]) == ("1", "false")

    assert out["e-a"]["flag_codes"] == "duplicate_resume,low_confidence"
    assert out["e-c"]["flag_codes"] == "jd_copy"
    assert out["e-a"]["criteria_version"] == "2"
    assert out["e-a"]["model_version"] == "gemini-001"
    assert "secret-hmac" not in audit_export_csv(rows)


def test_audit_export_newest_unscored_row_supersedes_an_older_score() -> None:
    rows = [
        audit_row("e1", "opp-a", 90, created_at="2026-10-01T00:00:00+00:00"),
        audit_row("e2", "opp-b", 60, created_at="2026-10-02T00:00:00+00:00"),
        audit_row("e3", "opp-c", 70, created_at="2026-10-03T00:00:00+00:00"),
        audit_row("e4", "opp-a", None, created_at="2026-10-05T00:00:00+00:00"),  # error
    ]
    parsed = parse_csv(audit_export_csv(rows))
    out = {r[0]: (r[AUDIT_COLUMNS.index("rank_in_posting")],
                  r[AUDIT_COLUMNS.index("above_posting_median")]) for r in parsed[1:]}
    # opp-a's current row is the error: neither it nor the superseded 90 is ranked, and
    # the median is 65 over (70, 60), not 70 over (90, 70, 60).
    assert out == {"e1": ("", ""), "e4": ("", ""), "e3": ("1", "true"), "e2": ("2", "false")}


def test_audit_export_group_with_no_current_score_is_unranked() -> None:
    rows = [audit_row("e1", "a", 80, created_at="2026-10-01T00:00:00+00:00"),
            audit_row("e2", "a", None, created_at="2026-10-02T00:00:00+00:00")]
    parsed = parse_csv(audit_export_csv(rows))
    idx = AUDIT_COLUMNS.index("rank_in_posting")
    assert [r[idx] for r in parsed[1:]] == ["", ""]


def test_audit_export_even_count_median() -> None:
    rows = [audit_row("e1", "a", 80), audit_row("e2", "b", 60)]
    out = parse_csv(audit_export_csv(rows))[1:]
    idx = AUDIT_COLUMNS.index("above_posting_median")
    assert [r[idx] for r in out] == ["true", "false"]  # median 70


def test_audit_export_formula_guard_and_stage_names() -> None:
    rows = [audit_row("=cmd|' /C calc'!A0", "@opp", 75, stage_id="stage-1"),
            audit_row("e2", "b", 65, stage="stage-unknown"),
            audit_row("e3", "c", 55)]
    parsed = parse_csv(audit_export_csv(rows, stage_names={"stage-1": "Phone screen"}))
    assert parsed[0] == [*AUDIT_COLUMNS, "stage"]
    assert parsed[1][0].startswith("'=")
    assert parsed[1][AUDIT_COLUMNS.index("opportunity_id")] == "'@opp"
    assert [r[-1] for r in parsed[1:]] == ["Phone screen", "stage-unknown", ""]


def test_audit_export_accepts_decoded_flag_codes() -> None:
    # SqliteStore.audit_rows() decodes flag_codes_json into a "flag_codes" list.
    row = audit_row("e1", "a", 80)
    del row["flag_codes_json"]
    row["flag_codes"] = ["hidden_text", "truncated"]
    parsed = parse_csv(audit_export_csv([row]))
    assert parsed[1][AUDIT_COLUMNS.index("flag_codes")] == "hidden_text,truncated"


def test_audit_export_empty() -> None:
    assert parse_csv(audit_export_csv([])) == [list(AUDIT_COLUMNS)]
