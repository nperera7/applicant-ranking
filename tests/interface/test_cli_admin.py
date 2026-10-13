"""``backtest``, ``fairness``, ``export-audit``, ``purge``, ``hold``, ``doctor``, ``serve`` and
settings loading."""

from __future__ import annotations

import csv
import io
import json
import re
import stat
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from typer.testing import CliRunner

from applicant_ranking import cli
from applicant_ranking.criteria import dump_criteria_yaml
from applicant_ranking.domain import RoleCriteria, RunMode
from applicant_ranking.fairness import FairnessReport, VariantResult
from applicant_ranking.lever.models import ArchiveReason, Opportunity, Stage
from applicant_ranking.store import PurgeResult, SqliteStore

from .conftest import CANDIDATE, NOW, SECRETS, CliHarness, make_evaluation

ADVISORY = "Advisory ranking - not a decision. Review every application."
EVAL_ONLY = "EVALUATION ONLY - not for hiring decisions"

_OPENED: list[str] | None = None


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if _OPENED is not None and event == "open" and args and isinstance(args[0], str | bytes):
        _OPENED.append(str(args[0]))


sys.addaudithook(_audit)


@pytest.fixture
def opened_files() -> Iterator[list[str]]:
    global _OPENED
    _OPENED = []
    try:
        yield _OPENED
    finally:
        _OPENED = None


@pytest.fixture
def criteria_file(tmp_path: Path, sample_criteria: RoleCriteria) -> Path:
    path = tmp_path / "role.yaml"
    path.write_text(dump_criteria_yaml(sample_criteria), encoding="utf-8")
    return path


@pytest.fixture
def resume_file(tmp_path: Path, sample_resume_text: str) -> Path:
    path = tmp_path / "resume.txt"
    path.write_text(sample_resume_text, encoding="utf-8")
    return path


# ------------------------------------------------------------------------- backtest


def _backtest_lever(harness: CliHarness) -> None:
    harness.lever.stages = [Stage(id="st-new", text="New applicant"),
                            Stage(id="st-onsite", text="On-site interview")]
    harness.lever.archive_reasons = [ArchiveReason(id="ar-hired", text="Hired", type="hired")]
    harness.lever.opportunities = [
        Opportunity.model_validate({"id": "opp-1", "stage": "st-onsite"}),
        Opportunity.model_validate({"id": "opp-2", "stage": "st-new"}),
        Opportunity.model_validate({"id": "opp-3", "stage": "st-new"}),
    ]


def test_backtest_prints_metrics_and_writes_json(harness: CliHarness, tmp_path: Path) -> None:
    _backtest_lever(harness)
    out = tmp_path / "reports" / "backtest.json"

    result = harness.invoke("backtest", "--posting", "posting-1", "--positive-stage",
                            "on-site interview", "--limit", "10", "--out", str(out))

    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[0] == EVAL_ONLY
    assert "ranked applications       2" in result.stdout
    assert "positives                 1" in result.stdout
    assert "AUC                       1.000" in result.stdout
    assert "Caveats:" in result.stdout
    assert CANDIDATE not in result.stdout  # names stay in the JSON file
    [(_, kwargs)] = harness.ranker.called("rank_posting")
    assert kwargs["mode"] == RunMode.EVALUATION
    assert kwargs["include_archived"] is True
    assert kwargs["limit"] == 10
    payload = json.loads(out.read_text())
    assert payload["watermark"] == EVAL_ONLY
    assert payload["metrics"]["auc"] == 1.0
    assert stat.S_IMODE(out.stat().st_mode) == 0o600


def test_backtest_unknown_stage_exits_2_before_ranking(harness: CliHarness) -> None:
    _backtest_lever(harness)

    result = harness.invoke("backtest", "--posting", "posting-1", "--positive-stage", "Offer")

    assert result.exit_code == 2
    assert "no positive stage name matches" in result.stderr
    assert harness.ranker.called("rank_posting") == []
    assert harness.built.closed == 1


def test_backtest_requires_a_stage(harness: CliHarness) -> None:
    result = harness.invoke("backtest", "--posting", "posting-1")

    assert result.exit_code == 2  # usage error
    assert harness.built.kwargs == []


# ------------------------------------------------------------------------- fairness


def test_fairness_with_stable_scores_passes(harness: CliHarness, resume_file: Path,
                                            criteria_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "fairness.json"

    result = harness.invoke("fairness", str(resume_file), "--criteria", str(criteria_file),
                            "--runs", "2", "--out", str(out))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[0] == EVAL_ONLY
    assert "PASSED" in result.stdout
    assert harness.built.kwargs == [{"need_lever": False}]
    [(_, gate)] = harness.ranker.called("check_criteria")
    assert gate == {"mode": RunMode.EVALUATION}
    calls = harness.ranker.called("evaluate_local")
    assert calls and all(k["criteria"].role_id == "senior-data-engineer" for _, k in calls)
    assert all(k["text"] and k["data"] is None for _, k in calls)
    assert json.loads(out.read_text())["watermark"] == EVAL_ONLY


def test_fairness_name_is_the_base_name(harness: CliHarness, tmp_path: Path,
                                        criteria_file: Path, sample_resume_text: str) -> None:
    resume = tmp_path / "named.txt"
    resume.write_text("Ana Real-Person\n" + sample_resume_text)

    result = harness.invoke("fairness", str(resume), "--criteria", str(criteria_file),
                            "--runs", "1", "--name", "Ana Real-Person")

    assert result.exit_code == 0, result.output
    calls = harness.ranker.called("evaluate_local")
    base = calls[0][1]
    assert base["candidate_names"] == ["Ana Real-Person"]
    names = {tuple(k["candidate_names"]) for _, k in calls}
    assert len(names) > 1  # the name variants replaced it
    assert "Ana Real-Person" not in result.output


def _report(*, flagged: bool) -> FairnessReport:
    results = [
        VariantResult(variant="base", group="base", scores=[80, 82], mean=81.0, std=1.4,
                      modal_tier="A", delta_vs_base=0.0, flagged=False),
        VariantResult(variant="grad_year_1995", group="grad_year", scores=[60, 62],
                      mean=61.0, std=1.4, modal_tier="C", delta_vs_base=-20.0,
                      flagged=flagged, reason="mean moved by 20.0 points" if flagged else ""),
    ]
    return FairnessReport(criteria_role_id="senior-data-engineer", criteria_version=2, runs=2,
                          tolerance=5.0, generated_at=NOW, results=results)


def test_fairness_without_name_warns_and_lists_the_caveat(harness: CliHarness,
                                                          resume_file: Path,
                                                          criteria_file: Path) -> None:
    result = harness.invoke("fairness", str(resume_file), "--criteria", str(criteria_file),
                            "--runs", "1")

    assert result.exit_code == 0, result.output
    assert cli.NO_NAME_FAIRNESS_WARNING in result.stderr
    assert "PASSED" in result.stdout  # a nameless synthetic resume is still a valid run
    lines = result.stdout.splitlines()
    assert "Caveats:" in lines
    caveat = lines[lines.index("Caveats:") + 1]
    assert caveat.startswith("  - the resume does not contain the name 'Jordan Lee'")
    assert "fairness --name" in caveat

    named = harness.invoke("fairness", str(resume_file), "--criteria", str(criteria_file),
                           "--runs", "1", "--name", "Jordan Lee")
    assert named.exit_code == 0, named.output
    assert cli.NO_NAME_WARNING not in named.stderr


def test_fairness_flagged_variant_exits_2(harness: CliHarness, resume_file: Path,
                                          criteria_file: Path, sample_resume_text: str,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_run(ranker: Any, criteria: RoleCriteria, base_text: str, *, runs: int,
                       tolerance: float) -> FairnessReport:
        seen.update(ranker=ranker, criteria=criteria, text=base_text, runs=runs,
                    tolerance=tolerance)
        return _report(flagged=True)

    monkeypatch.setattr(cli, "run_fairness", fake_run)

    result = harness.invoke("fairness", str(resume_file), "--criteria", str(criteria_file),
                            "--runs", "4", "--tolerance", "2.5")

    assert result.exit_code == 2
    assert "FLAGGED: 1 variant(s)" in result.stdout
    assert "grad_year_1995" in result.stdout
    assert seen["runs"] == 4 and seen["tolerance"] == 2.5
    assert seen["ranker"] is harness.ranker
    assert "Built production data pipelines on GCP" in seen["text"]
    assert "Built production data pipelines on GCP" not in result.output
    assert sample_resume_text.splitlines()[3] not in result.output


def test_fairness_unreadable_resume_exits_1(harness: CliHarness, criteria_file: Path,
                                            tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")

    result = harness.invoke("fairness", str(empty), "--criteria", str(criteria_file))

    assert result.exit_code == 1
    assert "could not read the resume" in result.stderr or "no text layer" in result.stderr
    assert harness.ranker.calls == []
    assert harness.built.closed == 1


# ---------------------------------------------------------------------- export-audit


def _record(store: SqliteStore, evaluation_id: str, opportunity_id: str, posting_id: str,
            score: int, created_at: datetime = NOW) -> None:
    store.record_evaluation(make_evaluation(evaluation_id=evaluation_id,
                                            opportunity_id=opportunity_id,
                                            posting_id=posting_id, score=score,
                                            created_at=created_at),
                            cache_key=f"key-{evaluation_id}")


def test_export_audit_writes_pseudonymous_csv(harness: CliHarness, tmp_path: Path) -> None:
    with SqliteStore(harness.settings.state_db) as store:
        _record(store, "ev-1", "opp-1", "posting-1", 90)
        _record(store, "ev-2", "opp-2", "posting-1", 70)
        _record(store, "ev-3", "opp-3", "posting-2", 50)
    out = tmp_path / "exports" / "audit.csv"

    result = harness.invoke("export-audit", "--out", str(out))
    only = harness.invoke("export-audit", "--posting", "posting-1", "--out",
                          str(tmp_path / "p1.csv"))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[0] == ADVISORY
    assert f"wrote 3 audit row(s) to {out}" in result.stdout
    text = out.read_text()
    rows = list(csv.DictReader(io.StringIO(text)))
    assert [r["evaluation_id"] for r in rows] == ["ev-1", "ev-2", "ev-3"]
    assert rows[0]["rank_in_posting"] == "1"
    assert CANDIDATE not in text
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert only.exit_code == 0
    assert "wrote 2 audit row(s)" in only.stdout
    assert harness.built.kwargs == []  # the store is opened directly


def test_export_audit_without_database_writes_header_only(harness: CliHarness,
                                                          tmp_path: Path) -> None:
    out = tmp_path / "audit.csv"

    result = harness.invoke("export-audit", "--out", str(out))

    assert result.exit_code == 0, result.output
    assert "wrote 0 audit row(s)" in result.stdout
    assert out.read_text().startswith("evaluation_id,")
    assert not harness.settings.state_db.exists()


def test_export_audit_requires_out(harness: CliHarness) -> None:
    assert harness.invoke("export-audit").exit_code == 2


# ------------------------------------------------------------------------------- purge


def test_purge_removes_expired_and_reconciles(harness: CliHarness) -> None:
    with SqliteStore(harness.settings.state_db) as store:
        _record(store, "ev-old", "opp-old", "posting-1", 60,
                created_at=datetime.now(UTC) - timedelta(days=2000))
        _record(store, "ev-new", "opp-new", "posting-1", 70, created_at=datetime.now(UTC))
    harness.ranker.deleted = [
        PurgeResult(opportunity_id="opp-a", on_hold=False, pii_rows_deleted=2),
        PurgeResult(opportunity_id="opp-b", on_hold=True),
    ]

    result = harness.invoke("purge", "--yes", "--reconcile-days", "45")

    assert result.exit_code == 0, result.output
    assert "retention: 1 evaluation(s) older than 1461 days removed" in result.stdout
    assert "2 candidate(s) deleted in Lever in the last 45 days; 1 purged (2 PII row(s)), " \
           "1 kept on legal hold" in result.stdout
    assert harness.ranker.called("reconcile_deletions") == [((), {"days": 45})]
    assert harness.built.kwargs == [{"need_llm": False, "need_lever": True}]
    with SqliteStore(harness.settings.state_db) as store:
        assert store.get_evaluation("ev-old") is None
        assert store.get_evaluation("ev-new") is not None


def test_purge_without_reconciliation_needs_no_lever(harness: CliHarness) -> None:
    result = harness.invoke("purge", "--yes", "--reconcile-days", "0")

    assert result.exit_code == 0, result.output
    assert "reconciliation skipped" in result.stdout
    assert "deleted candidates:" not in result.stdout  # nothing a hold had suspended
    assert harness.ranker.called("reconcile_deletions") == []
    assert harness.built.kwargs == [{"need_llm": False, "need_lever": False}]


def test_purge_completes_deletions_a_cleared_hold_suspended(harness: CliHarness) -> None:
    with SqliteStore(harness.settings.state_db) as store:
        _record(store, "ev-held", "opp-held", "posting-1", 60)
        _record(store, "ev-live", "opp-live", "posting-1", 70)
        store.set_legal_hold(scope="posting", scope_id="posting-1", reason="Matter 7",
                             set_by="legal", now=NOW)
        store.record_deletion("opp-held", source="webhook", deleted_at=NOW)
        assert store.purge_candidate("opp-held").on_hold
        store.clear_legal_hold(scope="posting", scope_id="posting-1", purge_deleted=False)

    result = harness.invoke("purge", "--yes", "--reconcile-days", "0")

    assert result.exit_code == 0, result.output
    assert ("deleted candidates: 1 purged (1 PII row(s)) whose deletion a legal hold had "
            "suspended, 0 still on legal hold") in result.stdout
    with SqliteStore(harness.settings.state_db) as store:
        assert store.get_evaluation("ev-held") is None and store.is_deleted("opp-held")
        assert store.get_evaluation("ev-live") is not None
        assert {row["evaluation_id"] for row in store.audit_rows()} == {"ev-held", "ev-live"}


def test_purge_asks_for_confirmation(harness: CliHarness) -> None:
    declined = harness.invoke("purge", input="n\n")

    assert declined.exit_code == 1
    assert "Permanently delete" in declined.stderr
    assert harness.built.kwargs == []

    accepted = harness.invoke("purge", input="y\n")
    assert accepted.exit_code == 0, accepted.output
    assert harness.ranker.called("reconcile_deletions") == [((), {"days": 30})]


# -------------------------------------------------------------------------------- hold


def test_hold_set_and_clear(harness: CliHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_current_user", lambda: "os-user")

    set_posting = harness.invoke("hold", "set", "--posting", "posting-1", "--reason",
                                 "Matter 2026-17")
    set_opp = harness.invoke("hold", "set", "--opportunity", "opp-9", "--reason", "Subpoena",
                             "--by", "Legal (A. Counsel)")

    assert set_posting.exit_code == 0, set_posting.output
    assert set_posting.stdout.splitlines()[0] == ADVISORY
    assert "legal hold set on posting posting-1 by os-user" in set_posting.stdout
    assert "by Legal (A. Counsel)" in set_opp.stdout
    with SqliteStore(harness.settings.state_db) as store:
        assert store.is_on_hold(posting_id="posting-1")
        assert store.is_on_hold(opportunity_id="opp-9")

    cleared = harness.invoke("hold", "clear", "--posting", "posting-1", "--reason", "closed")
    again = harness.invoke("hold", "clear", "--posting", "posting-1")

    assert cleared.exit_code == 0
    assert "legal hold on posting posting-1 cleared by os-user" in cleared.stdout
    assert again.exit_code == 0
    assert "no legal hold was set on posting posting-1" in again.stdout
    with SqliteStore(harness.settings.state_db) as store:
        assert not store.is_on_hold(posting_id="posting-1")
        assert store.is_on_hold(opportunity_id="opp-9")
    assert harness.built.kwargs == []


def test_hold_clear_completes_the_deletions_it_suspended(
        harness: CliHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_current_user", lambda: "os-user")
    with SqliteStore(harness.settings.state_db) as store:
        _record(store, "ev-gone", "opp-gone", "posting-1", 60)
        _record(store, "ev-live", "opp-live", "posting-1", 70)
        _record(store, "ev-also", "opp-also", "posting-2", 50)
        store.set_legal_hold(scope="posting", scope_id="posting-1", reason="m", set_by="legal",
                             now=NOW)
        store.set_legal_hold(scope="opportunity", scope_id="opp-also", reason="m",
                             set_by="legal", now=NOW)
        for opp in ("opp-gone", "opp-also"):
            store.record_deletion(opp, source="webhook", deleted_at=NOW)
            assert store.purge_candidate(opp).on_hold

    result = harness.invoke("hold", "clear", "--posting", "posting-1")

    assert result.exit_code == 0, result.output
    assert "legal hold on posting posting-1 cleared by os-user" in result.stdout
    assert ("deleted candidates: 1 purged (1 PII row(s)) now that the hold is released, "
            "1 still on legal hold") in result.stdout
    with SqliteStore(harness.settings.state_db) as store:
        assert store.get_evaluation("ev-gone") is None and store.is_deleted("opp-gone")
        assert store.get_evaluation("ev-live") is not None  # not deleted: kept
        assert store.get_evaluation("ev-also") is not None  # its own hold still stands
    assert "deleted candidates" not in harness.invoke("hold", "clear", "--posting",
                                                      "posting-1").stdout


@pytest.mark.parametrize("args", [
    ["hold", "set", "--reason", "x"],
    ["hold", "set", "--posting", "p", "--opportunity", "o", "--reason", "x"],
    ["hold", "set", "--posting", "p"],
    ["hold", "set", "--posting", "p", "--reason", "  "],
    ["hold", "clear"],
    ["hold", "clear", "--posting", " ", "--opportunity", ""],
])
def test_hold_usage_errors(harness: CliHarness, args: list[str]) -> None:
    result = harness.invoke(*args)

    assert result.exit_code == 2
    assert not harness.settings.state_db.exists()


# ------------------------------------------------------------------------------ doctor


def test_doctor_masks_secrets_and_reads_no_credentials(
        harness: CliHarness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        sample_criteria: RoleCriteria, opened_files: list[str]) -> None:
    (harness.settings.criteria_dir / "role.yaml").write_text(dump_criteria_yaml(sample_criteria))
    adc = tmp_path / "home" / ".config" / "gcloud" / "application_default_credentials.json"
    adc.parent.mkdir(parents=True)
    adc.write_text("{}")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(adc))

    def no_build(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("doctor must not build components")

    monkeypatch.setattr(cli, "_build", no_build)
    opened_files.clear()  # only what doctor itself opens

    result = harness.invoke("doctor")

    assert result.exit_code == 0, result.output
    out = result.stdout
    assert out.splitlines()[0] == ADVISORY
    for secret in SECRETS:
        assert secret not in result.output
    assert "AR_LEVER_API_KEY" in out and "set (hidden)" in out
    assert "applicationCreated=set (hidden)" in out
    assert "test-project" in out
    assert "global" in out
    assert "gemini-3.1-pro-preview (thinking: low)" in out
    assert "terms" in out and "version" in out
    assert "senior-data-engineer v2" in out and "approved" in out
    assert "no policy file" in out
    assert "No problems found" in out
    assert not harness.settings.state_db.exists()
    assert opened_files  # the audit hook saw the criteria, lexicon and policy reads
    assert not any("/.config/gcloud" in p or "application_default_credentials" in p
                   for p in opened_files), opened_files


def test_doctor_reports_problems_with_exit_3(harness: CliHarness, sample_criteria: RoleCriteria,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    prod = harness.settings.model_copy(update={"env": "prod", "hmac_key": None,
                                               "gcp_project": None, "lever_api_key": None,
                                               "lever_webhook_tokens": {}})
    monkeypatch.setattr(cli, "_load_settings", lambda: prod)
    harness.settings.policy_file.write_text("not_a_policy_key: true\n")
    bad = sample_criteria.model_copy(update={"summary": "Young engineers preferred."})
    (harness.settings.criteria_dir / "role.yaml").write_text(dump_criteria_yaml(bad))
    (harness.settings.criteria_dir / "broken.yaml").write_text("- not a mapping\n")

    result = harness.invoke("doctor")

    assert result.exit_code == 3
    out = result.stdout
    assert "PROBLEM: AR_HMAC_KEY is required outside dev" in out
    assert "PROBLEM: policy file" in out
    assert "PROBLEM: senior-data-engineer: " in out
    assert "broken.yaml" in out
    assert "AR_GCP_PROJECT is not set" in out
    assert "no webhook signing token for applicationCreated" in out
    assert "problem(s)" in out


def test_doctor_reports_unapproved_policy_as_warning(harness: CliHarness) -> None:
    harness.settings.policy_file.write_text("version: 3\nallowed_countries: [US]\n")

    result = harness.invoke("doctor")

    assert result.exit_code == 0, result.output
    assert "the policy file is not approved" in result.stdout
    assert re.search(r"version\s+3\n", result.stdout)


def test_doctor_bad_lexicon_is_a_problem(harness: CliHarness, monkeypatch: pytest.MonkeyPatch,
                                         tmp_path: Path) -> None:
    broken = harness.settings.model_copy(update={"protected_terms_file": tmp_path / "x.yaml"})
    monkeypatch.setattr(cli, "_load_settings", lambda: broken)

    result = harness.invoke("doctor")

    assert result.exit_code == 3
    assert "cannot read protected-term list" in result.stdout


# ------------------------------------------------------------------------------- serve


@pytest.fixture
def uvicorn_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, dict[str, Any]]]:
    import uvicorn

    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append((app, kw)))
    return calls


def test_serve_defaults_to_loopback(harness: CliHarness, monkeypatch: pytest.MonkeyPatch,
                                    uvicorn_calls: list[tuple[Any, dict[str, Any]]]) -> None:
    monkeypatch.delenv("PORT", raising=False)

    result = harness.invoke("serve")

    assert result.exit_code == 0, result.output
    [(application, kwargs)] = uvicorn_calls
    assert isinstance(application, FastAPI)
    assert application.state.settings is harness.settings
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8080
    assert harness.built.kwargs == []  # components are built by the app's lifespan


def test_serve_reads_port_from_environment(harness: CliHarness,
                                           monkeypatch: pytest.MonkeyPatch,
                                           uvicorn_calls: list[tuple[Any, dict[str, Any]]],
                                           ) -> None:
    monkeypatch.setenv("PORT", "9090")

    env = harness.invoke("serve", "--host", "0.0.0.0")
    explicit = harness.invoke("serve", "--port", "1234")

    assert env.exit_code == explicit.exit_code == 0
    assert uvicorn_calls[0][1]["host"] == "0.0.0.0"
    assert uvicorn_calls[0][1]["port"] == 9090
    assert uvicorn_calls[1][1]["port"] == 1234


# ---------------------------------------------------------------------------- settings


def test_invalid_environment_is_a_configuration_error(monkeypatch: pytest.MonkeyPatch,
                                                      tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # no project .env
    monkeypatch.setenv("AR_LEVER_RATE_PER_SECOND", "fast-please")

    result = CliRunner().invoke(cli.app, ["criteria", "lint"])

    assert result.exit_code == 3
    assert "lever_rate_per_second" in result.stderr
    assert "fast-please" not in result.output


def test_malformed_webhook_tokens_do_not_leak(monkeypatch: pytest.MonkeyPatch,
                                              tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AR_LEVER_WEBHOOK_TOKENS", '{"applicationCreated": "s3cr3t-value"')

    result = CliRunner().invoke(cli.app, ["doctor"])

    assert result.exit_code == 3
    assert "s3cr3t-value" not in result.output


def test_settings_come_from_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("AR_GEMINI_MODEL=gemini-from-dotenv\nAR_ENV=test\n")

    settings = cli._load_settings()

    assert settings.gemini_model == "gemini-from-dotenv"


def test_main_entry_point(monkeypatch: pytest.MonkeyPatch,
                          capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["applicant-ranking", "--help"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    assert "applicant-ranking" in capsys.readouterr().out


def test_every_documented_command_exists() -> None:
    runner = CliRunner()
    for args in (["criteria", "lint"], ["criteria", "draft"], ["postings", "list"], ["rank"],
                 ["evaluate"], ["evaluate-file"], ["backtest"], ["fairness"],
                 ["export-audit"], ["purge"], ["hold", "set"], ["hold", "clear"], ["doctor"],
                 ["serve"]):
        result = runner.invoke(cli.app, [*args, "--help"])
        assert result.exit_code == 0, (args, result.output)
