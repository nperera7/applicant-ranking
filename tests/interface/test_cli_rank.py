"""``rank``, ``evaluate`` and ``evaluate-file``: rendering, modes, gates and exit codes."""

from __future__ import annotations

import csv
import io
import json
import stat
from collections.abc import Callable
from pathlib import Path

import pytest

from applicant_ranking import cli
from applicant_ranking.config import ConfigError
from applicant_ranking.criteria import dump_criteria_yaml
from applicant_ranking.domain import Requirement, RoleCriteria, RunMode, Status
from applicant_ranking.lever.client import LeverAuthError
from applicant_ranking.llm.base import LLMConfigError
from applicant_ranking.pipeline import PolicyBlocked, RankingError
from applicant_ranking.policy import PolicyDecision

from .conftest import CANDIDATE, SECRETS, CliHarness, make_evaluation

ADVISORY = "Advisory ranking - not a decision. Review every application."
EVAL_ONLY = "EVALUATION ONLY - not for hiring decisions"


def test_rank_table_happy_path(harness: CliHarness) -> None:
    result = harness.invoke("rank", "--posting", "posting-1")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[0] == ADVISORY
    assert CANDIDATE in result.stdout
    assert "Error Person" in result.stdout  # unranked rows are listed too
    [(args, kwargs)] = harness.ranker.called("rank_posting")
    assert args == ("posting-1",)
    assert kwargs == {"mode": RunMode.DRY_RUN, "limit": None, "include_archived": False,
                      "force": False}
    assert harness.built.kwargs == [{}]  # Lever and the LLM are both needed
    assert harness.built.closed == 1


def test_rank_passes_options(harness: CliHarness) -> None:
    result = harness.invoke("rank", "--posting", "posting-1", "--limit", "5",
                            "--include-archived", "--force", "--mode", "evaluation")

    assert result.exit_code == 0, result.output
    [(_, kwargs)] = harness.ranker.called("rank_posting")
    assert kwargs == {"mode": RunMode.EVALUATION, "limit": 5, "include_archived": True,
                      "force": True}
    assert result.stdout.splitlines()[0] == EVAL_ONLY


@pytest.mark.parametrize("fmt", ["csv", "json", "md"])
def test_rank_formats(harness: CliHarness, fmt: str) -> None:
    result = harness.invoke("rank", "--posting", "posting-1", "--format", fmt)

    assert result.exit_code == 0, result.output
    out = result.stdout
    if fmt == "csv":
        rows = list(csv.reader(io.StringIO(out)))
        assert rows[0] == [ADVISORY]
        assert rows[1][0] == "rank"
        assert rows[2][1] == CANDIDATE
    elif fmt == "json":
        payload = json.loads(out)
        assert payload["watermark"] == ADVISORY
        assert len(payload["rows"]) == 3
    else:
        assert out.splitlines()[0] == ADVISORY
        assert "| Rank |" in out


def test_rank_out_writes_private_file(harness: CliHarness, tmp_path: Path) -> None:
    out = tmp_path / "reports" / "nested" / "posting-1.csv"

    result = harness.invoke("rank", "--posting", "posting-1", "--format", "csv",
                            "--out", str(out))

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == f"wrote {out}"
    assert ADVISORY in result.stderr
    assert CANDIDATE not in result.output  # the report went to the file only
    assert out.read_text(encoding="utf-8").startswith(ADVISORY)
    assert stat.S_IMODE(out.stat().st_mode) == 0o600


def test_rank_out_tightens_existing_file(harness: CliHarness, tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    out.write_text("old")
    out.chmod(0o644)

    result = harness.invoke("rank", "--posting", "posting-1", "--format", "md",
                            "--out", str(out))

    assert result.exit_code == 0, result.output
    assert "old" not in out.read_text()
    assert stat.S_IMODE(out.stat().st_mode) == 0o600


def test_live_mode_requires_write_enabled(harness: CliHarness) -> None:
    result = harness.invoke("rank", "--posting", "posting-1", "--mode", "live", "--yes")

    assert result.exit_code == 2
    assert "AR_WRITE_ENABLED" in result.stderr
    assert harness.ranker.calls == []
    assert harness.built.kwargs == []  # refused before building anything


def test_live_mode_asks_for_confirmation(harness: CliHarness, monkeypatch: pytest.MonkeyPatch,
                                         make_settings: Callable[..., object]) -> None:
    live = harness.settings.model_copy(update={"write_enabled": True})
    monkeypatch.setattr(cli, "_load_settings", lambda: live)

    declined = harness.invoke("rank", "--posting", "posting-1", "--mode", "live", input="n\n")
    assert declined.exit_code == 1
    assert "Aborted" in declined.output
    assert harness.ranker.calls == []

    accepted = harness.invoke("rank", "--posting", "posting-1", "--mode", "live", input="y\n")
    assert accepted.exit_code == 0, accepted.output
    assert "writes tags and notes to Lever" in accepted.stderr

    skipped = harness.invoke("evaluate", "--opportunity", "opp-7", "--mode", "live", "--yes")
    assert skipped.exit_code == 0, skipped.output
    modes = [k["mode"] for _, k in harness.ranker.called("rank_posting")]
    modes += [k["mode"] for _, k in harness.ranker.called("evaluate_opportunity")]
    assert modes == [RunMode.LIVE, RunMode.LIVE]


def test_policy_blocked_exits_2_with_blockers(harness: CliHarness) -> None:
    decision = PolicyDecision(allowed=False, regimes=["IL_HRA"],
                              blockers=["no jurisdiction policy file", "posting country unknown"])
    harness.ranker.error = PolicyBlocked(decision)

    result = harness.invoke("rank", "--posting", "posting-1")

    assert result.exit_code == 2
    assert "blocked by the jurisdiction policy:" in result.stderr
    assert "  - no jurisdiction policy file" in result.stderr
    assert "  - posting country unknown" in result.stderr
    assert "Traceback" not in result.output
    assert harness.built.closed == 1  # components closed even on failure


@pytest.mark.parametrize(("error", "code"), [
    (RankingError("no criteria for posting posting-1; run 'criteria draft'"), 2),
    (ConfigError("AR_GCP_PROJECT is not set"), 3),
    (LLMConfigError("model not found"), 3),
    (LeverAuthError("Lever GET /postings returned 401", status=401), 3),
    (RuntimeError("database is locked"), 1),
])
def test_exit_codes(harness: CliHarness, error: Exception, code: int) -> None:
    harness.ranker.error = error

    result = harness.invoke("rank", "--posting", "posting-1")

    assert result.exit_code == code
    assert str(error).splitlines()[0] in result.stderr
    assert "Traceback" not in result.output
    assert harness.built.closed == 1


def test_build_config_errors_exit_3(harness: CliHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    def invalid_policy(settings: object, **kwargs: object) -> object:
        raise ValueError("policy file config/policy.yaml is invalid: unknown key")

    monkeypatch.setattr(cli, "_build", invalid_policy)

    result = harness.invoke("rank", "--posting", "posting-1")

    assert result.exit_code == 3
    assert "policy file config/policy.yaml is invalid" in result.stderr


def test_debug_shows_traceback(harness: CliHarness) -> None:
    harness.ranker.error = RuntimeError("boom\nsecond line")

    quiet = harness.invoke("rank", "--posting", "posting-1")
    loud = harness.invoke("--debug", "rank", "--posting", "posting-1")

    assert quiet.exit_code == loud.exit_code == 1
    assert quiet.stderr.splitlines()[-1] == "error: boom"
    assert "Traceback" in loud.stderr
    assert "second line" in loud.stderr


def test_error_message_is_one_clean_line(harness: CliHarness) -> None:
    harness.ranker.error = RuntimeError("\x1b[31mred\x1b[0m " + "x" * 1000)

    result = harness.invoke("rank", "--posting", "posting-1")

    line = result.stderr.splitlines()[-1]
    assert line.startswith("error: ")
    assert "\x1b" not in line
    assert len(line) < 320


def test_evaluate_one_applicant(harness: CliHarness) -> None:
    result = harness.invoke("evaluate", "--opportunity", "opp-7", "--force", "--format", "json")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["watermark"] == ADVISORY
    assert payload["posting_id"] == "posting-1"
    assert payload["rows"][0]["evaluation"]["opportunity_id"] == "opp-7"
    [(args, kwargs)] = harness.ranker.called("evaluate_opportunity")
    assert args == ("opp-7",)
    assert kwargs == {"mode": RunMode.DRY_RUN, "force": True, "defer_pending": True}


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


def test_evaluate_file_runs_offline_in_evaluation_mode(
        harness: CliHarness, criteria_file: Path, resume_file: Path,
        sample_resume_text: str) -> None:
    result = harness.invoke("evaluate-file", str(resume_file), "--criteria", str(criteria_file),
                            "--name", "Jordan Lee")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[0] == EVAL_ONLY
    assert harness.built.kwargs == [{"need_lever": False}]
    [(_, kwargs)] = harness.ranker.called("evaluate_local")
    assert kwargs["data"] == resume_file.read_bytes()
    assert kwargs["file_name"] == "resume.txt"
    assert kwargs["candidate_names"] == ["Jordan Lee"]
    assert kwargs["criteria"].role_id == "senior-data-engineer"
    assert kwargs["opportunity_id"] == "local"
    for line in sample_resume_text.splitlines():  # resume text is never printed
        if len(line) > 30:
            assert line not in result.output


def test_evaluate_file_without_name_warns_on_stderr(harness: CliHarness, criteria_file: Path,
                                                    resume_file: Path) -> None:
    result = harness.invoke("evaluate-file", str(resume_file), "--criteria", str(criteria_file))

    assert result.exit_code == 0, result.output
    assert cli.NO_NAME_WARNING in result.stderr
    assert harness.ranker.called("evaluate_local")[0][1]["candidate_names"] == []

    named = harness.invoke("evaluate-file", str(resume_file), "--criteria", str(criteria_file),
                           "--name", "Jordan Lee")
    assert named.exit_code == 0, named.output
    assert cli.NO_NAME_WARNING not in named.stderr


def test_evaluate_file_unranked_row(harness: CliHarness, criteria_file: Path,
                                    resume_file: Path) -> None:
    harness.ranker.local_evaluation = make_evaluation(
        mode=RunMode.EVALUATION, status=Status.ERROR, score=None, tier=None,
        candidate_name=None)

    result = harness.invoke("evaluate-file", str(resume_file), "--criteria", str(criteria_file),
                            "--format", "json")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["watermark"] == EVAL_ONLY
    assert payload["rows"][0]["rank"] is None
    assert harness.ranker.called("evaluate_local")[0][1]["candidate_names"] == []


def test_evaluate_file_invalid_criteria_exits_2(harness: CliHarness, tmp_path: Path,
                                                resume_file: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("role_id: Not A Slug\n")

    result = harness.invoke("evaluate-file", str(resume_file), "--criteria", str(bad))

    assert result.exit_code == 2
    assert "bad.yaml" in result.stderr
    assert harness.built.kwargs == []


def test_evaluate_file_lint_error_from_ranker_exits_2(
        harness: CliHarness, criteria_file: Path, resume_file: Path) -> None:
    harness.ranker.error = RankingError("criteria have lint errors")

    result = harness.invoke("evaluate-file", str(resume_file), "--criteria", str(criteria_file))

    assert result.exit_code == 2
    assert "criteria have lint errors" in result.stderr


def test_evaluate_file_missing_resume_is_a_usage_error(harness: CliHarness,
                                                       criteria_file: Path,
                                                       tmp_path: Path) -> None:
    result = harness.invoke("evaluate-file", str(tmp_path / "missing.pdf"), "--criteria",
                            str(criteria_file))

    assert result.exit_code == 2
    assert harness.ranker.calls == []


def test_evaluate_file_refuses_oversized_resume(harness: CliHarness, criteria_file: Path,
                                                tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    small = harness.settings.model_copy(update={"max_resume_bytes": 1024})
    monkeypatch.setattr(cli, "_load_settings", lambda: small)
    big = tmp_path / "big.txt"
    big.write_bytes(b"a" * 2048)

    result = harness.invoke("evaluate-file", str(big), "--criteria", str(criteria_file))

    assert result.exit_code == 2
    assert "AR_MAX_RESUME_BYTES" in result.stderr
    assert harness.ranker.calls == []


def test_unapproved_criteria_reach_the_ranker(harness: CliHarness, tmp_path: Path,
                                              resume_file: Path) -> None:
    unapproved = RoleCriteria(role_id="draft-role", title="Draft role", version=1,
                              requirements=[Requirement(id="python",
                                                        description="Professional Python")])
    path = tmp_path / "draft.yaml"
    path.write_text(dump_criteria_yaml(unapproved))

    result = harness.invoke("evaluate-file", str(resume_file), "--criteria", str(path))

    assert result.exit_code == 0, result.output  # evaluation mode allows unapproved criteria


def test_no_secrets_in_any_output(harness: CliHarness) -> None:
    result = harness.invoke("rank", "--posting", "posting-1", "--format", "json")

    for secret in SECRETS:
        assert secret not in result.output
