"""The CLI on the real wiring and pipeline, with the scripted FakeEvaluator instead of
Gemini and no Lever client (offline commands only)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from applicant_ranking import cli, wiring
from applicant_ranking.config import Settings
from applicant_ranking.criteria import dump_criteria_yaml
from applicant_ranking.domain import RoleCriteria
from applicant_ranking.llm.fake import FakeEvaluator
from applicant_ranking.wiring import Components

EVAL_ONLY = "EVALUATION ONLY - not for hiring decisions"


@pytest.fixture
def real_cli(monkeypatch: pytest.MonkeyPatch, iface_settings: Settings,
             ) -> tuple[CliRunner, FakeEvaluator, list[dict[str, Any]]]:
    evaluator = FakeEvaluator()
    calls: list[dict[str, Any]] = []

    def build(settings: Settings, **kwargs: Any) -> Components:
        calls.append(kwargs)
        return wiring.build_components(settings, evaluator=evaluator, **kwargs)

    monkeypatch.setattr(cli, "_load_settings", lambda: iface_settings)
    monkeypatch.setattr(cli, "_build", build)
    return CliRunner(), evaluator, calls


def test_evaluate_file_end_to_end(real_cli: tuple[CliRunner, FakeEvaluator, list[Any]],
                                  tmp_path: Path, sample_criteria: RoleCriteria,
                                  sample_resume_text: str) -> None:
    runner, evaluator, calls = real_cli
    criteria = tmp_path / "role.yaml"
    criteria.write_text(dump_criteria_yaml(sample_criteria))
    resume = tmp_path / "resume.txt"
    resume.write_text("Jordan Lee\njordan@example.com\n" + sample_resume_text)

    result = runner.invoke(cli.app, ["evaluate-file", str(resume), "--criteria", str(criteria),
                                     "--name", "Jordan Lee", "--format", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["watermark"] == EVAL_ONLY
    [row] = payload["rows"]
    assert row["rank"] == 1
    assert row["evaluation"]["mode"] == "evaluation"
    assert row["evaluation"]["score"] is not None
    assert calls == [{"need_lever": False}]
    [request] = evaluator.requests
    assert "Jordan Lee" not in request.resume_text  # redacted before the model
    assert "jordan@example.com" not in request.resume_text
    assert evaluator.closed  # components were closed


def test_evaluate_file_lint_errors_block(real_cli: tuple[CliRunner, FakeEvaluator, list[Any]],
                                         tmp_path: Path, sample_criteria: RoleCriteria,
                                         sample_resume_text: str) -> None:
    runner, evaluator, _ = real_cli
    biased = sample_criteria.model_copy(update={"summary": "Ideal for a young team."})
    criteria = tmp_path / "role.yaml"
    criteria.write_text(dump_criteria_yaml(biased))
    resume = tmp_path / "resume.txt"
    resume.write_text(sample_resume_text)

    result = runner.invoke(cli.app, ["evaluate-file", str(resume), "--criteria", str(criteria)])

    assert result.exit_code == 2
    assert "lint" in result.stderr
    assert evaluator.requests == []


def test_purge_and_hold_without_credentials(
        real_cli: tuple[CliRunner, FakeEvaluator, list[Any]],
        monkeypatch: pytest.MonkeyPatch, make_settings: Callable[..., Settings]) -> None:
    runner, _, calls = real_cli
    bare = make_settings(lever_api_key=None, gcp_project=None)
    monkeypatch.setattr(cli, "_load_settings", lambda: bare)

    held = runner.invoke(cli.app, ["hold", "set", "--opportunity", "opp-1", "--reason", "x"])
    purged = runner.invoke(cli.app, ["purge", "--yes", "--reconcile-days", "0"])
    needs_lever = runner.invoke(cli.app, ["purge", "--yes"])

    assert held.exit_code == 0, held.output
    assert purged.exit_code == 0, purged.output
    assert "retention: 0 evaluation(s)" in purged.stdout
    assert needs_lever.exit_code == 3
    assert "AR_LEVER_API_KEY" in needs_lever.stderr
    assert calls[0] == {"need_llm": False, "need_lever": False}
