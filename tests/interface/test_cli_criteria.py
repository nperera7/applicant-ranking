"""``criteria lint``, ``criteria draft`` and ``postings list``."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from applicant_ranking import cli
from applicant_ranking.config import ConfigError
from applicant_ranking.criteria import load_criteria_file
from applicant_ranking.domain import Requirement, RoleCriteria
from applicant_ranking.lever.models import Posting

from .conftest import CliHarness

ADVISORY = "Advisory ranking - not a decision. Review every application."
# Lever lists only public-site postings unless both distribution channels are requested.
BOTH_SITES = ("public", "internal")

POSTING = Posting.model_validate({
    "id": "posting-9",
    "text": "Platform Engineer",
    "state": "published",
    "categories": {"location": "Chicago, IL", "allLocations": ["Chicago, IL", "Remote - US"]},
    "content": {
        "description": "Build the platform.",
        "lists": [
            {"text": "Requirements",
             "content": "<li>Five years operating Kubernetes in production</li>"
                        "<li>Terraform for multi-environment infrastructure</li>"},
            {"text": "Nice to have", "content": "<li>Go services in production</li>"},
            {"text": "Benefits", "content": "<li>Free lunch every day</li>"},
        ],
    },
})


def test_lint_clean_criteria_passes(harness: CliHarness, sample_criteria: RoleCriteria,
                                    write_criteria: Callable[..., Path]) -> None:
    path = write_criteria(sample_criteria)

    result = harness.invoke("criteria", "lint")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[0] == ADVISORY
    assert str(path) in result.stdout
    assert "senior-data-engineer v2 (approved by Pat Example; posting-1)" in result.stdout
    assert "1 file(s) checked: 0 error(s)" in result.stdout
    assert harness.built.kwargs == []  # no components (and no credentials) needed


def test_lint_reports_protected_terms_and_exits_2(
        harness: CliHarness, sample_criteria: RoleCriteria,
        write_criteria: Callable[..., Path]) -> None:
    bad = sample_criteria.model_copy(update={
        "role_id": "biased-role", "posting_ids": ["posting-2"],
        "requirements": [Requirement(id="young", description="Young and energetic engineer "
                                                               "who is a digital native")],
    })
    write_criteria(sample_criteria)
    write_criteria(bad)

    result = harness.invoke("criteria", "lint")

    assert result.exit_code == 2
    assert "requirements[young].description" in result.stdout
    assert "error" in result.stdout
    assert "2 file(s) checked" in result.stdout


def test_lint_reports_unparseable_and_duplicate_files(
        harness: CliHarness, sample_criteria: RoleCriteria,
        write_criteria: Callable[..., Path]) -> None:
    write_criteria(sample_criteria, "a.yaml")
    write_criteria(sample_criteria.model_copy(update={"role_id": "other-role"}), "b.yaml")
    (harness.settings.criteria_dir / "broken.yml").write_text("role_id: [unclosed\n")

    result = harness.invoke("criteria", "lint")

    assert result.exit_code == 2
    assert "broken.yml: cannot be loaded" in result.stdout
    assert "invalid YAML" in result.stdout
    # The two files claim the same posting: both are linted, and the clash is reported.
    assert "posting id 'posting-1' is listed in more than one file" in result.stdout
    assert "a.yaml: senior-data-engineer v2" in result.stdout
    assert "b.yaml: other-role v2" in result.stdout


def test_lint_dir_option_and_missing_dir(harness: CliHarness, tmp_path: Path,
                                         sample_criteria: RoleCriteria) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "role.yaml").write_text(json.dumps(sample_criteria.model_dump(mode="json")))

    ok = harness.invoke("criteria", "lint", "--dir", str(other))
    missing = harness.invoke("criteria", "lint", "--dir", str(tmp_path / "nope"))

    assert ok.exit_code == 0, ok.output
    assert "role.yaml" in ok.stdout
    assert missing.exit_code == 2
    assert "criteria directory not found" in missing.stderr


def test_lint_bad_lexicon_is_a_configuration_error(
        harness: CliHarness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        make_settings: Callable[..., object]) -> None:
    settings = make_settings(criteria_dir=harness.settings.criteria_dir,
                             protected_terms_file=tmp_path / "missing-terms.yaml")
    monkeypatch.setattr(cli, "_load_settings", lambda: settings)

    result = harness.invoke("criteria", "lint")

    assert result.exit_code == 3
    assert result.stderr.startswith("configuration error:")
    assert "Traceback" not in result.output


def test_draft_writes_skeleton_into_criteria_dir(harness: CliHarness) -> None:
    harness.lever.postings = [POSTING]

    result = harness.invoke("criteria", "draft", "--posting", "posting-9")

    assert result.exit_code == 0, result.output
    target = harness.settings.criteria_dir / "platform-engineer.yaml"
    assert f"wrote {target}" in result.stdout
    draft = load_criteria_file(target)
    assert draft.posting_ids == ["posting-9"]
    assert draft.source == "job_description"
    assert not draft.approved
    assert [r.importance for r in draft.requirements] == ["must_have", "must_have",
                                                          "nice_to_have"]
    assert target.read_text().startswith("# Per-role criteria")
    assert harness.built.kwargs == [{"need_llm": False}]
    assert harness.built.closed == 1


def test_draft_refuses_to_overwrite_without_force(harness: CliHarness, tmp_path: Path) -> None:
    harness.lever.postings = [POSTING]
    out = tmp_path / "drafts" / "draft.yaml"
    out.parent.mkdir()
    out.write_text("keep me")

    refused = harness.invoke("criteria", "draft", "--posting", "posting-9", "--out", str(out))
    assert refused.exit_code == 2
    assert "--force" in refused.stderr
    assert out.read_text() == "keep me"

    forced = harness.invoke("criteria", "draft", "--posting", "posting-9", "--out", str(out),
                            "--force")
    assert forced.exit_code == 0, forced.output
    assert load_criteria_file(out).role_id == "platform-engineer"
    assert harness.built.closed == 2


def test_draft_refuses_to_shadow_existing_criteria(
        harness: CliHarness, sample_criteria: RoleCriteria,
        write_criteria: Callable[..., Path], tmp_path: Path) -> None:
    existing = write_criteria(sample_criteria.model_copy(update={"posting_ids": ["posting-9"]}))
    harness.lever.postings = [POSTING]

    result = harness.invoke("criteria", "draft", "--posting", "posting-9")

    assert result.exit_code == 2
    assert str(existing) in result.stderr
    assert not (harness.settings.criteria_dir / "platform-engineer.yaml").exists()
    # Outside the criteria directory it is fine (nothing loads it).
    elsewhere = tmp_path / "draft.yaml"
    ok = harness.invoke("criteria", "draft", "--posting", "posting-9", "--out", str(elsewhere))
    assert ok.exit_code == 0, ok.output


def test_draft_unknown_posting_is_a_runtime_error(harness: CliHarness) -> None:
    result = harness.invoke("criteria", "draft", "--posting", "nope")

    assert result.exit_code == 1
    assert "404" in result.stderr
    assert harness.built.closed == 1


def test_draft_without_lever_key_is_a_configuration_error(
        harness: CliHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_key(settings: object, **kwargs: object) -> object:
        raise ConfigError("AR_LEVER_API_KEY is not set")

    monkeypatch.setattr(cli, "_build", no_key)

    result = harness.invoke("criteria", "draft", "--posting", "posting-9")

    assert result.exit_code == 3
    assert "AR_LEVER_API_KEY is not set" in result.stderr


def test_postings_list_shows_criteria_status(
        harness: CliHarness, sample_criteria: RoleCriteria,
        write_criteria: Callable[..., Path]) -> None:
    write_criteria(sample_criteria)
    harness.lever.postings = [
        POSTING,
        Posting.model_validate({"id": "posting-1", "text": "Senior Data Engineer",
                                "state": "published",
                                "categories": {"location": "New York, NY"}}),
        Posting.model_validate({"id": "posting-old", "text": "Old", "state": "closed"}),
    ]

    result = harness.invoke("postings", "list")

    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[0] == ADVISORY
    assert lines[1].split() == ["ID", "TITLE", "STATE", "LOCATION", "CRITERIA"]
    row = next(line for line in lines if line.startswith("posting-1 "))
    assert "senior-data-engineer v2 (approved)" in row
    assert "New York, NY" in row
    nine = next(line for line in lines if line.startswith("posting-9 "))
    assert "Chicago, IL; Remote - US" in nine
    assert "posting-old" not in result.stdout
    assert ("list_postings", {"state": "published", "distribution_channels": BOTH_SITES,
                              "confidentiality": None}) in harness.lever.calls

    everything = harness.invoke("postings", "list", "--state", "all")
    assert "posting-old" in everything.stdout
    assert ("list_postings", {"state": None, "distribution_channels": BOTH_SITES,
                              "confidentiality": None}) in harness.lever.calls


def test_postings_list_empty(harness: CliHarness) -> None:
    result = harness.invoke("postings", "list", "--state", "draft")

    assert result.exit_code == 0
    assert "no postings (state: draft)" in result.stdout


def test_postings_list_includes_confidential_postings_only_when_asked(
        harness: CliHarness) -> None:
    """Lever hides confidential postings unless confidentiality=all is sent; the flag asks
    for them (and documents that the key then needs confidential access)."""
    secret = POSTING.model_copy(update={"id": "posting-secret", "text": "Confidential Role",
                                        "confidentiality": "confidential"})
    harness.lever.postings = [POSTING, secret]

    hidden = harness.invoke("postings", "list")
    assert hidden.exit_code == 0, hidden.output
    assert "posting-secret" not in hidden.stdout
    assert ("list_postings", {"state": "published", "distribution_channels": BOTH_SITES,
                              "confidentiality": None}) in harness.lever.calls

    shown = harness.invoke("postings", "list", "--include-confidential")
    assert shown.exit_code == 0, shown.output
    assert "posting-secret" in shown.stdout
    assert ("list_postings", {"state": "published", "distribution_channels": BOTH_SITES,
                              "confidentiality": "all"}) in harness.lever.calls


def test_postings_list_follows_the_confidential_access_setting(
        harness: CliHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    """AR_LEVER_CONFIDENTIAL_ACCESS=true records that the key may read confidential
    postings, so the listing asks for them without the flag."""
    settings = harness.settings.model_copy(update={"lever_confidential_access": True})
    monkeypatch.setattr(cli, "_load_settings", lambda: settings)
    secret = POSTING.model_copy(update={"id": "posting-secret", "text": "Confidential Role",
                                        "confidentiality": "confidential"})
    harness.lever.postings = [POSTING, secret]

    result = harness.invoke("postings", "list")

    assert result.exit_code == 0, result.output
    assert "posting-secret" in result.stdout
    assert ("list_postings", {"state": "published", "distribution_channels": BOTH_SITES,
                              "confidentiality": "all"}) in harness.lever.calls
