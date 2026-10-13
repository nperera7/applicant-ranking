from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from applicant_ranking.criteria import (
    FIELD_GUIDE,
    MAX_CRITERIA_FILE_BYTES,
    CriteriaError,
    CriteriaRepository,
    classify_list_title,
    draft_from_posting,
    dump_criteria_yaml,
    load_criteria_file,
    parse_criteria_yaml,
    slugify,
)
from applicant_ranking.criteria_lint import has_errors, lint_criteria
from applicant_ranking.domain import SLUG_PATTERN, Knockout, Requirement, RoleCriteria
from applicant_ranking.lever.models import Posting
from applicant_ranking.scoring.lexicon import ProtectedTermLexicon

RSQUO = chr(0x2019)

# GET /postings/:id example from the Lever Data API docs (data field), verbatim.
LEVER_POSTING_EXAMPLE: dict[str, Any] = {
    "id": "f2f01e16-27f8-4711-a728-7d49499795a0",
    "text": "Infrastructure Engineer",
    "createdAt": 1407779365624,
    "updatedAt": 1407779365624,
    "user": "ec1cb1bb-8b58-4834-bc6e-b2af06296e4a",
    "owner": "ec1cb1bb-8b58-4834-bc6e-b2af06296e4a",
    "hiringManager": "ec1cb1bb-8b58-4834-bc6e-b2af06296e4a",
    "confidentiality": "non-confidential",
    "categories": {
        "team": "Platform",
        "department": "Engineering",
        "location": "San Francisco",
        "locationDetails": {"addressLine1": "185 Berry St", "addressLine2": "Suite 5000",
                            "city": "San Francisco", "region": "CA", "postalCode": "94107",
                            "country": "US"},
        "allLocations": ["San Francisco", "New York City"],
        "commitment": "Full-time",
        "level": "Senior",
    },
    "content": {
        "description": "The Infrastructure Engineer will act as...\\nSuperman.",
        "descriptionHtml": "<div>The <u><b>Infrastructure Engineer</b></u> will act as..."
                           "</div><div>Superman.</div>",
        "lists": [{"text": "Job requirements",
                   "content": "<li>Quick learner</li><li>Ambitious</li>"}],
        "closing": "Our company is proud to be an equal opportunity workplace.",
        "closingHtml": "<div>Our <a href=\"https://www.lever.co/\" class=\"postings-link\">"
                       "company</a> is <span style=\"font-size: 18px;\">proud</span> to be an "
                       "equal opportunity workplace.</div>",
    },
    "country": "US",
    "tags": [],
    "state": "published",
    "distributionChannels": ["internal", "public"],
    "reqCode": "7381912",
    "requisitionCodes": ["7381912"],
    "salaryDescription": "This is a salary description.",
    "salaryDescriptionHtml": "<p>This is a salary description.</p>",
    "salaryRange": {"max": 60000, "min": 40000, "currency": "USD",
                    "interval": "per-year-salary"},
    "urls": {"list": "https://jobs.lever.co/example",
             "show": "https://jobs.lever.co/example/f2f01e16-27f8-4711-a728-7d49499795a0/",
             "apply": "https://jobs.lever.co/example/f2f01e16-27f8-4711-a728-7d49499795a0/apply"},
    "workplaceType": "remote",
}

MINIMAL_YAML = """\
role_id: data-analyst
title: Data Analyst
version: 1
posting_ids: [p-1, p-2]
requirements:
  - id: sql
    description: Writes analytical SQL against a warehouse.
"""


def _posting(**content: Any) -> Posting:
    data = {"id": "post-1", "text": "Senior Data Engineer", "content": content}
    return Posting.model_validate(data)


def _criteria_yaml(role_id: str, posting_ids: list[str]) -> str:
    return yaml.safe_dump({
        "role_id": role_id, "title": role_id.title(), "version": 1, "posting_ids": posting_ids,
        "requirements": [{"id": "sql", "description": "Writes analytical SQL daily."}],
    })


# ------------------------------------------------------------------------- load


def test_load_criteria_file(tmp_path: Path) -> None:
    path = tmp_path / "analyst.yaml"
    path.write_text(MINIMAL_YAML + "approved_by: Pat Lee\napproved_at: 2026-10-01T15:30:00Z\n",
                    encoding="utf-8")
    criteria = load_criteria_file(path)
    assert criteria.role_id == "data-analyst"
    assert criteria.posting_ids == ["p-1", "p-2"]
    assert criteria.requirements[0].importance == "must_have"
    assert criteria.approved_at == datetime(2026, 10, 1, 15, 30, tzinfo=UTC)
    assert criteria.approved


@pytest.mark.parametrize(("value", "expected"), [
    ("2026-10-01", datetime(2026, 10, 1, tzinfo=UTC)),
    ("2026-10-01 08:00:00", datetime(2026, 10, 1, 8, tzinfo=UTC)),
    ("'2026-10-01T08:00:00+02:00'", datetime(2026, 10, 1, 6, tzinfo=UTC)),
])
def test_approved_at_forms(value: str, expected: datetime) -> None:
    criteria = parse_criteria_yaml(MINIMAL_YAML + f"approved_by: Pat\napproved_at: {value}\n")
    assert criteria.approved_at == expected
    assert criteria.approved_at.tzinfo is not None


def test_approved_terms_load_round_trip_and_change_the_content_hash() -> None:
    text = MINIMAL_YAML + (
        "approved_terms:\n"
        "  - term: credit history\n"
        "    justification: The product scores credit histories; the role builds it.\n"
        "    approved_by: Legal (A. Counsel)\n"
        "    approved_at: 2026-10-12\n"
        "  - term: Veterans Affairs\n"
        "    justification: The client is the Department of Veterans Affairs.\n"
        "    approved_by: HR\n"
        "    approved_at: 2026-10-12T09:00:00\n"
    )
    criteria = parse_criteria_yaml(text)
    assert [t.term for t in criteria.approved_terms] == ["credit history", "Veterans Affairs"]
    assert criteria.approved_terms[0].approved_at == date(2026, 10, 12)
    assert criteria.approved_terms[1].approved_at == datetime(2026, 10, 12, 9, tzinfo=UTC)
    plain = parse_criteria_yaml(MINIMAL_YAML)
    assert plain.approved_terms == []
    assert plain.content_hash() != criteria.content_hash()  # what the criteria may say changed
    assert parse_criteria_yaml(dump_criteria_yaml(criteria)) == criteria
    with pytest.raises(CriteriaError, match="approved_terms"):
        parse_criteria_yaml(MINIMAL_YAML + "approved_terms:\n  - term: x\n")


@pytest.mark.parametrize(("text", "message"), [
    ("", "file is empty"),
    ("- a\n- b\n", "expected a mapping"),
    ("role_id: [unclosed\n", r"invalid YAML \(line \d+, column \d+\)"),
    (MINIMAL_YAML.replace("id: sql", "id: Bad Id"), r"requirements\.0\.id: String should match"),
    (MINIMAL_YAML + "colour: blue\n", "colour: Extra inputs are not permitted"),
    (MINIMAL_YAML.replace("version: 1", "version: 0"), "version: Input should be greater"),
    ("role_id: x\ntitle: X\nversion: 1\nrequirements: []\n", "requirements: List should have"),
])
def test_load_errors_name_the_file_and_field(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(CriteriaError, match=message) as excinfo:
        load_criteria_file(path)
    assert str(path) in str(excinfo.value)


def test_load_errors_for_unreadable_files(tmp_path: Path) -> None:
    with pytest.raises(CriteriaError, match="cannot read file"):
        load_criteria_file(tmp_path / "missing.yaml")
    binary = tmp_path / "binary.yaml"
    binary.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(CriteriaError, match="not valid UTF-8"):
        load_criteria_file(binary)
    big = tmp_path / "big.yaml"
    big.write_text("#" * (MAX_CRITERIA_FILE_BYTES + 1), encoding="utf-8")
    with pytest.raises(CriteriaError, match="too large"):
        load_criteria_file(big)


def test_criteria_error_is_a_value_error() -> None:
    assert issubclass(CriteriaError, ValueError)


# ------------------------------------------------------------------------- dump


def test_dump_round_trips(sample_criteria: RoleCriteria, tmp_path: Path) -> None:
    text = dump_criteria_yaml(sample_criteria)
    path = tmp_path / "round.yaml"
    path.write_text(text, encoding="utf-8")
    loaded = load_criteria_file(path)
    assert loaded == sample_criteria
    assert loaded.content_hash() == sample_criteria.content_hash()
    assert loaded.approved_at == datetime(2026, 10, 1, tzinfo=UTC)
    assert "approved_at: 2026-10-01T00:00:00+00:00" in text


def test_dump_layout(sample_criteria: RoleCriteria) -> None:
    text = dump_criteria_yaml(sample_criteria)
    keys = [line.split(":")[0] for line in text.splitlines() if re.match(r"^[a-z_]+:", line)]
    assert keys == list(RoleCriteria.model_fields)
    assert "weight" not in text.split("id: streaming")[0]  # None weights omitted
    assert "weight: 2" in text
    assert "{" not in text  # block style
    assert "  - id: gcp-pipelines" in text


def test_dump_omits_none_and_keeps_unicode() -> None:
    criteria = RoleCriteria(
        role_id="analyst", title="Analista de Datos (São Paulo)", version=1,
        summary="Line one.\nLine two: café ≥ 3 años.",
        requirements=[Requirement(id="sql", description="SQL — window functions")],
    )
    text = dump_criteria_yaml(criteria)
    assert "approved_by" not in text and "approved_at" not in text and "weight" not in text
    assert "São Paulo" in text and "café ≥ 3 años" in text and "SQL — window" in text
    assert "summary: |" in text
    assert parse_criteria_yaml(text) == criteria


def test_dump_header_is_commented(sample_criteria: RoleCriteria) -> None:
    text = dump_criteria_yaml(sample_criteria, header=FIELD_GUIDE)
    head = text.split("\n\n")[0].splitlines()
    assert head and all(line.startswith("#") for line in head)
    assert parse_criteria_yaml(text) == sample_criteria


# ------------------------------------------------------------------------- repository


def test_missing_directory_is_empty(tmp_path: Path) -> None:
    repo = CriteriaRepository(tmp_path / "nope")
    repo.load()
    assert repo.all() == [] and repo.errors == []
    assert repo.for_posting("p") is None and repo.by_role("r") is None


def test_directory_that_is_a_file(tmp_path: Path) -> None:
    path = tmp_path / "criteria"
    path.write_text("x", encoding="utf-8")
    repo = CriteriaRepository(path)
    repo.load()
    assert repo.all() == []
    assert repo.errors == [f"{path}: not a directory"]


def test_repository_loads_matches_and_collects_errors(tmp_path: Path) -> None:
    (tmp_path / "b-engineer.yml").write_text(_criteria_yaml("engineer", ["p-2"]), "utf-8")
    (tmp_path / "a-analyst.yaml").write_text(_criteria_yaml("analyst", ["p-1", "p-1"]), "utf-8")
    (tmp_path / "c-broken.yaml").write_text("role_id: [", "utf-8")
    (tmp_path / "notes.txt").write_text("ignored", "utf-8")
    (tmp_path / ".hidden.yaml").write_text("ignored: [", "utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.yaml").write_text(_criteria_yaml("nested", ["p-9"]), "utf-8")

    repo = CriteriaRepository(tmp_path)
    repo.load()
    assert [c.role_id for c in repo.all()] == ["analyst", "engineer"]
    assert repo.for_posting("p-1").role_id == "analyst"
    assert repo.for_posting("p-2").role_id == "engineer"
    assert repo.for_posting("p-9") is None
    assert repo.by_role("engineer").posting_ids == ["p-2"]
    assert repo.path_for("engineer") == tmp_path / "b-engineer.yml"
    assert len(repo.errors) == 1 and "c-broken.yaml" in repo.errors[0]


def test_duplicate_role_ids_skip_every_file_involved(tmp_path: Path) -> None:
    (tmp_path / "one.yaml").write_text(_criteria_yaml("engineer", ["p-1"]), "utf-8")
    (tmp_path / "two.yaml").write_text(_criteria_yaml("engineer", ["p-2"]), "utf-8")
    (tmp_path / "three.yaml").write_text(_criteria_yaml("analyst", ["p-3"]), "utf-8")
    repo = CriteriaRepository(tmp_path)
    repo.load()
    assert [c.role_id for c in repo.all()] == ["analyst"]
    assert repo.for_posting("p-1") is None and repo.for_posting("p-2") is None
    [error] = repo.errors
    assert "role_id 'engineer'" in error and "one.yaml" in error and "two.yaml" in error


def test_duplicate_posting_ids_skip_every_file_involved(tmp_path: Path) -> None:
    (tmp_path / "one.yaml").write_text(_criteria_yaml("engineer", ["p-1", "p-2"]), "utf-8")
    (tmp_path / "two.yaml").write_text(_criteria_yaml("analyst", ["p-2"]), "utf-8")
    (tmp_path / "three.yaml").write_text(_criteria_yaml("manager", ["p-3"]), "utf-8")
    repo = CriteriaRepository(tmp_path)
    repo.load()
    assert [c.role_id for c in repo.all()] == ["manager"]
    assert repo.for_posting("p-1") is None  # its file was ambiguous, so skipped entirely
    [error] = repo.errors
    assert "posting id 'p-2'" in error and "one.yaml" in error and "two.yaml" in error


def test_reload_replaces_state(tmp_path: Path) -> None:
    path = tmp_path / "engineer.yaml"
    path.write_text(_criteria_yaml("engineer", ["p-1"]), "utf-8")
    repo = CriteriaRepository(tmp_path)
    repo.load()
    assert repo.for_posting("p-1") is not None
    path.write_text(_criteria_yaml("engineer", ["p-2"]), "utf-8")
    (tmp_path / "bad.yaml").write_text("", "utf-8")
    repo.load()
    assert repo.for_posting("p-1") is None and repo.for_posting("p-2") is not None
    assert len(repo.errors) == 1
    repo.errors.clear()  # a copy
    assert len(repo.errors) == 1


def test_repository_loads_the_shipped_example() -> None:
    root = Path(__file__).resolve().parents[2] / "criteria"
    repo = CriteriaRepository(root)
    repo.load()
    assert repo.errors == []
    example = repo.by_role("example-senior-data-engineer")
    assert example is not None and example.posting_ids == [] and not example.approved


# ------------------------------------------------------------------------- drafting


def test_draft_from_documented_lever_posting(lexicon: ProtectedTermLexicon) -> None:
    posting = Posting.model_validate(LEVER_POSTING_EXAMPLE)
    draft = draft_from_posting(posting)
    assert draft.role_id == "infrastructure-engineer"
    assert draft.title == "Infrastructure Engineer"
    assert draft.version == 1
    assert draft.source == "job_description"
    assert draft.posting_ids == ["f2f01e16-27f8-4711-a728-7d49499795a0"]
    assert not draft.approved and draft.approved_by is None
    assert draft.knockouts == [] and draft.guardrails == []
    assert [(r.id, r.description, r.importance) for r in draft.requirements] == [
        ("quick-learner", "Quick learner", "must_have"),
        ("ambitious", "Ambitious", "must_have"),
    ]
    issues = lint_criteria(draft, lexicon)
    assert not has_errors(issues)
    assert {i.location for i in issues} >= {"source", "approved_by"}
    assert parse_criteria_yaml(dump_criteria_yaml(draft)) == draft


@pytest.mark.parametrize(("title", "expected"), [
    ("Job requirements", "must_have"),
    ("Requirements", "must_have"),
    ("Qualifications", "must_have"),
    ("Minimum qualifications", "must_have"),
    ("Must haves", "must_have"),
    ("Must-have skills", "must_have"),
    ("What you'll need", "must_have"),
    (f"What you{RSQUO}ll need", "must_have"),
    ("What You Will Need", "must_have"),
    ("Preferred qualifications", "nice_to_have"),
    ("Nice to have", "nice_to_have"),
    ("Bonus points", "nice_to_have"),
    ("Pluses", "nice_to_have"),
    ("A plus if you have", "nice_to_have"),
    ("Ideally you also have", "nice_to_have"),
    ("The ideal candidate", "must_have"),
    ("Responsibilities", None),
    ("What you'll do", None),
    ("Benefits", None),
    ("Perks", None),
    ("About us", None),
    ("Our values", None),
    ("", None),
])
def test_list_title_classification(title: str, expected: str | None) -> None:
    assert classify_list_title(title) == expected


def test_draft_classifies_lists_and_cleans_items() -> None:
    posting = _posting(lists=[
        {"text": "Responsibilities", "content": "<li>Build pipelines</li>"},
        {"text": "Preferred qualifications",
         "content": "<li>Experience with <b>Kafka</b> &amp; Pub/Sub</li>"},
        {"text": "Requirements", "content": (
            "<li>  5+ years of\n  <i>Python</i>  </li>"
            "<li>• SQL on BigQuery</li>"
            "<li>GCP data services<ul><li>Dataflow</li><li>Dataproc</li></ul></li>"
            "<li>sql on bigquery</li><li>ok</li><li></li>")},
        {"text": "Benefits", "content": "<li>Free lunch</li>"},
    ])
    draft = draft_from_posting(posting)
    assert [(r.description, r.importance) for r in draft.requirements] == [
        ("5+ years of Python", "must_have"),
        ("SQL on BigQuery", "must_have"),
        ("GCP data services", "must_have"),
        ("Dataflow", "must_have"),
        ("Dataproc", "must_have"),
        ("Experience with Kafka & Pub/Sub", "nice_to_have"),
    ]


def test_draft_ids_are_unique_slugs_within_40_chars() -> None:
    long_item = "Experience designing and operating large scale distributed data platforms"
    posting = _posting(lists=[{"text": "Requirements", "content": (
        f"<li>{long_item} on GCP</li><li>{long_item} on AWS</li><li>{long_item} on Azure</li>"
        "<li>日本語でのコミュニケーション</li><li>Résumé parsing in Ünïcode</li>"
        "<li>!!! ???</li>")}])
    draft = draft_from_posting(posting)
    ids = [r.id for r in draft.requirements]
    assert len(ids) == len(set(ids)) == 6
    for req_id in ids:
        assert re.fullmatch(SLUG_PATTERN, req_id), req_id
        assert len(req_id) <= 40
    assert ids[0] == "experience-designing-and-operating-large"
    assert ids[1] == "experience-designing-and-operating-lar-2"
    assert ids[3] == "requirement"
    assert ids[4] == "resume-parsing-in-unicode"
    assert ids[5] == "requirement-2"


def test_draft_caps_at_25_with_must_haves_first() -> None:
    nice = "".join(f"<li>Nice skill number {i}</li>" for i in range(10))
    must = "".join(f"<li>Required skill number {i}</li>" for i in range(30))
    posting = _posting(lists=[{"text": "Nice to have", "content": nice},
                              {"text": "Requirements", "content": must}])
    draft = draft_from_posting(posting)
    assert len(draft.requirements) == 25
    assert all(r.importance == "must_have" for r in draft.requirements)
    assert draft.requirements[0].description == "Required skill number 0"


def test_draft_trims_long_items() -> None:
    posting = _posting(lists=[{"text": "Requirements", "content": f"<li>{'word ' * 300}</li>"}])
    [req] = draft_from_posting(posting).requirements
    assert len(req.description) <= 500


def test_draft_falls_back_to_the_description() -> None:
    posting = Posting.model_validate(LEVER_POSTING_EXAMPLE | {"content": {
        **LEVER_POSTING_EXAMPLE["content"],
        "lists": [{"text": "Benefits", "content": "<li>Lunch</li>"}],
    }})
    [req] = draft_from_posting(posting).requirements
    assert req.id == "job-description"
    assert req.importance == "must_have"
    assert req.description == "The Infrastructure Engineer will act as... Superman."


def test_draft_fallback_uses_html_and_trims_to_500() -> None:
    html = "<div>Build <b>pipelines</b>.</div><p>" + "Operate them well. " * 60 + "</p>"
    [req] = draft_from_posting(_posting(descriptionHtml=html)).requirements
    assert req.description.startswith("Build pipelines. Operate them well.")
    assert len(req.description) <= 500


def test_draft_with_no_content_gets_a_placeholder() -> None:
    draft = draft_from_posting(Posting.model_validate({"id": "p-77", "text": "  "}))
    assert draft.title == "Posting p-77"
    assert draft.role_id == "posting-p-77"
    [req] = draft.requirements
    assert "Replace with the requirements" in req.description


def test_role_id_slug_from_title() -> None:
    title = "Sr. Data Engineer – GCP / BigQuery (Remote, Contract-to-Hire) – Team Apollo"
    draft = draft_from_posting(Posting.model_validate({"id": "p", "text": title}))
    assert re.fullmatch(SLUG_PATTERN, draft.role_id)
    assert draft.role_id.startswith("sr-data-engineer-gcp-bigquery")


@pytest.mark.parametrize(("text", "expected"), [
    ("Senior Data Engineer", "senior-data-engineer"),
    ("  --Hello__World!!  ", "hello-world"),
    ("Café Ünïcode", "cafe-unicode"),
    ("日本語", ""),
    ("a" * 50, "a" * 40),
])
def test_slugify(text: str, expected: str) -> None:
    assert slugify(text) == expected


def test_drafted_criteria_validate_as_domain_models() -> None:
    draft = draft_from_posting(Posting.model_validate(LEVER_POSTING_EXAMPLE))
    rebuilt = RoleCriteria.model_validate(draft.model_dump())
    assert rebuilt == draft
    assert all(isinstance(r, Requirement) for r in draft.requirements)
    assert not any(isinstance(k, Knockout) for k in draft.knockouts)
