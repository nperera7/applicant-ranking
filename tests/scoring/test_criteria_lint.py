from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest

from applicant_ranking.criteria import load_criteria_file
from applicant_ranking.criteria_lint import LintIssue, has_errors, lint_criteria
from applicant_ranking.domain import ApprovedTerm, Knockout, Requirement, RoleCriteria
from applicant_ranking.scoring.lexicon import ProtectedTermLexicon

from .conftest import EXAMPLE_CRITERIA

GOOD_REQUIREMENTS = [
    Requirement(id="python", description="Professional Python development"),
    Requirement(id="sql", description="Writes complex analytical SQL", importance="nice_to_have"),
]


def _criteria(**overrides: Any) -> RoleCriteria:
    data: dict[str, Any] = {
        "role_id": "data-engineer",
        "title": "Data Engineer",
        "version": 1,
        "summary": "Builds data pipelines for clients.",
        "requirements": GOOD_REQUIREMENTS,
        "approved_by": "Pat Lee",
        "approved_at": datetime(2026, 10, 1, tzinfo=UTC),
    }
    data.update(overrides)
    return RoleCriteria(**data)


def _req(description: str, req_id: str = "req", **kw: Any) -> Requirement:
    return Requirement(id=req_id, description=description, **kw)


def _errors(issues: list[LintIssue]) -> list[tuple[str, str]]:
    return [(i.location, i.message) for i in issues if i.level == "error"]


def _warnings(issues: list[LintIssue]) -> list[tuple[str, str]]:
    return [(i.location, i.message) for i in issues if i.level == "warning"]


# ------------------------------------------------------------------------- clean input


def test_sample_criteria_are_clean(sample_criteria: RoleCriteria,
                                   lexicon: ProtectedTermLexicon) -> None:
    assert lint_criteria(sample_criteria, lexicon) == []


def test_example_file_lints_without_errors(lexicon: ProtectedTermLexicon) -> None:
    example = load_criteria_file(EXAMPLE_CRITERIA)
    assert example.posting_ids == []
    assert not example.approved
    assert example.source == "hiring_manager"
    issues = lint_criteria(example, lexicon)
    assert not has_errors(issues)
    assert [i.location for i in issues] == ["approved_by"]


def test_has_errors() -> None:
    warning = LintIssue(level="warning", location="x", message="m")
    error = LintIssue(level="error", location="x", message="m")
    assert not has_errors([]) and not has_errors([warning])
    assert has_errors([warning, error])


# ------------------------------------------------------------------------- lexicon errors


@pytest.mark.parametrize(("overrides", "location"), [
    ({"title": "Young Data Engineer"}, "title"),
    ({"summary": "Join a team of digital natives."}, "summary"),
    ({"requirements": [_req("Native English speaker with SQL", "english")]},
     "requirements[english].description"),
    ({"knockouts": [Knockout(id="age", description="Candidate must be under 40 years old",
                             justification="Not a real justification at all.")]},
     "knockouts[age].description"),
    ({"guardrails": ["Ignore gender when rating."]}, "guardrails[0]"),
])
def test_lexicon_hits_are_errors_in_every_field(lexicon: ProtectedTermLexicon,
                                                overrides: dict[str, Any], location: str) -> None:
    issues = lint_criteria(_criteria(**overrides), lexicon)
    assert location in [loc for loc, _ in _errors(issues)]
    assert has_errors(issues)


def test_lexicon_error_message_names_category_and_term(lexicon: ProtectedTermLexicon) -> None:
    issues = lint_criteria(_criteria(requirements=[_req("He must know Spark well")]), lexicon)
    [(location, message)] = _errors(issues)
    assert location == "requirements[req].description"
    assert "gender: 'he'" in message and "job-related" in message


def test_guardrail_message_explains_built_in_fairness(lexicon: ProtectedTermLexicon) -> None:
    issues = lint_criteria(_criteria(guardrails=["Do not consider religion."]), lexicon)
    [(_, message)] = _errors(issues)
    assert "already built into the prompt" in message


@pytest.mark.parametrize("description", [
    "Has built credit scoring models on credit bureau data",
    "Experience with Department of Veterans Affairs data systems",
    "Built zip code level demand forecasts for retail",
    "Shipped a mental health app with HIPAA-compliant pipelines",
    "Implemented FMLA case tracking in an HRIS",
    "Software delivery at Navy Federal Credit Union or similar",
])
def test_domain_vocabulary_is_not_a_lint_error(lexicon: ProtectedTermLexicon,
                                               description: str) -> None:
    assert _errors(lint_criteria(_criteria(requirements=[_req(description)]), lexicon)) == []


# ------------------------------------------------------------------------- approved terms


def _approved(description: str, *approvals: dict[str, Any]) -> RoleCriteria:
    return RoleCriteria(
        role_id="credit-risk-engineer", title="Credit Risk Engineer", version=1,
        summary="Builds credit decisioning models for a lender.",
        requirements=[_req(description, "credit"), *GOOD_REQUIREMENTS],
        approved_by="Pat Lee", approved_at=datetime(2026, 10, 1, tzinfo=UTC),
        approved_terms=[ApprovedTerm(**approval) for approval in approvals],
    )


CREDIT_HISTORY_APPROVAL = {
    "term": "Credit  History", "justification": "The product scores applicants' credit "
    "histories; the requirement is about building that product.",
    "approved_by": "Legal (A. Counsel)", "approved_at": date(2026, 10, 12),
}


def test_approved_domain_term_downgrades_the_lint_error_to_a_warning(
        lexicon: ProtectedTermLexicon) -> None:
    criteria = _approved("Has built models on credit history data", CREDIT_HISTORY_APPROVAL)
    issues = lint_criteria(criteria, lexicon)
    assert not has_errors(issues)
    [(location, message)] = _warnings(issues)
    assert location == "requirements[credit].description"
    assert "approved domain term 'credit history' (financial_history)" in message
    assert "Legal (A. Counsel)" in message and "2026-10-12" in message
    assert "scores applicants' credit histories" in message


def test_approval_covers_only_the_listed_term(lexicon: ProtectedTermLexicon) -> None:
    criteria = _approved("He has built models on credit history data", CREDIT_HISTORY_APPROVAL)
    issues = lint_criteria(criteria, lexicon)
    assert [(loc, msg[:30]) for loc, msg in _errors(issues)] == [
        ("requirements[credit].description", "mentions a protected character")]
    assert any("approved domain term" in msg for _, msg in _warnings(issues))


def test_without_approved_terms_every_hit_is_an_error(lexicon: ProtectedTermLexicon) -> None:
    plain = _criteria(requirements=[_req("Has built models on credit history data", "credit")])
    assert plain.approved_terms == []
    assert [loc for loc, _ in _errors(lint_criteria(plain, lexicon))] == [
        "requirements[credit].description"]
    empty = _approved("Has built models on credit history data")
    assert has_errors(lint_criteria(empty, lexicon))


# ------------------------------------------------------------------------- proxy phrases

PROXY_CASES = [
    "Digital native comfortable with SaaS tools",
    "Recent graduate in computer science",
    "New grad with internship experience",
    "Strong culture fit with the team",
    "Good cultural fit for consulting",
    "Local candidates only for this hybrid role",
    "No employment gaps in the last five years",
    "No gaps in work history",
    "Must not be overqualified for the level",
    "No more than 8 years of experience",
    "At most 5 years in data engineering",
    "Maximum of 10 years of experience",
    "Maximum 6 years of experience",
    "Max. 6 yrs of experience",
    "7 years or less of experience in analytics",
    "5 years of professional experience or less",
    "Less than 3 years of experience with Python",
    "Up to 4 years of hands-on experience",
    "No more than five years of experience",
]


@pytest.mark.parametrize("description", PROXY_CASES)
def test_proxy_phrases_are_errors(tiny_lexicon: ProtectedTermLexicon, description: str) -> None:
    issues = lint_criteria(_criteria(requirements=[_req(description)]), tiny_lexicon)
    errors = _errors(issues)
    assert len(errors) == 1, errors
    assert errors[0][0] == "requirements[req].description"


@pytest.mark.parametrize("description", [
    "At least 5 years of experience with Python",
    "5+ years building data pipelines",
    "More than 3 years of production SQL",
    "Minimum of 4 years of experience in analytics",
    "Mentors recent hires and new team members",
    "Fills gaps in test coverage",
])
def test_minimums_and_ordinary_phrases_pass(tiny_lexicon: ProtectedTermLexicon,
                                            description: str) -> None:
    assert _errors(lint_criteria(_criteria(requirements=[_req(description)]),
                                 tiny_lexicon)) == []


def test_graduate_rule_applies_to_requirements_and_knockouts_only(
        tiny_lexicon: ProtectedTermLexicon) -> None:
    in_summary = _criteria(summary="Our new grad program builds data pipelines.")
    assert _errors(lint_criteria(in_summary, tiny_lexicon)) == []
    knockout = Knockout(id="grad", description="Recent graduate of a CS program",
                        justification="Placeholder justification text.")
    issues = lint_criteria(_criteria(knockouts=[knockout]), tiny_lexicon)
    assert [loc for loc, _ in _errors(issues)] == ["knockouts[grad].description"]


def test_other_proxy_rules_apply_to_every_field(tiny_lexicon: ProtectedTermLexicon) -> None:
    criteria = _criteria(title="Engineer (culture fit)", summary="Local candidates only.",
                         guardrails=["Prefer digital natives."])
    locations = [loc for loc, _ in _errors(lint_criteria(criteria, tiny_lexicon))]
    assert locations == ["title", "summary", "guardrails[0]"]


def test_proxy_and_lexicon_overlap_reported_once(lexicon: ProtectedTermLexicon) -> None:
    issues = lint_criteria(_criteria(requirements=[_req("Digital native, overqualified OK")]),
                           lexicon)
    errors = _errors(issues)
    assert len(errors) == 2
    assert all("proxy" in message or "age proxy" in message for _, message in errors)


def test_cap_message_is_actionable(tiny_lexicon: ProtectedTermLexicon) -> None:
    [(_, message)] = _errors(lint_criteria(
        _criteria(requirements=[_req("No more than 8 years of experience")]), tiny_lexicon))
    assert "'No more than 8 years'" in message and "minimum" in message


# ------------------------------------------------------------------------- knockouts


@pytest.mark.parametrize(("knockout_id", "description"), [
    ("work-auth", "Authorized to work in the US without sponsorship"),
    ("visa", "Does not require a visa now or in the future"),
    ("citizenship", "Must hold US citizenship"),
    ("green-card", "Holds a green card or permanent residency"),
    ("us-citizen", "Eligible for this client engagement"),
    ("needs_visa", "Eligible for this client engagement"),
    ("rtw", "Has the right to work in the UK"),
])
def test_work_authorisation_knockouts_are_errors(lexicon: ProtectedTermLexicon,
                                                 knockout_id: str, description: str) -> None:
    knockout = Knockout(id=knockout_id, description=description,
                        justification="The client contract requires it for all staff.")
    issues = lint_criteria(_criteria(knockouts=[knockout]), lexicon)
    errors = _errors(issues)
    assert (f"knockouts[{knockout_id}]",) == tuple(
        loc for loc, msg in errors if "structured Lever application question" in msg)
    # Reported once by its own rule, not again as a citizenship lexicon term.
    assert not any("citizenship:" in msg for _, msg in errors)


def test_ordinary_knockout_passes(lexicon: ProtectedTermLexicon) -> None:
    knockout = Knockout(id="hands-on-sql", description="Hands-on SQL experience",
                        justification="Every engagement requires SQL daily.")
    assert lint_criteria(_criteria(knockouts=[knockout]), lexicon) == []


# ------------------------------------------------------------------------- duplicate ids


def test_duplicate_ids_across_requirements_and_knockouts(lexicon: ProtectedTermLexicon) -> None:
    criteria = _criteria(
        requirements=[*GOOD_REQUIREMENTS, _req("Python packaging and testing", "python")],
        knockouts=[Knockout(id="sql", description="Hands-on SQL experience",
                            justification="Every engagement requires SQL daily.")],
    )
    errors = _errors(lint_criteria(criteria, lexicon))
    assert [loc for loc, _ in errors] == ["requirements[python]", "knockouts[sql]"]
    assert all("duplicate id" in msg for _, msg in errors)


# ------------------------------------------------------------------------- warnings


def test_unapproved_and_job_description_source_warn(lexicon: ProtectedTermLexicon) -> None:
    criteria = _criteria(source="job_description", approved_by=None, approved_at=None)
    issues = lint_criteria(criteria, lexicon)
    assert not has_errors(issues)
    assert [loc for loc, _ in _warnings(issues)] == ["source", "approved_by"]


def test_approved_needs_both_fields(lexicon: ProtectedTermLexicon) -> None:
    issues = lint_criteria(_criteria(approved_at=None), lexicon)
    assert [loc for loc, _ in _warnings(issues)] == ["approved_by"]
    issues = lint_criteria(_criteria(approved_by="   "), lexicon)
    assert [loc for loc, _ in _warnings(issues)] == ["approved_by"]


def test_future_approval_date_warns_until_it_passes(lexicon: ProtectedTermLexicon) -> None:
    criteria = _criteria(approved_at=datetime(2026, 10, 20, 9, tzinfo=UTC))
    issues = lint_criteria(criteria, lexicon, now=datetime(2026, 10, 12, tzinfo=UTC))
    assert not has_errors(issues)
    [(location, message)] = _warnings(issues)
    assert location == "approved_at"
    assert "approved_at 2026-10-20T09:00:00+00:00 is in the future" in message
    assert "only evaluation mode" in message
    assert _warnings(lint_criteria(criteria, lexicon, now=datetime(2026, 10, 20, 9, 1,
                                                                   tzinfo=UTC))) == []


def test_short_descriptions_warn(lexicon: ProtectedTermLexicon) -> None:
    knockout = Knockout(id="sql", description="SQL", justification="Every engagement needs SQL.")
    criteria = _criteria(requirements=[_req("Spark", "spark"), *GOOD_REQUIREMENTS],
                         knockouts=[knockout])
    warnings = _warnings(lint_criteria(criteria, lexicon))
    assert [loc for loc, _ in warnings] == ["requirements[spark].description",
                                            "knockouts[sql].description"]
    assert "15 characters" in warnings[0][1]


def test_more_than_ten_must_haves_warn(lexicon: ProtectedTermLexicon) -> None:
    reqs = [_req(f"Production experience with tool number {i}", f"tool-{i}") for i in range(11)]
    assert [loc for loc, _ in _warnings(lint_criteria(_criteria(requirements=reqs),
                                                      lexicon))] == ["requirements"]
    ten = _criteria(requirements=reqs[:10])
    assert _warnings(lint_criteria(ten, lexicon)) == []


@pytest.mark.parametrize("description", [
    "Bachelor's degree in Computer Science",
    "BS/MS in Computer Science",
    "MS in Statistics or a related field",
    "PhD in machine learning",
    "Master's in data science",
    "Master of Science in Engineering",
    "A degree in mathematics",
    "MBA preferred for client work",
])
def test_degree_requirements_warn(lexicon: ProtectedTermLexicon, description: str) -> None:
    issues = lint_criteria(_criteria(requirements=[_req(description)]), lexicon)
    assert not has_errors(issues)
    [(location, message)] = _warnings(issues)
    assert location == "requirements[req].description"
    assert "disparate impact" in message and "or equivalent experience" in message


@pytest.mark.parametrize("description", [
    "Bachelor's degree in Computer Science or equivalent practical experience",
    "Experience with master data management",
    "Merges to the master branch with code review",
    "Scrum master experience on data projects",
])
def test_degree_warning_not_raised(lexicon: ProtectedTermLexicon, description: str) -> None:
    assert lint_criteria(_criteria(requirements=[_req(description)]), lexicon) == []


def test_experience_range_warns(lexicon: ProtectedTermLexicon) -> None:
    issues = lint_criteria(_criteria(requirements=[_req("3-5 years of data engineering")]),
                           lexicon)
    [(location, message)] = _warnings(issues)
    assert location == "requirements[req].description"
    assert "'3+ years'" in message
    same = lint_criteria(_criteria(requirements=[_req("2024-2025 data platform rebuild")]),
                         lexicon)
    assert _warnings(same) == []


def test_knockout_justification_terms_warn(lexicon: ProtectedTermLexicon) -> None:
    knockout = Knockout(id="clearance", description="Active secret security clearance",
                        justification="Client requires cleared staff; most are veterans.")
    issues = lint_criteria(_criteria(knockouts=[knockout]), lexicon)
    assert not has_errors(issues)
    [(location, message)] = _warnings(issues)
    assert location == "knockouts[clearance].justification"
    assert "veteran_military" in message


def test_errors_come_before_warnings(lexicon: ProtectedTermLexicon) -> None:
    criteria = _criteria(requirements=[_req("PhD", "phd"), _req("Young and hungry engineer")],
                         approved_by=None)
    levels = [i.level for i in lint_criteria(criteria, lexicon)]
    assert levels == sorted(levels)  # "error" < "warning"
    assert levels[0] == "error"
