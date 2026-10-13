"""PolicyEngine.evaluate: fail-closed gates, regime requirements and date boundaries."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime

import pytest

from applicant_ranking.domain import RunMode
from applicant_ranking.lever.models import Posting
from applicant_ranking.policy import PolicyConfig, PolicyDecision, PolicyEngine

PostingFactory = Callable[..., Posting]
EngineFactory = Callable[..., PolicyEngine]
GATED_MODES = (RunMode.DRY_RUN, RunMode.LIVE)


def decide(engine: PolicyEngine, posting: Posting, today: date,
           mode: RunMode = RunMode.DRY_RUN) -> PolicyDecision:
    return engine.evaluate(posting, mode=mode, today=today)


def assert_blocked_by(decision: PolicyDecision, *fragments: str) -> None:
    assert not decision.allowed
    for fragment in fragments:
        assert any(fragment in b for b in decision.blockers), (fragment, decision.blockers)


@pytest.fixture
def everywhere(make_posting: PostingFactory) -> Posting:
    """A US posting that triggers every US regime (NYC, CO, CA) plus IL."""
    return make_posting("New York, NY", "Denver, CO", "San Francisco, CA")


# ------------------------------------------------------------------------ happy path


@pytest.mark.parametrize("mode", GATED_MODES)
def test_complete_approved_policy_allows(make_engine: EngineFactory, everywhere: Posting,
                                         today: date, mode: RunMode) -> None:
    decision = decide(make_engine(), everywhere, today, mode)
    assert decision.allowed
    assert decision.blockers == []
    assert decision.regimes == ["IL_HRA", "NYC_LL144", "CO_ADMT", "CA_FEHA"]
    assert any("approved by Egen Legal" in n for n in decision.notes)
    assert any("4 years" in n for n in decision.notes)  # CA FEHA retention note
    assert any("valid through 2027-03-02" in n for n in decision.notes)


def test_today_may_be_a_datetime(make_engine: EngineFactory, everywhere: Posting) -> None:
    decision = make_engine().evaluate(everywhere, mode=RunMode.DRY_RUN,
                                      today=datetime(2026, 10, 8, 23, 59))
    assert decision.allowed


# --------------------------------------------------------------------- fail closed


@pytest.mark.parametrize("mode", GATED_MODES)
def test_no_policy_file_blocks(make_posting: PostingFactory, today: date,
                               mode: RunMode) -> None:
    engine = PolicyEngine(None, source="config/policy.yaml")
    decision = decide(engine, make_posting("Chicago, IL"), today, mode)
    assert_blocked_by(decision, "No jurisdiction policy file at config/policy.yaml")
    assert decision.regimes == ["IL_HRA"]


@pytest.mark.parametrize("field", ["approved_by", "approved_at"])
def test_unapproved_policy_blocks(make_engine: EngineFactory, make_posting: PostingFactory,
                                  today: date, field: str) -> None:
    decision = decide(make_engine(**{field: None}), make_posting("Chicago, IL"), today)
    assert_blocked_by(decision, "is not approved", "approved_by and approved_at")


def test_blank_approver_is_unapproved(make_engine: EngineFactory,
                                      make_posting: PostingFactory, today: date) -> None:
    decision = decide(make_engine(approved_by="  "), make_posting("Chicago, IL"), today)
    assert_blocked_by(decision, "is not approved")


def test_approval_dated_in_the_future_blocks(make_engine: EngineFactory,
                                             make_posting: PostingFactory,
                                             today: date) -> None:
    decision = decide(make_engine(approved_at=date(2026, 10, 9)), make_posting("Chicago, IL"),
                      today)
    assert_blocked_by(decision, "approved_at 2026-10-09 is in the future")


def test_unapproved_policy_still_reports_every_missing_piece(make_posting: PostingFactory,
                                                             today: date) -> None:
    engine = PolicyEngine(PolicyConfig())
    decision = decide(engine, make_posting("New York, NY"), today)
    assert_blocked_by(decision, "is not approved", "illinois.notice_url",
                      "illinois.posting_notice_confirmed", "nyc.bias_audit_date",
                      "nyc.audit_summary_url", "nyc.notice_published_at")


# ---------------------------------------------------------------------- country gate


def test_country_outside_allowed_list_blocks(make_engine: EngineFactory,
                                             make_posting: PostingFactory,
                                             today: date) -> None:
    for country in ("GB", "IN", "DE", "FR"):
        decision = decide(make_engine(), make_posting(country=country), today)
        assert_blocked_by(decision, f"posting country {country} is not in allowed_countries")


def test_unknown_country_blocks(make_engine: EngineFactory, make_posting: PostingFactory,
                                today: date) -> None:
    decision = decide(make_engine(), make_posting("Remote", country=None, workplace="remote"),
                      today)
    assert_blocked_by(decision, "has no country", "set the posting's country in Lever")


def test_missing_country_inferred_from_us_states_is_allowed(make_engine: EngineFactory,
                                                            make_posting: PostingFactory,
                                                            today: date) -> None:
    decision = decide(make_engine(), make_posting("Chicago, IL", "Austin, TX", country=None),
                      today)
    assert decision.allowed, decision.blockers


def test_foreign_location_blocks_even_with_us_country(make_engine: EngineFactory,
                                                      make_posting: PostingFactory,
                                                      today: date) -> None:
    decision = decide(make_engine(), make_posting("Naperville, IL", "Hyderabad, India"), today)
    assert_blocked_by(decision, "location 'Hyderabad, India' is in IN")
    decision = decide(make_engine(), make_posting("Chicago, IL", "Mississauga, ON"), today)
    assert_blocked_by(decision, "location 'Mississauga, ON' is in CA")


def test_canada_allowed_with_ontario_disclosure(make_engine: EngineFactory,
                                                make_posting: PostingFactory,
                                                today: date) -> None:
    engine = make_engine(allowed_countries=["us", "Canada"])
    assert engine.config is not None
    assert engine.config.allowed_countries == ["US", "CA"]
    decision = decide(engine, make_posting("Mississauga, ON", country="CA"), today)
    assert decision.allowed, decision.blockers
    assert decision.regimes == ["IL_HRA", "ON_ESA"]


def test_empty_allowed_countries_blocks_everything(make_engine: EngineFactory,
                                                   make_posting: PostingFactory,
                                                   today: date) -> None:
    decision = decide(make_engine(allowed_countries=[]), make_posting("Chicago, IL"), today)
    assert_blocked_by(decision, "posting country US is not in allowed_countries (none)")


# ----------------------------------------------------------------------------- Illinois


def test_illinois_notice_url_required(make_engine: EngineFactory,
                                      make_posting: PostingFactory, today: date) -> None:
    for url in (None, "", "   "):
        decision = decide(make_engine(illinois={"notice_url": url}),
                          make_posting("Chicago, IL"), today)
        assert_blocked_by(decision, "IL HRA: illinois.notice_url is not set")


def test_illinois_notice_url_must_be_http(make_engine: EngineFactory,
                                          make_posting: PostingFactory, today: date) -> None:
    for url in ("TBD", "javascript:alert(1)", "ftp://example.com/notice"):
        decision = decide(make_engine(illinois={"notice_url": url}),
                          make_posting("Chicago, IL"), today)
        assert_blocked_by(decision, "is not an http(s) URL")


def test_illinois_posting_notice_must_be_confirmed(make_engine: EngineFactory,
                                                   make_posting: PostingFactory,
                                                   today: date) -> None:
    decision = decide(make_engine(illinois={"posting_notice_confirmed": False}),
                      make_posting("Austin, TX"), today)
    assert decision.regimes == ["IL_HRA"]
    assert_blocked_by(decision, "IL HRA: illinois.posting_notice_confirmed is false")


# ---------------------------------------------------------------------------------- NYC


def test_nyc_requires_a_bias_audit(make_engine: EngineFactory, make_posting: PostingFactory,
                                   today: date) -> None:
    decision = decide(make_engine(nyc={"bias_audit_date": None}), make_posting("NYC"), today)
    assert_blocked_by(decision, "NYC LL144: nyc.bias_audit_date is not set")


def test_nyc_bias_audit_exactly_365_days_old_is_valid(make_engine: EngineFactory,
                                                      make_posting: PostingFactory,
                                                      today: date) -> None:
    engine = make_engine(nyc={"bias_audit_date": date(2025, 10, 8)})
    assert (today - date(2025, 10, 8)).days == 365
    decision = decide(engine, make_posting("NYC"), today)
    assert decision.allowed, decision.blockers


def test_nyc_bias_audit_366_days_old_blocks(make_engine: EngineFactory,
                                            make_posting: PostingFactory, today: date) -> None:
    engine = make_engine(nyc={"bias_audit_date": date(2025, 10, 7)})
    decision = decide(engine, make_posting("NYC"), today)
    assert_blocked_by(decision, "NYC LL144: bias audit dated 2025-10-07 is 366 days old "
                                "(limit 365)")


def test_nyc_bias_audit_in_the_future_blocks(make_engine: EngineFactory,
                                             make_posting: PostingFactory,
                                             today: date) -> None:
    engine = make_engine(nyc={"bias_audit_date": date(2026, 10, 9)})
    assert_blocked_by(decide(engine, make_posting("NYC"), today), "is in the future")


def test_nyc_requires_audit_summary_url(make_engine: EngineFactory,
                                        make_posting: PostingFactory, today: date) -> None:
    for url in (None, "", "summary.pdf"):
        decision = decide(make_engine(nyc={"audit_summary_url": url}), make_posting("NYC"),
                          today)
        assert_blocked_by(decision, "nyc.audit_summary_url")


def test_nyc_requires_notice(make_engine: EngineFactory, make_posting: PostingFactory,
                             today: date) -> None:
    decision = decide(make_engine(nyc={"notice_published_at": None}), make_posting("NYC"),
                      today)
    assert_blocked_by(decision, "NYC LL144: nyc.notice_published_at is not set")


def test_nyc_notice_too_recent_blocks_with_the_start_date(make_engine: EngineFactory,
                                                          make_posting: PostingFactory,
                                                          today: date) -> None:
    engine = make_engine(nyc={"notice_published_at": date(2026, 10, 1), "holidays": []})
    decision = decide(engine, make_posting("New York, NY"), today)
    assert_blocked_by(decision, "NYC LL144: notice published 2026-10-01 + 10 business days = "
                                "2026-10-15 is after today (2026-10-08)",
                      "can start on 2026-10-15")


def test_nyc_notice_boundary(make_engine: EngineFactory, make_posting: PostingFactory) -> None:
    engine = make_engine(nyc={"notice_published_at": date(2026, 10, 1), "holidays": [],
                              "bias_audit_date": date(2026, 9, 1)})
    posting = make_posting("NYC")
    assert not decide(engine, posting, date(2026, 10, 14)).allowed
    assert decide(engine, posting, date(2026, 10, 15)).allowed


def test_nyc_holidays_push_the_start_date(make_engine: EngineFactory,
                                          make_posting: PostingFactory) -> None:
    engine = make_engine(nyc={"notice_published_at": date(2026, 10, 1),
                              "holidays": [date(2026, 10, 12)],
                              "bias_audit_date": date(2026, 9, 1)})
    posting = make_posting("NYC")
    decision = decide(engine, posting, date(2026, 10, 15))
    assert_blocked_by(decision, "= 2026-10-16 is after today")
    assert decide(engine, posting, date(2026, 10, 16)).allowed


def test_nyc_notice_in_the_future_blocks(make_engine: EngineFactory,
                                         make_posting: PostingFactory, today: date) -> None:
    engine = make_engine(nyc={"notice_published_at": date(2026, 11, 1)})
    assert_blocked_by(decide(engine, make_posting("NYC"), today), "is after today")


def test_nyc_checks_only_apply_to_nyc_postings(make_engine: EngineFactory,
                                               make_posting: PostingFactory,
                                               today: date) -> None:
    engine = make_engine(nyc={"bias_audit_date": None, "notice_published_at": None})
    assert decide(engine, make_posting("Albany, NY"), today).allowed


def test_remote_nyc_flag_brings_in_nyc_checks(make_engine: EngineFactory,
                                              make_posting: PostingFactory,
                                              today: date) -> None:
    engine = make_engine(nyc={"remote_roles_associated_with_nyc_office": True,
                              "bias_audit_date": None})
    decision = decide(engine, make_posting("Remote - US", workplace="remote"), today)
    assert "NYC_LL144" in decision.regimes
    assert_blocked_by(decision, "nyc.bias_audit_date")


# ----------------------------------------------------------------------------- Colorado


def test_colorado_before_effective_date_is_a_note(make_engine: EngineFactory,
                                                  make_posting: PostingFactory) -> None:
    engine = make_engine(colorado={"adverse_outcome_workflow_ready": False})
    decision = decide(engine, make_posting("Denver, CO"), date(2026, 12, 31))
    assert decision.allowed, decision.blockers
    assert any("CO ADMT" in n and "2027-01-01" in n for n in decision.notes)


def test_colorado_from_effective_date_blocks_without_workflow(make_engine: EngineFactory,
                                                              make_posting: PostingFactory
                                                              ) -> None:
    engine = make_engine(colorado={"adverse_outcome_workflow_ready": False},
                         nyc={"bias_audit_date": date(2026, 6, 1)})
    decision = decide(engine, make_posting("Denver, CO"), date(2027, 1, 1))
    assert_blocked_by(decision, "CO ADMT: colorado.adverse_outcome_workflow_ready is false")


def test_colorado_from_effective_date_allows_with_workflow(make_engine: EngineFactory,
                                                           make_posting: PostingFactory
                                                           ) -> None:
    decision = decide(make_engine(), make_posting("Denver, CO"), date(2027, 1, 1))
    assert decision.allowed, decision.blockers
    assert any("adverse-outcome workflow confirmed" in n for n in decision.notes)


def test_colorado_applies_to_us_remote(make_engine: EngineFactory,
                                       make_posting: PostingFactory) -> None:
    engine = make_engine(colorado={"adverse_outcome_workflow_ready": False})
    decision = decide(engine, make_posting("Remote - US", workplace="remote"), date(2027, 2, 1))
    assert_blocked_by(decision, "CO ADMT")


# --------------------------------------------------------------------------- California


@pytest.mark.parametrize("applies", [None, True, False])
def test_california_before_2027_is_notes_only(make_engine: EngineFactory,
                                              make_posting: PostingFactory,
                                              applies: bool | None) -> None:
    engine = make_engine(california={"ccpa_admt_applies": applies, "ccpa_notice_ready": False})
    decision = decide(engine, make_posting("San Francisco, CA"), date(2026, 12, 31))
    assert decision.allowed, decision.blockers
    assert any("CA CCPA ADMT (from 2027-01-01)" in n for n in decision.notes)


def test_california_undecided_from_2027_blocks(make_engine: EngineFactory,
                                               make_posting: PostingFactory) -> None:
    engine = make_engine(california={"ccpa_admt_applies": None})
    decision = decide(engine, make_posting("San Francisco, CA"), date(2027, 1, 1))
    assert_blocked_by(decision, "CA CCPA ADMT: california.ccpa_admt_applies is undecided")


def test_california_applies_without_notice_from_2027_blocks(make_engine: EngineFactory,
                                                            make_posting: PostingFactory
                                                            ) -> None:
    engine = make_engine(california={"ccpa_admt_applies": True, "ccpa_notice_ready": False})
    decision = decide(engine, make_posting("San Francisco, CA"), date(2027, 1, 1))
    assert_blocked_by(decision, "california.ccpa_notice_ready is false")


@pytest.mark.parametrize(("applies", "ready"), [(True, True), (False, False)])
def test_california_decided_from_2027_allows(make_engine: EngineFactory,
                                             make_posting: PostingFactory, applies: bool,
                                             ready: bool) -> None:
    engine = make_engine(california={"ccpa_admt_applies": applies, "ccpa_notice_ready": ready})
    decision = decide(engine, make_posting("San Francisco, CA"), date(2027, 1, 1))
    assert decision.allowed, decision.blockers


def test_california_checks_only_apply_to_california(make_engine: EngineFactory,
                                                    make_posting: PostingFactory) -> None:
    engine = make_engine(california={"ccpa_admt_applies": None})
    assert decide(engine, make_posting("Austin, TX"), date(2027, 1, 1)).allowed


# ------------------------------------------------------------------------------ Ontario


def test_ontario_requires_posting_disclosure(make_engine: EngineFactory,
                                             make_posting: PostingFactory,
                                             today: date) -> None:
    engine = make_engine(allowed_countries=["US", "CA"],
                         ontario={"posting_disclosure_confirmed": False})
    decision = decide(engine, make_posting("Toronto, ON", country="CA"), today)
    assert_blocked_by(decision, "ON ESA: ontario.posting_disclosure_confirmed is false")


# ---------------------------------------------------------------------- EVALUATION mode


def test_evaluation_mode_is_never_blocked(make_posting: PostingFactory, today: date) -> None:
    engine = PolicyEngine(None)
    decision = decide(engine, make_posting(country="IN"), today, RunMode.EVALUATION)
    assert decision.allowed
    assert decision.blockers
    assert all(any(b in n for n in decision.notes) for b in decision.blockers)
    assert any(n.startswith("EVALUATION only - DRY_RUN/LIVE would be blocked") for n in
               decision.notes)


def test_evaluation_mode_reports_regime_blockers(make_engine: EngineFactory,
                                                 make_posting: PostingFactory,
                                                 today: date) -> None:
    engine = make_engine(approved_by=None, nyc={"bias_audit_date": None})
    decision = decide(engine, make_posting("NYC"), today, RunMode.EVALUATION)
    assert decision.allowed
    assert any("nyc.bias_audit_date" in n for n in decision.notes)
    assert any("is not approved" in n for n in decision.notes)


def test_evaluation_mode_without_blockers_adds_no_warning(make_engine: EngineFactory,
                                                          everywhere: Posting,
                                                          today: date) -> None:
    decision = decide(make_engine(), everywhere, today, RunMode.EVALUATION)
    assert decision.allowed and not decision.blockers
    assert not any(n.startswith("EVALUATION only") for n in decision.notes)


def test_blockers_name_the_setting_to_fix(make_posting: PostingFactory, today: date) -> None:
    """Every blocker from an empty policy names what to change (actionable)."""
    decision = decide(PolicyEngine(PolicyConfig()), make_posting("NYC", "Denver, CO"),
                      date(2027, 1, 4))
    for blocker in decision.blockers:
        assert blocker.endswith(".")
        assert any(word in blocker for word in ("set", "must", "publish", "commission",
                                                "confirm", "decide", "add")), blocker
