"""Where a posting is, beyond "City, ST": Lever's structured address
(``categories.locationDetails``), office labels that name no state, and the
``location_overrides`` Legal/HR use to approve such labels. Every case here is about the
fail-closed side of detection: an unmapped US label is a blocker, not a silent IL-only run."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

import pytest
from pydantic import ValidationError

from applicant_ranking.domain import RunMode
from applicant_ranking.lever.models import Posting
from applicant_ranking.policy import NYC_ZIP_PREFIXES, PolicyConfig, PolicyDecision, PolicyEngine

EngineFactory = Callable[..., PolicyEngine]

TODAY = date(2026, 10, 8)
NYC_DETAILS = {"addressLine1": "20 Hudson Yards", "city": "New York", "region": "NY",
               "postalCode": "10001", "country": "US"}
DENVER_DETAILS = {"city": "Denver", "region": "CO", "postalCode": "80202", "country": "US"}
SF_DETAILS = {"city": "San Francisco", "region": "CA", "postalCode": "94107", "country": "US"}
UNRESOLVED = "could not be mapped to a US state or city"
# Office labels Lever accounts actually use: none names a state or a known city.
OFFICE_LABELS = ("Hudson Yards Office", "Brooklyn Navy Yard", "Midtown",
                 "One World Trade Center", "Mountain Office", "Denver Tech Center",
                 "SF Office", "SoMa HQ", "HQ")


def posting(*locations: str, country: str | None = "US", workplace: str | None = "onsite",
            details: dict[str, Any] | None = None, posting_id: str = "posting-1") -> Posting:
    """Like the shared ``make_posting`` fixture, plus ``details`` for
    ``categories.locationDetails`` (None leaves the key out)."""
    categories: dict[str, Any] = {}
    if locations:
        categories = {"location": locations[0], "allLocations": list(locations)}
    if details is not None:
        categories["locationDetails"] = details
    return Posting.model_validate({
        "id": posting_id, "text": "Senior Data Engineer", "country": country,
        "workplaceType": workplace, "categories": categories,
    })


def live(engine: PolicyEngine, p: Posting, today: date = TODAY) -> PolicyDecision:
    return engine.evaluate(p, mode=RunMode.LIVE, today=today)


def mentioning(decision: PolicyDecision, fragment: str) -> list[str]:
    return [b for b in decision.blockers if fragment in b]


def with_overrides(engine: PolicyEngine, overrides: dict[str, list[str]]) -> PolicyEngine:
    """``engine`` with ``location_overrides`` set (validated like a policy file)."""
    assert engine.config is not None
    data = {**engine.config.model_dump(), "location_overrides": overrides}
    return PolicyEngine(PolicyConfig.model_validate(data), source=engine.source)


# ------------------------------------------------------------------- structured address


@pytest.mark.parametrize("label", ["Hudson Yards Office", "US - New York", "NY, NY",
                                   "New York Metro Area", "Brooklyn Navy Yard", "Midtown",
                                   "One World Trade Center"])
def test_nyc_structured_address_brings_in_nyc_checks(make_engine: EngineFactory,
                                                     label: str) -> None:
    """An NYC structured address behind an office label brings in the NYC checks: with no
    bias audit date a LIVE run is blocked, and once the NYC requirements are met it runs."""
    engine = make_engine(nyc={"bias_audit_date": None})
    decision = live(engine, posting(label, details=NYC_DETAILS))
    assert "NYC_LL144" in decision.regimes
    assert not decision.allowed
    assert mentioning(decision, "nyc.bias_audit_date")
    assert not mentioning(decision, UNRESOLVED)
    # With the NYC requirements met the same posting runs.
    complete = live(make_engine(), posting(label, details=NYC_DETAILS))
    assert complete.allowed, complete.blockers
    assert "NYC_LL144" in complete.regimes


def test_structured_address_is_unioned_with_the_text_locations(
        make_engine: EngineFactory) -> None:
    """The address describes the primary location only; allLocations keep their own
    regimes, and nothing the text found is removed."""
    engine = make_engine()
    decision = live(engine, posting("Hudson Yards Office", "Denver, CO", details=NYC_DETAILS))
    assert decision.regimes == ["IL_HRA", "NYC_LL144", "CO_ADMT"]
    assert decision.allowed, decision.blockers


@pytest.mark.parametrize("label", ["Mountain Office", "Denver Tech Center",
                                   "Greater Denver Area"])
def test_denver_structured_address_brings_in_colorado(make_engine: EngineFactory,
                                                      label: str) -> None:
    engine = make_engine(colorado={"adverse_outcome_workflow_ready": False})
    decision = live(engine, posting(label, details=DENVER_DETAILS), date(2027, 1, 5))
    assert "CO_ADMT" in decision.regimes
    assert mentioning(decision, "CO ADMT: colorado.adverse_outcome_workflow_ready is false")


@pytest.mark.parametrize("label", ["SF Office", "SoMa HQ"])
def test_san_francisco_structured_address_brings_in_california(make_engine: EngineFactory,
                                                               label: str) -> None:
    engine = make_engine(california={"ccpa_admt_applies": None})
    decision = live(engine, posting(label, details=SF_DETAILS), date(2027, 1, 5))
    assert "CA_FEHA" in decision.regimes
    assert mentioning(decision, "california.ccpa_admt_applies is undecided")


def test_structured_country_outside_allowed_list_blocks(make_engine: EngineFactory) -> None:
    london = {"city": "London", "postalCode": "EC2A 4BX", "country": "GB"}
    without_country = posting("London Office", country=None, details=london)
    assert make_engine().effective_country(without_country) == "GB"
    decision = live(make_engine(), without_country)
    assert mentioning(decision, "posting country GB is not in allowed_countries")
    # posting.country wins for the gate, but the address still points abroad.
    decision = live(make_engine(), posting("London Office", country="US", details=london))
    assert mentioning(decision, "is in GB")
    assert not decision.allowed


def test_structured_country_fills_a_blank_posting_country(make_engine: EngineFactory) -> None:
    p = posting("Hudson Yards Office", country=None, details=NYC_DETAILS)
    assert make_engine().effective_country(p) == "US"
    decision = live(make_engine(), p)
    assert decision.allowed, decision.blockers
    assert "NYC_LL144" in decision.regimes


def test_structured_province_brings_in_ontario(make_engine: EngineFactory) -> None:
    engine = make_engine(allowed_countries=["US", "CA"])
    p = posting("Head Office", country=None,
                details={"city": "Mississauga", "region": "ON", "country": "CA"})
    assert engine.effective_country(p) == "CA"
    decision = live(engine, p)
    assert decision.regimes == ["IL_HRA", "ON_ESA"]
    assert decision.allowed, decision.blockers


def test_structured_region_may_be_a_full_name(make_engine: EngineFactory) -> None:
    details = {"city": "Queens", "region": "New York", "country": "US"}
    assert "NYC_LL144" in make_engine().detect_regimes(posting("Office", details=details))
    details = {"city": "Boulder", "region": "Colorado", "country": "US"}
    assert "CO_ADMT" in make_engine().detect_regimes(posting("Office", details=details))


@pytest.mark.parametrize(("details", "nyc"), [
    ({"city": "New York", "region": "NY", "country": "US"}, True),
    ({"city": "Brooklyn", "region": "NY", "country": "US"}, True),
    ({"region": "NY", "postalCode": "11201", "country": "US"}, True),  # Brooklyn ZIP
    ({"city": "Floral Park", "region": "NY", "postalCode": "11004", "country": "US"}, True),
    ({"city": "Hudson Yards", "region": "NY", "country": "US"}, True),  # unknown, no ZIP
    ({"region": "NY", "country": "US"}, True),  # no city, no ZIP: fail closed
    ({"city": "Scarsdale", "region": "NY", "postalCode": "10583", "country": "US"}, False),
    ({"city": "Yonkers", "region": "NY", "country": "US"}, False),
    ({"city": "Albany", "region": "NY", "postalCode": "12207", "country": "US"}, False),
    ({"city": "Buffalo", "region": "NY", "country": "US"}, False),
    ({"city": "Manhattan", "region": "KS", "country": "US"}, False),
    ({"city": "New York", "region": "NY", "postalCode": 10001, "country": "US"}, True),
])
def test_structured_address_nyc_resolution(make_engine: EngineFactory, details: dict[str, Any],
                                           nyc: bool) -> None:
    regimes = make_engine().detect_regimes(posting("Office", details=details))
    assert ("NYC_LL144" in regimes) is nyc, regimes


def test_nyc_zip_prefixes_cover_the_boroughs_only() -> None:
    boroughs = {"100", "101", "102", "103", "104", "111", "112", "113", "114", "116"}
    assert boroughs <= NYC_ZIP_PREFIXES
    assert all(len(p) == 3 and p.isdigit() for p in NYC_ZIP_PREFIXES)
    suburbs = {"105", "106", "107", "108", "109", "115", "117", "118", "119"}
    assert not suburbs & NYC_ZIP_PREFIXES


def test_null_or_empty_details_keep_the_text_behaviour(make_engine: EngineFactory) -> None:
    for details in (None, {}, {"addressLine1": "123 Main St"}):
        data: dict[str, Any] = {"id": "p", "country": "US", "workplaceType": "onsite",
                                "categories": {"location": "Chicago, IL",
                                               "locationDetails": details}}
        decision = live(make_engine(), Posting.model_validate(data))
        assert decision.regimes == ["IL_HRA"]
        assert decision.allowed, decision.blockers


# -------------------------------------------------------------- unresolved US locations


@pytest.mark.parametrize("label", OFFICE_LABELS)
def test_unmapped_us_label_without_an_address_blocks(make_engine: EngineFactory,
                                                     label: str) -> None:
    """Country US but no state or city: the state-level regimes cannot be checked, so the
    posting is blocked instead of silently getting IL only."""
    decision = live(make_engine(), posting(label))
    assert decision.regimes == ["IL_HRA"]
    assert not decision.allowed
    [blocker] = mentioning(decision, UNRESOLVED)
    assert repr(label) in blocker
    assert "structured address" in blocker and "location_overrides" in blocker
    assert "test-policy.yaml" in blocker and blocker.endswith(".")


def test_unmapped_label_is_a_note_in_evaluation_mode(make_engine: EngineFactory) -> None:
    decision = make_engine().evaluate(posting("HQ"), mode=RunMode.EVALUATION, today=TODAY)
    assert decision.allowed
    assert mentioning(decision, UNRESOLVED)
    assert any(UNRESOLVED in n and n.startswith("EVALUATION only") for n in decision.notes)


def test_address_covers_only_the_primary_location(make_engine: EngineFactory) -> None:
    decision = live(make_engine(), posting("Hudson Yards Office", "Mountain Office",
                                          details=NYC_DETAILS))
    unresolved = mentioning(decision, UNRESOLVED)
    assert len(unresolved) == 1 and "'Mountain Office'" in unresolved[0]
    assert "'Hudson Yards Office'" not in unresolved[0]


def test_address_without_a_region_does_not_cover_the_label(make_engine: EngineFactory) -> None:
    decision = live(make_engine(), posting("HQ", details={"country": "US"}))
    assert mentioning(decision, UNRESOLVED)


@pytest.mark.parametrize("label", ["Remote - US", "Remote", "Anywhere", "Chicago, IL",
                                   "Austin, TX", "Buffalo", "Greater Denver Area",
                                   "New York Metro Area", "Toronto, ON", "Hyderabad, India"])
def test_mapped_remote_and_foreign_labels_are_not_unresolved(make_engine: EngineFactory,
                                                             label: str) -> None:
    decision = live(make_engine(allowed_countries=["US", "CA"]),
                    posting(label, workplace="remote" if "Remote" in label else "onsite"))
    assert not mentioning(decision, UNRESOLVED), decision.blockers


def test_country_only_label_is_unresolved(make_engine: EngineFactory) -> None:
    decision = live(make_engine(), posting("US Office"))
    assert mentioning(decision, UNRESOLVED)


def test_unresolved_check_applies_to_us_postings_only(make_engine: EngineFactory) -> None:
    engine = make_engine(allowed_countries=["US", "CA"])
    decision = live(engine, posting("Head Office", country="CA"))
    assert not mentioning(decision, UNRESOLVED)
    assert decision.regimes == ["IL_HRA", "ON_ESA"]
    decision = live(engine, posting("Head Office", country=None))
    assert mentioning(decision, "has no country")
    assert not mentioning(decision, UNRESOLVED)


def test_posting_without_locations_is_not_unresolved(make_engine: EngineFactory) -> None:
    assert live(make_engine(), posting()).allowed


# ------------------------------------------------------------------ location_overrides


def test_override_maps_a_label_to_regimes(make_engine: EngineFactory) -> None:
    engine = with_overrides(make_engine(nyc={"bias_audit_date": None}),
                            {"hudson  yards office": ["NYC_LL144"]})
    decision = live(engine, posting("Hudson Yards Office"))
    assert "NYC_LL144" in decision.regimes
    assert mentioning(decision, "nyc.bias_audit_date")
    assert not mentioning(decision, UNRESOLVED)
    assert engine.config is not None
    assert engine.config.override_for(" HUDSON YARDS OFFICE ") == ["NYC_LL144"]
    assert engine.config.override_for("Mountain Office") is None


def test_override_with_no_regimes_approves_the_label(make_engine: EngineFactory) -> None:
    decision = live(with_overrides(make_engine(), {"HQ": []}), posting("HQ"))
    assert decision.allowed, decision.blockers
    assert decision.regimes == ["IL_HRA"]


def test_override_never_removes_a_detected_regime(make_engine: EngineFactory) -> None:
    engine = with_overrides(make_engine(), {"New York, NY": [], "Denver, CO": ["IL_HRA"]})
    assert engine.detect_regimes(posting("New York, NY", "Denver, CO")) == [
        "IL_HRA", "NYC_LL144", "CO_ADMT"]


def test_override_applies_per_label_not_per_posting(make_engine: EngineFactory) -> None:
    engine = with_overrides(make_engine(), {"HQ": []})
    decision = live(engine, posting("HQ", "Mountain Office"))
    [blocker] = mentioning(decision, UNRESOLVED)
    assert "'Mountain Office'" in blocker


def test_override_does_not_match_the_structured_address(make_engine: EngineFactory) -> None:
    engine = with_overrides(make_engine(),
                            {"New York, NY, US (structured address)": ["CO_ADMT"]})
    assert "CO_ADMT" not in engine.detect_regimes(posting("Office", details=NYC_DETAILS))


@pytest.mark.parametrize(("overrides", "message"), [
    ({"HQ": ["NYC"]}, "unknown regime"),
    ({"HQ": ["NYC_LL144", "colorado"]}, "unknown regime"),
    ({"  ": []}, "must not be empty"),
    ({"HQ": [], "hq": ["CO_ADMT"]}, "more than once"),
])
def test_invalid_overrides_are_rejected(overrides: dict[str, list[str]], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        PolicyConfig.model_validate({"location_overrides": overrides})


def test_overrides_default_to_none() -> None:
    assert PolicyConfig().location_overrides == {}
    assert PolicyConfig().override_for("HQ") is None
