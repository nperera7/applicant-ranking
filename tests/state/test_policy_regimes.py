"""Regime detection and country inference from Lever posting locations."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from applicant_ranking.lever.models import Posting
from applicant_ranking.policy import PolicyEngine

PostingFactory = Callable[..., Posting]


@pytest.fixture
def engine(make_engine: Callable[..., PolicyEngine]) -> PolicyEngine:
    return make_engine()


def regimes(engine: PolicyEngine, posting: Posting) -> set[str]:
    return set(engine.detect_regimes(posting))


# ----------------------------------------------------------------------------- Illinois


def test_illinois_always_applies(engine: PolicyEngine, make_posting: PostingFactory) -> None:
    for posting in (make_posting(), make_posting("Austin, TX"), make_posting("Toronto, ON"),
                    make_posting("Hyderabad, India", country="IN")):
        assert "IL_HRA" in engine.detect_regimes(posting)


def test_posting_with_no_location_fields_gets_illinois_only(engine: PolicyEngine) -> None:
    bare = Posting(id="p-bare")
    assert engine.detect_regimes(bare) == ["IL_HRA"]
    assert engine.effective_country(bare) is None


def test_none_and_blank_locations_are_ignored(engine: PolicyEngine) -> None:
    posting = Posting.model_validate({"id": "p", "country": "US", "workplaceType": None,
                                      "categories": {"location": "  ", "allLocations": [""]}})
    assert engine.detect_regimes(posting) == ["IL_HRA"]


# ------------------------------------------------------------------------- New York City


@pytest.mark.parametrize("location", [
    "New York, NY",
    "New York, New York",
    "New York, NY 10001",
    "New York, NY, USA",
    "New York",
    "NYC",
    "New York City",
    "Manhattan, NY",
    "Brooklyn, NY",
    "Brooklyn",
    "Queens, New York",
    "The Bronx, NY",
    "Bronx",
    "Staten Island, NY",
    "Long Island City, NY",
    "Hybrid - New York, NY",
    "New York, NY (Hybrid)",
    "Remote (NYC office)",
    "N.Y.C.",
    # Forms Lever accounts use that name New York State but no non-NYC city: fail closed.
    "NY, NY",
    "NY",
    "US - New York",
    "New York Metro Area",
    "Greater New York Area",
    "Greater New York City Area",
    "Midtown Manhattan, NY",
    "Lower Manhattan",
    "Downtown Brooklyn, NY",
    "Hudson Yards, Manhattan, NY",
    "Tri-State Area, NY",
])
def test_nyc_locations(engine: PolicyEngine, make_posting: PostingFactory, location: str) -> None:
    assert "NYC_LL144" in regimes(engine, make_posting(location))


@pytest.mark.parametrize("location", [
    "Albany, NY",
    "Buffalo, New York",
    "Rochester, NY",
    "Syracuse, NY",
    "New York State",
    "Upstate New York",
    "Manhattan, KS",
    "Brooklyn Park, MN",
    "Manhattan Beach, CA",
    "Delhi, NY",
    "Newark, NJ",
    "Ithaca, NY",
    "White Plains, NY",
    "Yonkers, NY",
    "Buffalo",
    "Greater Albany Area",
    "Albany County, NY",
    "West New York, NJ",
    "Long Island, NY",
])
def test_not_nyc(engine: PolicyEngine, make_posting: PostingFactory, location: str) -> None:
    assert "NYC_LL144" not in regimes(engine, make_posting(location))


@pytest.mark.parametrize("location", ["Albany, NY", "Buffalo", "Rochester, New York",
                                      "Syracuse, NY", "Greater Albany Area"])
def test_known_upstate_cities_still_resolve_to_new_york_state(
        engine: PolicyEngine, make_posting: PostingFactory, location: str) -> None:
    """They are in the city table (so the posting is not an unresolved US label) but never
    bring in NYC LL144."""
    assert engine.effective_country(make_posting(location, country=None)) == "US"
    assert regimes(engine, make_posting(location)) == {"IL_HRA"}


def test_remote_posting_is_nyc_only_with_the_remote_flag(
        make_engine: Callable[..., PolicyEngine], make_posting: PostingFactory) -> None:
    posting = make_posting("Remote - US", workplace="remote")
    assert "NYC_LL144" not in regimes(make_engine(), posting)
    flagged = make_engine(nyc={"remote_roles_associated_with_nyc_office": True})
    assert "NYC_LL144" in regimes(flagged, posting)


def test_remote_flag_does_not_make_onsite_postings_nyc(
        make_engine: Callable[..., PolicyEngine], make_posting: PostingFactory) -> None:
    flagged = make_engine(nyc={"remote_roles_associated_with_nyc_office": True})
    assert "NYC_LL144" not in regimes(flagged, make_posting("Chicago, IL", workplace="onsite"))
    assert "NYC_LL144" not in regimes(flagged, make_posting("Chicago, IL", workplace="hybrid"))


def test_without_policy_file_remote_is_not_assumed_nyc(make_posting: PostingFactory) -> None:
    posting = make_posting("Remote - US", workplace="remote")
    assert "NYC_LL144" not in regimes(PolicyEngine(None), posting)


# ----------------------------------------------------------------------------- Colorado


@pytest.mark.parametrize("location", [
    "Denver, CO", "Boulder, Colorado", "Colorado", "Colorado Springs, CO", "Denver",
    "Fort Collins, CO, United States", "Remote - Colorado", "Greenwood Village, CO",
    "Denver Metro Area", "Greater Denver Area", "Downtown Denver, CO", "Boulder County, CO",
    "Denver Metropolitan Region",
])
def test_colorado_locations(engine: PolicyEngine, make_posting: PostingFactory,
                            location: str) -> None:
    assert "CO_ADMT" in regimes(engine, make_posting(location))


@pytest.mark.parametrize("location", ["Chicago, IL", "Aurora, IL", "Columbus, OH",
                                      "Toronto, ON", "Denver Road, Naperville, IL"])
def test_not_colorado(engine: PolicyEngine, make_posting: PostingFactory,
                      location: str) -> None:
    assert "CO_ADMT" not in regimes(engine, make_posting(location))


# --------------------------------------------------------------------------- California


@pytest.mark.parametrize("location", [
    "San Francisco, CA", "Los Angeles, California", "California", "Palo Alto",
    "Irvine, CA, USA", "San Jose, CA 95112", "Manhattan Beach, CA", "Ontario, CA",
    "Remote - California", "Los Angeles Metropolitan Area", "Greater Los Angeles",
    "Downtown Los Angeles, CA", "San Francisco Bay Area", "Bay Area",
])
def test_california_locations(engine: PolicyEngine, make_posting: PostingFactory,
                              location: str) -> None:
    assert "CA_FEHA" in regimes(engine, make_posting(location))


def test_ontario_california_is_not_ontario_canada(engine: PolicyEngine,
                                                  make_posting: PostingFactory) -> None:
    assert regimes(engine, make_posting("Ontario, CA")) == {"IL_HRA", "CA_FEHA"}


@pytest.mark.parametrize("location", ["Toronto, ON, CA", "Austin, TX", "Chicago, IL"])
def test_not_california(engine: PolicyEngine, make_posting: PostingFactory,
                        location: str) -> None:
    assert "CA_FEHA" not in regimes(engine, make_posting(location))


# ------------------------------------------------------------------------------ Ontario


@pytest.mark.parametrize("location", [
    "Mississauga, ON", "Toronto", "Toronto, Ontario, Canada", "Ontario", "Toronto, ON, CA",
    "Ottawa, Ontario", "London, ON",
])
def test_ontario_locations(engine: PolicyEngine, make_posting: PostingFactory,
                           location: str) -> None:
    assert "ON_ESA" in regimes(engine, make_posting(location, country="CA"))
    assert "ON_ESA" in regimes(engine, make_posting(location, country=None))


def test_country_canada_brings_in_ontario(engine: PolicyEngine,
                                          make_posting: PostingFactory) -> None:
    assert "ON_ESA" in regimes(engine, make_posting("Vancouver, BC", country="CA"))
    assert "ON_ESA" in regimes(engine, make_posting(country="ca"))


def test_canada_in_a_location_brings_in_ontario(engine: PolicyEngine,
                                                make_posting: PostingFactory) -> None:
    posting = make_posting("Remote - US & Canada", workplace="remote")
    assert "ON_ESA" in regimes(engine, posting)


@pytest.mark.parametrize("location", ["London, UK", "New Brunswick, NJ", "Chicago, IL"])
def test_not_ontario(engine: PolicyEngine, make_posting: PostingFactory,
                     location: str) -> None:
    assert "ON_ESA" not in regimes(engine, make_posting(location))


# ------------------------------------------------------------------------------- remote


def test_remote_us_posting_gets_colorado_and_california(engine: PolicyEngine,
                                                        make_posting: PostingFactory) -> None:
    posting = make_posting("Remote - US", workplace="remote")
    assert engine.detect_regimes(posting) == ["IL_HRA", "CO_ADMT", "CA_FEHA"]


def test_remote_detected_from_location_text(engine: PolicyEngine,
                                            make_posting: PostingFactory) -> None:
    posting = make_posting("Remote", workplace=None)
    assert engine.detect_regimes(posting) == ["IL_HRA", "CO_ADMT", "CA_FEHA"]


def test_workplace_type_remote_without_locations(engine: PolicyEngine,
                                                 make_posting: PostingFactory) -> None:
    assert engine.detect_regimes(make_posting(workplace="Remote")) == [
        "IL_HRA", "CO_ADMT", "CA_FEHA"]


def test_remote_with_unknown_country_takes_the_union(engine: PolicyEngine,
                                                     make_posting: PostingFactory) -> None:
    posting = make_posting("Remote", country=None, workplace="remote")
    assert {"CO_ADMT", "CA_FEHA"} <= regimes(engine, posting)


def test_remote_canada_is_not_us_remote(engine: PolicyEngine,
                                        make_posting: PostingFactory) -> None:
    posting = make_posting("Remote - Canada", country="CA", workplace="remote")
    assert engine.detect_regimes(posting) == ["IL_HRA", "ON_ESA"]


def test_remote_india_is_not_us_remote(engine: PolicyEngine,
                                       make_posting: PostingFactory) -> None:
    posting = make_posting("Remote - India", country="IN", workplace="remote")
    assert engine.detect_regimes(posting) == ["IL_HRA"]


def test_hybrid_and_onsite_are_not_remote(engine: PolicyEngine,
                                          make_posting: PostingFactory) -> None:
    for workplace in ("hybrid", "onsite", "unspecified", None):
        assert engine.detect_regimes(make_posting("Chicago, IL", workplace=workplace)) == [
            "IL_HRA"]


# ----------------------------------------------------------------------- multi-location


def test_multi_location_posting_unions_regimes(engine: PolicyEngine,
                                               make_posting: PostingFactory) -> None:
    posting = make_posting("New York, NY", "Denver, CO", "Naperville, IL")
    assert engine.detect_regimes(posting) == ["IL_HRA", "NYC_LL144", "CO_ADMT"]


def test_several_locations_in_one_string(engine: PolicyEngine,
                                         make_posting: PostingFactory) -> None:
    posting = make_posting("New York, NY / San Francisco, CA; Toronto, ON")
    assert engine.detect_regimes(posting) == ["IL_HRA", "NYC_LL144", "CA_FEHA", "ON_ESA"]


def test_primary_location_and_all_locations_are_both_used(engine: PolicyEngine) -> None:
    posting = Posting.model_validate({
        "id": "p", "country": "US",
        "categories": {"location": "Denver, CO", "allLocations": ["San Diego, CA"]},
    })
    assert engine.detect_regimes(posting) == ["IL_HRA", "CO_ADMT", "CA_FEHA"]


def test_regime_order_is_stable(engine: PolicyEngine, make_posting: PostingFactory) -> None:
    posting = make_posting("Toronto, ON", "San Francisco, CA", "Denver, CO", "NYC")
    assert engine.detect_regimes(posting) == [
        "IL_HRA", "NYC_LL144", "CO_ADMT", "CA_FEHA", "ON_ESA"]


# --------------------------------------------------------------------- country inference


@pytest.mark.parametrize(("locations", "country", "expected"), [
    (("Chicago, IL",), "US", "US"),
    (("Chicago, IL",), "us", "US"),
    (("Chicago, IL",), "USA", "US"),
    (("Chicago, IL",), " gb ", "GB"),
    (("Hyderabad, India",), "IN", "IN"),
    (("Chicago, IL",), None, "US"),
    (("Chicago, IL", "Denver, CO", "New York, NY"), None, "US"),
    (("Naperville",), None, "US"),
    (("Indianapolis, IN",), None, "US"),
    (("Remote (Colorado)",), None, "US"),
    (("Remote",), None, None),
    (("Remote - US",), None, None),
    ((), None, None),
    (("Toronto, ON",), None, None),
    (("Chicago, IL", "London, UK"), None, None),
    (("Chicago, IL", "Remote"), None, None),
    (("Hyderabad, India",), None, None),
    (("Springfield",), None, None),
    (("Greater New York Area",), None, "US"),
    (("Denver Metro Area",), None, "US"),
    (("Hudson Yards Office",), None, None),
    (("Hudson Yards Office", "Chicago, IL"), None, None),
])
def test_effective_country(engine: PolicyEngine, make_posting: PostingFactory,
                           locations: tuple[str, ...], country: str | None,
                           expected: str | None) -> None:
    assert engine.effective_country(make_posting(*locations, country=country)) == expected
