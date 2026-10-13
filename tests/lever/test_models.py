"""The lenient Lever models: nulls as defaults, the privacy fail-closed ``isAnonymized``,
the ``unreadable`` placeholder, and the structured posting fields from the docs example."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from applicant_ranking.lever.models import (
    DeletedOpportunity,
    LocationDetails,
    Opportunity,
    Posting,
    Resume,
)

from .conftest import load

# ------------------------------------------------------------------- nulls mean defaults


def test_opportunity_nulls_mean_defaults() -> None:
    opp = Opportunity.model_validate({
        "id": "o1", "name": None, "emails": None, "phones": None, "links": None,
        "tags": None, "sources": None, "applications": None, "stageChanges": None,
        "urls": None, "archived": None, "dataProtection": None,
    })
    assert (opp.emails, opp.phones, opp.links, opp.tags, opp.sources) == ([], [], [], [], [])
    assert opp.applications == [] and opp.stageChanges == [] and opp.urls == {}
    assert opp.name is None and opp.archived is None
    assert opp.storage_allowed  # isAnonymized absent -> False
    assert opp.record_valid
    assert not opp.is_archived and opp.profile_url is None


def test_posting_nulls_mean_defaults() -> None:
    posting = Posting.model_validate({"id": "p", "tags": None, "urls": None,
                                      "categories": None, "content": None})
    assert posting.tags == [] and posting.urls == {}
    assert posting.locations == [] and posting.content.lists == []
    posting = Posting.model_validate({"id": "p", "categories": {
        "location": "Chicago, IL", "allLocations": None, "locationDetails": None}})
    assert posting.locations == ["Chicago, IL"]
    assert posting.categories.locationDetails is None


def test_resume_nulls_mean_defaults() -> None:
    resume = Resume.model_validate({"id": "r", "file": None,
                                    "parsedData": {"positions": None, "schools": None}})
    assert resume.file is None
    assert resume.parsedData is not None
    assert resume.parsedData.positions == [] and resume.parsedData.schools == []


def test_shapes_that_are_not_nulls_still_fail() -> None:
    with pytest.raises(ValidationError):
        Opportunity.model_validate({"id": "o", "tags": [1, 2]})
    with pytest.raises(ValidationError):
        Opportunity.model_validate({"name": "no id"})
    with pytest.raises(ValidationError):
        Posting.model_validate({"id": "p", "categories": {"allLocations": "Chicago"}})


# --------------------------------------------------------------------- privacy fails closed


def test_is_anonymized_null_blocks_storage() -> None:
    """A null is not "not anonymized": the before-validator leaves it alone because the
    field accepts None, and storage_allowed then says no."""
    opp = Opportunity.model_validate({"id": "o", "name": "Jane Example", "isAnonymized": None})
    assert opp.isAnonymized is None
    assert opp.storage_allowed is False
    assert Opportunity(id="o").storage_allowed is True
    assert Opportunity(id="o", isAnonymized=False).storage_allowed is True
    assert Opportunity(id="o", isAnonymized=True).storage_allowed is False
    assert Opportunity(id="o", isAnonymized=None).storage_allowed is False


def test_data_protection_store_denied_blocks_storage() -> None:
    opp = Opportunity.model_validate({"id": "o", "dataProtection": {"store": {"allowed": False}}})
    assert opp.storage_allowed is False
    opp = Opportunity.model_validate({"id": "o", "dataProtection": {"store": None}})
    assert opp.storage_allowed is True


# ------------------------------------------------------------------ unreadable placeholder


def test_unreadable_placeholder_keeps_only_the_id() -> None:
    opp = Opportunity.unreadable("opp-9")
    assert opp.id == "opp-9"
    assert opp.record_valid is False
    assert opp.storage_allowed is False
    assert opp.isAnonymized is None
    assert opp.name is None and opp.headline is None and opp.contact is None
    assert (opp.emails, opp.phones, opp.links, opp.tags, opp.sources) == ([], [], [], [], [])
    assert opp.applications == [] and opp.stageChanges == [] and opp.urls == {}
    # Everything the pipeline reads from an opportunity works on the placeholder.
    assert opp.application is None and opp.posting_id is None
    assert opp.profile_url is None and not opp.is_archived and opp.archive_reason_id is None


def test_record_valid_is_true_for_validated_records() -> None:
    assert Opportunity.model_validate({"id": "o"}).record_valid is True
    assert Opportunity.model_validate({"id": "o", "record_valid": None}).record_valid is True


# ------------------------------------------------------------ structured posting fields


def test_deleted_opportunity_row() -> None:
    row = DeletedOpportunity.model_validate({"id": "opp-1", "deletedAt": "1700000000000",
                                             "deletedBy": None, "unexpected": 1})
    assert (row.id, row.deletedAt, row.deletedBy) == ("opp-1", 1_700_000_000_000, None)
    assert DeletedOpportunity.model_validate({"id": "opp-2"}).deletedAt is None
    with pytest.raises(ValidationError):
        DeletedOpportunity.model_validate({"deletedAt": 1})


def test_posting_docs_example_parses_location_details_and_confidentiality() -> None:
    posting = Posting.model_validate(load("posting.json")["data"])
    details = posting.categories.locationDetails
    assert isinstance(details, LocationDetails)
    assert (details.city, details.region, details.postalCode, details.country) == (
        "San Francisco", "CA", "94107", "US")
    assert details.addressLine1 == "185 Berry St" and details.addressLine2 == "Suite 5000"
    assert posting.confidentiality == "non-confidential"
    assert posting.locations == ["San Francisco", "New York City"]
    assert posting.country == "US"


def test_location_details_tolerate_numbers_and_omitted_components() -> None:
    details = LocationDetails.model_validate({"postalCode": 94107, "region": "CA"})
    assert details.postalCode == "94107"
    assert details.city is None and details.country is None and details.addressLine1 is None
    assert LocationDetails.model_validate({}).country is None


def test_posting_confidentiality_absent_or_null() -> None:
    assert Posting.model_validate({"id": "p"}).confidentiality is None
    assert Posting.model_validate({"id": "p", "confidentiality": None}).confidentiality is None
    assert Posting.model_validate({"id": "p", "confidentiality": "confidential"}
                                  ).confidentiality == "confidential"
