"""Lenient models of Lever Data API v1 responses.

Lever adds fields without a version bump and its docs and examples disagree in places
(``archived`` object vs top-level ``archivedAt``; ``posting`` vs ``postingId``; ``schools``
vs ``school``; sizes and timestamps sometimes strings). So: every field is optional,
unknown fields are kept (``extra="allow"``), ids are plain strings (some are not UUIDs),
and millisecond timestamps are coerced to int. A null where Lever documents a list, an
object or a string means "use the default" (see ``LeverModel``), so one null can't make a
whole record, and with it an applicant, disappear. An opportunity that still fails
validation is represented by ``Opportunity.unreadable`` (``record_valid=False``, storage
not allowed) so the applicant keeps a visible row instead of being dropped.
"""

from __future__ import annotations

import types
from typing import Any, Union, get_args, get_origin

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class LeverModel(BaseModel):
    """Base of every Lever model. A null for a field that has a default but does not
    accept None (lists, dicts, nested objects, strings) is dropped before validation, so
    the default applies instead of the whole record failing. Optional fields keep their
    null; ``Opportunity.isAnonymized`` is Optional on purpose (null must not read as
    "not anonymized")."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _null_means_default(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        keys = _null_default_keys(cls)
        if not any(data.get(key, ...) is None for key in keys):
            return data
        return {k: v for k, v in data.items() if not (v is None and k in keys)}


_NULL_DEFAULT_KEYS: dict[type[BaseModel], frozenset[str]] = {}


def _null_default_keys(model: type[BaseModel]) -> frozenset[str]:
    """Input keys (field names and string aliases) of the fields that have a default but
    do not accept None. Cached per model class."""
    keys = _NULL_DEFAULT_KEYS.get(model)
    if keys is None:
        found: set[str] = set()
        for name, info in model.model_fields.items():
            if info.is_required() or _accepts_none(info.annotation):
                continue
            found.add(name)
            aliases = [info.alias, info.validation_alias]
            if isinstance(info.validation_alias, AliasChoices):
                aliases.extend(info.validation_alias.choices)
            found.update(alias for alias in aliases if isinstance(alias, str))
        keys = _NULL_DEFAULT_KEYS[model] = frozenset(found)
    return keys


def _accepts_none(annotation: Any) -> bool:
    if annotation is None or annotation is type(None) or annotation is Any:
        return True
    if get_origin(annotation) in (Union, types.UnionType):
        return any(_accepts_none(arg) for arg in get_args(annotation))
    return False


def _to_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------- postings


class PostingList(LeverModel):
    text: str = ""
    content: str = ""  # HTML <li> items


class PostingContent(LeverModel):
    description: str = ""
    descriptionHtml: str = ""
    lists: list[PostingList] = Field(default_factory=list)
    closing: str = ""
    closingHtml: str = ""


class LocationDetails(LeverModel):
    """Structured address of the primary location (``categories.location`` only), present
    when the account has configured one. Lever omits components without a value."""

    addressLine1: str | None = None
    addressLine2: str | None = None
    city: str | None = None
    region: str | None = None  # state, province or district, e.g. "NY" or "ON"
    postalCode: str | None = None
    country: str | None = None  # ISO 3166-1 alpha-2, e.g. "US"

    @field_validator("*", mode="before")
    @classmethod
    def _number_to_str(cls, v: Any) -> Any:
        # A postal code sent as a JSON number must not fail the whole posting.
        if isinstance(v, int | float) and not isinstance(v, bool):
            return str(v)
        return v


class PostingCategories(LeverModel):
    team: str | None = None
    department: str | None = None
    location: str | None = None
    locationDetails: LocationDetails | None = None  # describes ``location`` only
    allLocations: list[str] = Field(default_factory=list)
    commitment: str | None = None
    level: str | None = None


class Posting(LeverModel):
    id: str
    text: str = ""  # title
    state: str | None = None  # published | internal | closed | draft | pending | rejected
    # "confidential" | "non-confidential". Lever lists a confidential posting's
    # opportunities only when asked for confidentiality=all.
    confidentiality: str | None = None
    content: PostingContent = Field(default_factory=PostingContent)
    categories: PostingCategories = Field(default_factory=PostingCategories)
    country: str | None = None  # ISO code, e.g. "US"
    workplaceType: str | None = None  # onsite | remote | hybrid | unspecified
    tags: list[str] = Field(default_factory=list)
    createdAt: int | None = None
    updatedAt: int | None = None
    urls: dict[str, str] = Field(default_factory=dict)

    _coerce_ts = field_validator("createdAt", "updatedAt", mode="before")(
        lambda cls, v: _to_int(v)
    )

    @property
    def locations(self) -> list[str]:
        locs = list(self.categories.allLocations)
        if self.categories.location and self.categories.location not in locs:
            locs.insert(0, self.categories.location)
        return locs


# ------------------------------------------------------------------------ opportunities


class Archived(LeverModel):
    archivedAt: int | None = None
    reason: str | None = None  # archive reason id

    _coerce_ts = field_validator("archivedAt", mode="before")(lambda cls, v: _to_int(v))


class StageChange(LeverModel):
    toStageId: str | None = None
    toStageIndex: int | None = None
    updatedAt: int | None = None
    userId: str | None = None

    _coerce_ts = field_validator("updatedAt", mode="before")(lambda cls, v: _to_int(v))


class Application(LeverModel):
    id: str
    opportunityId: str | None = None
    # The docs' examples use "posting"; the updatedAt description mentions "postingId".
    posting: str | None = Field(default=None, validation_alias=AliasChoices("posting", "postingId"))
    type: str | None = None  # posting | user | referral
    createdAt: int | None = None
    archived: Archived | None = None

    _coerce_ts = field_validator("createdAt", mode="before")(lambda cls, v: _to_int(v))

    @field_validator("posting", mode="before")
    @classmethod
    def _posting_id(cls, v: Any) -> Any:
        # An expanded posting is an object; keep only its id.
        if isinstance(v, dict):
            return v.get("id")
        return v


class DataProtectionStore(LeverModel):
    allowed: bool | None = None
    expiresAt: int | None = None


class DataProtection(LeverModel):
    store: DataProtectionStore | None = None
    contact: DataProtectionStore | None = None


class Opportunity(LeverModel):
    id: str
    name: str | None = None
    headline: str | None = None
    contact: str | None = None
    emails: list[str] = Field(default_factory=list)
    phones: list[dict[str, Any]] = Field(default_factory=list)  # [{type, value}]
    links: list[str] = Field(default_factory=list)
    location: str | None = None
    stage: str | None = None  # stage id (may be an object when expanded)
    stageChanges: list[StageChange] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    origin: str | None = None
    # With expand=applications these are objects; otherwise bare ids (strings).
    applications: list[Application | str] = Field(default_factory=list)
    archived: Archived | None = None
    archivedAt: int | None = None  # older shape seen in examples
    archiveReason: str | None = None
    createdAt: int | None = None
    updatedAt: int | None = None
    # Optional on purpose: a null must not read as "not anonymized" (see storage_allowed).
    isAnonymized: bool | None = False
    dataProtection: DataProtection | None = None
    urls: dict[str, str] = Field(default_factory=dict)
    # False only on the placeholder ``unreadable`` builds for a record that failed
    # validation; the pipeline then skips it visibly instead of dropping the applicant.
    record_valid: bool = True

    _coerce_ts = field_validator("createdAt", "updatedAt", "archivedAt", mode="before")(
        lambda cls, v: _to_int(v)
    )

    @classmethod
    def unreadable(cls, opportunity_id: str) -> Opportunity:
        """A stand-in for an opportunity whose Lever record did not validate: only the id
        is kept, storage is not allowed (``isAnonymized=None``) and ``record_valid`` is
        False. Built without validation, so nothing from the bad record is copied."""
        return cls.model_construct(
            id=opportunity_id, isAnonymized=None, record_valid=False, emails=[], phones=[],
            links=[], tags=[], sources=[], stageChanges=[], applications=[], urls={},
        )

    @field_validator("stage", mode="before")
    @classmethod
    def _stage_id(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return v.get("id")
        return v

    @property
    def application(self) -> Application | None:
        for app in self.applications:
            if isinstance(app, Application):
                return app
        return None

    @property
    def posting_id(self) -> str | None:
        app = self.application
        return app.posting if app else None

    @property
    def is_archived(self) -> bool:
        if self.archived is not None and self.archived.archivedAt:
            return True
        return bool(self.archivedAt)

    @property
    def archive_reason_id(self) -> str | None:
        if self.archived is not None and self.archived.reason:
            return self.archived.reason
        return self.archiveReason

    @property
    def storage_allowed(self) -> bool:
        """False when Lever's data-protection settings say we may not store this candidate,
        or when ``isAnonymized`` is unknown (null): privacy fails closed."""
        if self.isAnonymized is None or self.isAnonymized:
            return False
        dp = self.dataProtection
        if dp is not None and dp.store is not None and dp.store.allowed is False:
            return False
        return True

    @property
    def profile_url(self) -> str | None:
        return self.urls.get("show")


# ------------------------------------------------------------------------------ resumes


class ResumeFile(LeverModel):
    name: str | None = None
    ext: str | None = None  # "pdf" for resumes; files use ".pdf" — normalise with lstrip(".")
    downloadUrl: str | None = None  # web-app URL; never used (download via the API)
    uploadedAt: int | None = None
    status: str | None = None  # processing | processed | unsupported | error | None
    size: int | None = None

    _coerce = field_validator("uploadedAt", "size", mode="before")(lambda cls, v: _to_int(v))


class ParsedDate(LeverModel):
    year: int | None = None
    month: int | None = None


class ParsedPosition(LeverModel):
    org: str | None = None
    title: str | None = None
    summary: str | None = None
    location: str | None = None
    start: ParsedDate | None = None
    end: ParsedDate | None = None


class ParsedSchool(LeverModel):
    org: str | None = None
    degree: str | None = None
    field: str | None = None
    summary: str | None = None
    start: ParsedDate | None = None
    end: ParsedDate | None = None


class ParsedData(LeverModel):
    positions: list[ParsedPosition] = Field(default_factory=list)
    schools: list[ParsedSchool] = Field(
        default_factory=list, validation_alias=AliasChoices("schools", "school")
    )


class Resume(LeverModel):
    id: str
    createdAt: int | None = None
    file: ResumeFile | None = None
    parsedData: ParsedData | None = None

    _coerce_ts = field_validator("createdAt", mode="before")(lambda cls, v: _to_int(v))


class DownloadedFile(BaseModel):
    content: bytes
    content_type: str | None = None


# ------------------------------------------------------------------------- reference data


class Stage(LeverModel):
    id: str
    text: str = ""


class ArchiveReason(LeverModel):
    id: str
    text: str = ""
    type: str | None = None  # hired | non-hired


class User(LeverModel):
    id: str
    name: str | None = None
    email: str | None = None
    accessRole: str | None = None


class DeletedApplication(LeverModel):
    """A row of GET /applications/deleted: a deleted *application*, whose candidate may
    well still exist. Not used for candidate deletions (see ``DeletedOpportunity``)."""

    id: str
    deletedAt: int | None = None
    opportunityId: str | None = None
    postingId: str | None = None

    _coerce_ts = field_validator("deletedAt", mode="before")(lambda cls, v: _to_int(v))


class DeletedOpportunity(LeverModel):
    """A row of GET /opportunities/deleted: ``id`` is the deleted opportunity's id."""

    id: str
    deletedAt: int | None = None
    deletedBy: str | None = None

    _coerce_ts = field_validator("deletedAt", mode="before")(lambda cls, v: _to_int(v))
