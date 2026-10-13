"""Jurisdiction policy engine. FAILS CLOSED: if the policy file is missing, a regime's
required configuration is incomplete, or a posting is outside the approved countries,
DRY_RUN and LIVE runs are blocked for that posting. EVALUATION runs are never blocked
(their output is watermarked and can't be written back), but the decision's notes and
blockers are still reported.

Regimes (research 2026-10-08; not legal advice — Legal/HR own config/policy.yaml):
- IL_HRA (Illinois HB 3773, in force 2026-01-01): always applies (Egen HQ is in
  Naperville, IL). Needs ``illinois.notice_url`` and ``illinois.posting_notice_confirmed``.
- NYC_LL144 (AEDT): applies when a posting location is in New York City (New York, NY;
  NYC; Manhattan; Brooklyn; Queens; Bronx; Staten Island), or the posting is remote and
  ``nyc.remote_roles_associated_with_nyc_office`` is true. Needs a bias audit dated within
  365 days of today, ``nyc.audit_summary_url``, and ``nyc.notice_published_at`` at least
  10 NYC business days (weekdays; holidays in ``nyc.holidays``) before today.
- CO_ADMT (Colorado SB 26-189, decisions on/after 2027-01-01): applies when a location is
  in Colorado (", CO", "Colorado", Denver, Boulder, ...) or the posting is remote in the US
  (Colorado-resident applicants). From 2027-01-01 needs
  ``colorado.adverse_outcome_workflow_ready``; before that it is a note only.
- CA_FEHA (California ADS regulations, in force 2025-10-01): applies for California
  locations or US-remote postings. Note: 4-year record retention (store default). From
  2027-01-01, if ``california.ccpa_admt_applies`` is true, needs
  ``california.ccpa_notice_ready``.
- ON_ESA (Ontario Employment Standards Act, postings on/after 2026-01-01): applies for
  Ontario locations (Mississauga, Toronto, ", ON", "Ontario") or country CA (Canada).
  Needs ``ontario.posting_disclosure_confirmed``.
Country gate: ``posting.country`` (else the structured address's country, else "US" when
every location matches a US state) must be in ``allowed_countries`` (default ["US"]);
anything else — EU, UK, India, unknown — is blocked (EU AI Act / UK GDPR / DPDP not
assessed).

Where a posting is comes from two Lever sources, and their regimes are UNIONED (more
regimes is the fail-closed direction): the location tags (``categories.location`` and
``allLocations``, free text) and the structured address ``categories.locationDetails``
(city, region, postal code, country; Lever keeps one for the primary location only).

Interpretation choices (all fail closed):
- Location strings are matched conservatively: a city name must agree with any state or
  country named next to it ("Manhattan, KS" is not NYC; "Ontario, CA" is California).
  Metro qualifiers are ignored for the lookup ("Greater Denver Area", "Midtown
  Manhattan, NY", "Los Angeles Metropolitan Area").
- New York fails closed: a location that names New York State without a known non-NYC
  city ("New York", "NY", "NY, NY", "US - New York", "New York Metro Area") is treated as
  New York City. "Albany, NY", "Buffalo", "Upstate New York" are not (``_NON_NYC_NY``).
- The structured address resolves NYC from the city, else from the ZIP prefix
  (``NYC_ZIP_PREFIXES``), else - unknown New York city, no ZIP - as NYC.
- A US posting whose location tag maps to neither a state nor a city ("Hudson Yards
  Office", "HQ") is BLOCKED until Lever has a structured address for it (primary location
  only) or Legal/HR list the label under ``location_overrides`` with the regimes that
  apply ([] for none). Overrides only add regimes; they never remove detected ones.
- A remote posting whose country is unknown is treated as possibly US-remote (CO + CA).
- A location that names a country outside ``allowed_countries`` ("Hyderabad, India",
  "Toronto, ON") is a blocker even when ``posting.country`` is allowed.
- Canada anywhere (posting country or a location) brings in ON_ESA.
- From 2027-01-01 an undecided ``california.ccpa_admt_applies`` (null) is a blocker:
  Legal has to decide either way.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from applicant_ranking.domain import RunMode
from applicant_ranking.lever.models import LocationDetails, Posting

logger = logging.getLogger(__name__)

CO_ADMT_EFFECTIVE = date(2027, 1, 1)
CCPA_ADMT_EFFECTIVE = date(2027, 1, 1)
NYC_AUDIT_MAX_AGE_DAYS = 365
NYC_NOTICE_BUSINESS_DAYS = 10
# First three ZIP digits of the five boroughs: Manhattan 100-102, Staten Island 103, Bronx
# 104, Queens 110-111 and 113-116, Brooklyn 112. 110 and 116 also cover a few Nassau
# County addresses; over-inclusion fails closed.
NYC_ZIP_PREFIXES: frozenset[str] = frozenset(
    {"100", "101", "102", "103", "104", "110", "111", "112", "113", "114", "116"}
)

IL_HRA = "IL_HRA"
NYC_LL144 = "NYC_LL144"
CO_ADMT = "CO_ADMT"
CA_FEHA = "CA_FEHA"
ON_ESA = "ON_ESA"
REGIMES: tuple[str, ...] = (IL_HRA, NYC_LL144, CO_ADMT, CA_FEHA, ON_ESA)

_COUNTRY_ALIASES = {
    "USA": "US",
    "U.S.": "US",
    "U.S.A.": "US",
    "UNITED STATES": "US",
    "UNITED STATES OF AMERICA": "US",
    "CANADA": "CA",
    "UK": "GB",
    "UNITED KINGDOM": "GB",
}


def normalize_country(value: str) -> str:
    """ISO-3166 alpha-2 style code: upper-cased, with a few common spellings mapped
    ("USA" -> "US", "UK" -> "GB"). Note "CA" is Canada here, never California."""
    code = value.strip().upper()
    return _COUNTRY_ALIASES.get(code, code)


class IllinoisPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    notice_url: str | None = None
    posting_notice_confirmed: bool = False


class NycPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    remote_roles_associated_with_nyc_office: bool = False
    bias_audit_date: date | None = None
    audit_summary_url: str | None = None
    notice_published_at: date | None = None
    holidays: list[date] = Field(default_factory=list)


class ColoradoPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    adverse_outcome_workflow_ready: bool = False


class CaliforniaPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ccpa_admt_applies: bool | None = None  # Legal decides; None = undecided
    ccpa_notice_ready: bool = False


class OntarioPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    posting_disclosure_confirmed: bool = False


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = 1
    approved_by: str | None = None  # who in Legal/HR signed off this file
    approved_at: date | None = None
    allowed_countries: list[str] = Field(default_factory=lambda: ["US"])
    illinois: IllinoisPolicy = Field(default_factory=IllinoisPolicy)
    nyc: NycPolicy = Field(default_factory=NycPolicy)
    colorado: ColoradoPolicy = Field(default_factory=ColoradoPolicy)
    california: CaliforniaPolicy = Field(default_factory=CaliforniaPolicy)
    ontario: OntarioPolicy = Field(default_factory=OntarioPolicy)
    # Location labels the engine cannot map to a state or city, mapped by Legal/HR to the
    # regimes that apply there ([] = none). Matched case-insensitively against the
    # posting's location tags; the regimes are added to the detected ones, never removed.
    location_overrides: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("allowed_countries")
    @classmethod
    def _normalize_countries(cls, value: list[str]) -> list[str]:
        codes: list[str] = []
        for raw in value:
            code = normalize_country(raw)
            if not code:
                raise ValueError("allowed_countries must not contain empty entries")
            if code not in codes:
                codes.append(code)
        return codes

    @field_validator("location_overrides")
    @classmethod
    def _check_overrides(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        seen: set[str] = set()
        for label, regimes in value.items():
            key = _override_key(label)
            if not key:
                raise ValueError("location_overrides labels must not be empty")
            if key in seen:
                raise ValueError(f"location_overrides lists {label!r} more than once")
            seen.add(key)
            unknown = [r for r in regimes if r not in REGIMES]
            if unknown:
                raise ValueError(f"location_overrides[{label!r}]: unknown regime(s) "
                                 f"{', '.join(map(repr, unknown))}; valid: {', '.join(REGIMES)}")
        return value

    def override_for(self, label: str) -> list[str] | None:
        """The regimes Legal/HR assigned to a location label, or None when it is not listed."""
        wanted = _override_key(label)
        for key, regimes in self.location_overrides.items():
            if _override_key(key) == wanted:
                return list(regimes)
        return None


def _override_key(label: str) -> str:
    return re.sub(r"\s+", " ", label).strip().casefold()


class PolicyDecision(BaseModel):
    allowed: bool
    regimes: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------------ location parsing

_US_STATES: dict[str, str] = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas",
    "CA": "california", "CO": "colorado", "CT": "connecticut", "DE": "delaware",
    "FL": "florida", "GA": "georgia", "HI": "hawaii", "ID": "idaho", "IL": "illinois",
    "IN": "indiana", "IA": "iowa", "KS": "kansas", "KY": "kentucky", "LA": "louisiana",
    "ME": "maine", "MD": "maryland", "MA": "massachusetts", "MI": "michigan",
    "MN": "minnesota", "MS": "mississippi", "MO": "missouri", "MT": "montana",
    "NE": "nebraska", "NV": "nevada", "NH": "new hampshire", "NJ": "new jersey",
    "NM": "new mexico", "NY": "new york", "NC": "north carolina", "ND": "north dakota",
    "OH": "ohio", "OK": "oklahoma", "OR": "oregon", "PA": "pennsylvania",
    "RI": "rhode island", "SC": "south carolina", "SD": "south dakota",
    "TN": "tennessee", "TX": "texas", "UT": "utah", "VT": "vermont", "VA": "virginia",
    "WA": "washington", "WV": "west virginia", "WI": "wisconsin", "WY": "wyoming",
    "DC": "district of columbia",
}  # fmt: skip

_CA_PROVINCES: dict[str, str] = {
    "ON": "ontario", "QC": "quebec", "BC": "british columbia", "AB": "alberta",
    "MB": "manitoba", "SK": "saskatchewan", "NS": "nova scotia", "NB": "new brunswick",
    "NL": "newfoundland", "PE": "prince edward island", "YT": "yukon",
    "NT": "northwest territories", "NU": "nunavut",
}  # fmt: skip

_Region = tuple[str, str]  # (country, region code), e.g. ("US", "CO") or ("CA", "ON")


@dataclass(frozen=True)
class _City:
    country: str
    region: str | None
    nyc: bool = False


_NYC = _City("US", "NY", nyc=True)
_CITIES: dict[str, _City] = {
    **dict.fromkeys(
        ("new york", "new york city", "manhattan", "brooklyn", "queens", "bronx",
         "the bronx", "staten island", "long island city"),
        _NYC,
    ),
    **dict.fromkeys(("albany", "buffalo", "rochester", "syracuse"), _City("US", "NY")),
    **dict.fromkeys(
        ("denver", "boulder", "colorado springs", "fort collins", "greenwood village",
         "broomfield"),
        _City("US", "CO"),
    ),
    **dict.fromkeys(
        ("los angeles", "san francisco", "san jose", "san diego", "oakland", "sacramento",
         "palo alto", "mountain view", "sunnyvale", "santa clara", "santa monica",
         "irvine", "cupertino", "menlo park", "redwood city", "bay area",
         "san francisco bay area", "silicon valley"),
        _City("US", "CA"),
    ),
    **dict.fromkeys(("chicago", "naperville"), _City("US", "IL")),
    **dict.fromkeys(
        ("toronto", "mississauga", "ottawa", "brampton", "markham", "kitchener",
         "oakville", "vaughan", "greater toronto area", "gta"),
        _City("CA", "ON"),
    ),
    **dict.fromkeys(
        ("hyderabad", "bengaluru", "bangalore", "chennai", "pune", "mumbai", "delhi",
         "new delhi", "noida", "gurugram", "gurgaon", "kolkata"),
        _City("IN", None),
    ),
}  # fmt: skip

# Country (or region) names searched anywhere in a location. Pseudo-codes (EU, EMEA,
# APAC, LATAM) are never "allowed" unless Legal adds them explicitly.
_COUNTRY_WORDS: tuple[tuple[str, str], ...] = (
    ("US", r"united states(?: of america)?|usa|us"),
    ("CA", r"canada"),
    ("IN", r"india"),
    ("GB", r"united kingdom|uk|great britain|britain|england|scotland|wales|northern ireland"),
    ("IE", r"(?<!northern )ireland"),
    ("EU", r"european union|europe|eu"),
    ("EMEA", r"emea"),
    ("APAC", r"apac|asia"),
    ("LATAM", r"latam|latin america|south america"),
    ("MX", r"(?<!new )mexico"),
    ("DE", r"germany"), ("FR", r"france"), ("ES", r"spain"), ("IT", r"italy"),
    ("NL", r"netherlands"), ("PL", r"poland"), ("PT", r"portugal"), ("BE", r"belgium"),
    ("SE", r"sweden"), ("AT", r"austria"), ("DK", r"denmark"), ("FI", r"finland"),
    ("CZ", r"czechia|czech republic"), ("RO", r"romania"), ("GR", r"greece"),
    ("HU", r"hungary"), ("CH", r"switzerland"), ("NO", r"norway"), ("BR", r"brazil"),
    ("AU", r"australia"), ("SG", r"singapore"), ("PH", r"philippines"), ("JP", r"japan"),
    ("CN", r"china"), ("IL", r"israel"),
)  # fmt: skip
_COUNTRY_RES = tuple((code, re.compile(rf"\b(?:{pattern})\b")) for code, pattern in _COUNTRY_WORDS)


def _names_regex(names: dict[str, str]) -> re.Pattern[str]:
    alternatives = sorted(names.values(), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(map(re.escape, alternatives)) + r")\b")


_STATE_NAME_RE = _names_regex(_US_STATES)
_PROVINCE_NAME_RE = _names_regex(_CA_PROVINCES)
_STATE_BY_NAME = {name: code for code, name in _US_STATES.items()}
_PROVINCE_BY_NAME = {name: code for code, name in _CA_PROVINCES.items()}

_REMOTE_RE = re.compile(r"\b(?:remote|work from home|wfh|anywhere)\b")
_QUALIFIER_RE = re.compile(
    r"\b(?:fully|remote|hybrid|on-?site|on site|in-?office|in office|office|work from home|"
    r"wfh|anywhere|headquarters|hq|based)\b"
)
_NYC_RE = re.compile(r"\b(?:nyc|new york city)\b")
# Dropped before a part is looked up in the city table ("greater denver area" -> "denver").
# Table keys themselves keep these words ("bay area"), so the raw part is tried first.
_METRO_RE = re.compile(
    r"\b(?:greater|metro|metropolitan|area|region|county|downtown|midtown|uptown|lower|"
    r"upper|central)\b"
)
# New York State places that are not New York City. Any other city-less or unknown-city
# New York location is treated as NYC (fail closed; see the module docstring).
_NON_NYC_NY: frozenset[str] = frozenset((
    "albany", "buffalo", "rochester", "syracuse", "ithaca", "white plains", "yonkers",
    "delhi", "new york state", "upstate new york", "upstate", "western new york",
    "hudson valley", "long island", "westchester", "binghamton", "utica", "schenectady",
    "troy", "poughkeepsie", "new rochelle", "tarrytown", "armonk", "melville", "hauppauge",
    "garden city", "mineola", "hempstead", "stony brook", "saratoga springs", "plattsburgh",
    "niagara falls", "rye", "purchase",
))  # fmt: skip
_SUBLOCATION_SPLIT_RE = re.compile(r"\s*(?:;|\||/|&|\s\+\s)\s*")
_PART_SPLIT_RE = re.compile(r"\s*(?:,|\(|\)|\[|\]|:|\s[-\N{EN DASH}\N{EM DASH}]\s)\s*")
_PART_STRIP = " -\N{EN DASH}\N{EM DASH}'\"%*"


@dataclass(frozen=True)
class _Place:
    """What one Lever location string says about where the job is."""

    text: str
    regions: frozenset[_Region]
    countries: frozenset[str]
    nyc: bool
    remote: bool

    @property
    def names_us_state(self) -> bool:
        """A US state is named and no other country is."""
        return any(c == "US" for c, _ in self.regions) and self.countries <= {"US"}


def _normalize_location(text: str) -> str:
    text = text.casefold().replace(".", "")
    text = re.sub(r"\d+", " ", text)  # zip / postal codes
    return re.sub(r"\s+", " ", text).strip()


def _parts(sub_location: str) -> list[str]:
    parts = []
    for raw in _PART_SPLIT_RE.split(sub_location):
        part = re.sub(r"\s+", " ", _QUALIFIER_RE.sub(" ", raw)).strip(_PART_STRIP).strip()
        if part:
            parts.append(part)
    return parts


def _strip_metro(part: str) -> str:
    return re.sub(r"\s+", " ", _METRO_RE.sub(" ", part)).strip()


def _lookup_city(part: str) -> _City | None:
    """The city table entry for a normalised part, trying it as written first and then
    without metro qualifiers ("downtown denver" -> "denver")."""
    city = _CITIES.get(part)
    if city is None:
        stripped = _strip_metro(part)
        if stripped != part:
            city = _CITIES.get(stripped)
    return city


def _is_non_nyc_ny(part: str) -> bool:
    return part in _NON_NYC_NY or _strip_metro(part) in _NON_NYC_NY


def _analyze_sub_location(sub: str) -> tuple[set[_Region], set[str], bool]:
    parts = _parts(sub)
    if not parts:
        return set(), set(), False

    countries: set[str] = {code for code, rx in _COUNTRY_RES if rx.search(sub)}
    # The first of several parts is the city: "New Brunswick, NJ" and "Ontario, CA" are
    # cities, not the province / state of the same name.
    named = parts[1:] if len(parts) > 1 else parts
    provinces: set[_Region] = set()
    states: set[_Region] = set()
    for part in named:
        provinces |= {("CA", _PROVINCE_BY_NAME[m]) for m in _PROVINCE_NAME_RE.findall(part)}
        states |= {("US", _STATE_BY_NAME[m]) for m in _STATE_NAME_RE.findall(part)}
    for part in parts:
        code = part.upper()
        if code in _CA_PROVINCES:
            provinces.add(("CA", code))
    canadian = bool(provinces) or "CA" in countries
    for part in parts:
        code = part.upper()
        if code == "CA" and canadian:
            countries.add("CA")  # "Toronto, ON, CA": the country, not California
        elif code in _US_STATES:
            states.add(("US", code))
    regions = provinces | states

    nyc = bool(_NYC_RE.search(sub))
    # Every part may name a city ("Hudson Yards, Manhattan, NY"), but only one that agrees
    # with the state and country named next to it counts ("Manhattan, KS" is Kansas).
    city_known = False
    for index, part in enumerate(parts):
        if index and (part in _STATE_BY_NAME or part in _PROVINCE_BY_NAME):
            continue  # "Buffalo, New York": the state, not the city of the same name
        city = _lookup_city(part)
        if city is None:
            continue
        region_ok = not regions or (
            city.region is not None and (city.country, city.region) in regions
        )
        country_ok = not countries or city.country in countries
        if region_ok and country_ok:
            city_known = True
            if city.region is not None:
                regions.add((city.country, city.region))
            countries.add(city.country)
            nyc = nyc or city.nyc
    if not nyc and not city_known and ("US", "NY") in regions and countries <= {"US"}:
        # Fail closed: New York State without a known non-NYC city is New York City
        # ("NY, NY", "US - New York", "New York Metro Area").
        nyc = not any(_is_non_nyc_ny(part) for part in parts)
    if nyc:
        regions.add(("US", "NY"))
    countries |= {country for country, _ in regions}
    return regions, countries, nyc


def _analyze_location(text: str) -> _Place:
    normalized = _normalize_location(text)
    regions: set[_Region] = set()
    countries: set[str] = set()
    nyc = False
    for sub in _SUBLOCATION_SPLIT_RE.split(normalized):
        sub_regions, sub_countries, sub_nyc = _analyze_sub_location(sub)
        regions |= sub_regions
        countries |= sub_countries
        nyc = nyc or sub_nyc
    return _Place(
        text=text,
        regions=frozenset(regions),
        countries=frozenset(countries),
        nyc=nyc,
        remote=bool(_REMOTE_RE.search(normalized)),
    )


def _place_from_details(details: LocationDetails) -> _Place | None:
    """What Lever's structured address (``categories.locationDetails``) says about the
    primary location: the region from the state/province code or name, NYC from the city
    or ZIP prefix (``_details_in_nyc``). None when every component is blank."""
    raw_country = (details.country or "").strip()
    country = normalize_country(raw_country) if raw_country else None
    region_text = (details.region or "").strip()
    code, name = region_text.upper(), region_text.casefold()
    regions: set[_Region] = set()
    if country in (None, "US"):
        if code in _US_STATES:
            regions.add(("US", code))
        elif name in _STATE_BY_NAME:
            regions.add(("US", _STATE_BY_NAME[name]))
    if country in (None, "CA"):
        if code in _CA_PROVINCES:
            regions.add(("CA", code))
        elif name in _PROVINCE_BY_NAME:
            regions.add(("CA", _PROVINCE_BY_NAME[name]))
    countries = ({country} if country else set()) | {c for c, _ in regions}
    if not countries:
        return None
    nyc = ("US", "NY") in regions and _details_in_nyc(details.city, details.postalCode)
    shown = ", ".join(p for p in ((details.city or "").strip(), region_text, raw_country) if p)
    return _Place(text=f"{shown} (structured address)", regions=frozenset(regions),
                  countries=frozenset(countries), nyc=nyc, remote=False)


def _details_in_nyc(city: str | None, postal_code: str | None) -> bool:
    """For a New York State address: NYC when the ZIP prefix or the city says so; not NYC
    when the city is a known non-NYC one or a ZIP outside the boroughs is given; otherwise
    (unknown or missing city, no ZIP) NYC, failing closed."""
    prefix = re.sub(r"\D", "", postal_code or "")[:3]
    if prefix in NYC_ZIP_PREFIXES:
        return True
    key = _normalize_location(city or "")
    entry = _lookup_city(key) if key else None
    if entry is not None:
        return entry.nyc
    if key and _is_non_nyc_ny(key):
        return False
    return len(prefix) < 3


def _is_http_url(value: str | None) -> bool:
    if not value:
        return False
    return re.match(r"^https?://\S+$", value.strip(), re.IGNORECASE) is not None


# ---------------------------------------------------------------------------------- engine


@dataclass
class _Context:
    posting: Posting
    places: list[_Place]  # the location tags, plus ``structured`` when present
    structured: _Place | None  # from categories.locationDetails (primary location only)
    country: str | None
    remote: bool

    @property
    def text_places(self) -> list[_Place]:
        return [place for place in self.places if place is not self.structured]

    @property
    def location_countries(self) -> set[str]:
        return {c for place in self.places for c in place.countries}

    @property
    def regions(self) -> set[_Region]:
        return {r for place in self.places for r in place.regions}

    @property
    def remote_us(self) -> bool:
        if not self.remote:
            return False
        countries = self.location_countries
        # Unknown country and nothing pointing abroad: assume US-remote (union of regimes).
        return self.country == "US" or "US" in countries or (
            self.country is None and not countries
        )


class PolicyEngine:
    def __init__(self, config: PolicyConfig | None, *, source: str = "<memory>") -> None:
        """``config=None`` means no policy file: every DRY_RUN/LIVE decision is blocked."""
        self.config = config
        self.source = source

    @classmethod
    def from_file(cls, path: Path) -> PolicyEngine:
        """Missing file -> PolicyEngine(None). Invalid file -> ValueError (fail loudly)."""
        path = Path(path)
        source = str(path)
        if not path.exists():
            logger.warning("no jurisdiction policy file at %s: DRY_RUN/LIVE will be blocked",
                           source)
            return cls(None, source=source)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise ValueError(f"policy file {source} could not be read: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"policy file {source} must be a YAML mapping of settings")
        try:
            config = PolicyConfig.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"policy file {source} is invalid: {exc}") from exc
        engine = cls(config, source=source)
        logger.info("loaded jurisdiction policy %s (version %s, approved=%s)", source,
                    config.version, engine.approved)
        return engine

    @property
    def approved(self) -> bool:
        cfg = self.config
        return cfg is not None and bool(cfg.approved_by and cfg.approved_by.strip()
                                        and cfg.approved_at)

    def effective_country(self, posting: Posting) -> str | None:
        """``posting.country`` (normalised), else "US" when every location names a US
        state, else None (unknown)."""
        return self._context(posting).country

    def detect_regimes(self, posting: Posting) -> list[str]:
        return self._regimes(self._context(posting))

    def evaluate(self, posting: Posting, *, mode: RunMode, today: date) -> PolicyDecision:
        """An unapproved policy file (no approved_by/approved_at) is itself a blocker.

        ``allowed`` is the decision: always True in EVALUATION mode, otherwise True only
        without blockers. ``blockers`` are listed in every mode; in EVALUATION mode they
        are also repeated in ``notes`` (which reach the watermarked report).
        """
        if isinstance(today, datetime):
            today = today.date()
        ctx = self._context(posting)
        regimes = self._regimes(ctx)
        blockers: list[str] = []
        notes: list[str] = []
        cfg = self.config
        if cfg is None:
            blockers.append(
                f"No jurisdiction policy file at {self.source}: copy "
                "config/policy.example.yaml there and have Legal/HR complete and approve it."
            )
        else:
            self._check_approval(cfg, today, blockers, notes)
            self._check_country(ctx, cfg, blockers)
            self._check_locations(ctx, cfg, blockers)
            checks = {
                IL_HRA: _check_illinois,
                NYC_LL144: _check_nyc,
                CO_ADMT: _check_colorado,
                CA_FEHA: _check_california,
                ON_ESA: _check_ontario,
            }
            for regime in regimes:
                checks[regime](cfg, today, blockers, notes)

        evaluation = mode == RunMode.EVALUATION
        if evaluation:
            notes.extend(f"EVALUATION only - DRY_RUN/LIVE would be blocked: {b}"
                         for b in blockers)
        decision = PolicyDecision(allowed=evaluation or not blockers, regimes=regimes,
                                  blockers=blockers, notes=notes)
        if blockers:
            logger.info("policy for posting %s (%s): %d blocker(s)%s", posting.id, mode.value,
                        len(blockers), "" if decision.allowed else ", blocked")
        return decision

    # ------------------------------------------------------------------------- internals

    def _context(self, posting: Posting) -> _Context:
        places = [_analyze_location(loc) for loc in posting.locations if loc and loc.strip()]
        details = posting.categories.locationDetails
        structured = _place_from_details(details) if details is not None else None
        if structured is not None:
            places.append(structured)
        details_country = (details.country or "").strip() if details is not None else ""
        if posting.country and posting.country.strip():
            country: str | None = normalize_country(posting.country)
        elif details_country:
            country = normalize_country(details_country)
        elif places and all(place.names_us_state for place in places):
            country = "US"
        else:
            country = None
        remote = (posting.workplaceType or "").strip().lower() == "remote" or any(
            place.remote for place in places
        )
        return _Context(posting=posting, places=places, structured=structured, country=country,
                        remote=remote)

    def _regimes(self, ctx: _Context) -> list[str]:
        cfg = self.config or PolicyConfig()
        regions = ctx.regions
        found = {IL_HRA}
        for place in ctx.text_places:
            found.update(cfg.override_for(place.text) or ())
        if any(place.nyc for place in ctx.places) or (
            ctx.remote and cfg.nyc.remote_roles_associated_with_nyc_office
        ):
            found.add(NYC_LL144)
        if ("US", "CO") in regions or ctx.remote_us:
            found.add(CO_ADMT)
        if ("US", "CA") in regions or ctx.remote_us:
            found.add(CA_FEHA)
        if ("CA", "ON") in regions or ctx.country == "CA" or "CA" in ctx.location_countries:
            found.add(ON_ESA)
        return [regime for regime in REGIMES if regime in found]

    def _check_approval(self, cfg: PolicyConfig, today: date, blockers: list[str],
                        notes: list[str]) -> None:
        if not self.approved:
            blockers.append(
                f"Policy file {self.source} is not approved: Legal/HR must review it and set "
                "approved_by and approved_at."
            )
        elif cfg.approved_at is not None and cfg.approved_at > today:
            blockers.append(
                f"Policy file {self.source}: approved_at {cfg.approved_at} is in the future; "
                "set the date Legal/HR actually signed off."
            )
        else:
            notes.append(f"Policy file {self.source} v{cfg.version} approved by "
                         f"{cfg.approved_by} on {cfg.approved_at}.")

    def _check_country(self, ctx: _Context, cfg: PolicyConfig, blockers: list[str]) -> None:
        allowed = cfg.allowed_countries
        locations = ", ".join(repr(place.text) for place in ctx.places) or "none"
        if ctx.country is None:
            blockers.append(
                f"Country gate: posting {ctx.posting.id} has no country and not every "
                f"location ({locations}) names a US state; set the posting's country in "
                f"Lever (allowed: {', '.join(allowed) or 'none'})."
            )
        elif ctx.country not in allowed:
            blockers.append(
                f"Country gate: posting country {ctx.country} is not in allowed_countries "
                f"({', '.join(allowed) or 'none'}); rules there (EU AI Act, UK GDPR, India "
                f"DPDP, ...) have not been assessed. Legal must assess them before adding "
                f"{ctx.country} to allowed_countries."
            )
        already_reported = {ctx.country} if ctx.country else set()
        for place in ctx.places:
            foreign = sorted(place.countries - set(allowed) - already_reported)
            if foreign:
                blockers.append(
                    f"Country gate: location {place.text!r} is in {', '.join(foreign)}, "
                    f"which is not in allowed_countries ({', '.join(allowed) or 'none'}); "
                    "remove the location or have Legal assess and allow it."
                )

    def _check_locations(self, ctx: _Context, cfg: PolicyConfig, blockers: list[str]) -> None:
        """US postings only (the US regimes are state- and city-level): a location tag that
        is neither remote nor mapped to a state, a city or a non-US country must be covered
        by the structured address (primary location only) or by ``location_overrides``."""
        if ctx.country != "US":
            return
        primary = (ctx.posting.categories.location or "").strip()
        structured_ok = ctx.structured is not None and bool(ctx.structured.regions)
        for place in ctx.text_places:
            if place.remote or place.regions or place.countries - {"US"}:
                continue
            if cfg.override_for(place.text) is not None:
                continue
            if structured_ok and place.text.strip() == primary:
                continue
            blockers.append(
                f"Location {place.text!r} could not be mapped to a US state or city, so the "
                "NYC, Colorado and California rules cannot be checked; set the posting "
                "location's structured address in Lever (primary location only), name the "
                "state in the location tag, or list the label under location_overrides in "
                f"{self.source} with the regimes that apply."
            )


def _check_illinois(cfg: PolicyConfig, today: date, blockers: list[str],
                    notes: list[str]) -> None:
    il = cfg.illinois
    if not il.notice_url or not il.notice_url.strip():
        blockers.append("IL HRA: illinois.notice_url is not set; publish the AI-use notice "
                        "for applicants and set its URL.")
    elif not _is_http_url(il.notice_url):
        blockers.append(f"IL HRA: illinois.notice_url {il.notice_url!r} is not an http(s) "
                        "URL; set the URL of the published AI-use notice.")
    if not il.posting_notice_confirmed:
        blockers.append("IL HRA: illinois.posting_notice_confirmed is false; add the AI-use "
                        "notice to the Lever job posting and confirm it.")
    notes.append("IL HRA (HB 3773, in force 2026-01-01): AI use must not have a "
                 "discriminatory effect and zip codes must not be used as a proxy; IDHR "
                 "notice rules are not final.")


def _check_nyc(cfg: PolicyConfig, today: date, blockers: list[str], notes: list[str]) -> None:
    nyc = cfg.nyc
    audit = nyc.bias_audit_date
    if audit is None:
        blockers.append("NYC LL144: nyc.bias_audit_date is not set; commission an "
                        "independent bias audit (needed within the 365 days before use) and "
                        "set its date.")
    elif audit > today:
        blockers.append(f"NYC LL144: bias audit dated {audit} is in the future; set the date "
                        "the audit was completed.")
    else:
        age = (today - audit).days
        if age > NYC_AUDIT_MAX_AGE_DAYS:
            blockers.append(f"NYC LL144: bias audit dated {audit} is {age} days old (limit "
                            f"{NYC_AUDIT_MAX_AGE_DAYS}); commission a new audit and update "
                            "nyc.bias_audit_date.")
        else:
            expires = audit + timedelta(days=NYC_AUDIT_MAX_AGE_DAYS)
            notes.append(f"NYC LL144: bias audit dated {audit} is valid through {expires}.")
    if not _is_http_url(nyc.audit_summary_url):
        blockers.append("NYC LL144: nyc.audit_summary_url is not set to an http(s) URL; "
                        "publish the bias-audit summary and set its URL.")
    published = nyc.notice_published_at
    if published is None:
        blockers.append(f"NYC LL144: nyc.notice_published_at is not set; publish the "
                        f"candidate notice at least {NYC_NOTICE_BUSINESS_DAYS} business days "
                        "before use.")
    else:
        earliest = add_business_days(published, NYC_NOTICE_BUSINESS_DAYS, nyc.holidays)
        if earliest > today:
            blockers.append(f"NYC LL144: notice published {published} + "
                            f"{NYC_NOTICE_BUSINESS_DAYS} business days = {earliest} is after "
                            f"today ({today}); DRY_RUN/LIVE runs can start on {earliest}.")
    notes.append("NYC LL144: keep the audit summary published for at least 6 months after "
                 "the tool's last use; answer data requests within 30 days.")


def _check_colorado(cfg: PolicyConfig, today: date, blockers: list[str],
                    notes: list[str]) -> None:
    ready = cfg.colorado.adverse_outcome_workflow_ready
    if today < CO_ADMT_EFFECTIVE:
        state = ("is confirmed ready" if ready else
                 "must be confirmed (colorado.adverse_outcome_workflow_ready) before then, "
                 "or DRY_RUN/LIVE runs will be blocked")
        notes.append(f"CO ADMT (SB 26-189) applies to decisions on/after "
                     f"{CO_ADMT_EFFECTIVE}; the adverse-outcome workflow {state}.")
    elif not ready:
        blockers.append(f"CO ADMT: colorado.adverse_outcome_workflow_ready is false; decisions "
                        f"on/after {CO_ADMT_EFFECTIVE} need the adverse-outcome disclosure "
                        "(within 30 days) and human-review workflow; set it once both are in "
                        "place.")
    else:
        notes.append("CO ADMT: adverse-outcome workflow confirmed; send the disclosure "
                     "within 30 days of an adverse outcome.")


def _check_california(cfg: PolicyConfig, today: date, blockers: list[str],
                      notes: list[str]) -> None:
    ca = cfg.california
    notes.append("CA FEHA (ADS regulations, in force 2025-10-01): keep ADS data and selection "
                 "criteria for 4 years (retention_days >= 1461).")
    if today < CCPA_ADMT_EFFECTIVE:
        if ca.ccpa_admt_applies is None:
            state = ("Legal must decide california.ccpa_admt_applies before then, or "
                     "DRY_RUN/LIVE runs will be blocked")
        elif ca.ccpa_admt_applies and not ca.ccpa_notice_ready:
            state = ("california.ccpa_notice_ready must be true before then, or DRY_RUN/LIVE "
                     "runs will be blocked")
        elif ca.ccpa_admt_applies:
            state = "the pre-use notice is confirmed ready"
        else:
            state = "Legal decided the ADMT article does not apply"
        notes.append(f"CA CCPA ADMT (from {CCPA_ADMT_EFFECTIVE}): {state}.")
    elif ca.ccpa_admt_applies is None:
        blockers.append(f"CA CCPA ADMT: california.ccpa_admt_applies is undecided; since "
                        f"{CCPA_ADMT_EFFECTIVE} Legal must decide whether the ADMT article "
                        "applies.")
    elif ca.ccpa_admt_applies and not ca.ccpa_notice_ready:
        blockers.append("CA CCPA ADMT: california.ccpa_admt_applies is true but "
                        "california.ccpa_notice_ready is false; publish the pre-use notice and "
                        "appeal path, then confirm it.")
    elif not ca.ccpa_admt_applies:
        notes.append("CA CCPA ADMT: Legal decided the ADMT article does not apply.")


def _check_ontario(cfg: PolicyConfig, today: date, blockers: list[str],
                   notes: list[str]) -> None:
    if not cfg.ontario.posting_disclosure_confirmed:
        blockers.append("ON ESA: ontario.posting_disclosure_confirmed is false; the Lever "
                        "posting must state that AI is used to screen, assess or select "
                        "applicants.")
    notes.append("ON ESA: keep copies of the posting and application form (reportedly 3 "
                 "years; unverified).")


def add_business_days(start: date, days: int, holidays: list[date] | None = None) -> date:
    """The date ``days`` business days after ``start`` (before it when negative), skipping
    Saturdays, Sundays and ``holidays``. ``start`` itself is never counted."""
    skip = set(holidays or ())
    step = 1 if days >= 0 else -1
    current = start
    remaining = abs(days)
    while remaining:
        current += timedelta(days=step)
        if current.weekday() < 5 and current not in skip:
            remaining -= 1
    return current
