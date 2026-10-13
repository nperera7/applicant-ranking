"""Per-role criteria files (YAML, one role per file) and their repository.

Example file: criteria/example-senior-data-engineer.yaml. A posting is matched to criteria
through ``posting_ids``; a posting id listed in two files is a load error (ambiguous).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml
from lxml import html as lxml_html
from pydantic import ValidationError

from applicant_ranking.domain import Importance, Requirement, RoleCriteria
from applicant_ranking.lever.models import Posting

logger = logging.getLogger(__name__)

MAX_CRITERIA_FILE_BYTES = 256 * 1024
MAX_ID_CHARS = 40
MAX_REQUIREMENTS = 25
MAX_DESCRIPTION_CHARS = 500
CRITERIA_SUFFIXES = frozenset({".yaml", ".yml"})

FIELD_GUIDE = """\
Per-role criteria for Applicant Ranking (advisory only: it ranks, never rejects).
Fields:
  role_id       lowercase slug, unique across files (a-z, 0-9, - and _; max 40 chars)
  title         role title shown in reports
  version       bump by 1 whenever anything below changes (results record it)
  posting_ids   Lever posting ids this file applies to (each id in one file only)
  summary       2-3 sentences on what the role does (sent to the model)
  requirements  what to look for; each has id, description (specific and observable),
                importance (must_have | nice_to_have) and optional weight (0.5-10;
                defaults: must_have 3, nice_to_have 1)
  knockouts     hard requirements a human must confirm (never applied automatically);
                each needs a justification of why it is job-related. Work
                authorisation, visa and citizenship are NOT allowed here: use a
                structured Lever application question.
  guardrails    extra evaluation rules for the model (job-related only)
  source        hiring_manager | job_description (an uncurated draft)
  approved_terms
                domain vocabulary the linter would otherwise reject ("credit
                history" for a lender's risk role); each entry has term,
                justification, approved_by and approved_at, recorded by Legal/HR.
                Usually empty.
  approved_by / approved_at
                filled in by TA/HR on approval; unapproved criteria only run in
                evaluation mode
Never mention protected characteristics or proxies (age, graduation years, "native
speaker", "culture fit", "local candidates", gaps, maximum years of experience...).
Check with: applicant-ranking criteria lint"""


class CriteriaError(ValueError):
    pass


# ----------------------------------------------------------------------------- load/dump


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors(include_url=False, include_input=False):
        loc = ".".join(str(p) for p in err["loc"]) or "(top level)"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def _coerce_approved_at(data: dict[str, Any]) -> None:
    """YAML gives a ``date`` for ``2026-10-01`` and a naive datetime without an offset;
    both mean UTC here. The same reading applies to ``approved_terms[].approved_at``
    (where a plain date stays a date)."""
    value = data.get("approved_at")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            data["approved_at"] = value.replace(tzinfo=UTC)
    elif isinstance(value, date):
        data["approved_at"] = datetime(value.year, value.month, value.day, tzinfo=UTC)
    terms = data.get("approved_terms")
    for entry in terms if isinstance(terms, list) else ():
        if not isinstance(entry, dict):
            continue
        when = entry.get("approved_at")
        if isinstance(when, datetime) and when.tzinfo is None:
            entry["approved_at"] = when.replace(tzinfo=UTC)


def parse_criteria_yaml(text: str, *, source: str = "<string>") -> RoleCriteria:
    """Parse criteria YAML text; errors are CriteriaError naming ``source`` and the field."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark else ""
        problem = getattr(exc, "problem", None) or "syntax error"
        raise CriteriaError(f"{source}: invalid YAML{where}: {problem}") from None
    if data is None:
        raise CriteriaError(f"{source}: file is empty")
    if not isinstance(data, dict):
        raise CriteriaError(f"{source}: expected a mapping of fields at the top level")
    _coerce_approved_at(data)
    try:
        return RoleCriteria.model_validate(data)
    except ValidationError as exc:
        raise CriteriaError(f"{source}: {_format_validation_error(exc)}") from None


def load_criteria_file(path: Path) -> RoleCriteria:
    """yaml.safe_load + RoleCriteria.model_validate; errors re-raised as CriteriaError
    naming the file and the field."""
    try:
        size = path.stat().st_size
        if size > MAX_CRITERIA_FILE_BYTES:
            raise CriteriaError(f"{path}: file is too large ({size} bytes; "
                                f"max {MAX_CRITERIA_FILE_BYTES})")
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CriteriaError(f"{path}: cannot read file: {exc.strerror or exc}") from None
    except UnicodeDecodeError:
        raise CriteriaError(f"{path}: file is not valid UTF-8") from None
    return parse_criteria_yaml(text, source=str(path))


class _Dumper(yaml.SafeDumper):
    """Block style with indented list items and literal blocks for multi-line text."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


def _represent_datetime(dumper: yaml.SafeDumper, value: datetime) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:timestamp", value.isoformat())


_Dumper.add_representer(str, _represent_str)
_Dumper.add_representer(datetime, _represent_datetime)


def dump_criteria_yaml(criteria: RoleCriteria, *, header: str | None = None) -> str:
    """Readable YAML (key order as in the model, block style, unicode kept) that
    round-trips through load_criteria_file. Omits None-valued optional keys.

    ``header`` (e.g. ``FIELD_GUIDE``) is written first as ``#`` comment lines."""
    data = criteria.model_dump(mode="python", exclude_none=True)
    body = yaml.dump(data, Dumper=_Dumper, sort_keys=False, default_flow_style=False,
                     allow_unicode=True, width=100)
    if not header:
        return body
    comment = "\n".join(f"# {line}".rstrip() for line in header.splitlines())
    return f"{comment}\n\n{body}"


# ----------------------------------------------------------------------------- repository


class CriteriaRepository:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._items: list[RoleCriteria] = []
        self._by_role: dict[str, RoleCriteria] = {}
        self._by_posting: dict[str, RoleCriteria] = {}
        self._paths: dict[str, Path] = {}
        self._errors: list[str] = []

    def _files(self) -> list[Path]:
        return sorted(
            p for p in self.directory.iterdir()
            if p.suffix.lower() in CRITERIA_SUFFIXES and p.is_file()
            and not p.name.startswith(".")
        )

    def load(self) -> None:
        """(Re)load every *.yaml / *.yml in the directory (non-recursive, sorted). Bad files
        are recorded in ``errors`` and skipped; duplicate role_ids or posting ids across
        files are errors (all files involved are skipped). Missing directory = empty."""
        errors: list[str] = []
        parsed: list[tuple[Path, RoleCriteria]] = []
        if self.directory.exists() and not self.directory.is_dir():
            errors.append(f"{self.directory}: not a directory")
        elif self.directory.is_dir():
            try:
                files = self._files()
            except OSError as exc:
                files = []
                errors.append(f"{self.directory}: cannot list directory: {exc.strerror}")
            for path in files:
                try:
                    parsed.append((path, load_criteria_file(path)))
                except CriteriaError as exc:
                    errors.append(str(exc))

        by_role: dict[str, list[Path]] = defaultdict(list)
        by_posting: dict[str, list[Path]] = defaultdict(list)
        for path, criteria in parsed:
            by_role[criteria.role_id].append(path)
            for posting_id in dict.fromkeys(criteria.posting_ids):
                by_posting[posting_id].append(path)
        rejected: set[Path] = set()
        for role_id, paths in by_role.items():
            if len(paths) > 1:
                rejected.update(paths)
                errors.append(f"role_id {role_id!r} is defined in more than one file: "
                              f"{', '.join(p.name for p in paths)} (all skipped)")
        for posting_id, paths in by_posting.items():
            if len(paths) > 1:
                rejected.update(paths)
                errors.append(f"posting id {posting_id!r} is listed in more than one file: "
                              f"{', '.join(p.name for p in paths)} (all skipped)")

        items = [(path, c) for path, c in parsed if path not in rejected]
        self._items = [c for _, c in items]
        self._by_role = {c.role_id: c for c in self._items}
        self._by_posting = {pid: c for c in self._items for pid in c.posting_ids}
        self._paths = {c.role_id: path for path, c in items}
        self._errors = errors
        for error in errors:
            logger.warning("criteria: %s", error)
        logger.info("loaded %d criteria file(s) from %s", len(self._items), self.directory)

    @property
    def errors(self) -> list[str]:
        return list(self._errors)

    def all(self) -> list[RoleCriteria]:
        return list(self._items)

    def for_posting(self, posting_id: str) -> RoleCriteria | None:
        return self._by_posting.get(posting_id)

    def by_role(self, role_id: str) -> RoleCriteria | None:
        return self._by_role.get(role_id)

    def path_for(self, role_id: str) -> Path | None:
        """The file a loaded role came from (for lint output and error messages)."""
        return self._paths.get(role_id)


# ----------------------------------------------------------------------------- drafting

_NICE_RE = re.compile(
    r"\b(?:prefer\w*|nice|bonus|plus(?:es)?|desir\w*|optional|ideally|"
    r"good to have|great to have|extra credit|would be (?:a )?(?:great|nice|good))\b",
    re.IGNORECASE)
_SKIP_RE = re.compile(
    r"\b(?:responsibilit\w*|duties|what you(?:'ll| will) do|day[\s-]to[\s-]day|"
    r"benefits?|perks|about (?:us|the (?:company|team))|who we are|our (?:team|company|"
    r"culture|values)|compensation|salary|why (?:join|work)|equal opportunity)\b",
    re.IGNORECASE)
_MUST_RE = re.compile(
    r"\b(?:requirements?|required|qualifications?|must(?:[\s-]haves?)?|"
    r"what you(?:'ll| will)? need|what you bring|what we(?:'re| are) looking for|"
    r"who you are|about you|you have|ideal candidate|skills|experience)\b",
    re.IGNORECASE)
_BLOCK_TAGS = frozenset({"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5",
                         "h6", "tr", "section", "article", "blockquote"})
_BULLET_CHARS = " \t-*\u2022\u00b7\u25aa\u25cf\u2013\u2014>"


def _fold_quotes(text: str) -> str:
    return text.replace("\u2019", "'").replace("\u2018", "'")


def classify_list_title(title: str) -> Importance | None:
    """must_have / nice_to_have for a Lever ``content.lists`` title, None to skip it."""
    text = _fold_quotes(title)
    if _NICE_RE.search(text):
        return "nice_to_have"
    if _SKIP_RE.search(text):
        return None
    if _MUST_RE.search(text):
        return "must_have"
    return None


def _collapse(text: str) -> str:
    return " ".join(text.split())


def slugify(text: str, *, max_len: int = MAX_ID_CHARS) -> str:
    """ASCII slug matching domain.SLUG_PATTERN when non-empty (may be ""): lowercase,
    runs of other characters become "-", cut at a hyphen when possible."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    if len(slug) > max_len:
        cut = slug[:max_len]
        hyphen = cut.rfind("-")
        at_boundary = slug[max_len] == "-"
        slug = cut[:hyphen] if not at_boundary and hyphen >= max_len // 2 else cut
    return slug.strip("-")


def _unique_slug(base: str, used: set[str], *, fallback: str) -> str:
    base = base or fallback
    candidate = base
    n = 2
    while candidate in used:
        suffix = f"-{n}"
        candidate = base[:MAX_ID_CHARS - len(suffix)].rstrip("-") + suffix
        n += 1
    used.add(candidate)
    return candidate


def html_to_text(fragment: str) -> str:
    """Plain text of an HTML fragment, with line breaks between block elements."""
    if not fragment or not fragment.strip():
        return ""
    try:
        root = lxml_html.fragment_fromstring(fragment, create_parent="div")
    except Exception:  # lxml raises several types on garbage; fall back to tag stripping
        return _collapse(re.sub(r"<[^>]+>", " ", fragment))
    for element in root.iter():
        if isinstance(element.tag, str) and element.tag.lower() in _BLOCK_TAGS:
            element.tail = "\n" + (element.tail or "")
    return root.text_content()


def _li_items(fragment: str) -> list[str]:
    if not fragment or "<" not in fragment:
        return []
    try:
        root = lxml_html.fragment_fromstring(fragment, create_parent="div")
    except Exception:  # unparseable list content -> no items
        return []
    items: list[str] = []
    for li in root.iter("li"):
        parts = [li.text or ""]
        for child in li:
            if isinstance(child.tag, str) and child.tag.lower() not in ("ul", "ol"):
                parts.append(child.text_content())
            parts.append(child.tail or "")
        text = _collapse(" ".join(parts)).lstrip(_BULLET_CHARS).strip()
        if text:
            items.append(text)
    return items


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit - 1]
    space = cut.rfind(" ")
    if space >= int(limit * 0.6):
        cut = cut[:space]
    return cut.rstrip(" ,;:-") + "\u2026"


def _description_text(posting: Posting) -> str:
    plain = posting.content.description
    if plain.strip():
        # Lever's documented example carries a literal backslash-n; treat it as a break.
        return _collapse(plain.replace("\\n", "\n"))
    return _collapse(html_to_text(posting.content.descriptionHtml))


def draft_from_posting(posting: Posting) -> RoleCriteria:
    """Skeleton criteria from a Lever posting for the hiring manager to edit (no LLM: a
    model has no source for what the posting leaves out). Each ``content.lists`` entry's
    <li> items become requirements: lists titled like requirements/qualifications/
    must/what you'll need -> must_have; preferred/nice/bonus/plus -> nice_to_have; other
    lists (benefits, about us, responsibilities) are skipped. HTML stripped, whitespace
    collapsed, ids slugified and de-duplicated (max 40 chars), at most 25 requirements.
    If no list qualifies, one requirement holds the plain-text description (trimmed to
    500 chars). source="job_description", version=1, role_id slugified from the title,
    posting_ids=[posting.id], unapproved, no knockouts.

    Must-haves are listed before nice-to-haves (each in posting order) so the 25-item cap
    never drops a must-have in favour of a nice-to-have."""
    title = _collapse(posting.text) or f"Posting {posting.id}"
    collected: dict[Importance, list[str]] = {"must_have": [], "nice_to_have": []}
    seen: set[str] = set()
    for posting_list in posting.content.lists:
        importance = classify_list_title(_collapse(html_to_text(posting_list.text)))
        if importance is None:
            continue
        for item in _li_items(posting_list.content):
            key = item.casefold()
            if len(item) < 3 or key in seen:
                continue
            seen.add(key)
            collected[importance].append(_trim(item, MAX_DESCRIPTION_CHARS))

    used: set[str] = set()
    requirements: list[Requirement] = []
    for importance in ("must_have", "nice_to_have"):
        for description in collected[importance]:
            if len(requirements) >= MAX_REQUIREMENTS:
                break
            req_id = _unique_slug(slugify(description), used, fallback="requirement")
            requirements.append(Requirement(id=req_id, description=description,
                                            importance=importance))
    if not requirements:
        description = _trim(_description_text(posting), MAX_DESCRIPTION_CHARS)
        if len(description) < 3:
            description = f"Replace with the requirements for {title}"[:MAX_DESCRIPTION_CHARS]
        requirements.append(Requirement(id="job-description", description=description,
                                        importance="must_have"))

    role_id = slugify(title) or slugify(f"posting-{posting.id}") or "role"
    return RoleCriteria(
        role_id=role_id,
        title=title[:200],
        version=1,
        posting_ids=[posting.id],
        requirements=requirements,
        source="job_description",
    )
