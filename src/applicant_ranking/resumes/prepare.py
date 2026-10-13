"""Turn a resume (file bytes, or text from Lever parsedData / a vision transcript) into a
``PreparedResume``: extract -> normalize -> security scan -> redact -> fingerprints.

Flag rules applied here (in addition to flags from extraction and the security scan):
- ``truncated`` (info) when normalize truncated the text.
- ``resume_from_parsed_data`` (info) for the parsedData fallback.
- ``no_resume`` (review) when no usable text and no page images remain; ``error`` set.
- ``jd_copy`` when the resume repeats the posting's 8-word phrases: measured against
  ``job_description_lists`` (the posting's requirement/responsibility items) when that
  text has at least ``JD_LISTS_MIN_SHINGLES`` 8-grams, else against the whole
  ``job_description`` (whose "about us" and benefits boilerplate would otherwise dilute
  the share). Flagged when the share is >= ``JD_COPY_THRESHOLD`` AND at least
  ``JD_COPY_MIN_SHINGLES`` phrases match (about three bullets): "review" from
  ``JD_COPY_REVIEW_SHARE`` or ``JD_COPY_REVIEW_SHINGLES`` matches, "info" below.
HMACs use ``hmac_key`` (``Settings.require_hmac_key()``): ``file_hmac`` over the original
bytes, ``redacted_text_hmac`` over the redacted text. ``fingerprint`` is
``signals.fingerprint(redacted_text)``.
The security scan runs on the normalised visible text, the (normalised) hidden snippets
and the tag payload; the resume text passed on is the normalised, REDACTED visible text.

Details:
- "Usable" text has at least ``MIN_USABLE_WORDS`` words left besides placeholders; a
  resume that redacts down to "[CANDIDATE] [EMAIL]" is ``no_resume``, with empty
  ``redacted_text`` and no fingerprint (so empty resumes never look like duplicates).
- An extraction error is ``no_resume`` too (the extraction flag says why).
- Vision path: whatever text the file had (a footer, a hidden layer) is still normalised
  and scanned, so hidden-text injections are flagged before transcription.
- ``jd_overlap`` is measured on the normalised visible text before redaction.
- After redaction, ``redact.scrub_identity`` re-checks that no name token or extra
  identifier survived; a hit is replaced and logged (count only, never the text).
- ``prepare_text`` takes optional ``kind``, ``file_name`` and ``hidden_snippets`` so the
  pipeline can prepare a vision transcript under the original file's kind.
- ``redaction_version`` is REDACTION_VERSION plus the education-year and length settings
  (``ResumePreparer.redaction_version``), e.g. "redact-v1+eduyears=1+max=30000".
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
from collections.abc import Sequence
from typing import get_args

from applicant_ranking.config import Settings
from applicant_ranking.domain import (
    ExtractionResult,
    FileKind,
    Flag,
    FlagCode,
    NormalizedText,
    PreparedResume,
)
from applicant_ranking.lever.models import ParsedData, ParsedDate
from applicant_ranking.resumes.extract import ResumeExtractor
from applicant_ranking.resumes.normalize import MIN_MAX_CHARS, normalize_text
from applicant_ranking.resumes.redact import REDACTION_VERSION, redact, scrub_identity
from applicant_ranking.resumes.security import scan_text
from applicant_ranking.resumes.signals import fingerprint, jd_overlap_counts, shingles

logger = logging.getLogger(__name__)

JD_COPY_THRESHOLD = 0.3
JD_COPY_MIN_SHINGLES = 20  # matched 8-grams: roughly three 11-word bullets
JD_COPY_REVIEW_SHARE = 0.5
JD_COPY_REVIEW_SHINGLES = 60
JD_LISTS_MIN_SHINGLES = 20  # below this the lists are too short to measure against
MIN_USABLE_WORDS = 3

_SOURCES: frozenset[str] = frozenset(get_args(PreparedResume.model_fields["source"].annotation))
_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z_ ]*\]")
_WORD_RE = re.compile(r"[^\W\d_]{2,}")


def _fmt_month(d: ParsedDate | None) -> str | None:
    if d is None or not d.year:
        return None
    if d.month and 1 <= d.month <= 12:
        return f"{d.year:04d}-{d.month:02d}"
    return f"{d.year:04d}"


def _clean(value: str | None) -> str:
    return " ".join(value.split()) if value else ""


def parsed_data_to_text(parsed: ParsedData) -> str:
    """Render Lever's parsedData (positions, schools) as plain resume-like text:
    "EXPERIENCE" then "Title, Org (YYYY-MM - YYYY-MM|present)" + summary per position;
    "EDUCATION" then "Degree in Field, Org (years)" per school. Empty string if nothing."""
    sections: list[str] = []

    experience: list[str] = []
    for p in parsed.positions:
        head = ", ".join(x for x in (_clean(p.title), _clean(p.org)) if x)
        start, end = _fmt_month(p.start), _fmt_month(p.end)
        summary = (p.summary or "").strip()
        if not (head or summary):
            continue
        if start or end:
            head = f"{head} ({start or '?'} - {end or 'present'})".strip()
        experience.append("\n".join(x for x in (head, summary) if x))
    if experience:
        sections.append("EXPERIENCE\n" + "\n\n".join(experience))

    education: list[str] = []
    for s in parsed.schools:
        degree = " in ".join(x for x in (_clean(s.degree), _clean(s.field)) if x)
        head = ", ".join(x for x in (degree, _clean(s.org)) if x)
        summary = (s.summary or "").strip()
        if not (head or summary):
            continue
        years = list(dict.fromkeys(d.year for d in (s.start, s.end) if d is not None and d.year))
        if years:
            head = f"{head} ({' - '.join(str(y) for y in years)})".strip()
        education.append("\n".join(x for x in (head, summary) if x))
    if education:
        sections.append("EDUCATION\n" + "\n\n".join(education))

    return "\n\n".join(sections)


def _usable(redacted_text: str) -> bool:
    return len(_WORD_RE.findall(_PLACEHOLDER_RE.sub(" ", redacted_text))) >= MIN_USABLE_WORDS


def _check_source(source: str) -> str:
    if source not in _SOURCES:
        raise ValueError(f"source must be one of {sorted(_SOURCES)}, not {source!r}")
    return source


def _jd_copy_flag(resume_text: str, job_description: str, job_description_lists: str
                  ) -> Flag | None:
    """The ``jd_copy`` flag (see the module docstring), or None."""
    reference, label = job_description, "job description"
    if len(shingles(job_description_lists, 8)) >= JD_LISTS_MIN_SHINGLES:
        reference, label = job_description_lists, "job description's requirement lists"
    if not reference.strip():
        return None
    matched, total = jd_overlap_counts(resume_text, reference)
    if not total:
        return None
    share = matched / total
    if share < JD_COPY_THRESHOLD or matched < JD_COPY_MIN_SHINGLES:
        return None
    review = share >= JD_COPY_REVIEW_SHARE or matched >= JD_COPY_REVIEW_SHINGLES
    return Flag(code=FlagCode.JD_COPY, severity="review" if review else "info",
                detail=f"{share:.0%} of the {label} 8-word phrases ({matched} of {total}) "
                       "appear verbatim in the resume")


class ResumePreparer:
    def __init__(self, *, extractor: ResumeExtractor, hmac_key: bytes, max_chars: int,
                 redact_education_years: bool) -> None:
        if not hmac_key:
            raise ValueError("hmac_key must not be empty")
        if max_chars < MIN_MAX_CHARS:
            raise ValueError(f"max_chars must be >= {MIN_MAX_CHARS}")
        self._extractor = extractor
        self._hmac_key = bytes(hmac_key)
        self._max_chars = max_chars
        self._redact_education_years = redact_education_years

    @property
    def redaction_version(self) -> str:
        """``PreparedResume.redaction_version``: REDACTION_VERSION plus the preparation
        settings that change the text sent (education-year redaction, the length limit).
        Part of the evaluation cache key, which for a scanned resume is checked against
        the file HMAC before its text exists."""
        return self._versioned(REDACTION_VERSION)

    def _versioned(self, version: str) -> str:
        return f"{version}+eduyears={int(self._redact_education_years)}+max={self._max_chars}"

    @classmethod
    def from_settings(cls, settings: Settings, extractor: ResumeExtractor) -> ResumePreparer:
        return cls(extractor=extractor, hmac_key=settings.require_hmac_key(),
                   max_chars=settings.max_resume_chars,
                   redact_education_years=settings.redact_education_years)

    def _hmac(self, data: bytes) -> str:
        return hmac.new(self._hmac_key, data, hashlib.sha256).hexdigest()

    async def prepare_file(self, data: bytes, *, opportunity_id: str,
                           resume_id: str | None = None, file_name: str | None = None,
                           candidate_names: Sequence[str] = (),
                           extra_identifiers: Sequence[str] = (),
                           job_description: str = "",
                           source: str = "file",
                           job_description_lists: str = "") -> PreparedResume:
        """If extraction says needs_vision, return with ``needs_vision=True`` and
        ``page_images`` set (no text yet); the pipeline transcribes them and calls
        ``prepare_text`` with the transcript, then merges flags."""
        _check_source(source)
        file_hmac = self._hmac(data)
        try:
            extraction = await self._extractor.extract(data)
        except Exception as exc:  # the extractor promises not to raise; contain it if it does
            logger.warning("resume extraction raised %s for opportunity %s",
                           type(exc).__name__, opportunity_id)
            extraction = ExtractionResult(
                kind="unknown", error="extraction failed",
                flags=[Flag(code=FlagCode.UNREADABLE_FILE, severity="review",
                            detail="the file could not be processed")],
            )
        base = PreparedResume(opportunity_id=opportunity_id, resume_id=resume_id,
                              file_name=file_name, file_hmac=file_hmac, source=source,
                              kind=extraction.kind,
                              redaction_version=self.redaction_version)
        flags = list(extraction.flags)

        if extraction.error:
            return self._no_resume(base, flags, extraction.error)
        if extraction.needs_vision:
            normalized = normalize_text(extraction.visible_text, max_chars=self._max_chars)
            flags += self._scan(normalized, extraction.hidden_snippets)
            if not extraction.page_images:
                return self._no_resume(base, flags, "no page images to transcribe")
            logger.debug("resume for opportunity %s needs vision (%d page images)",
                         opportunity_id, len(extraction.page_images))
            return base.model_copy(update={"needs_vision": True,
                                           "page_images": list(extraction.page_images),
                                           "flags": flags})
        return self._prepare(extraction.visible_text, base=base, flags=flags,
                             hidden_snippets=extraction.hidden_snippets,
                             candidate_names=candidate_names,
                             extra_identifiers=extra_identifiers,
                             job_description=job_description,
                             job_description_lists=job_description_lists)

    def prepare_text(self, text: str, *, opportunity_id: str, resume_id: str | None = None,
                     candidate_names: Sequence[str] = (),
                     extra_identifiers: Sequence[str] = (),
                     job_description: str = "", source: str = "parsed_data",
                     file_hmac: str | None = None, kind: FileKind = "unknown",
                     file_name: str | None = None,
                     hidden_snippets: Sequence[str] = (),
                     job_description_lists: str = "") -> PreparedResume:
        _check_source(source)
        base = PreparedResume(opportunity_id=opportunity_id, resume_id=resume_id,
                              file_name=file_name, file_hmac=file_hmac, source=source,
                              kind=kind, redaction_version=self.redaction_version)
        return self._prepare(text, base=base, flags=[], hidden_snippets=hidden_snippets,
                             candidate_names=candidate_names,
                             extra_identifiers=extra_identifiers,
                             job_description=job_description,
                             job_description_lists=job_description_lists)

    # -- internals ---------------------------------------------------------------------

    def _scan(self, normalized: NormalizedText, hidden_snippets: Sequence[str]) -> list[Flag]:
        hidden = normalize_text("\n".join(hidden_snippets), max_chars=self._max_chars)
        payload = " ".join(p for p in (normalized.tag_payload, hidden.tag_payload) if p)
        return scan_text(normalized.text, hidden_text=hidden.text, tag_payload=payload,
                         removed_counts=normalized.removed).flags

    def _no_resume(self, base: PreparedResume, flags: list[Flag], reason: str) -> PreparedResume:
        flags = [*flags, Flag(code=FlagCode.NO_RESUME, severity="review", detail=reason)]
        logger.info("no usable resume for opportunity %s", base.opportunity_id)
        return base.model_copy(update={"flags": flags, "error": reason})

    def _prepare(self, text: str, *, base: PreparedResume, flags: list[Flag],
                 hidden_snippets: Sequence[str], candidate_names: Sequence[str],
                 extra_identifiers: Sequence[str], job_description: str,
                 job_description_lists: str = "") -> PreparedResume:
        normalized = normalize_text(text, max_chars=self._max_chars)
        flags = [*flags, *self._scan(normalized, hidden_snippets)]
        if normalized.truncated:
            flags.append(Flag(code=FlagCode.TRUNCATED, severity="info",
                              detail=f"longer than {self._max_chars:,} characters after "
                                     "normalisation; the middle was omitted"))
        if base.source == "parsed_data":
            flags.append(Flag(code=FlagCode.RESUME_FROM_PARSED_DATA, severity="info",
                              detail="text rebuilt from Lever's parsed resume data"))

        redaction = redact(normalized.text, candidate_names=candidate_names,
                           extra_identifiers=extra_identifiers,
                           redact_education_years=self._redact_education_years)
        redacted, leaked = scrub_identity(redaction.text, candidate_names=candidate_names,
                                          extra_identifiers=extra_identifiers)
        if leaked:
            logger.warning("redaction backstop replaced %d identity token(s) for opportunity %s",
                           leaked, base.opportunity_id)
        if not _usable(redacted):
            return self._no_resume(base, flags, "no usable resume text")

        jd_flag = _jd_copy_flag(normalized.text, job_description, job_description_lists)
        if jd_flag is not None:
            flags.append(jd_flag)

        logger.debug("prepared resume for opportunity %s: %d chars, %d flags, redactions %s",
                     base.opportunity_id, len(redacted), len(flags), redaction.counts)
        return base.model_copy(update={
            "redacted_text": redacted,
            "redacted_text_hmac": self._hmac(redacted.encode()),
            "fingerprint": fingerprint(redacted),
            "redaction_version": self._versioned(redaction.version),
            "flags": flags,
        })
