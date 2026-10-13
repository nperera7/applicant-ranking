"""End-to-end harness for the pipeline, wiring and fairness tests.

A fake Lever API (``FakeLever``) is served through ``httpx.MockTransport`` to a real
``LeverClient``. Everything else is real too: SqliteStore in tmp_path, the resume
preparer (extraction inline), the packaged lexicon, validation and write-back. Only the
model is scripted (``FakeEvaluator``): ``ScriptedResponder`` rates each requirement from
a keyword and quotes the redacted resume line that contains it, so evidence grounding
passes. Resumes are generated with reportlab/Pillow at test time; every name and contact
detail is fictional.

The fixtures here are not autouse and have names no other module in this directory uses,
so the write-back, report and backtest tests are unaffected.
"""

from __future__ import annotations

import inspect
import io
import json
import re
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from applicant_ranking.config import Settings
from applicant_ranking.criteria import CriteriaRepository, dump_criteria_yaml
from applicant_ranking.domain import RoleCriteria
from applicant_ranking.lever.client import LeverClient
from applicant_ranking.lever.ratelimit import AsyncTokenBucket
from applicant_ranking.llm.base import AssessmentRequest, LLMError
from applicant_ranking.llm.fake import FakeEvaluator
from applicant_ranking.llm.schema import LLMAssessment
from applicant_ranking.pipeline import Ranker
from applicant_ranking.policy import IllinoisPolicy, PolicyConfig, PolicyEngine
from applicant_ranking.resumes.extract import ResumeExtractor
from applicant_ranking.resumes.prepare import ResumePreparer
from applicant_ranking.scoring.lexicon import ProtectedTermLexicon
from applicant_ranking.store import SqliteStore
from applicant_ranking.writeback import LeverWriter, WritebackPlanner

NOW = datetime(2026, 10, 8, 15, 0, tzinfo=UTC)
POSTING_ID = "posting-1"
OTHER_POSTING_ID = "posting-2"

# ----------------------------------------------------------------------------- resumes

ALICE_LINES = [
    "Alice Novak",
    "alice.novak@example.com | (312) 555-0100 | Chicago, IL",
    "SUMMARY",
    "Senior data engineer with 8 years of experience.",
    "EXPERIENCE",
    "Lead Data Engineer, Acme Analytics (2019 - present)",
    "Built production data pipelines on GCP with BigQuery and Dataflow for retail clients.",
    "Wrote Python services and SQL transformations for client reporting.",
    "Ran client workshops as the consulting lead for three accounts.",
    "Operated Kafka and Pub/Sub streaming jobs in production.",
]
ALICE_OLD_LINES = [
    "Alice Novak",
    "EXPERIENCE",
    "Junior Analyst, Old Firm (2015 - 2019)",
    "Prepared weekly spreadsheets for the operations team.",
]
BOB_LINES = [
    "Bob Marsh",
    "bob.marsh@example.com | (312) 555-0101",
    "EXPERIENCE",
    "Data Engineer, Beta Corp (2018 - present)",
    "Maintained BigQuery tables and scheduled loads for the finance team.",
    "Wrote Python scripts to validate nightly loads.",
    "Documented data definitions for analysts.",
]
DANA_LINES = [
    "Dana Kerr",
    "dana.kerr@example.com | 312-555-0101",
    *BOB_LINES[2:],
]
ERIN_PARSED_DATA = {
    "positions": [{
        "org": "Gamma Retail", "title": "Analytics Engineer",
        "summary": "Built BigQuery models and Python pipelines; ran Kafka consumers for "
                   "clickstream data.",
        "start": {"year": 2020, "month": 3}, "end": None,
    }],
    "schools": [{"org": "State University", "degree": "B.S.", "field": "Statistics",
                 "end": {"year": 2016}}],
}
FRANK_SCAN_LINES = ["Frank Osei", "Reporting Analyst, Delta Inc", "Python and client workshops"]
FRANK_TRANSCRIPT = (
    "Frank Osei\nfrank.osei@example.com\nEXPERIENCE\n"
    "Reporting Analyst, Delta Inc (2020 - present)\n"
    "Automated finance reports in Python.\n"
    "Ran client workshops on reporting needs.\n"
)

# Marker text that survives redaction, used to tell candidates apart in model requests.
ALICE_MARKER = "Lead Data Engineer"
BOB_MARKER = "Beta Corp"
ERIN_MARKER = "Gamma Retail"
FRANK_MARKER = "Delta Inc"

# Expected scores with ScriptedResponder and the sample criteria (weights 3, 3, 1, 2).
EXPECTED_SCORES = {"opp-alice": 85, "opp-erin": 76, "opp-bob": 57, "opp-dana": 57,
                   "opp-frank": 38}


def text_pdf(lines: Sequence[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setFont("Helvetica", 11)
    y = letter[1] - 72
    for line in lines:
        c.drawString(72, y, line)
        y -= 16
    c.showPage()
    c.save()
    return buf.getvalue()


def scanned_pdf(lines: Sequence[str]) -> bytes:
    """An image-only page (no text layer), like a scanned resume."""
    image = Image.new("L", (1275, 1650), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    for i, line in enumerate(lines):
        draw.text((120, 120 + 70 * i), line, fill=0, font=font)
    png = io.BytesIO()
    image.save(png, format="PNG")
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawImage(ImageReader(io.BytesIO(png.getvalue())), 0, 0, *letter)
    c.showPage()
    c.save()
    return buf.getvalue()


# --------------------------------------------------------------------------- fake Lever


@dataclass
class Recorded:
    method: str
    path: str  # without the /v1 prefix
    params: dict[str, str]
    json: Any


@dataclass
class FakeLever:
    """An in-memory Lever account. Opportunities keep their insertion order; tags and
    notes change as the client writes. ``failures`` maps (method, path) to an HTTP
    status to answer instead."""

    postings: dict[str, dict[str, Any]] = field(default_factory=dict)
    opportunities: dict[str, dict[str, Any]] = field(default_factory=dict)
    resumes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    files: dict[tuple[str, str], bytes | int] = field(default_factory=dict)
    notes: dict[str, dict[str, str]] = field(default_factory=dict)
    # GET /applications/deleted rows (deleted applications) and GET /opportunities/deleted
    # rows (deleted candidates: {"id", "deletedAt", "deletedBy"}).
    deleted: list[dict[str, Any]] = field(default_factory=list)
    deleted_opportunities: list[dict[str, Any]] = field(default_factory=list)
    failures: dict[tuple[str, str], int] = field(default_factory=dict)
    requests: list[Recorded] = field(default_factory=list)
    _note_seq: int = 0

    # -- setup helpers ---------------------------------------------------------------

    def add_opportunity(self, opp_id: str, *, name: str | None, created: int,
                        emails: Sequence[str] = (), phones: Sequence[str] = (),
                        posting_id: str = POSTING_ID, archived: bool = False,
                        anonymized: bool = False, tags: Sequence[str] = ()) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": opp_id, "name": name, "emails": list(emails),
            "phones": [{"type": "mobile", "value": p} for p in phones],
            "tags": list(tags), "createdAt": created, "isAnonymized": anonymized,
            "archived": {"archivedAt": created + 1000, "reason": "reason-1"} if archived
            else None,
            "applications": [{"id": f"app-{opp_id}", "opportunityId": opp_id,
                              "posting": posting_id, "type": "posting",
                              "createdAt": created}],
            "urls": {"show": f"https://hire.lever.co/candidates/{opp_id}"},
        }
        self.opportunities[opp_id] = record
        self.resumes.setdefault(opp_id, [])
        return record

    def add_resume(self, opp_id: str, resume_id: str, content: bytes | int | None, *,
                   uploaded: int, status: str = "processed", name: str = "resume.pdf",
                   parsed: dict[str, Any] | None = None) -> None:
        self.resumes[opp_id].append({
            "id": resume_id, "createdAt": uploaded,
            "file": {"name": name, "ext": "pdf", "uploadedAt": uploaded, "status": status,
                     "size": len(content) if isinstance(content, bytes) else None},
            "parsedData": parsed or {"positions": [], "schools": []},
        })
        if content is not None:
            self.files[(opp_id, resume_id)] = content

    # -- inspection ----------------------------------------------------------------------

    @property
    def writes(self) -> list[Recorded]:
        return [r for r in self.requests if r.method != "GET"]

    def paths(self, method: str = "GET") -> list[str]:
        return [r.path for r in self.requests if r.method == method]

    # -- transport -----------------------------------------------------------------------

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1")
        body = json.loads(request.content) if request.content else None
        params = dict(request.url.params)
        self.requests.append(Recorded(request.method, path, params, body))
        status = self.failures.get((request.method, path))
        if status is not None:
            return httpx.Response(status, json={"code": "Failure", "message": "scripted"})
        route = self._route(request.method, path)
        if route is None:
            return httpx.Response(404, json={"code": "NotFound", "message": "no route"})
        handler, groups = route
        return handler(params, body, *groups)

    def _route(self, method: str, path: str) -> tuple[Callable[..., httpx.Response],
                                                      tuple[str, ...]] | None:
        table: list[tuple[str, str, Callable[..., httpx.Response]]] = [
            ("GET", r"/postings/([^/]+)", self._get_posting),
            ("GET", r"/opportunities", self._list_opportunities),
            ("GET", r"/opportunities/deleted", self._deleted_opportunities),
            ("GET", r"/opportunities/([^/]+)", self._get_opportunity),
            ("GET", r"/opportunities/([^/]+)/resumes", self._list_resumes),
            ("GET", r"/opportunities/([^/]+)/resumes/([^/]+)/download", self._download),
            ("POST", r"/opportunities/([^/]+)/addTags", self._add_tags),
            ("POST", r"/opportunities/([^/]+)/removeTags", self._remove_tags),
            ("POST", r"/opportunities/([^/]+)/notes", self._create_note),
            ("PUT", r"/opportunities/([^/]+)/notes/([^/]+)", self._update_note),
            ("GET", r"/applications/deleted", self._deleted),
        ]
        for verb, pattern, handler in table:
            match = re.fullmatch(pattern, path)
            if verb == method and match:
                return handler, match.groups()
        return None

    @staticmethod
    def _page(items: list[dict[str, Any]]) -> httpx.Response:
        return httpx.Response(200, json={"data": items, "hasNext": False})

    def _get_posting(self, params: dict[str, str], body: Any, posting_id: str) -> httpx.Response:
        if posting_id not in self.postings:
            return httpx.Response(404, json={"code": "NotFound", "message": "posting"})
        return httpx.Response(200, json={"data": self.postings[posting_id]})

    def _list_opportunities(self, params: dict[str, str], body: Any) -> httpx.Response:
        wanted = params.get("archived")
        items = []
        for opp in self.opportunities.values():
            if not any(app["posting"] == params.get("posting_id")
                       for app in opp["applications"]):
                continue
            archived = opp["archived"] is not None
            if wanted is not None and archived != (wanted == "true"):
                continue
            items.append(opp)
        return self._page(items)

    def _get_opportunity(self, params: dict[str, str], body: Any, opp_id: str) -> httpx.Response:
        if opp_id not in self.opportunities:
            return httpx.Response(404, json={"code": "NotFound", "message": "opportunity"})
        return httpx.Response(200, json={"data": self.opportunities[opp_id]})

    def _list_resumes(self, params: dict[str, str], body: Any, opp_id: str) -> httpx.Response:
        return self._page(self.resumes.get(opp_id, []))

    def _download(self, params: dict[str, str], body: Any, opp_id: str,
                  resume_id: str) -> httpx.Response:
        content = self.files.get((opp_id, resume_id))
        if content is None:
            return httpx.Response(404, json={"code": "NotFound", "message": "file"})
        if isinstance(content, int):
            return httpx.Response(content, json={"code": "Unprocessable", "message": "file"})
        return httpx.Response(200, content=content,
                              headers={"content-type": "application/pdf"})

    def _add_tags(self, params: dict[str, str], body: Any, opp_id: str) -> httpx.Response:
        opp = self.opportunities[opp_id]
        tags = list(opp.get("tags") or [])  # Lever may serve null for "no tags"
        tags.extend(t for t in body["tags"] if t not in tags)
        opp["tags"] = tags
        return httpx.Response(200, json={"data": opp})

    def _remove_tags(self, params: dict[str, str], body: Any, opp_id: str) -> httpx.Response:
        opp = self.opportunities[opp_id]
        opp["tags"] = [t for t in (opp.get("tags") or []) if t not in body["tags"]]
        return httpx.Response(200, json={"data": opp})

    def _create_note(self, params: dict[str, str], body: Any, opp_id: str) -> httpx.Response:
        self._note_seq += 1
        note_id = f"note-{self._note_seq}"
        self.notes.setdefault(opp_id, {})[note_id] = body["value"]
        return httpx.Response(201, json={"data": {"noteId": note_id}})

    def _update_note(self, params: dict[str, str], body: Any, opp_id: str,
                     note_id: str) -> httpx.Response:
        notes = self.notes.get(opp_id, {})
        if note_id not in notes:
            return httpx.Response(404, json={"code": "NotFound", "message": "note"})
        notes[note_id] = body["values"][0]["value"]
        return httpx.Response(200, json={"data": {"noteId": note_id}})

    def _deleted(self, params: dict[str, str], body: Any) -> httpx.Response:
        start, end = int(params["deleted_at_start"]), int(params["deleted_at_end"])
        return self._page([d for d in self.deleted if start <= d["deletedAt"] < end])

    def _deleted_opportunities(self, params: dict[str, str], body: Any) -> httpx.Response:
        """Both bounds optional and inclusive, as Lever documents them."""
        start = int(params.get("deleted_at_start", 0))
        end = int(params.get("deleted_at_end", 2**62))
        return self._page([d for d in self.deleted_opportunities
                           if start <= d["deletedAt"] <= end])


def _ms(when: datetime) -> int:
    return int(when.timestamp() * 1000)


def populate_posting(api: FakeLever) -> FakeLever:
    """posting-1 (Chicago, IL, US) with seven applicants:
    - opp-alice: two processed resumes (the newer one is strong) plus a newer one still
      processing; her first assessment mentions a protected term (corrective retry);
    - opp-bob / opp-dana: the same resume under different names, sharing a phone number;
    - opp-erin: the download answers 422, Lever's parsedData is used;
    - opp-frank: a scanned (image-only) PDF that needs vision;
    - opp-gina: archived; opp-hal: anonymized."""
    api.postings[POSTING_ID] = {
        "id": POSTING_ID, "text": "Senior Data Engineer", "state": "published",
        "country": "US", "workplaceType": "onsite",
        "categories": {"location": "Chicago, IL", "allLocations": ["Chicago, IL"],
                       "team": "Data"},
        "content": {
            "description": "Egen builds data platforms for clients.",
            "lists": [{"text": "Requirements",
                       "content": "<li>Production data pipelines on GCP</li><li>Python</li>"}],
            "closing": "Egen is an equal opportunity employer.",
        },
    }
    base = _ms(datetime(2026, 9, 1, tzinfo=UTC))
    api.add_opportunity("opp-alice", name="Alice Novak", created=base + 1,
                        emails=["alice.novak@example.com"], phones=["(312) 555-0100"],
                        tags=["Referral", "AI Rank: Tier C"])
    api.add_resume("opp-alice", "res-alice-old", text_pdf(ALICE_OLD_LINES), uploaded=base + 10)
    api.add_resume("opp-alice", "res-alice-new", text_pdf(ALICE_LINES), uploaded=base + 20)
    api.add_resume("opp-alice", "res-alice-processing", None, uploaded=base + 30,
                   status="processing")
    api.add_opportunity("opp-bob", name="Bob Marsh", created=base + 2,
                        emails=["bob.marsh@example.com"], phones=["+1 (312) 555-0101"])
    api.add_resume("opp-bob", "res-bob", text_pdf(BOB_LINES), uploaded=base + 10)
    api.add_opportunity("opp-dana", name="Dana Kerr", created=base + 3,
                        emails=["dana.kerr@example.com"], phones=["312-555-0101"])
    api.add_resume("opp-dana", "res-dana", text_pdf(DANA_LINES), uploaded=base + 10)
    api.add_opportunity("opp-erin", name="Erin Patel", created=base + 4,
                        emails=["erin.patel@example.com"])
    api.add_resume("opp-erin", "res-erin", 422, uploaded=base + 10, parsed=ERIN_PARSED_DATA)
    api.add_opportunity("opp-frank", name="Frank Osei", created=base + 5,
                        emails=["frank.osei@example.com"])
    api.add_resume("opp-frank", "res-frank", scanned_pdf(FRANK_SCAN_LINES), uploaded=base + 10)
    api.add_opportunity("opp-gina", name="Gina Ruiz", created=base + 6, archived=True,
                        emails=["gina.ruiz@example.com"])
    api.add_resume("opp-gina", "res-gina", text_pdf(BOB_LINES[2:]), uploaded=base + 10)
    api.add_opportunity("opp-hal", name="Hal Berg", created=base + 7, anonymized=True)
    api.add_resume("opp-hal", "res-hal", text_pdf(ALICE_LINES), uploaded=base + 10)
    return api


# -------------------------------------------------------------------------- the model

REQUIREMENT_KEYWORDS = {"gcp-pipelines": "BigQuery", "python": "Python",
                        "client-facing": "client workshops", "streaming": "Kafka"}
KNOCKOUT_KEYWORDS = {"sql": "SQL"}
PROTECTED_SUMMARY = "She is a strong match for the role."


def evidence_line(text: str, keyword: str) -> str | None:
    for line in text.splitlines():
        if keyword.casefold() in line.casefold():
            return line.strip()
    return None


class ScriptedResponder:
    """Grounded assessments: a requirement whose keyword appears in the redacted resume
    is rated ``rating`` (default "meets") with that line quoted verbatim, else "not_met".

    - ``protected_first``: resumes containing this marker get a protected term in the
      summary on the first call (no corrective note) only;
    - ``protected_always``: ... on every call;
    - ``errors``: marker -> exception (an LLMError is returned, i.e. raised by the fake;
      any other exception is raised directly);
    - ``ratings``: marker -> rating override.
    """

    def __init__(self, *, protected_first: str | None = None,
                 protected_always: str | None = None,
                 errors: dict[str, Exception] | None = None,
                 ratings: dict[str, str] | None = None,
                 on_call: Callable[[AssessmentRequest], None] | None = None) -> None:
        self.protected_first = protected_first
        self.protected_always = protected_always
        self.errors = dict(errors or {})
        self.ratings = dict(ratings or {})
        self.on_call = on_call

    def __call__(self, request: AssessmentRequest) -> LLMAssessment | LLMError:
        text = request.resume_text
        if self.on_call is not None:
            self.on_call(request)
        for marker, error in self.errors.items():
            if marker in text:
                if isinstance(error, LLMError):
                    return error
                raise error
        rating = next((r for m, r in self.ratings.items() if m in text), "meets")
        requirements = []
        found = 0
        for req in request.criteria.requirements:
            line = evidence_line(text, REQUIREMENT_KEYWORDS.get(req.id, req.id))
            found += line is not None
            requirements.append({
                "requirement_id": req.id, "evidence": [line] if line else [],
                "rating": rating if line else "not_met",
                "note": "Relevant experience is described." if line else "Not described.",
            })
        knockouts = []
        for ko in request.criteria.knockouts:
            line = evidence_line(text, KNOCKOUT_KEYWORDS.get(ko.id, ko.id))
            knockouts.append({"knockout_id": ko.id, "evidence": [line] if line else [],
                              "result": "met" if line else "unclear",
                              "note": "Shown in the resume." if line else "Not shown."})
        summary = f"The candidate shows evidence for {found} of the listed requirements."
        protected = (self.protected_always and self.protected_always in text) or (
            self.protected_first and self.protected_first in text
            and request.corrective_note is None)
        if protected:
            summary = PROTECTED_SUMMARY
        return LLMAssessment.model_validate({
            "requirements": requirements,
            "knockouts": knockouts,
            "strengths": ["Production data engineering experience."] if found else [],
            "gaps": [] if found == len(requirements) else ["Some requirements not shown."],
            "red_flags": [],
            "injection_attempt_detected": False,
            "summary": summary,
            "confidence": "high",
            "overall_fit": max(1, round(100 * found / len(requirements))),
        })


class Clock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class NoWaitBucket(AsyncTokenBucket):
    """Rate limiting is the client's own concern (tested there); never wait here."""

    def __init__(self) -> None:
        super().__init__(1000.0, 20)

    async def acquire(self, tokens: float = 1.0) -> None:
        return None


# ------------------------------------------------------------------------------ harness


def approved_policy(*, illinois: bool = True) -> PolicyEngine:
    config = PolicyConfig(
        approved_by="Legal and HR", approved_at=date(2026, 9, 1), allowed_countries=["US"],
        illinois=IllinoisPolicy(notice_url="https://careers.example.com/ai-notice",
                                posting_notice_confirmed=True) if illinois
        else IllinoisPolicy(),
    )
    return PolicyEngine(config, source="test-policy")


@dataclass
class Harness:
    settings: Settings
    api: FakeLever
    lever: LeverClient | None
    evaluator: FakeEvaluator
    store: SqliteStore
    criteria: CriteriaRepository
    policy: PolicyEngine
    extractor: ResumeExtractor
    preparer: ResumePreparer
    planner: WritebackPlanner
    writer: LeverWriter
    ranker: Ranker
    clock: Clock
    lexicon: ProtectedTermLexicon


_DEFAULT = object()


@pytest.fixture(scope="session")
def pipeline_lexicon() -> ProtectedTermLexicon:
    return ProtectedTermLexicon.load()


@pytest.fixture
async def make_harness(make_settings: Callable[..., Settings], tmp_path: Path,
                       pipeline_lexicon: ProtectedTermLexicon,
                       sample_criteria: RoleCriteria) -> AsyncIterator[Callable[..., Harness]]:
    """Build a Harness. Keyword arguments: ``criteria`` (None for no criteria file),
    ``policy``, ``responder``, ``transcriber``, ``evaluator``, ``with_lever``, ``api``
    (a FakeLever; default: ``populate_posting``), ``store``, ``lexicon`` (default: the
    packaged one) and any Settings override."""
    cleanups: list[Callable[[], Any]] = []
    seq = 0

    def build(*, criteria: Any = _DEFAULT, policy: PolicyEngine | None = None,
              responder: Any = None, transcriber: Any = None,
              evaluator: Any = None, with_lever: bool = True, api: FakeLever | None = None,
              store: SqliteStore | None = None, io_concurrency: int = 4,
              lexicon: ProtectedTermLexicon | None = None,
              **overrides: Any) -> Harness:
        nonlocal seq
        seq += 1
        criteria_dir = tmp_path / f"criteria-{seq}"
        criteria_dir.mkdir()
        settings = make_settings(criteria_dir=criteria_dir, **overrides)
        role = sample_criteria if criteria is _DEFAULT else criteria
        if role is not None:
            (criteria_dir / f"{role.role_id}.yaml").write_text(dump_criteria_yaml(role),
                                                              encoding="utf-8")
        repo = CriteriaRepository(criteria_dir)
        repo.load()
        api = api if api is not None else populate_posting(FakeLever())
        lever = LeverClient(api_key="test-lever-key", max_retries=0,
                            transport=httpx.MockTransport(api.handler), bucket=NoWaitBucket(),
                            backoff_base_seconds=0) if with_lever else None
        if lever is not None:
            cleanups.append(lever.aclose)
        fake = evaluator if evaluator is not None else FakeEvaluator(
            responder if responder is not None else ScriptedResponder(
                protected_first=ALICE_MARKER),
            transcriber=transcriber if transcriber is not None
            else (lambda images: FRANK_TRANSCRIPT))
        the_store = store if store is not None else SqliteStore(settings.state_db)
        cleanups.append(the_store.close)
        extractor = ResumeExtractor.from_settings(settings)
        cleanups.append(extractor.close)
        preparer = ResumePreparer.from_settings(settings, extractor)
        planner = WritebackPlanner(settings)
        clock = Clock()
        writer = LeverWriter(lever, the_store, settings, clock=clock)
        the_lexicon = lexicon if lexicon is not None else pipeline_lexicon
        the_policy = policy or approved_policy()
        ranker = Ranker(settings=settings, lever=lever, evaluator=fake, store=the_store,
                        criteria=repo, policy=the_policy, preparer=preparer,
                        lexicon=the_lexicon, planner=planner, writer=writer, clock=clock,
                        io_concurrency=io_concurrency)
        harness = Harness(settings=settings, api=api, lever=lever, evaluator=fake,
                          store=the_store, criteria=repo, policy=the_policy,
                          extractor=extractor, preparer=preparer, planner=planner,
                          writer=writer, ranker=ranker, clock=clock, lexicon=the_lexicon)
        return harness

    yield build

    for cleanup in reversed(cleanups):
        result = cleanup()
        if inspect.isawaitable(result):
            await result


@pytest.fixture(scope="session")
def pipeline_kit() -> SimpleNamespace:
    """This module's public constants, classes and helpers (test modules can't import a
    conftest under --import-mode=importlib)."""
    return SimpleNamespace(**{name: value for name, value in globals().items()
                              if name.isidentifier() and not name.startswith("_")})
