"""Fixtures for the CLI and the webhook service: fake ranker and Lever client, plus
real store/criteria/policy/lexicon objects on temporary files."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from applicant_ranking import cli
from applicant_ranking.config import Settings
from applicant_ranking.criteria import CriteriaRepository, dump_criteria_yaml
from applicant_ranking.domain import (
    Evaluation,
    RankingReport,
    RankingRow,
    RoleCriteria,
    RunMode,
    Status,
    Tier,
)
from applicant_ranking.lever.models import ArchiveReason, Opportunity, Posting, Stage
from applicant_ranking.pipeline import SKIP_RESUME_PENDING
from applicant_ranking.policy import PolicyEngine
from applicant_ranking.resumes.extract import ResumeExtractor
from applicant_ranking.resumes.prepare import ResumePreparer
from applicant_ranking.scoring.lexicon import ProtectedTermLexicon
from applicant_ranking.store import PurgeResult, SqliteStore
from applicant_ranking.wiring import Components
from applicant_ranking.writeback import LeverWriter, WritebackPlanner

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
SECRET_LEVER_KEY = "lever-key-SHOULD-NOT-APPEAR"
SECRET_HMAC_KEY = "hmac-key-SHOULD-NOT-APPEAR"
WEBHOOK_TOKENS = {
    "applicationCreated": "whsec-app-SHOULD-NOT-APPEAR",
    "candidateDeleted": "whsec-del-SHOULD-NOT-APPEAR",
    "candidateHired": "whsec-hired-SHOULD-NOT-APPEAR",
}
SECRETS = (SECRET_LEVER_KEY, SECRET_HMAC_KEY, *WEBHOOK_TOKENS.values())
CANDIDATE = "Jane Candidate"


def make_evaluation(**overrides: Any) -> Evaluation:
    base: dict[str, Any] = {
        "evaluation_id": "ev-1",
        "created_at": NOW,
        "mode": RunMode.DRY_RUN,
        "posting_id": "posting-1",
        "opportunity_id": "opp-1",
        "candidate_name": CANDIDATE,
        "profile_url": "https://hire.lever.co/candidates/opp-1",
        "criteria_role_id": "senior-data-engineer",
        "criteria_version": 2,
        "status": Status.SCORED,
        "score": 84,
        "tier": Tier.A,
        "strengths": ["Production GCP pipelines"],
        "gaps": ["No consulting experience"],
    }
    base.update(overrides)
    return Evaluation(**base)


def make_report(mode: RunMode = RunMode.DRY_RUN, posting_id: str = "posting-1") -> RankingReport:
    rows = [
        RankingRow(rank=1, evaluation=make_evaluation(mode=mode, posting_id=posting_id)),
        RankingRow(rank=2, evaluation=make_evaluation(
            mode=mode, posting_id=posting_id, evaluation_id="ev-2", opportunity_id="opp-2",
            candidate_name="Bob Applicant", score=61, tier=Tier.C)),
        RankingRow(rank=None, evaluation=make_evaluation(
            mode=mode, posting_id=posting_id, evaluation_id="ev-3", opportunity_id="opp-3",
            candidate_name="Error Person", status=Status.ERROR, score=None, tier=None,
            strengths=[], gaps=[])),
    ]
    return RankingReport(posting_id=posting_id, posting_title="Senior Data Engineer",
                         criteria_role_id="senior-data-engineer", criteria_version=2,
                         mode=mode, rows=rows)


class FakeRanker:
    """Async methods return canned results and record their calls. Set ``error`` to an
    exception instance to make every call raise it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.error: BaseException | None = None
        self.report_factory: Callable[[str, RunMode], RankingReport] = (
            lambda posting_id, mode: make_report(mode, posting_id))
        self.local_evaluation: Evaluation | None = None
        self.deleted: list[PurgeResult] = []
        self.on_evaluate: Callable[[str], Any] | None = None
        # evaluate_opportunity answers "resume still processing" this many times while the
        # caller defers (a last, undeferred attempt always scores).
        self.pending_rows = 0

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))
        if self.error is not None:
            raise self.error

    def called(self, name: str) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
        return [(a, k) for n, a, k in self.calls if n == name]

    def clock(self) -> datetime:
        return NOW

    def check_criteria(self, criteria: RoleCriteria, *, mode: RunMode) -> None:
        self._record("check_criteria", criteria, mode=mode)

    async def rank_posting(self, posting_id: str, *, mode: RunMode = RunMode.DRY_RUN,
                           limit: int | None = None, include_archived: bool = False,
                           force: bool = False) -> RankingReport:
        self._record("rank_posting", posting_id, mode=mode, limit=limit,
                     include_archived=include_archived, force=force)
        return self.report_factory(posting_id, mode)

    async def evaluate_opportunity(self, opportunity_id: str, *,
                                   mode: RunMode = RunMode.DRY_RUN,
                                   force: bool = False,
                                   defer_pending: bool = True) -> RankingRow:
        self._record("evaluate_opportunity", opportunity_id, mode=mode, force=force,
                     defer_pending=defer_pending)
        if self.on_evaluate is not None:
            await self.on_evaluate(opportunity_id)
        if defer_pending and self.pending_rows > 0:
            self.pending_rows -= 1
            return RankingRow(rank=None, evaluation=make_evaluation(
                opportunity_id=opportunity_id, mode=mode, status=Status.SKIPPED, score=None,
                tier=None, strengths=[], gaps=[], review_reasons=[SKIP_RESUME_PENDING]))
        return RankingRow(rank=1, evaluation=make_evaluation(opportunity_id=opportunity_id,
                                                             mode=mode))

    async def evaluate_local(self, *, criteria: RoleCriteria, data: bytes | None = None,
                             text: str | None = None, file_name: str | None = None,
                             candidate_names: Any = (),
                             opportunity_id: str = "local") -> Evaluation:
        self._record("evaluate_local", criteria=criteria, data=data, text=text,
                     file_name=file_name, candidate_names=list(candidate_names),
                     opportunity_id=opportunity_id)
        if self.local_evaluation is not None:
            return self.local_evaluation
        return make_evaluation(mode=RunMode.EVALUATION, opportunity_id=opportunity_id,
                               posting_id=None, candidate_name=None, profile_url=None,
                               criteria_role_id=criteria.role_id,
                               criteria_version=criteria.version)

    async def handle_candidate_deleted(self, opportunity_id: str, *,
                                       source: str = "webhook") -> PurgeResult:
        self._record("handle_candidate_deleted", opportunity_id, source=source)
        return PurgeResult(opportunity_id=opportunity_id, on_hold=False)

    async def reconcile_deletions(self, *, days: int = 30) -> list[PurgeResult]:
        self._record("reconcile_deletions", days=days)
        return list(self.deleted)


class FakeLever:
    """Just the read methods the CLI (and backtest) use."""

    def __init__(self) -> None:
        self.postings: list[Posting] = []
        self.opportunities: list[Opportunity] = []
        self.stages: list[Stage] = []
        self.archive_reasons: list[ArchiveReason] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    async def get_posting(self, posting_id: str) -> Posting:
        self.calls.append(("get_posting", {"posting_id": posting_id}))
        for posting in self.postings:
            if posting.id == posting_id:
                return posting
        from applicant_ranking.lever.client import LeverNotFound
        raise LeverNotFound(f"Lever GET /postings/{posting_id} returned 404", status=404)

    async def list_postings(self, *, state: str | None = None,
                            distribution_channels: Sequence[str] | None = ("public", "internal"),
                            confidentiality: str | None = None) -> AsyncIterator[Posting]:
        """Mirrors Lever's documented default: confidential postings are listed only when
        ``confidentiality`` is "confidential" or "all"."""
        channels = tuple(distribution_channels) if distribution_channels else None
        self.calls.append(("list_postings", {"state": state, "distribution_channels": channels,
                                             "confidentiality": confidentiality}))
        for posting in self.postings:
            if state is not None and posting.state != state:
                continue
            if posting.confidentiality == "confidential" and confidentiality not in (
                    "confidential", "all"):
                continue
            yield posting

    async def list_opportunities(self, *, posting_id: str, archived: bool | None = False,
                                 created_at_start_ms: int | None = None,
                                 confidentiality: str | None = None,
                                 ) -> AsyncIterator[Opportunity]:
        self.calls.append(("list_opportunities", {"posting_id": posting_id,
                                                  "archived": archived,
                                                  "confidentiality": confidentiality}))
        for opportunity in self.opportunities:
            yield opportunity

    async def list_stages(self) -> list[Stage]:
        return list(self.stages)

    async def list_archive_reasons(self) -> list[ArchiveReason]:
        return list(self.archive_reasons)

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """The CLI configures logging for its process (root handler and level, httpx level);
    undo that so other test areas see the default logging state."""
    root = logging.getLogger()
    httpx_logger = logging.getLogger("httpx")
    handlers, level, httpx_level = list(root.handlers), root.level, httpx_logger.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    httpx_logger.setLevel(httpx_level)


@pytest.fixture(scope="session")
def lexicon() -> ProtectedTermLexicon:
    return ProtectedTermLexicon.load()


@pytest.fixture
def iface_settings(make_settings: Callable[..., Settings], tmp_path: Path) -> Settings:
    criteria_dir = tmp_path / "criteria"
    criteria_dir.mkdir()
    return make_settings(
        lever_api_key=SECRET_LEVER_KEY,
        hmac_key=SECRET_HMAC_KEY,
        lever_webhook_tokens=WEBHOOK_TOKENS,
        criteria_dir=criteria_dir,
    )


@pytest.fixture
def write_criteria(iface_settings: Settings) -> Callable[..., Path]:
    def _write(criteria: RoleCriteria, name: str | None = None) -> Path:
        path = iface_settings.criteria_dir / (name or f"{criteria.role_id}.yaml")
        path.write_text(dump_criteria_yaml(criteria), encoding="utf-8")
        return path

    return _write


@dataclass
class Built:
    """What a fake ``cli._build`` produced, and how it was asked."""

    components: list[Components] = field(default_factory=list)
    kwargs: list[dict[str, Any]] = field(default_factory=list)
    closed: int = 0


@pytest.fixture
def make_components(lexicon: ProtectedTermLexicon) -> Iterator[Callable[..., Components]]:
    opened: list[Components] = []

    def _make(settings: Settings, *, ranker: Any, lever: Any = None,
              store: SqliteStore | None = None) -> Components:
        store = store or SqliteStore(settings.state_db)
        criteria = CriteriaRepository(settings.criteria_dir)
        criteria.load()
        extractor = ResumeExtractor.from_settings(settings)
        components = Components(
            settings=settings,
            store=store,
            criteria=criteria,
            policy=PolicyEngine.from_file(settings.policy_file),
            lexicon=lexicon,
            extractor=extractor,
            preparer=ResumePreparer.from_settings(settings, extractor),
            planner=WritebackPlanner(settings),
            writer=LeverWriter(lever, store, settings),
            ranker=ranker,
            lever=lever,
        )
        opened.append(components)
        return components

    yield _make
    for components in opened:
        components.extractor.close()
        components.store.close()


@dataclass
class CliHarness:
    runner: CliRunner
    settings: Settings
    ranker: FakeRanker
    lever: FakeLever
    built: Built

    def invoke(self, *args: str, input: str | None = None) -> Any:
        return self.runner.invoke(cli.app, list(args), input=input)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, iface_settings: Settings,
            make_components: Callable[..., Components]) -> CliHarness:
    """``cli`` wired to a FakeRanker/FakeLever and a real store on tmp files."""
    ranker = FakeRanker()
    lever = FakeLever()
    built = Built()

    def fake_build(settings: Settings, **kwargs: Any) -> Components:
        built.kwargs.append(kwargs)
        use_lever = lever if kwargs.get("need_lever", True) else None
        components = make_components(settings, ranker=ranker, lever=use_lever)
        built.components.append(components)
        return components

    async def fake_close(components: Components) -> None:
        built.closed += 1
        components.store.close()

    monkeypatch.setattr(cli, "_load_settings", lambda: iface_settings)
    monkeypatch.setattr(cli, "_build", fake_build)
    monkeypatch.setattr(cli, "_close", fake_close)
    return CliHarness(runner=CliRunner(), settings=iface_settings, ranker=ranker, lever=lever,
                      built=built)
