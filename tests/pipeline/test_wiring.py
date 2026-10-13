"""build_components / close_components. The real Gemini evaluator is never constructed
here: ``GeminiEvaluator`` is replaced in the wiring module by a recording stand-in."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

from applicant_ranking import wiring
from applicant_ranking.config import ConfigError, Settings
from applicant_ranking.criteria import dump_criteria_yaml
from applicant_ranking.domain import Evaluation, RoleCriteria, RunMode, Status
from applicant_ranking.lever.client import LeverClient
from applicant_ranking.llm.base import AssessmentRequest, LLMConfigError
from applicant_ranking.llm.fake import FakeEvaluator
from applicant_ranking.llm.prompts import PROMPT_VERSION
from applicant_ranking.pipeline import Ranker, RankingError
from applicant_ranking.policy import PolicyEngine
from applicant_ranking.scoring.lexicon import LexiconError, ProtectedTermLexicon
from applicant_ranking.store import SqliteStore, StoreError
from applicant_ranking.wiring import Components, build_components, close_components

POLICY_YAML = """\
version: 3
approved_by: Legal and HR
approved_at: 2026-09-01
allowed_countries: [US]
illinois:
  notice_url: https://careers.example.com/ai-notice
  posting_notice_confirmed: true
"""


class RecordingGemini:
    """Stands in for GeminiEvaluator (which would build a Vertex AI client)."""

    instances: ClassVar[list[RecordingGemini]] = []
    fail: ClassVar[Exception | None] = None

    def __init__(self, settings: Settings) -> None:
        if RecordingGemini.fail is not None:
            raise RecordingGemini.fail
        self.settings = settings
        self.model_id = settings.gemini_model
        self.prompt_version = "p1-recording"
        self.closed = False
        RecordingGemini.instances.append(self)

    async def assess(self, request: AssessmentRequest) -> Any:
        raise AssertionError("no model calls in wiring tests")

    async def transcribe(self, page_images: list[bytes]) -> str:
        raise AssertionError("no model calls in wiring tests")

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _no_real_gemini(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    RecordingGemini.instances = []
    RecordingGemini.fail = None
    monkeypatch.setattr(wiring, "GeminiEvaluator", RecordingGemini)
    yield


@pytest.fixture
def configured(make_settings: Callable[..., Settings],
               sample_criteria: RoleCriteria) -> Callable[..., Settings]:
    """Settings with a criteria directory (the sample criteria) and an approved policy."""

    def build(**overrides: Any) -> Settings:
        settings = make_settings(**overrides)
        settings.criteria_dir.mkdir(parents=True, exist_ok=True)
        (settings.criteria_dir / "role.yaml").write_text(dump_criteria_yaml(sample_criteria),
                                                         encoding="utf-8")
        settings.policy_file.write_text(POLICY_YAML, encoding="utf-8")
        return settings

    return build


async def test_builds_everything_from_settings(configured: Callable[..., Settings],
                                               sample_criteria: RoleCriteria) -> None:
    settings = configured()
    comps = build_components(settings)
    try:
        assert isinstance(comps, Components) and comps.settings is settings
        assert isinstance(comps.lever, LeverClient)
        assert comps.evaluator is RecordingGemini.instances[0]
        assert RecordingGemini.instances[0].settings is settings
        assert isinstance(comps.store, SqliteStore)
        assert Path(comps.store.path) == settings.state_db and settings.state_db.exists()
        assert comps.criteria.for_posting("posting-1") == sample_criteria
        assert isinstance(comps.policy, PolicyEngine) and comps.policy.approved
        assert comps.policy.config is not None and comps.policy.config.version == 3
        assert isinstance(comps.lexicon, ProtectedTermLexicon) and comps.lexicon.term_count
        assert comps.extractor.in_subprocess is False  # from settings
        assert isinstance(comps.ranker, Ranker)
        assert comps.planner.namespace() == "AI Rank: "
    finally:
        await close_components(comps)
    assert RecordingGemini.instances[0].closed
    with pytest.raises(StoreError):
        comps.store.audit_rows()
    with pytest.raises(RuntimeError, match="closed"):
        await comps.extractor.extract(b"some resume text")


async def test_store_resume_text_setting_reaches_the_store(
        configured: Callable[..., Settings]) -> None:
    comps = build_components(configured(store_resume_text=True), need_lever=False,
                             need_llm=False)
    try:
        assert isinstance(comps.store, SqliteStore)
        comps.store.record_evaluation(
            Evaluation(evaluation_id="ev-1", mode=RunMode.EVALUATION, opportunity_id="opp-1",
                       status=Status.ERROR),
            cache_key=None, redacted_text="[CANDIDATE] built pipelines")
        assert comps.store.get_redacted_text("ev-1") == "[CANDIDATE] built pipelines"
    finally:
        await close_components(comps)


async def test_need_flags_skip_lever_and_llm(configured: Callable[..., Settings],
                                             sample_criteria: RoleCriteria) -> None:
    settings = configured(lever_api_key=None, gcp_project=None)
    comps = build_components(settings, need_lever=False, need_llm=False)
    try:
        assert comps.lever is None and comps.evaluator is None
        assert RecordingGemini.instances == []
        with pytest.raises(RankingError, match="AR_LEVER_API_KEY"):
            await comps.ranker.rank_posting("posting-1")
        # Systemic (the deployment, not the applicant): it propagates; the CLI maps it to
        # exit 3 rather than recording an ERROR row per candidate.
        with pytest.raises(LLMConfigError, match="AR_GCP_PROJECT"):
            await comps.ranker.evaluate_local(
                criteria=sample_criteria,
                text="Built BigQuery pipelines in Python for clients.")
    finally:
        await close_components(comps)


async def test_unavailable_evaluator_explains_what_is_missing(
        make_settings: Callable[..., Settings], sample_criteria: RoleCriteria) -> None:
    no_project = wiring._UnavailableEvaluator(make_settings(gcp_project=None))
    request = AssessmentRequest(criteria=sample_criteria, resume_text="text")
    with pytest.raises(LLMConfigError, match="AR_GCP_PROJECT is not set"):
        await no_project.assess(request)
    with pytest.raises(LLMConfigError, match="AR_GCP_PROJECT is not set"):
        await no_project.transcribe([b"png"])
    assert (no_project.model_id, no_project.prompt_version) == (
        "gemini-3.1-pro-preview", PROMPT_VERSION)
    await no_project.aclose()

    with_project = wiring._UnavailableEvaluator(make_settings())
    with pytest.raises(LLMConfigError, match="started without the LLM"):
        await with_project.assess(request)


@pytest.mark.parametrize(("overrides", "kwargs", "error", "match"), [
    ({"lever_api_key": None}, {}, ConfigError, "AR_LEVER_API_KEY"),
    ({"gcp_project": None}, {}, ConfigError, "AR_GCP_PROJECT"),
    ({"env": "prod"}, {"need_lever": False, "need_llm": False}, ConfigError, "AR_HMAC_KEY"),
    ({"tag_prefix": " "}, {"need_lever": False, "need_llm": False}, ConfigError,
     "AR_TAG_PREFIX"),
])
def test_configuration_errors_leave_nothing_open(configured: Callable[..., Settings],
                                                 overrides: dict[str, Any],
                                                 kwargs: dict[str, Any],
                                                 error: type[Exception], match: str) -> None:
    settings = configured(**overrides)
    with pytest.raises(error, match=match):
        build_components(settings, **kwargs)
    assert not settings.state_db.exists()
    assert RecordingGemini.instances == []


def test_invalid_policy_and_lexicon_files_fail_loudly(configured: Callable[..., Settings],
                                                      tmp_path: Path) -> None:
    settings = configured()
    settings.policy_file.write_text("approved_by: [not, a, string]\nunknown: 1\n",
                                    encoding="utf-8")
    with pytest.raises(ValueError, match="policy file"):
        build_components(settings)

    lexicon_settings = configured(protected_terms_file=tmp_path / "missing-terms.yaml")
    with pytest.raises(LexiconError):
        build_components(lexicon_settings)
    assert isinstance(LexiconError("x"), ConfigError)
    assert not settings.state_db.exists()


async def test_missing_policy_file_blocks_real_runs(configured: Callable[..., Settings]) -> None:
    settings = configured()
    settings.policy_file.unlink()
    comps = build_components(settings, need_lever=False, need_llm=False)
    try:
        assert comps.policy.config is None
    finally:
        await close_components(comps)


async def test_injected_objects_win(make_settings: Callable[..., Settings],
                                    tmp_path: Path) -> None:
    settings = make_settings(lever_api_key=None, gcp_project=None)
    lever = LeverClient(api_key="injected-key",
                        transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    evaluator = FakeEvaluator()
    store = SqliteStore(tmp_path / "injected.sqlite3")
    comps = build_components(settings, lever=lever, evaluator=evaluator, store=store)
    try:
        assert comps.lever is lever and comps.evaluator is evaluator and comps.store is store
        assert RecordingGemini.instances == []
        assert not settings.state_db.exists()
        assert comps.policy.config is None  # no policy file: DRY_RUN/LIVE will be blocked
        assert comps.criteria.all() == []
    finally:
        await close_components(comps)
    assert evaluator.closed


async def test_evaluator_failure_closes_what_was_opened(configured: Callable[..., Settings],
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[str] = []

    class SpyStore(SqliteStore):
        def close(self) -> None:
            closed.append("store")
            super().close()

    monkeypatch.setattr(wiring, "SqliteStore", SpyStore)
    RecordingGemini.fail = RuntimeError("client construction failed")
    with pytest.raises(RuntimeError, match="client construction failed"):
        build_components(configured())
    assert closed == ["store"]


class _BrokenLever:
    async def aclose(self) -> None:
        raise RuntimeError("socket already gone")


async def test_close_components_closes_everything_despite_a_failure(
        configured: Callable[..., Settings]) -> None:
    evaluator = FakeEvaluator()
    comps = build_components(configured(), lever=_BrokenLever(),  # type: ignore[arg-type]
                             evaluator=evaluator)
    with pytest.raises(RuntimeError, match="socket already gone"):
        await close_components(comps)
    assert evaluator.closed
    with pytest.raises(StoreError):
        comps.store.audit_rows()
    with pytest.raises(RuntimeError, match="closed"):
        await comps.extractor.extract(b"text")


def test_one_ranker_serves_several_event_loops(configured: Callable[..., Settings],
                                               sample_criteria: RoleCriteria) -> None:
    evaluator = FakeEvaluator()
    comps = build_components(configured(llm_concurrency=1), need_lever=False,
                             evaluator=evaluator)
    try:
        for _ in range(2):  # a CLI may call asyncio.run more than once
            evaluation = asyncio.run(comps.ranker.evaluate_local(
                criteria=sample_criteria, text="Built BigQuery pipelines in Python daily."))
            assert evaluation.mode is RunMode.EVALUATION
            assert evaluation.status is not Status.ERROR
        assert len(evaluator.requests) == 2
    finally:
        asyncio.run(close_components(comps))
