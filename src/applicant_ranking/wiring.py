"""Build and close the object graph (shared by the CLI and the web service)."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass

from applicant_ranking.config import Settings
from applicant_ranking.criteria import CriteriaRepository
from applicant_ranking.lever.client import LeverClient
from applicant_ranking.llm.base import AssessmentRequest, Evaluator, LLMCallResult, LLMConfigError
from applicant_ranking.llm.gemini import GeminiEvaluator
from applicant_ranking.llm.prompts import PROMPT_VERSION
from applicant_ranking.pipeline import Ranker
from applicant_ranking.policy import PolicyEngine
from applicant_ranking.resumes.extract import ResumeExtractor
from applicant_ranking.resumes.prepare import ResumePreparer
from applicant_ranking.scoring.lexicon import ProtectedTermLexicon
from applicant_ranking.store import SqliteStore, Store
from applicant_ranking.writeback import LeverWriter, WritebackPlanner

logger = logging.getLogger(__name__)


@dataclass
class Components:
    settings: Settings
    store: Store
    criteria: CriteriaRepository
    policy: PolicyEngine
    lexicon: ProtectedTermLexicon
    extractor: ResumeExtractor
    preparer: ResumePreparer
    planner: WritebackPlanner
    writer: LeverWriter
    ranker: Ranker
    lever: LeverClient | None = None
    evaluator: Evaluator | None = None


class _UnavailableEvaluator:
    """Placeholder used when a command doesn't need the LLM; any call raises
    LLMConfigError explaining which setting is missing.

    It reports the configured model id and the current prompt version, so cache keys
    computed with it match those of real runs."""

    def __init__(self, settings: Settings) -> None:
        self.model_id = settings.gemini_model
        self.prompt_version = PROMPT_VERSION
        if settings.gcp_project:
            self._reason = ("the model is not available to this command (it was started "
                            "without the LLM); use a command that evaluates resumes")
        else:
            self._reason = ("the model is not configured: AR_GCP_PROJECT is not set (an "
                            "Egen GCP project with Vertex AI enabled)")

    async def assess(self, request: AssessmentRequest) -> LLMCallResult:
        raise LLMConfigError(self._reason)

    async def transcribe(self, page_images: list[bytes]) -> str:
        raise LLMConfigError(self._reason)

    async def aclose(self) -> None:
        return None


def build_components(settings: Settings, *, need_lever: bool = True, need_llm: bool = True,
                     lever: LeverClient | None = None, evaluator: Evaluator | None = None,
                     store: Store | None = None) -> Components:
    """Construct everything from settings. Injected ``lever``/``evaluator``/``store`` win
    (tests). ``need_lever``/``need_llm`` False -> those are not constructed (no key or GCP
    project required). Loads criteria (``criteria.load()``), the policy file and the
    lexicon (``settings.protected_terms_file``).

    Configuration is validated before anything that holds a resource (database
    connection, HTTP client, model client) is created, so a ConfigError (or a ValueError
    for an invalid policy file) leaves nothing open. ``Components.evaluator`` is None
    when the LLM was not needed and none was injected; the ranker then holds a
    placeholder that raises LLMConfigError."""
    # Configuration and files first: these raise without leaving anything open.
    criteria = CriteriaRepository(settings.criteria_dir)
    criteria.load()
    policy = PolicyEngine.from_file(settings.policy_file)
    lexicon = ProtectedTermLexicon.load(settings.protected_terms_file)
    planner = WritebackPlanner(settings)
    settings.require_hmac_key()
    build_lever = lever is None and need_lever
    build_llm = evaluator is None and need_llm
    if build_lever:
        settings.require_lever_key()
    if build_llm:
        settings.require_gcp_project()

    extractor = ResumeExtractor.from_settings(settings)
    preparer = ResumePreparer.from_settings(settings, extractor)
    own_store = store is None
    the_store: Store = store if store is not None else SqliteStore(
        settings.state_db, store_resume_text=settings.store_resume_text)
    try:
        if build_llm:
            evaluator = GeminiEvaluator(settings)
        if build_lever:
            lever = LeverClient.from_settings(settings)
    except BaseException:
        extractor.close()
        if own_store:
            the_store.close()
        raise

    writer = LeverWriter(lever, the_store, settings)
    ranker = Ranker(
        settings=settings,
        lever=lever,
        evaluator=evaluator if evaluator is not None else _UnavailableEvaluator(settings),
        store=the_store,
        criteria=criteria,
        policy=policy,
        preparer=preparer,
        lexicon=lexicon,
        planner=planner,
        writer=writer,
        io_concurrency=settings.io_concurrency,
    )
    logger.info("components ready: lever=%s, llm=%s, criteria files=%d, policy approved=%s",
                lever is not None, evaluator is not None, len(criteria.all()), policy.approved)
    return Components(
        settings=settings,
        store=the_store,
        criteria=criteria,
        policy=policy,
        lexicon=lexicon,
        extractor=extractor,
        preparer=preparer,
        planner=planner,
        writer=writer,
        ranker=ranker,
        lever=lever,
        evaluator=evaluator,
    )


async def close_components(components: Components) -> None:
    """Close the Lever client, the evaluator, the extractor's worker pool and the store,
    in that order. Each is closed even if closing an earlier one fails; the first
    failure is re-raised at the end."""
    first_error: BaseException | None = None

    async def attempt(name: str, close: Callable[[], object]) -> None:
        nonlocal first_error
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.warning("closing %s failed: %s", name, type(exc).__name__)
            first_error = first_error or exc

    if components.lever is not None:
        await attempt("the Lever client", components.lever.aclose)
    if components.evaluator is not None:
        await attempt("the evaluator", components.evaluator.aclose)
    await attempt("the resume extractor", components.extractor.close)
    await attempt("the store", components.store.close)
    if first_error is not None:
        raise first_error
