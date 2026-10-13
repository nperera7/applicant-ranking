"""Command-line interface (Typer). Entry point: ``applicant-ranking`` (``main``).

Commands (all print the report watermark; exit code 0 ok, 1 runtime error, 2 blocked by
gates/lint/policy, 3 configuration error):

- ``criteria lint [--dir DIR]``: lint every criteria file; show errors/warnings; exit 2 on
  errors. No credentials needed.
- ``criteria draft --posting ID [--out FILE]``: write a skeleton YAML drafted from the
  Lever posting (refuses to overwrite an existing file without ``--force``).
- ``postings list [--state published] [--include-confidential]``: id, title, state,
  location, whether criteria exist (both job sites; confidential postings on request or
  with AR_LEVER_CONFIDENTIAL_ACCESS=true).
- ``rank --posting ID [--mode dry-run|live|evaluation] [--limit N] [--include-archived]
  [--force] [--format table|csv|json|md] [--out FILE]``: rank a posting. ``--mode live``
  also requires AR_WRITE_ENABLED=true and asks for confirmation unless ``--yes``.
- ``evaluate --opportunity ID [--mode ...] [--force]``: one applicant.
- ``evaluate-file RESUME --criteria FILE [--name "Full Name"] [--format ...]``: offline
  evaluation of a local resume against a criteria file (EVALUATION mode; no Lever).
- ``backtest --posting ID --positive-stage NAME [--positive-stage NAME ...] [--limit N]
  [--out FILE]``: metrics vs past recruiter decisions (EVALUATION mode).
- ``fairness RESUME --criteria FILE [--runs 3] [--tolerance 5]``: counterfactual check.
- ``export-audit [--posting ID] --out FILE``: pseudonymous CSV for adverse-impact analysis.
- ``purge [--reconcile-days 30] [--yes]``: retention purge + deletion reconciliation (GET
  /opportunities/deleted) + completion of deletions a since-cleared legal hold suspended.
- ``hold set|clear --posting ID | --opportunity ID [--reason TEXT] [--by NAME]``;
  ``clear`` also completes the deletions the hold suspended and reports them.
- ``doctor``: show effective configuration (secrets masked), criteria/policy status,
  lexicon version, and which GCP project/location/model would be used. Makes no network
  calls and never reads credential files.
- ``serve [--host 0.0.0.0] [--port 8080]``: run the webhook service with uvicorn.

Reports containing candidate names go to stdout or ``--out``; the default ``reports/``
directory is git-ignored. Never log resume text.

Details:
- Report commands (rank, evaluate, evaluate-file) carry the watermark inside the rendered
  report; with ``--out`` it is also printed to stderr. Every other command prints the
  watermark as its first line.
- Exit 2 also covers deliberate refusals: ``--mode live`` without AR_WRITE_ENABLED,
  overwriting a file without ``--force``, a draft that would clash with existing criteria,
  a resume over AR_MAX_RESUME_BYTES, unknown backtest stage names, and a fairness check
  with flagged variants. Click usage errors also exit 2. Declining a confirmation prompt
  exits 1 ("Aborted!").
- Exit 3 also covers invalid AR_* settings, Lever rejecting the API key (401/403) and the
  LLM rejecting the configuration (LLMConfigError).
- Errors are one line on stderr (``--debug`` adds the traceback). A pydantic validation
  error is reported by field location only, never with its input values.
- Files written with ``--out`` (reports, audit exports) are created owner-only (0600).
- ``fairness`` extracts the resume's text with the production extractor (a scanned
  resume is refused) and accepts ``--name``, the name written in the resume. Without it
  (and likewise for ``evaluate-file``) the resume's own name reaches the model
  unredacted, which is said on stderr; the fairness report then also carries a caveat.
- ``hold`` and ``export-audit`` open the state store directly, so a broken criteria or
  policy file can't stop a legal hold. ``hold set`` requires ``--reason``; ``--by``
  defaults to the OS user.
- ``serve`` binds 127.0.0.1 unless ``--host`` says otherwise; ``--port`` falls back to
  ``$PORT`` (Cloud Run) and then 8080.
"""

from __future__ import annotations

import asyncio
import contextvars
import getpass
import json
import logging
import os
import sys
import traceback
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from pydantic import BaseModel, SecretBytes, SecretStr, ValidationError
from pydantic_settings import SettingsError

from applicant_ranking import wiring
from applicant_ranking.backtest import BacktestReport, run_backtest
from applicant_ranking.config import ConfigError, Settings, TierThresholds
from applicant_ranking.criteria import (
    CRITERIA_SUFFIXES,
    FIELD_GUIDE,
    CriteriaError,
    CriteriaRepository,
    draft_from_posting,
    dump_criteria_yaml,
    load_criteria_file,
)
from applicant_ranking.criteria_lint import LintIssue, has_errors, lint_criteria
from applicant_ranking.domain import (
    Evaluation,
    RankingReport,
    RankingRow,
    RoleCriteria,
    RunMode,
    utcnow,
)
from applicant_ranking.fairness import FairnessReport, run_fairness
from applicant_ranking.lever.client import LeverAuthError, LeverClient
from applicant_ranking.lever.webhooks import HANDLED_EVENTS
from applicant_ranking.llm.base import LLMConfigError
from applicant_ranking.pipeline import PolicyBlocked, RankingError
from applicant_ranking.policy import PolicyEngine
from applicant_ranking.report import (
    audit_export_csv,
    render_console,
    report_to_csv,
    report_to_json,
    report_to_markdown,
)
from applicant_ranking.scoring.lexicon import ProtectedTermLexicon
from applicant_ranking.store import PurgeResult, SqliteStore
from applicant_ranking.wiring import Components
from applicant_ranking.writeback import shorten

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_BLOCKED = 2
EXIT_CONFIG = 3

_MESSAGE_CHARS = 300
_CELL_CHARS = 60
_LOOPBACK = "127.0.0.1"
NO_NAME_WARNING = ("warning: without --name the resume's real name reaches the model "
                   "unredacted (the candidate name is what real runs redact)")
NO_NAME_FAIRNESS_WARNING = (NO_NAME_WARNING + " and the name variants test nothing; pass "
                            "--name unless the resume is synthetic and has no name")

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False,
                  help="Advisory ranking of Lever applications (Egen Talent Acquisition).")
criteria_app = typer.Typer(no_args_is_help=True, help="Lint and draft per-role criteria files.")
postings_app = typer.Typer(no_args_is_help=True, help="Lever postings.")
hold_app = typer.Typer(no_args_is_help=True,
                       help="Legal holds: suspend purges for a posting or an opportunity.")
app.add_typer(criteria_app, name="criteria")
app.add_typer(postings_app, name="postings")
app.add_typer(hold_app, name="hold")


class CommandBlocked(Exception):
    """The command refused to proceed under its own rules (exit 2)."""


class ModeOption(StrEnum):
    DRY_RUN = "dry-run"
    LIVE = "live"
    EVALUATION = "evaluation"

    @property
    def run_mode(self) -> RunMode:
        return {ModeOption.DRY_RUN: RunMode.DRY_RUN, ModeOption.LIVE: RunMode.LIVE,
                ModeOption.EVALUATION: RunMode.EVALUATION}[self]


class ReportFormat(StrEnum):
    TABLE = "table"
    CSV = "csv"
    JSON = "json"
    MD = "md"


_RENDERERS: dict[ReportFormat, Callable[[RankingReport], str]] = {
    ReportFormat.TABLE: render_console,
    ReportFormat.CSV: report_to_csv,
    ReportFormat.JSON: report_to_json,
    ReportFormat.MD: report_to_markdown,
}

# ------------------------------------------------------------------- shared plumbing


def _load_settings() -> Settings:
    """Settings from AR_* environment variables and ``.env``."""
    try:
        return Settings()
    except ValidationError as exc:
        raise ConfigError(f"invalid settings: {_validation_summary(exc)}") from None
    except SettingsError as exc:
        raise ConfigError(f"invalid settings: {exc}") from None


def _build(settings: Settings, **kwargs: Any) -> Components:
    """Indirection over ``wiring.build_components`` (tests substitute fake components)."""
    return wiring.build_components(settings, **kwargs)


async def _close(components: Components) -> None:
    await wiring.close_components(components)


def _open_store(settings: Settings) -> SqliteStore:
    return SqliteStore(settings.state_db, store_resume_text=settings.store_resume_text)


@asynccontextmanager
async def _components(settings: Settings, **kwargs: Any) -> AsyncIterator[Components]:
    try:
        components = _build(settings, **kwargs)
    except ValueError as exc:  # e.g. an invalid policy file: still configuration
        raise ConfigError(str(exc)) from exc
    try:
        yield components
    finally:
        await _close(components)


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def _require_lever(components: Components) -> LeverClient:
    if components.lever is None:
        raise ConfigError("this command needs the Lever API (set AR_LEVER_API_KEY)")
    return components.lever


def _validation_summary(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors(include_url=False, include_input=False):
        loc = ".".join(str(p) for p in err["loc"]) or "(top level)"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def _describe(exc: BaseException) -> str:
    if isinstance(exc, ValidationError):
        text = f"{exc.title}: {_validation_summary(exc)}"
    else:
        text = str(exc).strip() or type(exc).__name__
    return shorten(text.splitlines()[0] if text else type(exc).__name__, _MESSAGE_CHARS)


_DEBUG: contextvars.ContextVar[bool] = contextvars.ContextVar("applicant_ranking_debug",
                                                              default=False)


def _debug_enabled() -> bool:
    return _DEBUG.get()


def _echo_err(message: str) -> None:
    typer.echo(message, err=True)


def _fail(exc: BaseException, code: int, prefix: str = "error") -> NoReturn:
    _echo_err(f"{prefix}: {_describe(exc)}")
    if _debug_enabled():
        traceback.print_exception(exc, file=sys.stderr)
    raise typer.Exit(code)


@contextmanager
def _handle_errors() -> Iterator[None]:
    """Map exceptions to the documented exit codes, with a one-line message."""
    try:
        yield
    except (typer.Exit, typer.Abort, typer.TyperException):
        raise
    except PolicyBlocked as exc:
        _echo_err("blocked by the jurisdiction policy:")
        for blocker in exc.decision.blockers:
            _echo_err(f"  - {shorten(blocker, _MESSAGE_CHARS)}")
        if _debug_enabled():
            traceback.print_exception(exc, file=sys.stderr)
        raise typer.Exit(EXIT_BLOCKED) from None
    except (ConfigError, LLMConfigError, LeverAuthError) as exc:
        _fail(exc, EXIT_CONFIG, "configuration error")
    except (RankingError, CriteriaError, CommandBlocked) as exc:
        _fail(exc, EXIT_BLOCKED, "blocked")
    except Exception as exc:
        _fail(exc, EXIT_ERROR)


class _StderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes to whatever ``sys.stderr`` is at emit time (test runners swap it)."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property  # type: ignore[override]
    def stream(self) -> Any:
        return sys.stderr

    @stream.setter
    def stream(self, value: Any) -> None:
        pass


def _configure_logging(level: int) -> None:
    root = logging.getLogger()
    if not any(isinstance(h, _StderrHandler) for h in root.handlers):
        handler = _StderrHandler()
        root.addHandler(handler)
    for handler in root.handlers:
        if isinstance(handler, _StderrHandler):
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s" if level < logging.WARNING
                else "%(levelname)s: %(message)s"))
    root.setLevel(level)
    # httpx logs every request line at INFO; keep it quiet unless debugging.
    logging.getLogger("httpx").setLevel(logging.DEBUG if level <= logging.DEBUG
                                        else logging.WARNING)


def _watermark(mode: RunMode = RunMode.DRY_RUN) -> str:
    return RankingReport(posting_id="-", mode=mode).watermark or ""


def _banner(mode: RunMode = RunMode.DRY_RUN) -> None:
    typer.echo(_watermark(mode))


def _write_private(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` (parents created), owner read/write only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(text)
    return path


def _emit(text: str, out: Path | None, *, watermark: str) -> None:
    if out is None:
        typer.echo(text, nl=False)
        return
    _write_private(out, text)
    if watermark:
        _echo_err(watermark)
    typer.echo(f"wrote {out}")


def _emit_report(report: RankingReport, fmt: ReportFormat, out: Path | None) -> None:
    _emit(_RENDERERS[fmt](report), out, watermark=report.watermark or "")


def _dump_json(model: BaseModel, watermark: str) -> str:
    payload = {"watermark": watermark, **model.model_dump(mode="json")}
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _table(header: Sequence[str], rows: Sequence[Sequence[object]], *,
           width: int = _CELL_CHARS) -> str:
    cells = [[shorten("-" if v is None or v == "" else str(v), width) for v in row]
             for row in [header, *rows]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]

    def fmt(row: Sequence[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()

    lines = [fmt(cells[0]), fmt(["-" * w for w in widths]), *(fmt(r) for r in cells[1:])]
    return "\n".join(lines)


def _confirm_live(settings: Settings, mode: RunMode, yes: bool, target: str) -> None:
    if mode is not RunMode.LIVE:
        return
    if not settings.write_enabled:
        raise CommandBlocked("--mode live needs AR_WRITE_ENABLED=true; write-back is off "
                             "(use --mode dry-run to see the planned write-back)")
    if not yes:
        typer.confirm(f"Live mode writes tags and notes to Lever for {target}. Continue?",
                      abort=True, err=True)


def _read_resume(path: Path, max_bytes: int) -> bytes:
    size = path.stat().st_size
    if size > max_bytes:
        raise CommandBlocked(f"{path.name} is {size} bytes, over AR_MAX_RESUME_BYTES "
                             f"({max_bytes})")
    return path.read_bytes()


# ------------------------------------------------------------------------ callback


@app.callback()
def _main(
    debug: Annotated[bool, typer.Option("--debug", help="Show tracebacks and debug logs.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show progress logs.")] = False,
) -> None:
    _DEBUG.set(debug)
    _configure_logging(logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING)


# ------------------------------------------------------------------------ criteria


def _criteria_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir()
                  if p.suffix.lower() in CRITERIA_SUFFIXES and p.is_file()
                  and not p.name.startswith("."))


def _issue_line(issue: LintIssue) -> str:
    return f"  {issue.level:<8} {issue.location}: {issue.message}"


def _lint_directory(directory: Path, lexicon: ProtectedTermLexicon,
                    ) -> tuple[list[str], int, int]:
    """Lint every criteria file individually, then the cross-file rules (duplicate role or
    posting ids). Returns (output lines, errors, warnings)."""
    lines: list[str] = []
    errors = warnings = 0
    parse_errors: list[str] = []
    for path in _criteria_files(directory):
        try:
            criteria = load_criteria_file(path)
        except CriteriaError as exc:
            parse_errors.append(str(exc))
            lines.append(f"{path}: cannot be loaded")
            lines.append(f"  {'error':<8} {exc}")
            errors += 1
            continue
        issues = lint_criteria(criteria, lexicon)
        n_err = sum(1 for i in issues if i.level == "error")
        errors += n_err
        warnings += len(issues) - n_err
        approval = (f"approved by {criteria.approved_by}" if criteria.approved
                    else "NOT approved")
        postings = ", ".join(criteria.posting_ids) or "no postings"
        lines.append(f"{path}: {criteria.role_id} v{criteria.version} ({approval}; "
                     f"{postings})")
        lines.extend(_issue_line(issue) for issue in issues)
        if not issues:
            lines.append("  ok")
    repo = CriteriaRepository(directory)
    repo.load()
    for message in repo.errors:
        if message not in parse_errors:
            lines.append(f"{directory}: {message}")
            errors += 1
    return lines, errors, warnings


@criteria_app.command("lint")
def criteria_lint(
    directory: Annotated[Path | None, typer.Option(
        "--dir", help="Criteria directory (default: AR_CRITERIA_DIR).")] = None,
) -> None:
    """Lint every criteria file; exit 2 on errors. No credentials needed."""
    with _handle_errors():
        settings = _load_settings()
        _banner()
        directory = directory or settings.criteria_dir
        if not directory.is_dir():
            raise CommandBlocked(f"criteria directory not found: {directory}")
        lexicon = ProtectedTermLexicon.load(settings.protected_terms_file)
        lines, errors, warnings = _lint_directory(directory, lexicon)
        for line in lines:
            typer.echo(line)
        files = len(_criteria_files(directory))
        typer.echo(f"{files} file(s) checked: {errors} error(s), {warnings} warning(s)")
        if errors:
            raise typer.Exit(EXIT_BLOCKED)


def _same_file(a: Path, b: Path) -> bool:
    return a.resolve() == b.resolve()


def _check_draft_target(repo: CriteriaRepository, draft: RoleCriteria, target: Path,
                        criteria_dir: Path, force: bool) -> None:
    if target.exists() and not force:
        raise CommandBlocked(f"{target} already exists; use --force to overwrite it")
    if not (criteria_dir.is_dir() and _same_file(target.parent, criteria_dir)):
        return
    # A second file in the criteria directory for the same posting or role would make
    # both files fail to load, silently disabling the existing (maybe approved) criteria.
    clashes: list[tuple[str, RoleCriteria | None]] = [
        (f"posting {draft.posting_ids[0]}", repo.for_posting(draft.posting_ids[0])),
        (f"role_id {draft.role_id}", repo.by_role(draft.role_id)),
    ]
    for what, existing in clashes:
        path = repo.path_for(existing.role_id) if existing is not None else None
        if path is not None and not _same_file(path, target):
            raise CommandBlocked(
                f"{what} already has criteria in {path}; a second file in {criteria_dir} "
                "would make both fail to load. Edit that file, or write the draft "
                "elsewhere with --out")


async def _draft(settings: Settings, posting_id: str, out: Path | None, force: bool) -> Path:
    async with _components(settings, need_llm=False) as components:
        lever = _require_lever(components)
        posting = await lever.get_posting(posting_id)
        draft = draft_from_posting(posting)
        target = out or settings.criteria_dir / f"{draft.role_id}.yaml"
        _check_draft_target(components.criteria, draft, target, settings.criteria_dir, force)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(dump_criteria_yaml(draft, header=FIELD_GUIDE), encoding="utf-8")
        return target


@criteria_app.command("draft")
def criteria_draft(
    posting: Annotated[str, typer.Option("--posting", help="Lever posting id.")],
    out: Annotated[Path | None, typer.Option(
        "--out", dir_okay=False,
        help="Output file (default: AR_CRITERIA_DIR/<role_id>.yaml).")] = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Draft a skeleton criteria file from a Lever posting (unapproved, for editing)."""
    with _handle_errors():
        settings = _load_settings()
        _banner()
        target = _run(_draft(settings, posting, out, force))
        typer.echo(f"wrote {target}")
        typer.echo("Next: rewrite it with the hiring manager, run 'applicant-ranking criteria "
                   "lint', then have TA/HR approve it (approved_by / approved_at).")


# ------------------------------------------------------------------------ postings


def _criteria_status(repo: CriteriaRepository, posting_id: str) -> str:
    criteria = repo.for_posting(posting_id)
    if criteria is None:
        return "-"
    state = "approved" if criteria.approved else "unapproved"
    return f"{criteria.role_id} v{criteria.version} ({state})"


async def _postings(settings: Settings, state: str | None, *,
                    include_confidential: bool = False) -> list[list[object]]:
    """Published postings are listed from both job sites (public and internal-only).
    Confidential postings appear only when asked for, and only with a key that has
    confidential access (Lever answers an access error otherwise)."""
    async with _components(settings, need_llm=False) as components:
        lever = _require_lever(components)
        rows: list[list[object]] = []
        postings = lever.list_postings(state=state,
                                       confidentiality="all" if include_confidential else None)
        async for posting in postings:
            rows.append([posting.id, posting.text, posting.state,
                         "; ".join(posting.locations),
                         _criteria_status(components.criteria, posting.id)])
        return rows


@postings_app.command("list")
def postings_list(
    state: Annotated[str, typer.Option(
        "--state", help="published, internal, closed, draft, pending, rejected or 'all'."),
    ] = "published",
    include_confidential: Annotated[bool, typer.Option(
        "--include-confidential",
        help="Also list confidential postings (needs an API key with confidential access; "
             "Lever hides them by default). Default: AR_LEVER_CONFIDENTIAL_ACCESS.")] = False,
) -> None:
    """List Lever postings and whether criteria exist for them."""
    with _handle_errors():
        settings = _load_settings()
        _banner()
        rows = _run(_postings(
            settings, None if state == "all" else state,
            include_confidential=include_confidential or settings.lever_confidential_access))
        if not rows:
            typer.echo(f"no postings (state: {state})")
            return
        typer.echo(_table(["ID", "TITLE", "STATE", "LOCATION", "CRITERIA"], rows))


# ------------------------------------------------------------------------ ranking

ModeOpt = Annotated[ModeOption, typer.Option(
    "--mode", case_sensitive=False,
    help="dry-run (default: plan the write-back, send nothing), live (send it; needs "
         "AR_WRITE_ENABLED=true) or evaluation (watermarked, never written back).")]
ForceOpt = Annotated[bool, typer.Option(
    "--force", help="Re-evaluate even when a cached result exists.")]
FormatOpt = Annotated[ReportFormat, typer.Option("--format", case_sensitive=False,
                                                 help="Report format.")]
OutOpt = Annotated[Path | None, typer.Option(
    "--out", dir_okay=False, help="Write the report to this file (e.g. reports/...).")]
YesOpt = Annotated[bool, typer.Option("--yes", "-y", help="Don't ask for confirmation.")]


async def _rank(settings: Settings, posting_id: str, mode: RunMode, limit: int | None,
                include_archived: bool, force: bool) -> RankingReport:
    async with _components(settings) as components:
        return await components.ranker.rank_posting(
            posting_id, mode=mode, limit=limit, include_archived=include_archived, force=force)


@app.command()
def rank(
    posting: Annotated[str, typer.Option("--posting", help="Lever posting id.")],
    mode: ModeOpt = ModeOption.DRY_RUN,
    limit: Annotated[int | None, typer.Option(
        "--limit", min=1, help="Evaluate at most N applications.")] = None,
    include_archived: Annotated[bool, typer.Option(
        "--include-archived", help="Include archived applications.")] = False,
    force: ForceOpt = False,
    fmt: FormatOpt = ReportFormat.TABLE,
    out: OutOpt = None,
    yes: YesOpt = False,
) -> None:
    """Rank every application for a posting (advisory: nobody is rejected or hidden)."""
    with _handle_errors():
        settings = _load_settings()
        run_mode = mode.run_mode
        _confirm_live(settings, run_mode, yes, f"every application to posting {posting}")
        report = _run(_rank(settings, posting, run_mode, limit, include_archived, force))
        _emit_report(report, fmt, out)


async def _evaluate(settings: Settings, opportunity_id: str, mode: RunMode,
                    force: bool) -> RankingRow:
    async with _components(settings) as components:
        return await components.ranker.evaluate_opportunity(opportunity_id, mode=mode,
                                                            force=force)


@app.command()
def evaluate(
    opportunity: Annotated[str, typer.Option("--opportunity", help="Lever opportunity id.")],
    mode: ModeOpt = ModeOption.DRY_RUN,
    force: ForceOpt = False,
    fmt: FormatOpt = ReportFormat.TABLE,
    out: OutOpt = None,
    yes: YesOpt = False,
) -> None:
    """Evaluate one applicant."""
    with _handle_errors():
        settings = _load_settings()
        run_mode = mode.run_mode
        _confirm_live(settings, run_mode, yes, f"opportunity {opportunity}")
        row = _run(_evaluate(settings, opportunity, run_mode, force))
        evaluation = row.evaluation
        report = RankingReport(posting_id=evaluation.posting_id or "-", mode=run_mode,
                               criteria_role_id=evaluation.criteria_role_id,
                               criteria_version=evaluation.criteria_version, rows=[row])
        _emit_report(report, fmt, out)


async def _evaluate_local(settings: Settings, criteria: RoleCriteria, data: bytes,
                          file_name: str, names: Sequence[str]) -> Evaluation:
    async with _components(settings, need_lever=False) as components:
        return await components.ranker.evaluate_local(
            criteria=criteria, data=data, file_name=file_name, candidate_names=names)


ResumeArg = Annotated[Path, typer.Argument(
    exists=True, dir_okay=False, readable=True, help="Resume file (PDF, DOCX, DOC, RTF, "
    "TXT, ODT or image).")]
CriteriaOpt = Annotated[Path, typer.Option(
    "--criteria", exists=True, dir_okay=False, readable=True, help="Criteria YAML file.")]


@app.command("evaluate-file")
def evaluate_file(
    resume: ResumeArg,
    criteria_file: CriteriaOpt,
    name: Annotated[str | None, typer.Option(
        "--name", help="Candidate's full name, so it is redacted as in real runs; without "
        "it the name in the resume reaches the model unredacted.")] = None,
    fmt: FormatOpt = ReportFormat.TABLE,
    out: OutOpt = None,
) -> None:
    """Evaluate a local resume against a criteria file (EVALUATION mode; no Lever)."""
    with _handle_errors():
        settings = _load_settings()
        criteria = load_criteria_file(criteria_file)
        data = _read_resume(resume, settings.max_resume_bytes)
        names = [name] if name and name.strip() else []
        if not names:
            _echo_err(NO_NAME_WARNING)
        evaluation = _run(_evaluate_local(settings, criteria, data, resume.name, names))
        row = RankingRow(rank=1 if evaluation.is_ranked else None, evaluation=evaluation)
        report = RankingReport(posting_id="local", posting_title=criteria.title,
                               criteria_role_id=criteria.role_id,
                               criteria_version=criteria.version, mode=RunMode.EVALUATION,
                               rows=[row])
        _emit_report(report, fmt, out)


# ------------------------------------------------------------------- backtest/fairness


async def _backtest(settings: Settings, posting_id: str, stages: Sequence[str],
                    limit: int | None, force: bool) -> BacktestReport:
    async with _components(settings) as components:
        lever = _require_lever(components)
        try:
            return await run_backtest(components.ranker, lever, posting_id,
                                      positive_stage_names=stages, limit=limit, force=force)
        except ValueError as exc:  # stage names (checked before any model call)
            raise CommandBlocked(str(exc)) from exc


def _fmt_float(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _render_backtest(report: BacktestReport) -> str:
    m = report.metrics
    lines = [
        _watermark(RunMode.EVALUATION),
        f"Backtest of posting {report.posting_id} "
        f"(positive stages: {', '.join(report.positive_stage_names)})",
        "",
        f"  ranked applications       {m.n}",
        f"  positives                 {m.positives}",
        f"  not ranked                {m.unranked}",
        f"  AUC                       {_fmt_float(m.auc)}",
    ]
    lines.extend(f"  precision@{k:<15} {_fmt_float(v)}" for k, v in sorted(m.precision_at.items()))
    lines.append(f"  mean rank of positives    {_fmt_float(m.mean_rank_positive, 1)}")
    lines.append(f"  median rank of positives  {_fmt_float(m.median_rank_positive, 1)}")
    if m.tier_positive_rate:
        lines.append("  positive rate by tier:")
        lines.extend(f"    Tier {tier}: {count} ranked, {rate:.0%} positive"
                     for tier, (count, rate) in sorted(m.tier_positive_rate.items()))
    lines.append("")
    lines.append("Caveats:")
    lines.extend(f"  - {caveat}" for caveat in report.caveats)
    return "\n".join(lines) + "\n"


@app.command()
def backtest(
    posting: Annotated[str, typer.Option("--posting", help="Lever posting id.")],
    positive_stage: Annotated[list[str], typer.Option(
        "--positive-stage", help="Lever stage name that counts as a positive outcome "
        "(repeat for several).")],
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
    force: ForceOpt = False,
    out: Annotated[Path | None, typer.Option(
        "--out", dir_okay=False, help="Write the full report (JSON) to this file.")] = None,
) -> None:
    """Compare the ranking with past recruiter decisions (EVALUATION mode)."""
    with _handle_errors():
        settings = _load_settings()
        report = _run(_backtest(settings, posting, positive_stage, limit, force))
        typer.echo(_render_backtest(report), nl=False)
        if out is not None:
            _write_private(out, _dump_json(report, _watermark(RunMode.EVALUATION)))
            typer.echo(f"wrote {out}")


async def _resume_text(components: Components, data: bytes) -> str:
    result = await components.extractor.extract(data)
    if result.error:
        raise RuntimeError(f"could not read the resume: {result.error}")
    if result.needs_vision or not result.visible_text.strip():
        raise RuntimeError("the resume has no text layer (scanned or image-only); the "
                           "fairness check needs a text resume")
    return result.visible_text


async def _fairness(settings: Settings, criteria: RoleCriteria, data: bytes, runs: int,
                    tolerance: float, name: str | None) -> FairnessReport:
    async with _components(settings, need_lever=False) as components:
        text = await _resume_text(components, data)
        if name:
            return await run_fairness(components.ranker, criteria, text, runs=runs,
                                      tolerance=tolerance, base_name=name)
        return await run_fairness(components.ranker, criteria, text, runs=runs,
                                  tolerance=tolerance)


def _render_fairness(report: FairnessReport) -> str:
    rows = [[r.variant, r.group, _fmt_float(r.mean, 1), _fmt_float(r.std, 1),
             _fmt_float(r.delta_vs_base, 1), r.modal_tier,
             "FLAGGED" if r.flagged else "ok", r.reason]
            for r in report.results]
    flagged = sum(1 for r in report.results if r.flagged)
    verdict = ("PASSED: no variant moved the score beyond the tolerance" if report.passed
               else f"FLAGGED: {flagged} variant(s) moved the score beyond the tolerance; "
                    "review the criteria before approval")
    lines = [
        _watermark(RunMode.EVALUATION),
        f"Counterfactual fairness check: {report.criteria_role_id} v{report.criteria_version}"
        f" (runs={report.runs}, tolerance={report.tolerance:g})",
        "",
        _table(["VARIANT", "GROUP", "MEAN", "STD", "DELTA", "TIER", "RESULT", "REASON"],
               rows),
        "",
        verdict,
    ]
    if report.caveats:
        lines.append("")
        lines.append("Caveats:")
        lines.extend(f"  - {caveat}" for caveat in report.caveats)
    return "\n".join(lines) + "\n"


@app.command()
def fairness(
    resume: ResumeArg,
    criteria_file: CriteriaOpt,
    runs: Annotated[int, typer.Option("--runs", min=1, max=20,
                                      help="Evaluations per variant.")] = 3,
    tolerance: Annotated[float, typer.Option(
        "--tolerance", min=0, help="Allowed mean score change (points).")] = 5.0,
    name: Annotated[str | None, typer.Option(
        "--name", help="The name written in the resume: the name variants replace it and "
        "it is redacted as in real runs. Without it the resume's own name reaches the "
        "model unredacted and the name variants test nothing (a placeholder is added).",
    )] = None,
    out: Annotated[Path | None, typer.Option(
        "--out", dir_okay=False, help="Write the full report (JSON) to this file.")] = None,
) -> None:
    """Counterfactual fairness check (EVALUATION mode); exit 2 when a variant is flagged."""
    with _handle_errors():
        settings = _load_settings()
        criteria = load_criteria_file(criteria_file)
        data = _read_resume(resume, settings.max_resume_bytes)
        base_name = name.strip() if name and name.strip() else None
        if base_name is None:
            _echo_err(NO_NAME_FAIRNESS_WARNING)
        report = _run(_fairness(settings, criteria, data, runs, tolerance, base_name))
        typer.echo(_render_fairness(report), nl=False)
        if out is not None:
            _write_private(out, _dump_json(report, _watermark(RunMode.EVALUATION)))
            typer.echo(f"wrote {out}")
        if not report.passed:
            raise typer.Exit(EXIT_BLOCKED)


# ------------------------------------------------------------------------ audit & data


@app.command("export-audit")
def export_audit(
    out: Annotated[Path, typer.Option("--out", dir_okay=False, help="CSV file to write.")],
    posting: Annotated[str | None, typer.Option("--posting",
                                                help="Only this posting.")] = None,
) -> None:
    """Export the pseudonymous audit log (CSV) for adverse-impact analysis."""
    with _handle_errors():
        settings = _load_settings()
        _banner()
        rows: list[dict[str, Any]] = []
        if settings.state_db.exists():
            store = _open_store(settings)
            try:
                rows = store.audit_rows(posting_id=posting)
            finally:
                store.close()
        _write_private(out, audit_export_csv(rows))
        typer.echo(f"wrote {len(rows)} audit row(s) to {out}")


async def _purge(settings: Settings, reconcile_days: int,
                 ) -> tuple[int, list[PurgeResult], list[PurgeResult]]:
    """(expired evaluations removed, reconciliation results, tombstoned purges). The
    tombstoned purge (deletions a legal hold suspended, now released) needs no Lever
    access and always runs."""
    async with _components(settings, need_llm=False,
                           need_lever=reconcile_days > 0) as components:
        removed = components.store.purge_expired(now=utcnow(),
                                                 retention_days=settings.retention_days)
        results: list[PurgeResult] = []
        if reconcile_days > 0:
            results = await components.ranker.reconcile_deletions(days=reconcile_days)
        tombstoned = components.store.purge_tombstoned()
        return removed, results, tombstoned


def _tombstoned_line(results: Sequence[PurgeResult], *, released: str) -> str:
    held = sum(1 for r in results if r.on_hold)
    purged = [r for r in results if not r.on_hold]
    rows = sum(r.pii_rows_deleted for r in purged)
    return (f"deleted candidates: {len(purged)} purged ({rows} PII row(s)) {released}, "
            f"{held} still on legal hold")


@app.command()
def purge(
    reconcile_days: Annotated[int, typer.Option(
        "--reconcile-days", min=0, max=3650,
        help="Also purge candidates deleted in Lever (via GET /opportunities/deleted) in "
             "the last N days (0: skip).")] = 30,
    yes: YesOpt = False,
) -> None:
    """Retention purge plus deletion reconciliation (legal holds are respected)."""
    with _handle_errors():
        settings = _load_settings()
        _banner()
        if not yes:
            what = f"evaluations older than {settings.retention_days} days"
            if reconcile_days:
                what += (f" and the data of candidates deleted in Lever in the last "
                         f"{reconcile_days} days")
            typer.confirm(f"Permanently delete {what}? Records on legal hold are kept.",
                          abort=True, err=True)
        removed, results, tombstoned = _run(_purge(settings, reconcile_days))
        typer.echo(f"retention: {removed} evaluation(s) older than {settings.retention_days} "
                   "days removed")
        if reconcile_days:
            held = [r for r in results if r.on_hold]
            purged = [r for r in results if not r.on_hold]
            rows = sum(r.pii_rows_deleted for r in purged)
            typer.echo(f"deletions: {len(results)} candidate(s) deleted in Lever in the last "
                       f"{reconcile_days} days; {len(purged)} purged ({rows} PII row(s)), "
                       f"{len(held)} kept on legal hold")
        else:
            typer.echo("deletions: reconciliation skipped (--reconcile-days 0)")
        if tombstoned:
            typer.echo(_tombstoned_line(tombstoned,
                                        released="whose deletion a legal hold had suspended"))


def _hold_scope(posting: str | None, opportunity: str | None) -> tuple[str, str]:
    posting = (posting or "").strip() or None
    opportunity = (opportunity or "").strip() or None
    if (posting is None) == (opportunity is None):
        raise typer.BadParameter("pass exactly one of --posting or --opportunity")
    if posting is not None:
        return "posting", posting
    assert opportunity is not None
    return "opportunity", opportunity


def _current_user() -> str:
    try:
        return getpass.getuser()
    except (OSError, KeyError):
        return "unknown"


HoldPostingOpt = Annotated[str | None, typer.Option("--posting", help="Lever posting id.")]
HoldOpportunityOpt = Annotated[str | None, typer.Option("--opportunity",
                                                        help="Lever opportunity id.")]
HoldByOpt = Annotated[str | None, typer.Option("--by", help="Who (default: the OS user).")]


@hold_app.command("set")
def hold_set(
    reason: Annotated[str, typer.Option("--reason", help="Why (e.g. the matter reference).")],
    posting: HoldPostingOpt = None,
    opportunity: HoldOpportunityOpt = None,
    by: HoldByOpt = None,
) -> None:
    """Put a posting or an opportunity on legal hold (suspends every purge)."""
    with _handle_errors():
        scope, scope_id = _hold_scope(posting, opportunity)
        if not reason.strip():
            raise typer.BadParameter("--reason must not be empty")
        settings = _load_settings()
        _banner()
        set_by = (by or "").strip() or _current_user()
        store = _open_store(settings)
        try:
            store.set_legal_hold(scope=scope, scope_id=scope_id,  # type: ignore[arg-type]
                                 reason=reason.strip(), set_by=set_by, now=utcnow())
        finally:
            store.close()
        typer.echo(f"legal hold set on {scope} {scope_id} by {set_by}; purges of its records "
                   "are suspended")


@hold_app.command("clear")
def hold_clear(
    posting: HoldPostingOpt = None,
    opportunity: HoldOpportunityOpt = None,
    reason: Annotated[str | None, typer.Option("--reason", help="Why (logged).")] = None,
    by: HoldByOpt = None,
) -> None:
    """Release a legal hold."""
    with _handle_errors():
        scope, scope_id = _hold_scope(posting, opportunity)
        settings = _load_settings()
        _banner()
        cleared_by = (by or "").strip() or _current_user()
        store = _open_store(settings)
        results: list[PurgeResult] = []
        try:
            cleared = store.clear_legal_hold(scope=scope, scope_id=scope_id,  # type: ignore[arg-type]
                                             purge_deleted=False)
            if cleared:
                # Complete the deletions the hold suspended, and say what happened.
                results = store.purge_tombstoned()
        finally:
            store.close()
        if cleared:
            logger.info("legal hold on %s %s cleared by %s (reason: %s)", scope, scope_id,
                        cleared_by, reason or "-")
            typer.echo(f"legal hold on {scope} {scope_id} cleared by {cleared_by}")
            if results:
                typer.echo(_tombstoned_line(results, released="now that the hold is released"))
        else:
            typer.echo(f"no legal hold was set on {scope} {scope_id}")


# ------------------------------------------------------------------------ doctor


def _mask(value: Any) -> str:
    """Display form of a setting value; secrets never appear."""
    if isinstance(value, SecretStr | SecretBytes):
        return "set (hidden)" if value.get_secret_value() else "empty"
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        if not value:
            return "(none)"
        return ", ".join(f"{k}={_mask(v)}" for k, v in value.items())
    if isinstance(value, list | tuple | set):
        return ", ".join(_mask(v) for v in value) or "(none)"
    if isinstance(value, TierThresholds):
        return f"A>={value.a}, B>={value.b}, C>={value.c}"
    return str(value)


class _Doctor:
    def __init__(self) -> None:
        self.problems: list[str] = []
        self.warnings: list[str] = []

    def section(self, title: str) -> None:
        typer.echo("")
        typer.echo(title)

    def item(self, label: str, value: object) -> None:
        typer.echo(f"  {label:<30} {value}")

    def problem(self, message: str) -> None:
        self.problems.append(message)
        typer.echo(f"  PROBLEM: {message}")

    def warning(self, message: str) -> None:
        self.warnings.append(message)
        typer.echo(f"  warning: {message}")


def _doctor_settings(doc: _Doctor, settings: Settings) -> None:
    doc.section("Configuration (AR_* environment and .env; secrets hidden)")
    for name in type(settings).model_fields:
        doc.item(f"AR_{name.upper()}", _mask(getattr(settings, name)))
    if settings.hmac_key is None or not settings.hmac_key.get_secret_value():
        if settings.env == "prod":
            doc.problem("AR_HMAC_KEY is required outside dev (audit fingerprints)")
        else:
            doc.warning(f"AR_HMAC_KEY is not set: env={settings.env} uses the built-in "
                        "dev-only key")


def _doctor_services(doc: _Doctor, settings: Settings) -> None:
    doc.section("Gemini on Vertex AI (credentials are not checked here)")
    doc.item("project", settings.gcp_project or "NOT SET")
    doc.item("location", settings.gcp_location)
    doc.item("model", f"{settings.gemini_model} (thinking: {settings.gemini_thinking_level})")
    doc.item("credentials", "Application Default Credentials, resolved at run time")
    if not settings.gcp_project:
        doc.warning("AR_GCP_PROJECT is not set: rank, evaluate, evaluate-file, backtest and "
                    "fairness need it")

    doc.section("Lever")
    doc.item("base URL", settings.lever_base_url)
    doc.item("API key", "NOT SET" if settings.lever_api_key is None
             else _mask(settings.lever_api_key))
    doc.item("write-back", "ENABLED for --mode live" if settings.write_enabled
             else "disabled (planned and reported only)")
    for event in sorted(HANDLED_EVENTS):
        configured = settings.webhook_token(event) is not None
        doc.item(f"webhook {event}", "token configured" if configured else "NOT SET")
        if not configured:
            doc.warning(f"no webhook signing token for {event}: the service rejects it (401)")
    if settings.lever_api_key is None or not settings.lever_api_key.get_secret_value():
        doc.warning("AR_LEVER_API_KEY is not set: rank, evaluate, backtest, postings, "
                    "criteria draft and purge (reconciliation) need it")


def _doctor_lexicon(doc: _Doctor, settings: Settings) -> ProtectedTermLexicon | None:
    doc.section("Protected-term lexicon")
    try:
        lexicon = ProtectedTermLexicon.load(settings.protected_terms_file)
    except ConfigError as exc:
        doc.problem(_describe(exc))
        return None
    doc.item("version", lexicon.version)
    doc.item("source", lexicon.source)
    doc.item("terms", lexicon.term_count)
    return lexicon


def _doctor_criteria(doc: _Doctor, settings: Settings,
                     lexicon: ProtectedTermLexicon | None) -> None:
    directory = settings.criteria_dir
    doc.section(f"Criteria ({directory})")
    if not directory.is_dir():
        doc.warning(f"criteria directory not found: {directory}")
        return
    repo = CriteriaRepository(directory)
    repo.load()
    doc.item("files loaded", len(repo.all()))
    for message in repo.errors:
        doc.problem(message)
    for criteria in repo.all():
        approval = "approved" if criteria.approved else "NOT approved"
        postings = ", ".join(criteria.posting_ids) or "no postings"
        lint = "not linted (no lexicon)"
        if lexicon is not None:
            issues = lint_criteria(criteria, lexicon)
            n_err = sum(1 for i in issues if i.level == "error")
            lint = f"{n_err} lint error(s), {len(issues) - n_err} warning(s)"
            if has_errors(issues):
                doc.problem(f"{criteria.role_id}: {n_err} lint error(s); run 'applicant-ranking"
                            " criteria lint'")
        doc.item(f"{criteria.role_id} v{criteria.version}", f"{approval}; {postings}; {lint}")


def _doctor_policy(doc: _Doctor, settings: Settings) -> None:
    path = settings.policy_file
    doc.section(f"Jurisdiction policy ({path})")
    if not path.exists():
        doc.item("file", "MISSING")
        doc.warning("no policy file: DRY_RUN and LIVE runs are blocked (EVALUATION still "
                    "works); see config/policy.example.yaml")
        return
    try:
        engine = PolicyEngine.from_file(path)
    except ValueError as exc:
        doc.problem(_describe(exc))
        return
    config = engine.config
    assert config is not None
    doc.item("file", "present")
    doc.item("version", config.version)
    doc.item("approved", f"yes, by {config.approved_by} on {config.approved_at}"
             if engine.approved else "NO")
    doc.item("allowed countries", ", ".join(config.allowed_countries) or "(none)")
    if not engine.approved:
        doc.warning("the policy file is not approved: DRY_RUN and LIVE runs are blocked")


def _doctor_state(doc: _Doctor, settings: Settings) -> None:
    doc.section("State")
    doc.item("database", f"{settings.state_db} "
             f"({'exists' if settings.state_db.exists() else 'not created yet'})")
    doc.item("retention", f"{settings.retention_days} days")
    doc.item("store redacted resume text", _mask(settings.store_resume_text))


@app.command()
def doctor() -> None:
    """Show the effective configuration (secrets hidden); no network, no credential files.

    Exit 3 when a problem is found (warnings alone exit 0)."""
    with _handle_errors():
        settings = _load_settings()
        _banner()
        doc = _Doctor()
        _doctor_settings(doc, settings)
        _doctor_services(doc, settings)
        lexicon = _doctor_lexicon(doc, settings)
        _doctor_criteria(doc, settings, lexicon)
        _doctor_policy(doc, settings)
        _doctor_state(doc, settings)
        typer.echo("")
        if doc.problems:
            typer.echo(f"{len(doc.problems)} problem(s), {len(doc.warnings)} warning(s)")
            raise typer.Exit(EXIT_CONFIG)
        typer.echo(f"No problems found ({len(doc.warnings)} warning(s)).")


# ------------------------------------------------------------------------ service


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Interface to bind (0.0.0.0 in a "
                                      "container).")] = _LOOPBACK,
    port: Annotated[int, typer.Option("--port", envvar="PORT", min=1, max=65535,
                                      help="Port (default: $PORT, else 8080).")] = 8080,
) -> None:
    """Run the Lever webhook service (uvicorn, one process: the job queue is in-process)."""
    with _handle_errors():
        settings = _load_settings()
        _banner()
        import uvicorn

        from applicant_ranking.service.app import create_app

        debug = _debug_enabled()
        if not debug:
            logging.getLogger().setLevel(logging.INFO)
        uvicorn.run(create_app(settings), host=host, port=port,
                    log_level="debug" if debug else "info", server_header=False)


def main() -> None:
    app(prog_name="applicant-ranking")
