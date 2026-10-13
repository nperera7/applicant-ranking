"""Dispatch a resume file to the right extractor, with size caps and process isolation."""

from __future__ import annotations

import asyncio
import functools
import logging
import multiprocessing
import platform
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from applicant_ranking.config import Settings
from applicant_ranking.domain import ExtractionResult, FileKind, Flag, FlagCode
from applicant_ranking.resumes.extract_doc import extract_doc
from applicant_ranking.resumes.extract_docx import extract_docx
from applicant_ranking.resumes.extract_pdf import extract_pdf
from applicant_ranking.resumes.extract_rtf import extract_rtf
from applicant_ranking.resumes.extract_text import (
    clean_unicode,
    extract_odt,
    extract_txt,
    unreadable,
)
from applicant_ranking.resumes.image import extract_image
from applicant_ranking.resumes.sniff import sniff

logger = logging.getLogger(__name__)

TIMEOUT_ERROR = "extraction timed out"
DEFAULT_MAX_TASKS_PER_CHILD = 20
# Address-space cap for an extraction worker: a hostile DOCX/ODT inflates to roughly 25x
# its XML size as an lxml tree, and the sniffer's 8 MB XML cap leaves this much headroom
# for the PDF renders (8M greyscale pixels per page) and the libraries themselves.
WORKER_MEMORY_LIMIT_BYTES = 1024 * 1024 * 1024

Worker = Callable[..., ExtractionResult]


def _limit_worker_memory() -> None:
    """Pool initializer: cap the worker's address space on Linux. RLIMIT_AS is not
    enforced reliably on macOS, and Windows has no equivalent, so elsewhere this is a
    no-op and the sniffer's size caps are the only guard."""
    if platform.system() != "Linux":
        return
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (WORKER_MEMORY_LIMIT_BYTES,
                                                WORKER_MEMORY_LIMIT_BYTES))
    except (ImportError, ValueError, OSError) as exc:  # a stricter limit may already apply
        logger.info("could not cap extraction worker memory (%s)", type(exc).__name__)


def extract_resume_sync(data: bytes, *, max_bytes: int, max_pdf_pages: int,
                        vision_max_pages: int) -> ExtractionResult:
    """Pure function (safe to run in a worker process): enforce ``max_bytes`` (-> error +
    ``unreadable_file`` review flag), ``sniff`` the type, call the matching extractor.
    Unknown/unsupported type -> ``unsupported_file`` (review) with ``error`` set.
    Any unexpected exception from an extractor is caught and reported the same way as an
    unreadable file (never propagates). An empty file is unreadable. Text is returned as
    valid Unicode (no lone surrogates)."""
    kind: FileKind = "unknown"
    try:
        if len(data) > max_bytes:
            return _too_large(len(data), max_bytes)
        if not data:
            return unreadable("unknown", "file is empty")
        kind = sniff(data)
        result = _dispatch(kind, data, max_pdf_pages=max_pdf_pages,
                           vision_max_pages=vision_max_pages)
        return result.model_copy(update={
            "visible_text": clean_unicode(result.visible_text),
            "hidden_snippets": [clean_unicode(s) for s in result.hidden_snippets],
        })
    except Exception as exc:  # never propagate: a hostile file must not break the batch
        logger.warning("resume extraction failed (kind=%s, %s)", kind, type(exc).__name__)
        return unreadable(kind, f"could not read {kind} file ({type(exc).__name__})")


def _dispatch(kind: FileKind, data: bytes, *, max_pdf_pages: int,
              vision_max_pages: int) -> ExtractionResult:
    match kind:
        case "pdf":
            return extract_pdf(data, max_pages=max_pdf_pages, vision_max_pages=vision_max_pages)
        case "docx":
            return extract_docx(data)
        case "doc":
            return extract_doc(data)
        case "rtf":
            return extract_rtf(data)
        case "txt":
            return extract_txt(data)
        case "odt":
            return extract_odt(data)
        case "image":
            return extract_image(data)
    error = "unsupported or unrecognised file type"
    return ExtractionResult(
        kind="unknown", error=error,
        flags=[Flag(code=FlagCode.UNSUPPORTED_FILE, severity="review", detail=error)],
    )


def _too_large(size: int, max_bytes: int) -> ExtractionResult:
    return unreadable("unknown", f"file is too large ({size} bytes; limit {max_bytes})")


def _timed_out() -> ExtractionResult:
    return unreadable("unknown", TIMEOUT_ERROR)


def _kill_pool(pool: ProcessPoolExecutor, *, cancel_futures: bool = False) -> None:
    """Stop a pool whose worker may be stuck on a hostile file. Work still queued on it
    fails with BrokenProcessPool or is cancelled; ``_run_in_pool`` retries both."""
    kill_workers = getattr(pool, "kill_workers", None)  # Python 3.14+ (cancels queued work)
    if callable(kill_workers):
        kill_workers()
        return
    processes = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=cancel_futures)
    for process in processes:  # Python 3.13 has no public API to stop a running worker
        if process.is_alive():
            process.kill()
    for process in processes:
        process.join(timeout=5)


class ResumeExtractor:
    """Runs ``extract_resume_sync`` in a ``ProcessPoolExecutor`` (``max_tasks_per_child``
    set, so a leaky/hostile file can't poison a worker for long) with a timeout, or inline
    when ``in_subprocess`` is False (tests). A timeout yields an ExtractionResult with
    ``error="extraction timed out"`` and an ``unreadable_file`` review flag.

    The pool uses the "spawn" start method and is created on first use; each worker caps
    its own address space at ``WORKER_MEMORY_LIMIT_BYTES`` where the platform enforces
    that (Linux). At most ``max_workers`` tasks are submitted at once; the rest wait for
    a free worker, and that wait doesn't count toward the timeout. A timed-out worker
    can't be cancelled, so the pool is killed and replaced; a task that loses its worker
    that way (or to a crash) is retried once on the new pool. "Inline" runs in a worker
    thread of this process, so the
    event loop stays responsive and the timeout still applies (PDFium calls are serialised
    by ``PDFIUM_LOCK``). ``worker`` is injectable for tests; in subprocess mode it must be
    a picklable module-level function.
    """

    def __init__(self, *, max_bytes: int, max_pdf_pages: int, vision_max_pages: int,
                 timeout_seconds: float, in_subprocess: bool = True,
                 max_workers: int = 2,
                 max_tasks_per_child: int = DEFAULT_MAX_TASKS_PER_CHILD,
                 worker: Worker = extract_resume_sync) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_workers < 1 or max_tasks_per_child < 1:
            raise ValueError("max_workers and max_tasks_per_child must be at least 1")
        self.max_bytes = max_bytes
        self.max_pdf_pages = max_pdf_pages
        self.vision_max_pages = vision_max_pages
        self.timeout_seconds = timeout_seconds
        self.in_subprocess = in_subprocess
        self.max_workers = max_workers
        self.max_tasks_per_child = max_tasks_per_child
        self._worker = worker
        self._pool: ProcessPoolExecutor | None = None
        self._slots_for: tuple[asyncio.AbstractEventLoop, asyncio.Semaphore] | None = None
        self._closed = False

    @classmethod
    def from_settings(cls, settings: Settings) -> ResumeExtractor:
        return cls(
            max_bytes=settings.max_resume_bytes,
            max_pdf_pages=settings.max_pdf_pages,
            vision_max_pages=settings.vision_max_pages,
            timeout_seconds=settings.extraction_timeout_seconds,
            in_subprocess=settings.extraction_in_subprocess,
            max_workers=settings.extraction_max_workers,
        )

    async def extract(self, data: bytes) -> ExtractionResult:
        if self._closed:
            raise RuntimeError("ResumeExtractor is closed")
        if len(data) > self.max_bytes:  # don't ship oversized files to a worker
            return _too_large(len(data), self.max_bytes)
        call = functools.partial(self._worker, data, max_bytes=self.max_bytes,
                                 max_pdf_pages=self.max_pdf_pages,
                                 vision_max_pages=self.vision_max_pages)
        if self.in_subprocess:
            return await self._run_in_pool(call)
        return await self._run_inline(call)

    async def _run_inline(self, call: Callable[[], ExtractionResult]) -> ExtractionResult:
        try:
            return await asyncio.wait_for(asyncio.to_thread(call), self.timeout_seconds)
        except TimeoutError:
            logger.warning("resume extraction timed out after %.1fs", self.timeout_seconds)
            return _timed_out()
        except Exception as exc:
            logger.warning("resume extraction failed (%s)", type(exc).__name__)
            return unreadable("unknown", f"extraction failed ({type(exc).__name__})")

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=multiprocessing.get_context("spawn"),
                max_tasks_per_child=self.max_tasks_per_child,
                initializer=_limit_worker_memory,
            )
        return self._pool

    def _discard_pool(self, pool: ProcessPoolExecutor) -> None:
        if self._pool is pool:
            self._pool = None
        _kill_pool(pool)

    def _worker_slots(self) -> asyncio.Semaphore:
        """One slot per pool worker. Per event loop: asyncio primitives are loop-bound and
        one extractor can serve several ``asyncio.run`` calls."""
        loop = asyncio.get_running_loop()
        if self._slots_for is None or self._slots_for[0] is not loop:
            self._slots_for = (loop, asyncio.Semaphore(self.max_workers))
        return self._slots_for[1]

    async def _run_in_pool(self, call: Callable[[], ExtractionResult]) -> ExtractionResult:
        for attempt in range(2):
            # wait for a free worker here, not in the pool's queue, so only the run is timed
            async with self._worker_slots():
                if self._closed:
                    return unreadable("unknown", "extraction cancelled (extractor closed)")
                pool = self._get_pool()
                try:
                    future = asyncio.wrap_future(pool.submit(call))
                    return await asyncio.wait_for(future, self.timeout_seconds)
                except TimeoutError:
                    logger.warning("resume extraction timed out after %.1fs; restarting "
                                   "workers", self.timeout_seconds)
                    self._discard_pool(pool)
                    return _timed_out()
                except BrokenProcessPool:
                    logger.warning("extraction worker died (attempt %d)", attempt + 1)
                    self._discard_pool(pool)
                except asyncio.CancelledError:
                    task = asyncio.current_task()
                    if task is not None and task.cancelling():
                        raise  # our caller cancelled us
                    # the work item itself was cancelled: its pool was shut down under it
                    logger.warning("extraction work item was cancelled (attempt %d)",
                                   attempt + 1)
                    self._discard_pool(pool)
                except Exception as exc:  # e.g. pickling errors
                    logger.warning("resume extraction failed (%s)", type(exc).__name__)
                    return unreadable("unknown", f"extraction failed ({type(exc).__name__})")
        return unreadable("unknown", "extraction worker crashed")

    def close(self) -> None:
        self._closed = True
        pool, self._pool = self._pool, None
        if pool is not None:
            _kill_pool(pool, cancel_futures=True)
