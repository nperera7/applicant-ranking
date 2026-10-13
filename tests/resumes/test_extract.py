from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from concurrent.futures.process import BrokenProcessPool
from typing import ClassVar

import pytest

from applicant_ranking.domain import FlagCode
from applicant_ranking.resumes import extract as extract_module
from applicant_ranking.resumes.extract import (
    TIMEOUT_ERROR,
    ResumeExtractor,
    extract_resume_sync,
)

LIMITS = {"max_bytes": 60 * 1024 * 1024, "max_pdf_pages": 10, "vision_max_pages": 5}


def flags_by_code(result):
    return {flag.code: flag for flag in result.flags}


def make_extractor(**overrides) -> ResumeExtractor:
    options = {**LIMITS, "timeout_seconds": 30.0, "in_subprocess": False}
    options.update(overrides)
    return ResumeExtractor(**options)


# ----------------------------------------------------------------------------------------
# extract_resume_sync
# ----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def samples(pdfs, docxs, build_doc, build_odt, build_photo):
    return {
        "pdf": pdfs.one_column,
        "docx": docxs.full,
        "doc": build_doc([("Word 97 resume text\r", True)]),
        "rtf": rb"{\rtf1\ansi Rich text resume\par}",
        "txt": b"Plain text resume with Python",
        "odt": build_odt("<text:p>ODT resume</text:p>"),
        "image": build_photo(fmt="JPEG"),
    }


@pytest.mark.parametrize("kind", ["pdf", "docx", "doc", "rtf", "txt", "odt", "image"])
def test_dispatches_by_sniffed_type(samples, kind):
    result = extract_resume_sync(samples[kind], **LIMITS)

    assert result.kind == kind
    assert result.error is None
    assert result.visible_text or result.needs_vision


def test_xml_bomb_docx_is_unsupported_not_parsed(xml_bomb_docx, monkeypatch):
    monkeypatch.setattr(extract_module, "extract_docx",
                        lambda data: pytest.fail("the XML bomb reached the DOCX parser"))

    result = extract_resume_sync(xml_bomb_docx, **LIMITS)

    assert result.kind == "unknown"
    assert flags_by_code(result)[FlagCode.UNSUPPORTED_FILE].severity == "review"


def test_docx_with_large_image_is_still_extracted(docx_with_large_image):
    result = extract_resume_sync(docx_with_large_image, **LIMITS)

    assert result.kind == "docx"
    assert "large portrait photo" in result.visible_text


def test_worker_memory_limit_is_installed_as_pool_initializer():
    extractor = make_extractor(in_subprocess=True)
    pool = extractor._get_pool()
    try:
        assert pool._initializer is extract_module._limit_worker_memory
    finally:
        extractor.close()


def test_worker_memory_limit_is_harmless_where_unsupported(monkeypatch):
    monkeypatch.setattr(extract_module.platform, "system", lambda: "Darwin")
    extract_module._limit_worker_memory()  # a no-op outside Linux: must not raise


def test_worker_memory_limit_failure_is_logged_not_raised(monkeypatch, caplog):
    monkeypatch.setattr(extract_module.platform, "system", lambda: "Linux")
    import resource

    def refuse(*args):
        raise ValueError("limit too low")

    monkeypatch.setattr(resource, "setrlimit", refuse)
    with caplog.at_level(logging.INFO):
        extract_module._limit_worker_memory()
    assert "could not cap" in caplog.text


def test_docx_named_pdf_is_read_as_docx(docxs):
    result = extract_resume_sync(docxs.full, **LIMITS)

    assert result.kind == "docx"
    assert "BODYTEXT" in result.visible_text


def test_page_limits_are_passed_to_the_pdf_extractor(pdfs):
    result = extract_resume_sync(pdfs.scanned(pages=4), max_bytes=LIMITS["max_bytes"],
                                 max_pdf_pages=3, vision_max_pages=2)

    assert result.pages == 4
    assert len(result.page_images) == 2
    assert FlagCode.TRUNCATED in flags_by_code(result)


def test_oversized_file_is_rejected_before_sniffing(monkeypatch):
    monkeypatch.setattr(extract_module, "sniff", lambda data: pytest.fail("sniffed"))

    result = extract_resume_sync(b"x" * 11, max_bytes=10, max_pdf_pages=10, vision_max_pages=5)

    assert result.error == "file is too large (11 bytes; limit 10)"
    assert [(f.code, f.severity) for f in result.flags] == [(FlagCode.UNREADABLE_FILE, "review")]


def test_empty_file_is_unreadable():
    result = extract_resume_sync(b"", **LIMITS)

    assert result.error == "file is empty"
    assert FlagCode.UNREADABLE_FILE in flags_by_code(result)


@pytest.mark.parametrize("data", [b"\x00\x01\x02binary blob", b"BM\x00\x00bitmap"])
def test_unknown_type_is_unsupported(data):
    result = extract_resume_sync(data, **LIMITS)

    assert result.kind == "unknown"
    assert result.error == "unsupported or unrecognised file type"
    assert [(f.code, f.severity) for f in result.flags] == [(FlagCode.UNSUPPORTED_FILE, "review")]


def test_zip_bomb_is_unsupported_not_extracted(zip_bomb_size, monkeypatch):
    monkeypatch.setattr(extract_module, "extract_docx", lambda data: pytest.fail("extracted"))

    result = extract_resume_sync(zip_bomb_size, **LIMITS)

    assert FlagCode.UNSUPPORTED_FILE in flags_by_code(result)


def test_unexpected_exceptions_never_propagate(pdfs, monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("SECRET RESUME TEXT in an exception message")

    monkeypatch.setattr(extract_module, "extract_pdf", boom)
    with caplog.at_level(logging.DEBUG):
        result = extract_resume_sync(pdfs.one_column, **LIMITS)

    assert result.kind == "pdf"
    assert result.error == "could not read pdf file (RuntimeError)"
    assert [(f.code, f.severity) for f in result.flags] == [(FlagCode.UNREADABLE_FILE, "review")]
    assert "SECRET RESUME TEXT" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_text_is_returned_as_valid_unicode(monkeypatch):
    from applicant_ranking.domain import ExtractionResult

    monkeypatch.setattr(extract_module, "extract_txt", lambda data: ExtractionResult(
        kind="txt", visible_text="lone \ud800 surrogate", hidden_snippets=["\udfff"]))

    result = extract_resume_sync(b"plain text", **LIMITS)

    assert result.visible_text == "lone \ufffd surrogate"
    assert result.hidden_snippets == ["\ufffd"]
    assert result.model_dump_json()


def test_sniff_failure_is_contained(monkeypatch):
    monkeypatch.setattr(extract_module, "sniff", lambda data: (_ for _ in ()).throw(MemoryError))

    result = extract_resume_sync(b"anything", **LIMITS)

    assert result.kind == "unknown"
    assert result.error == "could not read unknown file (MemoryError)"


# ----------------------------------------------------------------------------------------
# ResumeExtractor
# ----------------------------------------------------------------------------------------


def test_from_settings(make_settings):
    settings = make_settings(max_resume_bytes=2048, max_pdf_pages=3, vision_max_pages=2,
                             extraction_timeout_seconds=12.5, extraction_in_subprocess=True)

    extractor = ResumeExtractor.from_settings(settings)

    assert extractor.max_bytes == 2048
    assert extractor.max_pdf_pages == 3
    assert extractor.vision_max_pages == 2
    assert extractor.timeout_seconds == 12.5
    assert extractor.in_subprocess is True
    extractor.close()


@pytest.mark.parametrize("kwargs", [{"timeout_seconds": 0}, {"max_workers": 0},
                                    {"max_tasks_per_child": 0}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        make_extractor(**kwargs)


async def test_inline_extraction(pdfs):
    extractor = make_extractor()

    result = await extractor.extract(pdfs.hidden("white_on_white"))

    assert result.kind == "pdf"
    assert "HIDDENMARK" not in result.visible_text
    assert FlagCode.HIDDEN_TEXT in flags_by_code(result)
    extractor.close()


async def test_worker_receives_the_limits():
    seen = {}

    def worker(data, **kwargs):
        seen.update(kwargs, data=data)
        return extract_resume_sync(data, **kwargs)

    extractor = make_extractor(worker=worker, max_pdf_pages=4, vision_max_pages=3)
    await extractor.extract(b"hello")

    assert seen == {"data": b"hello", "max_bytes": LIMITS["max_bytes"], "max_pdf_pages": 4,
                    "vision_max_pages": 3}


async def test_oversized_input_never_reaches_the_worker():
    def worker(data, **kwargs):
        raise AssertionError("worker called")

    extractor = make_extractor(worker=worker, max_bytes=10)

    result = await extractor.extract(b"x" * 11)

    assert result.error == "file is too large (11 bytes; limit 10)"


async def test_inline_timeout_returns_unreadable_result():
    release = threading.Event()
    started = threading.Event()

    def stuck_worker(data, **kwargs):
        started.set()
        release.wait(30)
        return extract_resume_sync(data, **kwargs)

    extractor = make_extractor(worker=stuck_worker, timeout_seconds=0.05)
    try:
        result = await extractor.extract(b"text")
    finally:
        release.set()

    assert started.is_set()
    assert result.error == TIMEOUT_ERROR
    assert [(f.code, f.severity) for f in result.flags] == [(FlagCode.UNREADABLE_FILE, "review")]


async def test_inline_worker_exception_is_contained():
    def failing_worker(data, **kwargs):
        raise RuntimeError("SECRET")

    result = await make_extractor(worker=failing_worker).extract(b"text")

    assert result.error == "extraction failed (RuntimeError)"
    assert FlagCode.UNREADABLE_FILE in flags_by_code(result)


async def test_closed_extractor_refuses_work():
    extractor = make_extractor()
    extractor.close()

    with pytest.raises(RuntimeError):
        await extractor.extract(b"text")


class FakePool:
    """Stands in for ProcessPoolExecutor; ``outcomes`` says how each pool's tasks end."""

    instances: ClassVar[list[FakePool]] = []
    outcomes: ClassVar[list[str]] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.shutdown_calls: list[tuple[bool, bool]] = []
        self.submitted: list[concurrent.futures.Future] = []
        self._processes: dict = {}
        self.outcome = FakePool.outcomes[len(FakePool.instances)]
        FakePool.instances.append(self)

    def submit(self, fn):
        future: concurrent.futures.Future = concurrent.futures.Future()
        self.submitted.append(future)
        if self.outcome == "broken":
            future.set_exception(BrokenProcessPool("worker died"))
        elif self.outcome == "cancelled":  # e.g. queued work dropped by a pool shutdown
            future.cancel()
        elif self.outcome == "ok":
            future.set_result(fn())
        return future  # "pending": never completes

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


@pytest.fixture
def fake_pool(monkeypatch):
    FakePool.instances = []
    FakePool.outcomes = []
    monkeypatch.setattr(extract_module, "ProcessPoolExecutor", FakePool)
    return FakePool


async def test_pool_is_spawn_based_with_task_limit(fake_pool):
    fake_pool.outcomes = ["ok"]
    extractor = make_extractor(in_subprocess=True, max_workers=3, max_tasks_per_child=7)

    result = await extractor.extract(b"plain text")

    assert result.kind == "txt"
    (pool,) = fake_pool.instances
    assert pool.kwargs["max_workers"] == 3
    assert pool.kwargs["max_tasks_per_child"] == 7
    assert pool.kwargs["mp_context"].get_start_method() == "spawn"
    await extractor.extract(b"again")
    assert len(fake_pool.instances) == 1  # reused
    extractor.close()
    assert pool.shutdown_calls


async def test_crashed_worker_is_retried_once_on_a_new_pool(fake_pool):
    fake_pool.outcomes = ["broken", "ok"]
    extractor = make_extractor(in_subprocess=True)

    result = await extractor.extract(b"plain text")

    assert result.kind == "txt"
    assert len(fake_pool.instances) == 2
    assert fake_pool.instances[0].shutdown_calls  # the broken pool was discarded


async def test_repeated_crash_is_reported(fake_pool):
    fake_pool.outcomes = ["broken", "broken"]
    extractor = make_extractor(in_subprocess=True)

    result = await extractor.extract(b"plain text")

    assert result.error == "extraction worker crashed"
    assert FlagCode.UNREADABLE_FILE in flags_by_code(result)


async def test_cancelled_work_item_is_retried(fake_pool):
    fake_pool.outcomes = ["cancelled", "ok"]
    extractor = make_extractor(in_subprocess=True)

    result = await extractor.extract(b"plain text")

    assert result.kind == "txt"
    assert len(fake_pool.instances) == 2


async def test_cancelling_the_caller_propagates(fake_pool):
    fake_pool.outcomes = ["pending"]
    extractor = make_extractor(in_subprocess=True)
    task = asyncio.ensure_future(extractor.extract(b"plain text"))
    await asyncio.sleep(0)  # let the task run up to its first await (the pool future)
    assert fake_pool.instances

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(fake_pool.instances) == 1  # no retry


async def test_close_during_extraction_reports_an_error(fake_pool):
    fake_pool.outcomes = ["pending"]
    extractor = make_extractor(in_subprocess=True)
    task = asyncio.ensure_future(extractor.extract(b"plain text"))
    await asyncio.sleep(0)  # let the task run up to its first await (the pool future)
    assert fake_pool.instances
    pool = fake_pool.instances[0]
    pending = pool.submitted[-1]

    extractor.close()
    pending.cancel()  # what shutdown(cancel_futures=True) does to queued work
    result = await task

    assert result.error == "extraction cancelled (extractor closed)"
    assert pool.shutdown_calls[0] == (False, True)  # queued work cancelled, no waiting


class ThreadedPool:
    """Stands in for ProcessPoolExecutor with a real queue: ``max_workers`` threads, and
    work submitted beyond that waits for one. ``kill_workers`` drops queued work like
    Python 3.14's; a thread can't be killed, so a hung task holds it until released."""

    instances: ClassVar[list[ThreadedPool]] = []

    def __init__(self, *, max_workers: int, **kwargs) -> None:
        self._threads = concurrent.futures.ThreadPoolExecutor(max_workers)
        self.killed = False
        ThreadedPool.instances.append(self)

    def submit(self, fn):
        return self._threads.submit(fn)

    def kill_workers(self) -> None:
        self.killed = True
        self._threads.shutdown(wait=False, cancel_futures=True)


@pytest.fixture
def threaded_pool(monkeypatch):
    ThreadedPool.instances = []
    monkeypatch.setattr(extract_module, "ProcessPoolExecutor", ThreadedPool)
    return ThreadedPool


async def test_task_queued_behind_a_hung_one_is_not_timed_out(threaded_pool):
    release = threading.Event()

    def worker(data, **kwargs):
        if data == b"hung":
            release.wait(10)
        return extract_resume_sync(data, **kwargs)

    extractor = make_extractor(in_subprocess=True, max_workers=1, worker=worker,
                               timeout_seconds=0.2)
    try:
        hung, healthy = await asyncio.gather(extractor.extract(b"hung"),
                                             extractor.extract(b"plain text"))
    finally:
        release.set()
        extractor.close()

    assert hung.error == TIMEOUT_ERROR
    assert healthy.error is None
    assert healthy.kind == "txt"
    assert len(threaded_pool.instances) == 2  # the hung pool was killed; a new one ran it
    assert threaded_pool.instances[0].killed


async def test_time_waiting_for_a_worker_does_not_count_toward_the_timeout(threaded_pool):
    def worker(data, **kwargs):
        time.sleep(0.25)  # each run is under the timeout; two back to back are not
        return extract_resume_sync(data, **kwargs)

    extractor = make_extractor(in_subprocess=True, max_workers=2, worker=worker,
                               timeout_seconds=0.4)
    try:
        results = await asyncio.gather(*(extractor.extract(b"plain text") for _ in range(4)))
    finally:
        extractor.close()

    assert [result.error for result in results] == [None] * 4
    assert len(threaded_pool.instances) == 1  # nothing timed out, so no pool was killed


def test_worker_slots_work_across_event_loops(threaded_pool):
    extractor = make_extractor(in_subprocess=True, max_workers=1)

    async def two_at_once():
        return await asyncio.gather(extractor.extract(b"plain text"),
                                    extractor.extract(b"more text"))

    try:
        for _ in range(2):  # e.g. two CLI commands sharing one extractor
            assert [result.kind for result in asyncio.run(two_at_once())] == ["txt", "txt"]
    finally:
        extractor.close()


async def test_real_subprocess_extraction_and_timeout(pdfs):
    """Runs in real spawned worker processes: a timeout far shorter than worker start-up
    kills and replaces the pool; the next file is then extracted normally."""
    extractor = ResumeExtractor(**LIMITS, timeout_seconds=0.001, in_subprocess=True,
                                max_workers=1)
    try:
        timed_out = await extractor.extract(pdfs.hidden("white_on_white"))
        assert timed_out.error == TIMEOUT_ERROR
        assert FlagCode.UNREADABLE_FILE in flags_by_code(timed_out)

        extractor.timeout_seconds = 120
        result = await extractor.extract(pdfs.hidden("white_on_white"))
    finally:
        extractor.close()

    assert result.error is None
    assert result.kind == "pdf"
    assert "Jordan Example - Senior Data Engineer" in result.visible_text
    assert "HIDDENMARK" not in result.visible_text
    assert result.hidden_char_count > 0
