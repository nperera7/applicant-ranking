"""Loading the policy file, the shipped example, and business-day arithmetic."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel

from applicant_ranking.domain import RunMode
from applicant_ranking.lever.models import Posting
from applicant_ranking.policy import PolicyConfig, PolicyEngine, add_business_days

EXAMPLE = Path(__file__).resolve().parents[2] / "config" / "policy.example.yaml"


# ---------------------------------------------------------------------------- from_file


def test_missing_file_means_no_policy(tmp_path: Path) -> None:
    engine = PolicyEngine.from_file(tmp_path / "absent.yaml")
    assert engine.config is None
    assert engine.source == str(tmp_path / "absent.yaml")
    assert not engine.approved


@pytest.mark.parametrize(("content", "message"), [
    ("", "must be a YAML mapping"),
    ("- just\n- a list\n", "must be a YAML mapping"),
    ("version: [unclosed\n", "could not be read"),
    ("versoin: 1\n", "is invalid"),
    ("nyc:\n  bias_audit_date: not-a-date\n", "is invalid"),
    ("illinois:\n  unknown_key: true\n", "is invalid"),
    ("allowed_countries: ['']\n", "is invalid"),
    ("approved_at: 2026-10-01T10:30:00\n", "is invalid"),
    ("location_overrides:\n  HQ: [NYC]\n", "is invalid"),
    ("location_overrides:\n  ' ': []\n", "is invalid"),
    ("location_overrides:\n  HQ: NYC_LL144\n", "is invalid"),
])
def test_invalid_file_fails_loudly(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        PolicyEngine.from_file(path)


def test_directory_is_invalid(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="could not be read"):
        PolicyEngine.from_file(tmp_path)


def test_yaml_tags_are_not_executed(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text("version: !!python/object/apply:os.system ['true']\n", encoding="utf-8")
    with pytest.raises(ValueError, match="could not be read"):
        PolicyEngine.from_file(path)


def test_complete_file_loads_and_allows(tmp_path: Path,
                                        make_posting: Callable[..., Posting]) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(
        """
version: 2
approved_by: Egen Legal
approved_at: 2026-10-01
allowed_countries: [us]
illinois: {notice_url: "https://careers.example.com/ai", posting_notice_confirmed: true}
nyc:
  bias_audit_date: 2026-05-01
  audit_summary_url: https://careers.example.com/audit
  notice_published_at: 2026-09-01
""",
        encoding="utf-8",
    )
    engine = PolicyEngine.from_file(path)
    assert engine.approved
    assert engine.config is not None and engine.config.allowed_countries == ["US"]
    decision = engine.evaluate(make_posting("New York, NY"), mode=RunMode.LIVE,
                               today=date(2026, 10, 8))
    assert decision.allowed, decision.blockers


def test_location_overrides_load_from_the_file(tmp_path: Path,
                                               make_posting: Callable[..., Posting]) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(
        """
approved_by: Egen Legal
approved_at: 2026-10-01
illinois: {notice_url: "https://careers.example.com/ai", posting_notice_confirmed: true}
location_overrides:
  Hudson Yards Office: [NYC_LL144]
  HQ: []
""",
        encoding="utf-8",
    )
    engine = PolicyEngine.from_file(path)
    assert engine.config is not None
    assert engine.config.location_overrides == {"Hudson Yards Office": ["NYC_LL144"], "HQ": []}
    assert engine.config.override_for("hudson yards office") == ["NYC_LL144"]
    assert engine.config.override_for("Mountain Office") is None
    assert engine.detect_regimes(make_posting("Hudson Yards Office")) == ["IL_HRA", "NYC_LL144"]
    decision = engine.evaluate(make_posting("HQ"), mode=RunMode.LIVE, today=date(2026, 10, 8))
    assert decision.allowed, decision.blockers
    blocked = engine.evaluate(make_posting("Mountain Office"), mode=RunMode.LIVE,
                              today=date(2026, 10, 8))
    assert any("could not be mapped" in b and str(path) in b for b in blocked.blockers)


# ------------------------------------------------------------------------------ example


def _field_paths(model: type[BaseModel], prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for name, info in model.model_fields.items():
        path = f"{prefix}{name}"
        paths.add(path)
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            paths |= _field_paths(annotation, f"{path}.")
    return paths


def _yaml_paths(data: dict[str, Any], prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for key, value in data.items():
        path = f"{prefix}{key}"
        paths.add(path)
        if isinstance(value, dict):
            paths |= _yaml_paths(value, f"{path}.")
    return paths


def test_example_lists_every_field() -> None:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    assert _yaml_paths(raw) == _field_paths(PolicyConfig)


def test_example_loads_unapproved_and_us_only() -> None:
    engine = PolicyEngine.from_file(EXAMPLE)
    cfg = engine.config
    assert cfg is not None
    assert cfg.approved_by is None and cfg.approved_at is None
    assert not engine.approved
    assert cfg.allowed_countries == ["US"]
    assert cfg.california.ccpa_admt_applies is None
    assert all(day.weekday() < 5 for day in cfg.nyc.holidays)


def test_example_blocks_live_runs(make_posting: Callable[..., Posting]) -> None:
    engine = PolicyEngine.from_file(EXAMPLE)
    decision = engine.evaluate(make_posting("Naperville, IL"), mode=RunMode.DRY_RUN,
                               today=date(2026, 10, 8))
    assert not decision.allowed
    assert any("is not approved" in b for b in decision.blockers)


def test_example_comments_cover_every_regime() -> None:
    text = EXAMPLE.read_text(encoding="utf-8")
    for marker in ("IL_HRA", "NYC_LL144", "CO_ADMT", "CA_FEHA", "ON_ESA", "NOT legal advice",
                   "Canada, not California"):
        assert marker in text


# ------------------------------------------------------------------------ business days


@pytest.mark.parametrize(("start", "days", "holidays", "expected"), [
    (date(2026, 10, 1), 10, [], date(2026, 10, 15)),  # Thu + 10 -> two weekends skipped
    (date(2026, 10, 1), 10, [date(2026, 10, 12)], date(2026, 10, 16)),
    (date(2026, 10, 1), 10, [date(2026, 10, 10)], date(2026, 10, 15)),  # Saturday holiday
    (date(2026, 10, 1), 0, [], date(2026, 10, 1)),
    (date(2026, 10, 3), 0, [], date(2026, 10, 3)),  # start itself is never moved
    (date(2026, 10, 2), 1, [], date(2026, 10, 5)),  # Friday -> Monday
    (date(2026, 10, 3), 1, [], date(2026, 10, 5)),  # Saturday -> Monday
    (date(2026, 10, 4), 1, [], date(2026, 10, 5)),  # Sunday -> Monday
    (date(2026, 10, 9), 1, [date(2026, 10, 12)], date(2026, 10, 13)),
    (date(2026, 12, 24), 2, [date(2026, 12, 25), date(2027, 1, 1)], date(2026, 12, 29)),
    (date(2026, 10, 15), -10, [], date(2026, 10, 1)),
    (date(2026, 10, 5), -1, [], date(2026, 10, 2)),
])
def test_add_business_days(start: date, days: int, holidays: list[date],
                           expected: date) -> None:
    assert add_business_days(start, days, holidays) == expected


def test_add_business_days_holidays_default() -> None:
    assert add_business_days(date(2026, 10, 1), 10) == date(2026, 10, 15)
    assert add_business_days(date(2026, 10, 1), 10, None) == date(2026, 10, 15)
