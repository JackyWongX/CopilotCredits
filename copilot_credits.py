#!/usr/bin/env python3
"""Report GitHub Copilot Credits and usage from local VS Code logs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import webbrowser
from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Iterator


CREDIT_FIELDS = {
    "credit",
    "credits",
    "creditusage",
    "creditsusage",
    "creditused",
    "creditsused",
    "usedcredit",
    "usedcredits",
    "creditconsumed",
    "creditsconsumed",
    "consumedcredit",
    "consumedcredits",
    "creditcost",
    "creditscost",
    "creditamount",
    "creditsamount",
    "copilotcredit",
    "copilotcredits",
    "copilotcreditusage",
}
TIMESTAMP_FIELDS = {
    "timestamp",
    "time",
    "date",
    "createdat",
    "createdon",
    "requesttime",
    "requesttimestamp",
    "eventtime",
    "eventtimestamp",
    "startedat",
}
MODEL_FIELD_PRIORITY = (
    "resolvedmodel",
    "modeldisplayname",
    "modelname",
    "model",
    "modelid",
    "selectedmodel",
    "family",
)
MODEL_ALIASES = {
    "copilotauto": "GPT-6 Luna",
    "auto": "GPT-6 Luna",
    "gpt6luna": "GPT-6 Luna",
}
CACHE_VERSION = 7
CACHE_FILENAME = "index.json"
AGENT_USAGE_EVENT = "session.usage_record"
NANO_AIU_PER_CREDIT = Decimal("1000000000")


def _normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _as_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, Decimal, str)):
        try:
            result = Decimal(str(value))
        except InvalidOperation:
            return None
        return result if result.is_finite() else None
    return None


def parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO date/time or Unix timestamp and return local time."""
    parsed: datetime
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = float(value)
        if abs(seconds) >= 100_000_000_000:
            seconds /= 1000
        try:
            parsed = datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = datetime.combine(date.fromisoformat(text), datetime.min.time())
            except ValueError:
                return None
    else:
        return None
    return parsed.astimezone()


def _timestamp_in(record: dict[str, Any]) -> datetime | None:
    for key, value in record.items():
        if _normalize_key(key) in TIMESTAMP_FIELDS:
            parsed = parse_timestamp(value)
            if parsed is not None:
                return parsed
    return None


def _model_in(record: dict[str, Any]) -> str | None:
    fields = {_normalize_key(key): value for key, value in record.items()}
    for field in MODEL_FIELD_PRIORITY:
        value = fields.get(field)
        candidates = [value] if isinstance(value, str) else []
        if isinstance(value, dict):
            nested_fields = {_normalize_key(key): item for key, item in value.items()}
            candidates.extend(
                nested_fields.get(nested_field)
                for nested_field in ("resolvedmodel", "modelname", "name", "id", "modelid", "family")
            )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                model_name = candidate.strip()
                return MODEL_ALIASES.get(_normalize_key(model_name), model_name)
    if record.get("type") == AGENT_USAGE_EVENT:
        data = record.get("data")
        usage = data.get("usage") if isinstance(data, dict) else None
        if isinstance(usage, dict):
            return _model_in(usage)
    return None


def _credit_values(record: dict[str, Any]) -> Iterator[tuple[str, Decimal | None]]:
    for key, value in record.items():
        normalized = _normalize_key(key)
        if normalized in CREDIT_FIELDS:
            amount = _as_decimal(value)
            if amount is not None:
                yield str(key), amount
            elif isinstance(value, dict):
                nested_values = {
                    _normalize_key(nested_key): (nested_key, nested_value)
                    for nested_key, nested_value in value.items()
                }
                for nested_key in ("used", "consumed", "amount", "value", "count", "cost", "total"):
                    if nested_key in nested_values:
                        original_key, nested_value = nested_values[nested_key]
                        yield f"{key}.{original_key}", _as_decimal(nested_value)
                        break
                else:
                    yield str(key), None
            else:
                yield str(key), None
    if record.get("type") == AGENT_USAGE_EVENT:
        data = record.get("data")
        usage = data.get("usage") if isinstance(data, dict) else None
        copilot_usage = usage.get("copilotUsage") if isinstance(usage, dict) else None
        nano_aiu = (
            _as_decimal(copilot_usage.get("totalNanoAiu"))
            if isinstance(copilot_usage, dict) and usage.get("aiCreditsStatus") == "complete"
            else None
        )
        if nano_aiu is not None and nano_aiu >= 0:
            yield "usage.copilotUsage.totalNanoAiu", nano_aiu / NANO_AIU_PER_CREDIT


def _has_unpriced_agent_usage(record: dict[str, Any]) -> bool:
    if record.get("type") != AGENT_USAGE_EVENT:
        return False
    data = record.get("data")
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict) or usage.get("aiCreditsStatus") != "complete":
        return False
    copilot_usage = usage.get("copilotUsage")
    nano_aiu = (
        _as_decimal(copilot_usage.get("totalNanoAiu"))
        if isinstance(copilot_usage, dict)
        else None
    )
    return nano_aiu is None or nano_aiu < 0


def _session_creation_timestamp(value: Any) -> datetime | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if _normalize_key(key) == "creationdate":
                parsed = parse_timestamp(child)
                if parsed is not None:
                    return parsed
        for child in value.values():
            parsed = _session_creation_timestamp(child)
            if parsed is not None:
                return parsed
    elif isinstance(value, list):
        for child in value:
            parsed = _session_creation_timestamp(child)
            if parsed is not None:
                return parsed
    return None


def _week_key(day: date) -> str:
    iso = day.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _walk_records(
    value: Any,
    inherited_timestamp: datetime | None = None,
    inherited_model: str | None = None,
) -> Iterator[tuple[dict[str, Any], datetime | None, str | None]]:
    if isinstance(value, dict):
        timestamp = _timestamp_in(value) or inherited_timestamp
        model = _model_in(value) or inherited_model
        yield value, timestamp, model
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from _walk_records(child, timestamp, model)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_records(child, inherited_timestamp, inherited_model)


def _load_file(path: Path) -> tuple[list[Any], int]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return [], 1

    if path.suffix.lower() == ".jsonl":
        documents: list[Any] = []
        malformed = 0
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
        return documents, malformed

    try:
        return [json.loads(text)], 0
    except json.JSONDecodeError:
        documents = []
        malformed = 0
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
        return documents, malformed


def _files_from_paths(paths: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_file():
            if path.suffix.lower() in {".json", ".jsonl"}:
                files.add(path)
        elif path.is_dir():
            files.update(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in {".json", ".jsonl"}
            )
    return sorted(files)


def discover_default_paths() -> list[Path]:
    appdata = os.environ.get("APPDATA")
    roots: list[Path] = []
    if appdata:
        storage = Path(appdata) / "Code" / "User" / "workspaceStorage"
        if storage.is_dir():
            for workspace in storage.iterdir():
                chat_sessions = workspace / "chatSessions"
                if chat_sessions.is_dir():
                    roots.append(chat_sessions)
                extension_dir = workspace / "GitHub.copilot-chat"
                for folder_name in ("debug-logs", "transcripts"):
                    candidate = extension_dir / folder_name
                    if candidate.is_dir():
                        roots.append(candidate)
    agent_sessions = Path.home() / ".copilot" / "session-state"
    if agent_sessions.is_dir():
        roots.append(agent_sessions)
    return roots


def _file_contribution(path: Path) -> dict[str, Any]:
    documents, malformed = _load_file(path)
    try:
        file_timestamp = parse_timestamp(path.stat().st_mtime)
    except OSError:
        file_timestamp = None
    totals: dict[str, dict[str, Decimal]] = {
        "daily": defaultdict(Decimal),
        "weekly": defaultdict(Decimal),
        "monthly": defaultdict(Decimal),
    }
    session_periods: dict[str, dict[str, int]] = {
        "daily": defaultdict(int),
        "weekly": defaultdict(int),
        "monthly": defaultdict(int),
    }
    model_periods: dict[str, dict[str, dict[str, Decimal]]] = {
        "daily": defaultdict(lambda: defaultdict(Decimal)),
        "weekly": defaultdict(lambda: defaultdict(Decimal)),
        "monthly": defaultdict(lambda: defaultdict(Decimal)),
    }
    unpriced_usage_periods: dict[str, dict[str, int]] = {
        "daily": defaultdict(int),
        "weekly": defaultdict(int),
        "monthly": defaultdict(int),
    }
    undated_unpriced_usage_records = 0
    field_names: set[str] = set()
    field_count = 0
    record_count = 0
    file_timestamp_records = 0
    undated_total = Decimal(0)
    undated_records = 0
    undated_by_model: dict[str, Decimal] = defaultdict(Decimal)
    session_dates: set[date] = set()
    for document in documents:
        session_created = _session_creation_timestamp(document)
        if session_created is not None:
            session_dates.add(session_created.date())
        if isinstance(document, dict) and document.get("type") == "session.start":
            timestamp = _timestamp_in(document)
            if timestamp is not None:
                session_dates.add(timestamp.date())
    for session_day in session_dates:
        session_periods["daily"][session_day.isoformat()] += 1
        session_periods["weekly"][_week_key(session_day)] += 1
        session_periods["monthly"][session_day.strftime("%Y-%m")] += 1
    for document in documents:
        for record, timestamp, model in _walk_records(document):
            if (
                timestamp is None
                and file_timestamp is not None
                and path.suffix.lower() == ".jsonl"
                and record.get("kind") == "subagent"
                and "credits" in record
            ):
                timestamp = file_timestamp
                file_timestamp_records += 1
            if record.get("type") == "session.usage_record":
                if _has_unpriced_agent_usage(record):
                    if timestamp is None:
                        undated_unpriced_usage_records += 1
                    else:
                        day = timestamp.date()
                        unpriced_usage_periods["daily"][day.isoformat()] += 1
                        unpriced_usage_periods["weekly"][_week_key(day)] += 1
                        unpriced_usage_periods["monthly"][day.strftime("%Y-%m")] += 1
            for field_name, amount in _credit_values(record):
                field_names.add(field_name)
                field_count += 1
                if amount is None:
                    continue
                record_count += 1
                model_name = model or "未标注模型"
                if timestamp is None:
                    undated_total += amount
                    undated_records += 1
                    undated_by_model[model_name] += amount
                    continue
                day = timestamp.date()
                totals["daily"][day.isoformat()] += amount
                totals["weekly"][_week_key(day)] += amount
                totals["monthly"][day.strftime("%Y-%m")] += amount
                model_periods["daily"][day.isoformat()][model_name] += amount
                model_periods["weekly"][_week_key(day)][model_name] += amount
                model_periods["monthly"][day.strftime("%Y-%m")][model_name] += amount

    return {
        "malformed": malformed,
        "field_names": sorted(field_names),
        "field_count": field_count,
        "record_count": record_count,
        "file_timestamp_records": file_timestamp_records,
        "undated_total": str(undated_total),
        "undated_records": undated_records,
        "undated_by_model": {key: str(value) for key, value in undated_by_model.items()},
        "periods": {
            period: {key: str(value) for key, value in groups.items()}
            for period, groups in totals.items()
        },
        "session_periods": {period: dict(groups) for period, groups in session_periods.items()},
        "unpriced_usage_periods": {
            period: dict(groups) for period, groups in unpriced_usage_periods.items()
        },
        "undated_unpriced_usage_records": undated_unpriced_usage_records,
        "model_periods": {
            period: {
                key: {model: str(value) for model, value in models.items()}
                for key, models in groups.items()
            }
            for period, groups in model_periods.items()
        },
    }


def _cache_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _read_cache(cache_dir: Path) -> dict[str, Any]:
    try:
        payload = json.loads((cache_dir / CACHE_FILENAME).read_text(encoding="utf-8"))
        if payload.get("version") != CACHE_VERSION or not isinstance(payload.get("files"), dict):
            return {}
        return payload["files"]
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, TypeError):
        return {}


def _write_cache(cache_dir: Path, entries: dict[str, Any]) -> None:
    temporary_path: str | None = None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=cache_dir, delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name
            json.dump({"version": CACHE_VERSION, "files": entries}, temporary_file)
        os.replace(temporary_path, cache_dir / CACHE_FILENAME)
    except (OSError, TypeError, ValueError):
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _valid_cache_entry(entry: Any, size: int, mtime_ns: int) -> bool:
    if not (
        isinstance(entry, dict)
        and entry.get("size") == size
        and entry.get("mtime_ns") == mtime_ns
        and isinstance(entry.get("contribution"), dict)
    ):
        return False
    contribution = entry["contribution"]
    try:
        for value in (contribution["undated_total"],):
            if not Decimal(value).is_finite():
                return False
        for key in (
            "malformed",
            "field_count",
            "record_count",
            "undated_records",
            "undated_unpriced_usage_records",
        ):
            if not isinstance(contribution[key], int) or contribution[key] < 0:
                return False
        if not isinstance(contribution["field_names"], list) or not all(
            isinstance(name, str) for name in contribution["field_names"]
        ):
            return False
        for values in (contribution["undated_by_model"], *contribution["periods"].values()):
            if not isinstance(values, dict) or any(
                not isinstance(key, str) or not Decimal(value).is_finite()
                for key, value in values.items()
            ):
                return False
        for groups in contribution["model_periods"].values():
            if not isinstance(groups, dict):
                return False
            for models in groups.values():
                if not isinstance(models, dict) or any(
                    not isinstance(model, str) or not Decimal(value).is_finite()
                    for model, value in models.items()
                ):
                    return False
        for groups in contribution["session_periods"].values():
            if not isinstance(groups, dict) or any(
                not isinstance(key, str) or not isinstance(value, int) or value < 0
                for key, value in groups.items()
            ):
                return False
        for groups in contribution["unpriced_usage_periods"].values():
            if not isinstance(groups, dict) or any(
                not isinstance(key, str) or not isinstance(value, int) or value < 0
                for key, value in groups.items()
            ):
                return False
    except (AttributeError, KeyError, TypeError, ValueError, InvalidOperation):
        return False
    return True


def analyze_paths(paths: Iterable[Path], cache_dir: Path | None = None) -> dict[str, Any]:
    files = _files_from_paths(paths)
    cache_entries = _read_cache(cache_dir) if cache_dir is not None else {}
    contributions: list[dict[str, Any]] = []
    updated_entries = dict(cache_entries)
    cache_hits = 0
    files_reparsed = 0

    for path in files:
        key = _cache_key(path)
        try:
            stat = path.stat()
            size, mtime_ns = stat.st_size, stat.st_mtime_ns
        except OSError:
            size = mtime_ns = -1
        entry = cache_entries.get(key)
        if size >= 0 and _valid_cache_entry(entry, size, mtime_ns):
            contribution = entry["contribution"]
            cache_hits += 1
        else:
            contribution = _file_contribution(path)
            files_reparsed += 1
            try:
                after_stat = path.stat()
                if size >= 0 and (after_stat.st_size, after_stat.st_mtime_ns) == (size, mtime_ns):
                    updated_entries[key] = {
                        "size": size,
                        "mtime_ns": mtime_ns,
                        "contribution": contribution,
                    }
            except OSError:
                pass
        contributions.append(contribution)

    if cache_dir is not None and files_reparsed:
        _write_cache(cache_dir, updated_entries)

    totals: dict[str, dict[str, Decimal]] = {
        "daily": defaultdict(Decimal),
        "weekly": defaultdict(Decimal),
        "monthly": defaultdict(Decimal),
    }
    session_periods: dict[str, dict[str, int]] = {
        "daily": defaultdict(int),
        "weekly": defaultdict(int),
        "monthly": defaultdict(int),
    }
    unpriced_usage_periods: dict[str, dict[str, int]] = {
        "daily": defaultdict(int),
        "weekly": defaultdict(int),
        "monthly": defaultdict(int),
    }
    model_periods: dict[str, dict[str, dict[str, Decimal]]] = {
        "daily": defaultdict(lambda: defaultdict(Decimal)),
        "weekly": defaultdict(lambda: defaultdict(Decimal)),
        "monthly": defaultdict(lambda: defaultdict(Decimal)),
    }
    field_names: set[str] = set()
    field_count = record_count = undated_records = malformed_count = 0
    file_timestamp_records = 0
    undated_unpriced_usage_records = 0
    undated_total = Decimal(0)
    undated_by_model: dict[str, Decimal] = defaultdict(Decimal)
    for contribution in contributions:
        malformed_count += contribution["malformed"]
        field_names.update(contribution["field_names"])
        field_count += contribution["field_count"]
        record_count += contribution["record_count"]
        file_timestamp_records += contribution.get("file_timestamp_records", 0)
        undated_unpriced_usage_records += contribution["undated_unpriced_usage_records"]
        undated_total += Decimal(contribution["undated_total"])
        undated_records += contribution["undated_records"]
        for model, amount in contribution["undated_by_model"].items():
            undated_by_model[model] += Decimal(amount)
        for period, groups in contribution["periods"].items():
            for key, amount in groups.items():
                totals[period][key] += Decimal(amount)
        for period, groups in contribution["session_periods"].items():
            for key, count in groups.items():
                session_periods[period][key] += count
        for period, groups in contribution["unpriced_usage_periods"].items():
            for key, count in groups.items():
                unpriced_usage_periods[period][key] += count
        for period, groups in contribution["model_periods"].items():
            for key, models in groups.items():
                for model, amount in models.items():
                    model_periods[period][key][model] += Decimal(amount)

    has_fields = field_count > 0
    status = "credits_found" if has_fields else "no_credit_fields"
    if not has_fields:
        message = (
            "无可统计的 Credits 字段；仅完整记录中的 Nano AIU 按 10 亿:1 折算，"
            "不将 tokens、cost、模型名或请求次数换算为 Credits。"
        )
    elif record_count == 0:
        message = "发现明确的 Credits 字段，但没有可解析的数值。"
    elif any(
        field_name == "usage.copilotUsage.totalNanoAiu"
        for field_name in field_names
    ):
        message = "已将完整用量记录中的 Nano AIU 汇总并按 10 亿:1 折算为 Credits。"
        if undated_records:
            message += " 部分 Credits 记录缺少可解析时间，已单独列出未分组用量。"
    elif undated_records:
        message = "部分 Credits 记录缺少可解析时间，已单独列出未分组用量。"
    else:
        message = "已按日志中明确记录的 Credits 数值汇总。"
    unpriced_count = sum(unpriced_usage_periods["daily"].values())
    if unpriced_count:
        message += f" 另有 {unpriced_count} 条记录缺少 Nano AIU，未计入 Credits。"
    if undated_unpriced_usage_records:
        message += (
            f" 另有 {undated_unpriced_usage_records} 条缺少可解析时间的记录缺少 Nano AIU，"
            "无法按日期分组。"
        )
    if file_timestamp_records:
        message += (
            f" 另有 {file_timestamp_records} 条未带时间戳的子代理 Credits，"
            "按日志文件修改日期归档。"
        )

    result = {
        "status": status,
        "message": message,
        "files_scanned": len(files),
        "malformed_records": malformed_count,
        "credit_fields": sorted(field_names),
        "credit_field_count": field_count,
        "credit_records": record_count,
        "file_timestamp_records": file_timestamp_records,
        "undated_credits": _json_number(undated_total),
        "undated_records": undated_records,
        "undated_by_model": {
            model: _json_number(amount) for model, amount in sorted(undated_by_model.items())
        },
        "periods": {
            period: {key: _json_number(value) for key, value in sorted(groups.items())}
            for period, groups in totals.items()
        },
        "session_periods": {
            period: dict(sorted(groups.items()))
            for period, groups in session_periods.items()
        },
        "unpriced_usage_periods": {
            period: dict(sorted(groups.items()))
            for period, groups in unpriced_usage_periods.items()
        },
        "undated_unpriced_usage_records": undated_unpriced_usage_records,
        "model_periods": {
            period: {
                key: {model: _json_number(amount) for model, amount in sorted(models.items())}
                for key, models in sorted(groups.items())
            }
            for period, groups in model_periods.items()
        },
    }
    if cache_dir is not None:
        result["cache_hits"] = cache_hits
        result["files_reparsed"] = files_reparsed
    return result


def _json_number(value: Decimal) -> int | float:
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _current_usage(result: dict[str, Any], generated_at: datetime) -> dict[str, dict[str, Any]]:
    today = generated_at.date()
    daily_key = today.isoformat()
    monthly_key = today.strftime("%Y-%m")
    periods = result["periods"]
    unpriced = result.get("unpriced_usage_periods", {})
    today_unpriced = unpriced.get("daily", {}).get(daily_key, 0)
    month_unpriced = unpriced.get("monthly", {}).get(monthly_key, 0)
    undated_unpriced = result.get("undated_unpriced_usage_records", 0)
    return {
        "today": {
            "label": daily_key,
            "value": (
                None
                if today_unpriced or undated_unpriced
                else periods["daily"].get(daily_key, 0)
            ),
            "unpriced_records": today_unpriced + undated_unpriced,
        },
        "month": {
            "label": monthly_key,
            "value": (
                None
                if month_unpriced or undated_unpriced
                else periods["monthly"].get(monthly_key, 0)
            ),
            "unpriced_records": month_unpriced + undated_unpriced,
        },
    }


def _format_credits(value: int | float | Decimal) -> str:
    rounded = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{rounded:.2f}"


def format_table(result: dict[str, Any]) -> str:
    lines = [result["message"]]
    if "cache_hits" in result:
        lines.append(
            f"缓存: 命中 {result['cache_hits']} 个，重新解析 {result['files_reparsed']} 个"
        )
    for day, count in result.get("unpriced_usage_periods", {}).get("daily", {}).items():
        lines.append(f"未计价的用量事件: {day} {count} 条（未计入 Credits）")
    undated_unpriced = result.get("undated_unpriced_usage_records", 0)
    if undated_unpriced:
        lines.append(f"未分组的未计价用量事件: {undated_unpriced} 条（日期未知，未计入 Credits）")
    labels = (("daily", "每日"), ("weekly", "每周"), ("monthly", "每月"))
    if result["status"] == "no_credit_fields":
        lines.append(f"扫描文件: {result['files_scanned']}  |  坏记录: {result['malformed_records']}")
    else:
        if result["credit_fields"]:
            lines.append("Credits 字段: " + ", ".join(result["credit_fields"]))
        lines.append("")
        lines.append(f"{'周期':<8} {'日期/周/月':<12} Credits")
        lines.append("-" * 34)
        rows = 0
        for period, label in labels:
            for bucket, amount in result["periods"][period].items():
                lines.append(f"{label:<8} {bucket:<12} {_format_credits(amount)}")
                rows += 1
        if rows == 0:
            lines.append("没有带有效时间的 Credits 记录。")
    lines.append("")
    lines.append("会话数:")
    session_rows = 0
    for period, label in labels:
        for bucket, count in result["session_periods"][period].items():
            lines.append(f"{label:<8} {bucket:<12} {count} 个会话")
            session_rows += 1
    if session_rows == 0:
        lines.append("没有带创建时间的会话记录。")
    if result["status"] == "no_credit_fields":
        return "\n".join(lines)
    if result["undated_records"]:
        lines.append(
            f"未分组（缺少时间）: {_format_credits(result['undated_credits'])} Credits"
            f"，{result['undated_records']} 条记录"
        )
    lines.append(f"有效记录: {result['credit_records']}  |  坏记录: {result['malformed_records']}")
    return "\n".join(lines)


def write_html_report(result: dict[str, Any], output_path: Path) -> Path:
    generated_at = datetime.now().astimezone()
    report = {
        **result,
        "generated_at": generated_at.isoformat(),
        "current_usage": _current_usage(result, generated_at),
    }
    report_data = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    report_data = (
        report_data.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    html = r'''<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Copilot Credits 用量报告</title>
    <style>
        :root {
            color-scheme: light;
            --canvas: #fff6fa;
            --paper: #fffdfa;
            --ink: #33242d;
            --muted: #756570;
            --line: #ead9e1;
            --accent: #ff3d8e;
            --accent-light: #ffe1ed;
            --coral: #e47758;
            --yellow: #e4c45e;
            --blue: #568ba3;
            --shadow: 0 12px 32px rgba(82, 37, 58, .07);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            background: linear-gradient(145deg, #fff2f7, #f8f6f5 70%);
            color: var(--ink);
            font-family: "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
            font-size: 14px;
        }
        button, select { font: inherit; }
        .shell { max-width: 1260px; margin: 0 auto; padding: 34px 38px 56px; }
        .toolbar { display: flex; align-items: center; justify-content: space-between; gap: 18px; margin: 20px 0; }
        .segmented { display: inline-flex; padding: 4px; background: var(--accent-light); border-radius: 7px; }
        .segmented button {
            min-width: 62px; border: 0; border-radius: 5px; padding: 8px 13px;
            background: transparent; color: var(--muted); cursor: pointer;
        }
        .segmented button[aria-pressed="true"] { background: var(--paper); color: var(--ink); box-shadow: 0 2px 7px rgba(55, 35, 45, .1); }
        .filter { display: flex; align-items: center; gap: 10px; color: var(--muted); }
        .filter[hidden] { display: none; }
        select { min-width: 190px; padding: 9px 32px 9px 11px; border: 1px solid var(--line); border-radius: 5px; color: var(--ink); background: var(--paper); }
        .metrics { display: flex; flex-wrap: wrap; gap: 8px 24px; margin: 0 0 20px; font-size: 25px; }
        .metric { display: inline-flex; align-items: baseline; gap: 5px; }
        .metric-label { color: var(--muted); font-size: 14px; }
        .metric-value { color: var(--accent); font-size: 26px; font-weight: 700; font-variant-numeric: tabular-nums; }
        .selection-summary { display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 4px 0 24px; color: var(--muted); font-size: 12px; }
        .selection-summary strong { color: var(--ink); font-variant-numeric: tabular-nums; }
        h2 { margin: 0; font-size: 22px; line-height: 1.2; font-weight: 600; }
        .section-note { margin: 0; color: var(--muted); font-size: 12px; }
        .empty { padding: 32px 16px; color: var(--muted); text-align: center; }
        .content-grid { display: grid; gap: 28px; margin-top: 6px; }
        .chart-section { min-width: 0; }
        .chart-frame { position: relative; margin-top: 12px; }
        .chart-scroll { overflow-x: auto; border-top: 1px solid var(--line); }
        .period-chart { display: block; height: 340px; }
        .chart-gridline { stroke: #eadfe4; stroke-width: 1; }
        .chart-axis { fill: var(--muted); font-size: 11px; }
        .chart-period { fill: var(--ink); font-size: 11px; }
        .chart-group { cursor: pointer; outline: none; }
        .chart-group:focus .chart-hit, .chart-group:hover .chart-hit, .chart-group.selected .chart-hit { fill: #fff0f6; }
        .chart-hit { fill: transparent; }
        .chart-bar { stroke: var(--paper); stroke-width: 1; }
        .chart-value { fill: var(--ink); font-size: 10px; font-weight: 600; font-variant-numeric: tabular-nums; paint-order: stroke; stroke: var(--paper); stroke-width: 3px; stroke-linejoin: round; pointer-events: none; }
        .chart-legend { display: flex; flex-wrap: wrap; gap: 16px; padding-top: 10px; color: var(--muted); font-size: 12px; }
        .legend-item { display: inline-flex; align-items: center; gap: 6px; }
        .legend-swatch { width: 10px; height: 10px; border-radius: 2px; }
        .chart-tooltip { position: absolute; z-index: 2; width: max-content; max-width: min(340px, calc(100% - 12px)); padding: 6px 10px; border: 1px solid var(--line); border-radius: 7px; background: var(--paper); box-shadow: var(--shadow); color: var(--ink); font-size: 12px; line-height: 1.5; pointer-events: none; }
        .chart-tooltip[hidden] { display: none; }
        .chart-tooltip-table { width: 100%; border-collapse: collapse; }
        .chart-tooltip-table th, .chart-tooltip-table td { padding: 4px 0; border-bottom: 1px solid var(--line); }
        .chart-tooltip-table tr:last-child th, .chart-tooltip-table tr:last-child td { border-bottom: 0; }
        .chart-tooltip-table th { padding-right: 18px; color: var(--muted); font-weight: 500; text-align: left; white-space: nowrap; }
        .chart-tooltip-table td { max-width: 210px; font-variant-numeric: tabular-nums; text-align: left; overflow-wrap: anywhere; }
        .model-chart { display: grid; gap: 16px; margin-top: 12px; }
        .model-chart-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; grid-template-rows: auto 10px; align-items: center; gap: 6px 12px; }
        .model-chart-label { grid-column: 1; grid-row: 1; min-width: 0; overflow-wrap: anywhere; }
        .model-chart-track { grid-column: 1 / -1; grid-row: 2; direction: ltr; width: 100%; height: 10px; overflow: hidden; border-radius: 3px; background: #f0e8ec; }
        .model-chart-bar { height: 100%; border-radius: inherit; background: var(--accent); }
        .model-chart-value { grid-column: 2; grid-row: 1; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
        .clear-selection { padding: 5px 9px; border: 1px solid var(--line); border-radius: 4px; color: #c51b62; background: transparent; cursor: pointer; }
        .footer { margin-top: 34px; padding-top: 14px; border-top: 1px solid var(--line); color: var(--muted); font-size: 11px; }
        @media (max-width: 760px) {
            .shell { padding: 24px 17px 40px; }
            .toolbar { align-items: flex-start; flex-direction: column; }
            .content-grid { gap: 22px; }
            .filter { width: 100%; justify-content: space-between; }
            select { flex: 1; min-width: 0; }
            .model-chart-row { gap: 6px 8px; }
        }
        @media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition-duration: .01ms !important; } }
    </style>
</head>
<body>
    <main class="shell">
        <section class="metrics" aria-label="今日和本月用量">
            <span class="metric"><span class="metric-label" id="today-label">今日用量</span><strong class="metric-value" id="today-value">0</strong> Credits</span>
            <span class="metric"><span class="metric-label" id="month-label">本月用量</span><strong class="metric-value" id="month-value">0</strong> Credits</span>
        </section>
        <section class="toolbar" aria-label="统计筛选">
            <div class="segmented" id="period-control" role="group" aria-label="统计周期">
                <button type="button" data-period="daily" aria-pressed="true">每日</button>
                <button type="button" data-period="weekly" aria-pressed="false">每周</button>
                <button type="button" data-period="monthly" aria-pressed="false">每月</button>
            </div>
            <label class="filter" id="month-filter-label" hidden>月份
                <select id="month-filter" aria-label="选择月份"></select>
            </label>
            <label class="filter">模型筛选
                <select id="model-filter" aria-label="筛选模型"></select>
            </label>
        </section>
        <div class="selection-summary" aria-live="polite">
            <span><span id="total-label">当前区间</span> <strong id="total-value">0</strong> Credits</span>
            <span>当前区间会话 <strong id="session-count">0</strong></span>
            <span>峰值 <strong id="peak-value">-</strong> <span id="peak-note">Credits</span></span>
            <span>模型 <strong id="model-count">0</strong></span>
            <span>未分组 <strong id="undated-value">0</strong> Credits</span>
            <button class="clear-selection" id="clear-selection" type="button" hidden>显示全部周期</button>
        </div>
        <div class="content-grid">
            <section class="chart-section" aria-label="周期用量">
                <div class="chart-frame" id="period-chart-frame">
                    <div class="chart-scroll"><svg class="period-chart" id="period-chart" role="img" aria-label="周期 Credits 与会话数"></svg></div>
                    <div class="chart-tooltip" id="chart-tooltip" role="tooltip" hidden></div>
                </div>
                <div class="chart-legend" aria-label="图例">
                    <span class="legend-item"><span class="legend-swatch" style="background:#ff3d8e"></span>Credits（向上）</span>
                    <span class="legend-item"><span class="legend-swatch" style="background:#568ba3"></span>会话数（向下）</span>
                </div>
                <div class="empty" id="period-empty" hidden>暂无周期记录。</div>
            </section>
            <section class="chart-section" aria-label="模型 Credits">
                <div class="model-chart" id="model-chart" role="list"></div>
                <div class="empty" id="model-empty" hidden>当前周期没有模型用量。</div>
            </section>
        </div>
        <footer class="footer" id="footer"></footer>
    </main>
    <script type="application/json" id="report-data">__REPORT_DATA__</script>
    <script>
        const data = JSON.parse(document.getElementById("report-data").textContent);
        const labels = { daily: "每日", weekly: "每周", monthly: "每月" };
        const colors = ["#ff3d8e", "#e47758", "#568ba3", "#b69a3b", "#6d8e64", "#c06b7b"];
        const chartColors = { credits: "#ff3d8e", sessions: "#a855f7" };
        const state = { period: "daily", model: "*", bucket: null, month: "" };
        const byId = (id) => document.getElementById(id);
        const number = (value) => new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 6 }).format(value || 0);
        const creditsNumber = (value) => new Intl.NumberFormat("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value || 0);
        const sum = (values) => Object.values(values || {}).reduce((total, value) => total + value, 0);
        const allBucketNames = () => [...new Set([
            ...Object.keys(data.periods[state.period] || {}),
            ...Object.keys(data.session_periods?.[state.period] || {})
        ])].sort();
        const availableMonths = () => [...new Set([
            ...Object.keys(data.periods.daily || {}),
            ...Object.keys(data.session_periods?.daily || {})
        ].map((bucket) => bucket.slice(0, 7)))].sort();
        const bucketNames = () => {
            const buckets = allBucketNames();
            return state.period === "daily" && state.month
                ? buckets.filter((bucket) => bucket.startsWith(`${state.month}-`))
                : buckets;
        };
        const modelNames = () => {
            const names = new Set(Object.keys(data.undated_by_model || {}));
            for (const buckets of Object.values(data.model_periods[state.period] || {})) {
                for (const name of Object.keys(buckets)) names.add(name);
            }
            for (const period of Object.values(data.model_periods || {})) {
                for (const buckets of Object.values(period)) for (const name of Object.keys(buckets)) names.add(name);
            }
            return [...names].sort((a, b) => a.localeCompare(b, "zh-CN"));
        };
        const selectedAmount = (bucket) => {
            const total = data.periods[state.period]?.[bucket] || 0;
            if (state.model === "*") return total;
            return data.model_periods[state.period]?.[bucket]?.[state.model] || 0;
        };
        function populateModelFilter() {
            const select = byId("model-filter");
            select.replaceChildren();
            for (const [value, text] of [["*", "全部模型"], ...modelNames().map((name) => [name, name])]) {
                const option = document.createElement("option");
                option.value = value;
                option.textContent = text;
                select.append(option);
            }
            select.value = state.model;
        }
        function populateMonthFilter() {
            const select = byId("month-filter");
            const months = availableMonths();
            if (!months.includes(state.month)) state.month = months.at(-1) || "";
            select.replaceChildren();
            for (const month of months) {
                const option = document.createElement("option");
                option.value = month;
                option.textContent = month;
                select.append(option);
            }
            select.value = state.month;
            select.disabled = months.length === 0;
            byId("month-filter-label").hidden = state.period !== "daily";
        }
        function modelBreakdown() {
            const result = [];
            const buckets = data.model_periods[state.period] || {};
            const addModels = (models, unknownDate = false) => {
                for (const [name, amount] of Object.entries(models)) {
                    if (state.model === "*" || state.model === name) {
                        result.push([unknownDate ? `${name}（日期未知）` : name, amount]);
                    }
                }
            };
            if (state.bucket) addModels(buckets[state.bucket] || {});
            else {
                const totals = {};
                for (const bucket of bucketNames()) {
                    const models = buckets[bucket] || {};
                    for (const [name, amount] of Object.entries(models)) {
                        totals[name] = (totals[name] || 0) + amount;
                    }
                }
                addModels(totals);
                addModels(data.undated_by_model || {}, true);
            }
            return result;
        }
        function svgElement(name, attributes = {}) {
            const element = document.createElementNS("http://www.w3.org/2000/svg", name);
            for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
            return element;
        }
        function showTooltip(bucket, clientX, clientY) {
            const tooltip = byId("chart-tooltip");
            const frame = byId("period-chart-frame");
            const modelLabel = state.model === "*" ? "全部模型" : state.model;
            const sessionValue = data.session_periods?.[state.period]?.[bucket] || 0;
            const table = document.createElement("table");
            table.className = "chart-tooltip-table";
            const rows = [
                ["周期", bucket],
                ["Credits", creditsNumber(selectedAmount(bucket))],
                ["会话数（全部模型）", number(sessionValue)],
                ["模型筛选", modelLabel],
            ];
            for (const [label, value] of rows) {
                const row = table.insertRow();
                const heading = document.createElement("th");
                heading.scope = "row";
                heading.textContent = label;
                const cell = row.insertCell();
                cell.textContent = value;
                row.prepend(heading);
            }
            tooltip.replaceChildren(table);
            tooltip.hidden = false;
            const bounds = frame.getBoundingClientRect();
            tooltip.style.left = `${Math.min(Math.max(clientX - bounds.left + 12, 6), frame.clientWidth - tooltip.offsetWidth - 6)}px`;
            tooltip.style.top = `${Math.min(Math.max(clientY - bounds.top + 12, 6), frame.clientHeight - tooltip.offsetHeight - 6)}px`;
        }
        function renderPeriodChart() {
            const chart = byId("period-chart");
            const buckets = bucketNames();
            const empty = byId("period-empty");
            chart.replaceChildren();
            empty.hidden = buckets.length > 0;
            const width = Math.max(640, buckets.length * 88 + 100, chart.parentElement.clientWidth);
            const height = 340;
            const margin = { top: 20, right: 58, bottom: 52, left: 62 };
            const plotWidth = width - margin.left - margin.right;
            const plotHeight = height - margin.top - margin.bottom;
            const creditPlotHeight = plotHeight * 0.62;
            const sessionPlotHeight = plotHeight - creditPlotHeight;
            const baselineY = margin.top + creditPlotHeight;
            const creditMaximum = Math.max(...buckets.map(selectedAmount), 0);
            const sessionMaximum = Math.max(...buckets.map((bucket) => data.session_periods?.[state.period]?.[bucket] || 0), 0);
            const creditScale = creditMaximum || 1;
            const sessionScale = sessionMaximum || 1;
            const step = plotWidth / Math.max(buckets.length, 1);
            chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
            chart.setAttribute("width", width);
            chart.setAttribute("height", height);
            chart.style.width = `${width}px`;
            chart.setAttribute("aria-label", `${labels[state.period]} Credits 与会话数，X 轴日期共用，Credits 向上，会话数向下，分别独立刻度`);
            for (let tick = 0; tick <= 2; tick += 1) {
                const creditY = margin.top + creditPlotHeight * tick / 2;
                const sessionY = baselineY + sessionPlotHeight * tick / 2;
                chart.append(svgElement("line", { x1: margin.left, x2: width - margin.right, y1: creditY, y2: creditY, class: "chart-gridline" }));
                if (tick > 0) chart.append(svgElement("line", { x1: margin.left, x2: width - margin.right, y1: sessionY, y2: sessionY, class: "chart-gridline" }));
                const creditTick = creditScale * (2 - tick) / 2;
                const leftLabel = svgElement("text", { x: margin.left - 8, y: creditY + 4, "text-anchor": "end", class: "chart-axis" });
                leftLabel.textContent = creditsNumber(creditTick);
                chart.append(leftLabel);
                const sessionTick = sessionScale * tick / 2;
                const rightLabel = svgElement("text", { x: width - margin.right + 8, y: sessionY + 4, "text-anchor": "start", class: "chart-axis" });
                rightLabel.textContent = number(sessionTick);
                chart.append(rightLabel);
            }
            const creditAxisY = margin.top + creditPlotHeight / 2;
            const leftTitle = svgElement("text", { x: 14, y: creditAxisY, transform: `rotate(-90 14 ${creditAxisY})`, "text-anchor": "middle", class: "chart-axis" });
            leftTitle.textContent = "Credits";
            const sessionAxisY = baselineY + sessionPlotHeight / 2;
            const rightTitle = svgElement("text", { x: width - 12, y: sessionAxisY, transform: `rotate(90 ${width - 12} ${sessionAxisY})`, "text-anchor": "middle", class: "chart-axis" });
            rightTitle.textContent = "会话数";
            chart.append(leftTitle, rightTitle);
            buckets.forEach((bucket, index) => {
                const center = margin.left + step * (index + 0.5);
                const creditValue = selectedAmount(bucket);
                const sessionValue = data.session_periods?.[state.period]?.[bucket] || 0;
                const barWidth = Math.min(step * 0.8, 48, Math.max(12, 96 / Math.sqrt(buckets.length)));
                const barX = center - barWidth / 2;
                const group = svgElement("g", {
                    class: `chart-group${state.bucket === bucket ? " selected" : ""}`,
                    "data-bucket": bucket,
                    tabindex: "0",
                    role: "button",
                    "aria-pressed": String(state.bucket === bucket),
                    "aria-label": `${bucket}，${creditsNumber(creditValue)} Credits，${number(sessionValue)} 个会话`,
                    "aria-describedby": "chart-tooltip",
                });
                group.append(svgElement("rect", { x: center - step / 2 + 2, y: margin.top, width: Math.max(step - 4, 1), height: plotHeight + 36, class: "chart-hit" }));
                const creditHeight = creditValue / creditScale * creditPlotHeight;
                const sessionHeight = sessionValue / sessionScale * sessionPlotHeight;
                group.append(svgElement("rect", {
                    x: barX, y: baselineY - creditHeight,
                    width: barWidth, height: creditHeight, fill: chartColors.credits, class: "chart-bar",
                    "data-series": "credits", "data-value": creditValue,
                    "aria-label": `${creditsNumber(creditValue)} Credits`,
                }));
                group.append(svgElement("rect", {
                    x: barX, y: baselineY,
                    width: barWidth, height: sessionHeight, fill: chartColors.sessions, class: "chart-bar",
                    "data-series": "sessions", "data-value": sessionValue,
                    "aria-label": `${number(sessionValue)} 个会话`,
                }));
                const creditLabel = svgElement("text", {
                    x: center, y: Math.max(margin.top - 5, baselineY - creditHeight - 6),
                    "text-anchor": "middle", class: "chart-value",
                });
                creditLabel.textContent = creditsNumber(creditValue);
                const sessionLabel = svgElement("text", {
                    x: center, y: baselineY + sessionHeight + 13,
                    "text-anchor": "middle", class: "chart-value",
                });
                sessionLabel.textContent = number(sessionValue);
                group.append(creditLabel, sessionLabel);
                const periodLabel = svgElement("text", { x: center, y: height - 18, "text-anchor": "middle", class: "chart-period" });
                periodLabel.textContent = bucket;
                group.append(periodLabel);
                const activate = () => { state.bucket = state.bucket === bucket ? null : bucket; render(); };
                group.addEventListener("click", activate);
                group.addEventListener("keydown", (event) => {
                    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activate(); }
                });
                group.addEventListener("pointerenter", (event) => showTooltip(bucket, event.clientX, event.clientY));
                group.addEventListener("pointermove", (event) => showTooltip(bucket, event.clientX, event.clientY));
                group.addEventListener("pointerleave", () => { byId("chart-tooltip").hidden = true; });
                group.addEventListener("focus", () => {
                    const bounds = group.getBoundingClientRect();
                    showTooltip(bucket, bounds.left + bounds.width / 2, bounds.top + bounds.height / 2);
                });
                group.addEventListener("blur", () => { byId("chart-tooltip").hidden = true; });
                chart.append(group);
            });
        }
        function renderModelChart() {
            const models = modelBreakdown();
            const modelChart = byId("model-chart");
            modelChart.replaceChildren();
            const entries = models.sort((a, b) => b[1] - a[1]);
            byId("model-empty").hidden = entries.length > 0;
            const maximum = Math.max(...entries.map((entry) => entry[1]), 0);
            for (const [index, [name, amount]] of entries.entries()) {
                const row = document.createElement("div"); row.className = "model-chart-row"; row.setAttribute("role", "listitem");
                const model = document.createElement("span"); model.className = "model-chart-label"; model.textContent = name;
                const track = document.createElement("div"); track.className = "model-chart-track"; track.setAttribute("aria-hidden", "true");
                const fill = document.createElement("div"); fill.className = "model-chart-bar";
                fill.style.width = `${maximum ? amount / maximum * 100 : 0}%`;
                fill.style.background = colors[index % colors.length];
                const value = document.createElement("span"); value.className = "model-chart-value"; value.textContent = `${creditsNumber(amount)} Credits`;
                track.append(fill); row.append(model, track, value); modelChart.append(row);
            }
        }
        function renderMetrics() {
            const buckets = bucketNames();
            const total = buckets.reduce((amount, bucket) => amount + selectedAmount(bucket), 0);
            const sessionTotal = buckets.reduce((count, bucket) => count + (data.session_periods?.[state.period]?.[bucket] || 0), 0);
            const peak = buckets.map((bucket) => ({ bucket, value: selectedAmount(bucket) })).sort((a, b) => b.value - a.value)[0];
            const current = data.current_usage;
            byId("today-label").textContent = `今日用量 · ${current.today.label}${current.today.unpriced_records ? `（${number(current.today.unpriced_records)} 条 Credits 未知）` : ""}`;
            byId("today-value").textContent = current.today.value === null ? "—" : creditsNumber(current.today.value);
            byId("today-value").title = current.today.unpriced_records
                ? `有 ${number(current.today.unpriced_records)} 条用量记录未提供可确认的 Credits 数值。`
                : "";
            byId("month-label").textContent = `本月用量 · ${current.month.label}${current.month.unpriced_records ? `（${number(current.month.unpriced_records)} 条 Credits 未知）` : ""}`;
            byId("month-value").textContent = current.month.value === null ? "—" : creditsNumber(current.month.value);
            byId("month-value").title = current.month.unpriced_records
                ? `有 ${number(current.month.unpriced_records)} 条用量记录未提供可确认的 Credits 数值。`
                : "";
            byId("total-value").textContent = creditsNumber(total);
            byId("total-label").textContent = state.model === "*" ? `${labels[state.period]}区间用量` : "筛选模型用量";
            byId("session-count").textContent = number(sessionTotal);
            byId("peak-value").textContent = peak?.value ? peak.bucket : "-";
            byId("peak-note").textContent = peak?.value ? `${creditsNumber(peak.value)} Credits` : "Credits";
            byId("model-count").textContent = modelNames().length;
            byId("undated-value").textContent = creditsNumber(data.undated_credits);
            byId("clear-selection").hidden = !state.bucket;
        }
        function render() {
            for (const button of document.querySelectorAll("[data-period]")) {
                button.setAttribute("aria-pressed", String(button.dataset.period === state.period));
            }
            populateMonthFilter();
            populateModelFilter();
            renderMetrics();
            renderPeriodChart();
            renderModelChart();
        }
        document.querySelectorAll("[data-period]").forEach((button) => button.addEventListener("click", () => {
            state.period = button.dataset.period; state.bucket = null; render();
        }));
        byId("month-filter").addEventListener("change", (event) => { state.month = event.target.value; state.bucket = null; render(); });
        byId("model-filter").addEventListener("change", (event) => { state.model = event.target.value; render(); });
        byId("clear-selection").addEventListener("click", () => { state.bucket = null; render(); });
        window.addEventListener("resize", renderPeriodChart);
        byId("footer").textContent = `扫描 ${data.files_scanned} 个文件 · ${data.credit_records} 条有效记录 · ${data.malformed_records} 条无效记录 · 未分组的未计价用量事件 ${number(data.undated_unpriced_usage_records || 0)} 条 · Credits 字段：${(data.credit_fields || []).join(", ") || "未发现"}`;
        render();
    </script>
</body>
</html>
'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html.replace("__REPORT_DATA__", report_data), encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "汇总 VS Code Copilot 日志中明确记录的 Credits。"
            "不会根据 tokens、模型名或请求次数推算 Credits。"
        )
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "扫描指定文件或目录（可重复）；默认扫描 VS Code workspaceStorage 和 "
            "~/.copilot/session-state 中的 Copilot 日志。"
        ),
    )
    parser.add_argument("--format", choices=("table", "json"), default="table", help="输出格式。")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("copilot_credits_report.html"),
        help="HTML 报告保存路径（默认当前目录下 copilot_credits_report.html）。",
    )
    parser.add_argument("--no-open", action="store_true", help="生成 HTML 但不自动打开浏览器。")
    args = parser.parse_args(argv)

    custom_paths = [Path(os.path.expandvars(path)).expanduser() for path in args.path]
    paths = custom_paths if custom_paths else discover_default_paths()
    cache_dir = Path(__file__).resolve().parent / ".copilot_credits_cache"
    result = analyze_paths(paths, cache_dir=cache_dir)
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_table(result))
    output_path = Path(os.path.expandvars(str(args.output))).expanduser().resolve()
    try:
        write_html_report(result, output_path)
    except OSError as error:
        print(f"无法写入 HTML 报告: {error}", file=sys.stderr)
        return 1
    print(f"HTML 报告: {output_path}")
    if not args.no_open:
        try:
            opened = webbrowser.open(output_path.as_uri())
        except webbrowser.Error:
            opened = False
        if opened:
            print("已在默认浏览器中打开报告。")
        else:
            print("无法自动打开浏览器，请手动打开上述 HTML 文件。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
