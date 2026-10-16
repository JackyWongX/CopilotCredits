#!/usr/bin/env python3
"""Report explicitly recorded GitHub Copilot Credits from local VS Code logs."""

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
from decimal import Decimal, InvalidOperation
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
MODEL_FIELD_PRIORITY = ("resolvedmodel", "modelname", "model", "modelid", "selectedmodel", "family")
MODEL_ALIASES = {
    "copilotauto": "GPT-6 Luna",
    "auto": "GPT-6 Luna",
    "gpt6luna": "GPT-6 Luna",
}
CACHE_VERSION = 1
CACHE_FILENAME = "index.json"


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
    if not appdata:
        return []
    storage = Path(appdata) / "Code" / "User" / "workspaceStorage"
    if not storage.is_dir():
        return []
    roots: list[Path] = []
    for workspace in storage.iterdir():
        chat_sessions = workspace / "chatSessions"
        if chat_sessions.is_dir():
            roots.append(chat_sessions)
        extension_dir = workspace / "GitHub.copilot-chat"
        for folder_name in ("debug-logs", "transcripts"):
            candidate = extension_dir / folder_name
            if candidate.is_dir():
                roots.append(candidate)
    return roots


def _file_contribution(path: Path) -> dict[str, Any]:
    documents, malformed = _load_file(path)
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
    field_names: set[str] = set()
    field_count = 0
    record_count = 0
    undated_total = Decimal(0)
    undated_records = 0
    undated_by_model: dict[str, Decimal] = defaultdict(Decimal)
    session_created = next(
        (created for document in documents if (created := _session_creation_timestamp(document))),
        None,
    )
    if session_created is not None:
        session_day = session_created.date()
        session_periods["daily"][session_day.isoformat()] += 1
        session_periods["weekly"][_week_key(session_day)] += 1
        session_periods["monthly"][session_day.strftime("%Y-%m")] += 1
    for document in documents:
        for record, timestamp, model in _walk_records(document):
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
        "undated_total": str(undated_total),
        "undated_records": undated_records,
        "undated_by_model": {key: str(value) for key, value in undated_by_model.items()},
        "periods": {
            period: {key: str(value) for key, value in groups.items()}
            for period, groups in totals.items()
        },
        "session_periods": {period: dict(groups) for period, groups in session_periods.items()},
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
        for key in ("malformed", "field_count", "record_count", "undated_records"):
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
    model_periods: dict[str, dict[str, dict[str, Decimal]]] = {
        "daily": defaultdict(lambda: defaultdict(Decimal)),
        "weekly": defaultdict(lambda: defaultdict(Decimal)),
        "monthly": defaultdict(lambda: defaultdict(Decimal)),
    }
    field_names: set[str] = set()
    field_count = record_count = undated_records = malformed_count = 0
    undated_total = Decimal(0)
    undated_by_model: dict[str, Decimal] = defaultdict(Decimal)
    for contribution in contributions:
        malformed_count += contribution["malformed"]
        field_names.update(contribution["field_names"])
        field_count += contribution["field_count"]
        record_count += contribution["record_count"]
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
        for period, groups in contribution["model_periods"].items():
            for key, models in groups.items():
                for model, amount in models.items():
                    model_periods[period][key][model] += Decimal(amount)

    has_fields = field_count > 0
    status = "credits_found" if has_fields else "no_credit_fields"
    if not has_fields:
        message = "无可统计的 Credits 字段；未将 tokens、模型名或请求次数换算为 Credits。"
    elif record_count == 0:
        message = "发现明确的 Credits 字段，但没有可解析的数值。"
    elif undated_records:
        message = "部分 Credits 记录缺少可解析时间，已单独列出未分组用量。"
    else:
        message = "已按日志中明确记录的 Credits 数值汇总。"

    result = {
        "status": status,
        "message": message,
        "files_scanned": len(files),
        "malformed_records": malformed_count,
        "credit_fields": sorted(field_names),
        "credit_field_count": field_count,
        "credit_records": record_count,
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
    return {
        "today": {"label": daily_key, "value": periods["daily"].get(daily_key, 0)},
        "month": {"label": monthly_key, "value": periods["monthly"].get(monthly_key, 0)},
    }


def format_table(result: dict[str, Any]) -> str:
    lines = [result["message"]]
    if "cache_hits" in result:
        lines.append(
            f"缓存: 命中 {result['cache_hits']} 个，重新解析 {result['files_reparsed']} 个"
        )
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
                lines.append(f"{label:<8} {bucket:<12} {amount}")
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
            f"未分组（缺少时间）: {result['undated_credits']} Credits"
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
        .masthead {
            display: flex; align-items: flex-start; justify-content: space-between; gap: 24px;
            padding: 4px 0 24px; border-bottom: 1px solid var(--line);
        }
        .eyebrow { margin: 0 0 8px; color: var(--accent); font-size: 11px; font-weight: 700; text-transform: uppercase; }
        h1 { margin: 0; font-size: 36px; line-height: 1.1; font-weight: 600; }
        .subtitle { margin: 10px 0 0; color: var(--muted); }
        .stamp { color: var(--muted); font-size: 12px; text-align: right; white-space: nowrap; }
        .status { margin: 18px 0 22px; color: var(--muted); font-size: 13px; }
        .toolbar { display: flex; align-items: center; justify-content: space-between; gap: 18px; margin: 20px 0; }
        .segmented { display: inline-flex; padding: 4px; background: var(--accent-light); border-radius: 7px; }
        .segmented button {
            min-width: 62px; border: 0; border-radius: 5px; padding: 8px 13px;
            background: transparent; color: var(--muted); cursor: pointer;
        }
        .segmented button[aria-pressed="true"] { background: var(--paper); color: var(--ink); box-shadow: 0 2px 7px rgba(55, 35, 45, .1); }
        .filter { display: flex; align-items: center; gap: 10px; color: var(--muted); }
        select { min-width: 190px; padding: 9px 32px 9px 11px; border: 1px solid var(--line); border-radius: 5px; color: var(--ink); background: var(--paper); }
        .metrics { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin: 0 0 20px; }
        .metric { min-height: 112px; padding: 17px 18px; border: 1px solid var(--line); border-radius: 6px; background: rgba(251, 252, 248, .8); }
        .metric-label { color: var(--muted); font-size: 12px; }
        .metric-value { margin-top: 13px; font-size: 25px; line-height: 1; font-weight: 500; overflow-wrap: anywhere; }
        .metric-note { margin-top: 7px; color: var(--muted); font-size: 11px; }
        .selection-summary { display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 4px 0 24px; color: var(--muted); font-size: 12px; }
        .selection-summary strong { color: var(--ink); font-variant-numeric: tabular-nums; }
        .section-head { display: flex; align-items: end; justify-content: space-between; gap: 16px; margin: 30px 0 12px; }
        h2 { margin: 0; font-size: 22px; line-height: 1.2; font-weight: 600; }
        .section-note { margin: 0; color: var(--muted); font-size: 12px; }
        .chart-panel { position: relative; padding: 18px 18px 10px; border: 1px solid var(--line); border-radius: 6px; background: var(--paper); box-shadow: var(--shadow); }
        #trend-chart { display: block; width: 100%; height: 270px; overflow: visible; }
        .chart-grid { stroke: #efe5ea; stroke-width: 1; }
        .chart-label { fill: #756570; font: 11px "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif; }
        .chart-hover-band { fill: var(--accent); opacity: .08; pointer-events: none; }
        .chart-bar { cursor: pointer; transition: opacity .16s ease, y .16s ease; }
        .chart-bar:hover, .chart-bar:focus { opacity: .74; outline: none; }
        .chart-tooltip { position: absolute; z-index: 2; min-width: 176px; max-width: 260px; padding: 10px 12px; border: 1px solid #c51b62; border-radius: 5px; color: #fff8fb; background: #4a2636; box-shadow: 0 8px 22px rgba(82, 37, 58, .22); pointer-events: none; font-size: 12px; }
        .chart-tooltip[hidden] { display: none; }
        .tooltip-date { color: #ffd0e2; }
        .tooltip-value { margin: 5px 0; font-size: 16px; font-weight: 700; font-variant-numeric: tabular-nums; }
        .tooltip-model { color: #f3dce6; line-height: 1.5; overflow-wrap: anywhere; }
        .empty { padding: 32px 16px; color: var(--muted); text-align: center; }
        .content-grid { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(320px, .9fr); gap: 26px; margin-top: 6px; }
        .table-wrap { overflow-x: auto; border-top: 1px solid var(--line); }
        table { width: 100%; border-collapse: collapse; text-align: left; }
        th { padding: 11px 10px; color: var(--muted); font-size: 11px; font-weight: 600; }
        td { padding: 11px 10px; border-top: 1px solid #efe5ea; }
        tbody tr[data-bucket] { cursor: pointer; }
        tbody tr[data-bucket]:hover, tbody tr.selected { background: #fff0f6; }
        td.amount, th.amount { text-align: right; font-variant-numeric: tabular-nums; }
        .model-name { max-width: 260px; overflow-wrap: anywhere; }
        .model-value { min-width: 150px; }
        .bar-track { height: 6px; margin-top: 7px; overflow: hidden; border-radius: 4px; background: #f0e8ec; }
        .bar-fill { height: 100%; border-radius: inherit; background: var(--accent); }
        .clear-selection { padding: 5px 9px; border: 1px solid var(--line); border-radius: 4px; color: #c51b62; background: transparent; cursor: pointer; }
        .footer { margin-top: 34px; padding-top: 14px; border-top: 1px solid var(--line); color: var(--muted); font-size: 11px; }
        @media (max-width: 760px) {
            .shell { padding: 24px 17px 40px; }
            .masthead { display: block; }
            h1 { font-size: 31px; }
            .stamp { margin-top: 13px; text-align: left; }
            .toolbar { align-items: flex-start; flex-direction: column; }
            .metrics { gap: 8px; }
            .metric { min-height: 98px; padding: 13px 10px; }
            .metric-value { font-size: 20px; }
            .content-grid { grid-template-columns: 1fr; gap: 8px; }
            .section-head { align-items: flex-start; flex-direction: column; }
            .filter { width: 100%; justify-content: space-between; }
            select { flex: 1; min-width: 0; }
            #trend-chart { height: 230px; }
        }
        @media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition-duration: .01ms !important; } }
    </style>
</head>
<body>
    <main class="shell">
        <header class="masthead">
            <div>
                <p class="eyebrow">GitHub Copilot / Usage report</p>
                <h1>Copilot Credits</h1>
                <p class="subtitle">本地会话记录中的 Credits 使用情况</p>
            </div>
            <div class="stamp"><div>离线报告</div><div id="generated-at"></div></div>
        </header>
        <p class="status" id="status"></p>
        <section class="metrics" aria-label="今日和本月用量">
            <article class="metric"><div class="metric-label" id="today-label">今日用量</div><div class="metric-value" id="today-value">0</div><div class="metric-note">Credits</div></article>
            <article class="metric"><div class="metric-label" id="month-label">本月用量</div><div class="metric-value" id="month-value">0</div><div class="metric-note">Credits</div></article>
        </section>
        <section class="toolbar" aria-label="统计筛选">
            <div class="segmented" id="period-control" role="group" aria-label="统计周期">
                <button type="button" data-period="daily" aria-pressed="true">每日</button>
                <button type="button" data-period="weekly" aria-pressed="false">每周</button>
                <button type="button" data-period="monthly" aria-pressed="false">每月</button>
            </div>
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
        </div>
        <section>
            <div class="section-head"><h2>Credits 趋势</h2><p class="section-note" id="trend-note"></p></div>
            <div class="chart-panel"><svg id="trend-chart" viewBox="0 0 1000 270" role="img" aria-label="Credits 用量柱状图" aria-describedby="chart-tooltip"></svg><div class="chart-tooltip" id="chart-tooltip" role="tooltip" hidden></div><div class="empty" id="chart-empty" hidden>当前筛选下没有可展示的用量。</div></div>
        </section>
        <div class="content-grid">
            <section>
                <div class="section-head"><h2>周期明细</h2><button class="clear-selection" id="clear-selection" type="button" hidden>显示全部周期</button></div>
                <div class="table-wrap"><table><thead><tr><th>周期</th><th class="amount">Credits</th><th class="amount">会话数</th></tr></thead><tbody id="period-rows"></tbody></table><div class="empty" id="period-empty" hidden>暂无周期记录。</div></div>
            </section>
            <section>
                <div class="section-head"><h2>模型用量</h2><p class="section-note" id="model-scope"></p></div>
                <div class="table-wrap"><table><thead><tr><th>模型</th><th class="amount">Credits</th></tr></thead><tbody id="model-rows"></tbody></table><div class="empty" id="model-empty" hidden>当前周期没有模型用量。</div></div>
            </section>
        </div>
        <footer class="footer" id="footer"></footer>
    </main>
    <script type="application/json" id="report-data">__REPORT_DATA__</script>
    <script>
        const data = JSON.parse(document.getElementById("report-data").textContent);
        const labels = { daily: "每日", weekly: "每周", monthly: "每月" };
        const colors = ["#ff3d8e", "#e47758", "#568ba3", "#b69a3b", "#6d8e64", "#c06b7b"];
        const state = { period: "daily", model: "*", bucket: null };
        const byId = (id) => document.getElementById(id);
        const number = (value) => new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 6 }).format(value || 0);
        const sum = (values) => Object.values(values || {}).reduce((total, value) => total + value, 0);
        const bucketNames = () => [...new Set([
            ...Object.keys(data.periods[state.period] || {}),
            ...Object.keys(data.session_periods?.[state.period] || {})
        ])].sort();
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
        function drawChart() {
            const svg = byId("trend-chart");
            byId("chart-tooltip").hidden = true;
            svg.onpointermove = null;
            svg.onpointerleave = null;
            svg.replaceChildren();
            const rows = bucketNames().map((bucket) => ({ bucket, value: selectedAmount(bucket) })).slice(-40);
            byId("chart-empty").hidden = rows.length > 0;
            svg.hidden = rows.length === 0;
            if (!rows.length) return;
            const ns = "http://www.w3.org/2000/svg";
            const make = (tag, attributes, text) => {
                const element = document.createElementNS(ns, tag);
                for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
                if (text !== undefined) element.textContent = text;
                return element;
            };
            const width = 1000, height = 270, left = 54, right = 12, top = 14, bottom = 40;
            const plotWidth = width - left - right, plotHeight = height - top - bottom;
            const maximum = Math.max(...rows.map((row) => row.value), 0);
            const scale = maximum || 1;
            for (let index = 0; index <= 4; index++) {
                const y = top + plotHeight * index / 4;
                svg.append(make("line", { x1: left, y1: y, x2: width - right, y2: y, class: "chart-grid" }));
                svg.append(make("text", { x: left - 8, y: y + 4, "text-anchor": "end", class: "chart-label" }, number(scale * (4 - index) / 4)));
            }
            const slot = plotWidth / rows.length;
            const barWidth = Math.max(3, Math.min(30, slot * .62));
            const labelStep = Math.max(1, Math.ceil(rows.length / 8));
            const hoverBand = make("rect", {
                x: left, y: top, width: slot, height: plotHeight,
                class: "chart-hover-band", visibility: "hidden"
            });
            svg.append(hoverBand);
            const showTooltip = (event, row, anchor) => {
                const tooltip = byId("chart-tooltip");
                const panel = svg.parentElement;
                const breakdown = data.model_periods[state.period]?.[row.bucket] || {};
                const modelDetails = state.model === "*"
                    ? Object.entries(breakdown).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([name, amount]) => `${name} ${number(amount)}`).join(" · ")
                    : `${state.model} ${number(row.value)}`;
                tooltip.replaceChildren();
                const dateLine = document.createElement("div"); dateLine.className = "tooltip-date"; dateLine.textContent = row.bucket;
                const valueLine = document.createElement("div"); valueLine.className = "tooltip-value"; valueLine.textContent = `${number(row.value)} Credits`;
                const sessionLine = document.createElement("div"); sessionLine.className = "tooltip-model"; sessionLine.textContent = `${number(data.session_periods?.[state.period]?.[row.bucket] || 0)} 个会话`;
                const modelLine = document.createElement("div"); modelLine.className = "tooltip-model"; modelLine.textContent = modelDetails || "模型未标注";
                tooltip.append(dateLine, valueLine, sessionLine, modelLine);
                tooltip.hidden = false;
                const panelRect = panel.getBoundingClientRect();
                const anchorRect = anchor ? anchor.getBoundingClientRect() : svg.getBoundingClientRect();
                const pointerX = event.clientX || anchorRect.left + anchorRect.width / 2;
                const pointerY = event.clientY || anchorRect.top;
                tooltip.style.left = `${Math.max(8, Math.min(pointerX - panelRect.left + 12, panel.clientWidth - tooltip.offsetWidth - 8))}px`;
                tooltip.style.top = `${Math.max(8, Math.min(pointerY - panelRect.top - tooltip.offsetHeight - 12, panel.clientHeight - tooltip.offsetHeight - 8))}px`;
            };
            svg.onpointermove = (event) => {
                const transform = svg.getScreenCTM();
                if (!transform) return;
                const point = svg.createSVGPoint();
                point.x = event.clientX;
                point.y = event.clientY;
                const chartPoint = point.matrixTransform(transform.inverse());
                const index = Math.floor((chartPoint.x - left) / slot);
                if (chartPoint.y < top || chartPoint.y > height - bottom || index < 0 || index >= rows.length) {
                    byId("chart-tooltip").hidden = true;
                    hoverBand.setAttribute("visibility", "hidden");
                    return;
                }
                hoverBand.setAttribute("x", left + index * slot);
                hoverBand.setAttribute("visibility", "visible");
                showTooltip(event, rows[index], null);
            };
            svg.onpointerleave = () => {
                byId("chart-tooltip").hidden = true;
                hoverBand.setAttribute("visibility", "hidden");
            };
            rows.forEach((row, index) => {
                const barHeight = row.value / scale * plotHeight;
                const x = left + index * slot + (slot - barWidth) / 2;
                const y = top + plotHeight - barHeight;
                const bar = make("rect", {
                    x, y, width: barWidth, height: Math.max(1, barHeight), rx: 2,
                    fill: state.bucket === row.bucket ? "#c51b62" : "#ff3d8e",
                    class: "chart-bar", tabindex: 0, role: "button",
                    "aria-label": `${row.bucket}: ${number(row.value)} Credits, ${number(data.session_periods?.[state.period]?.[row.bucket] || 0)} 个会话`
                });
                bar.append(make("title", {}, `${row.bucket} · ${number(row.value)} Credits · ${number(data.session_periods?.[state.period]?.[row.bucket] || 0)} 个会话`));
                const activate = () => { state.bucket = state.bucket === row.bucket ? null : row.bucket; render(); };
                bar.addEventListener("focus", (event) => showTooltip(event, row, bar));
                bar.addEventListener("blur", () => { byId("chart-tooltip").hidden = true; });
                bar.addEventListener("click", activate);
                bar.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activate(); } });
                svg.append(bar);
                if (index % labelStep === 0 || index === rows.length - 1) {
                    svg.append(make("text", { x: left + index * slot + slot / 2, y: height - 12, "text-anchor": "middle", class: "chart-label" }, row.bucket));
                }
            });
        }
        function modelBreakdown() {
            const result = {};
            const buckets = data.model_periods[state.period] || {};
            if (state.bucket) return { ...(buckets[state.bucket] || {}) };
            for (const models of Object.values(buckets)) {
                for (const [name, amount] of Object.entries(models)) result[name] = (result[name] || 0) + amount;
            }
            return result;
        }
        function renderTables() {
            const periodRows = byId("period-rows");
            periodRows.replaceChildren();
            const buckets = bucketNames().slice().reverse();
            byId("period-empty").hidden = buckets.length > 0;
            for (const bucket of buckets) {
                const row = document.createElement("tr");
                row.dataset.bucket = bucket;
                if (state.bucket === bucket) row.classList.add("selected");
                row.tabIndex = 0;
                row.setAttribute("aria-label", `${bucket}: ${number(selectedAmount(bucket))} Credits`);
                const period = document.createElement("td"); period.textContent = bucket;
                const amount = document.createElement("td"); amount.className = "amount"; amount.textContent = number(selectedAmount(bucket));
                const sessions = document.createElement("td"); sessions.className = "amount";
                sessions.textContent = number(data.session_periods?.[state.period]?.[bucket] || 0);
                row.append(period, amount, sessions);
                const activate = () => { state.bucket = state.bucket === bucket ? null : bucket; render(); };
                row.addEventListener("click", activate);
                row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activate(); } });
                periodRows.append(row);
            }
            const models = modelBreakdown();
            const modelRows = byId("model-rows");
            modelRows.replaceChildren();
            const entries = Object.entries(models).sort((a, b) => b[1] - a[1]);
            byId("model-empty").hidden = entries.length > 0;
            byId("model-scope").textContent = state.bucket || "当前区间";
            const maximum = Math.max(...entries.map((entry) => entry[1]), 0);
            for (const [index, [name, amount]] of entries.entries()) {
                const row = document.createElement("tr");
                const model = document.createElement("td"); model.className = "model-name"; model.textContent = name;
                const value = document.createElement("td"); value.className = "amount model-value";
                value.append(document.createTextNode(number(amount)));
                const track = document.createElement("div"); track.className = "bar-track";
                const fill = document.createElement("div"); fill.className = "bar-fill";
                fill.style.width = `${maximum ? amount / maximum * 100 : 0}%`;
                fill.style.background = colors[index % colors.length];
                track.append(fill); value.append(track); row.append(model, value); modelRows.append(row);
            }
        }
        function renderMetrics() {
            const buckets = bucketNames();
            const total = buckets.reduce((amount, bucket) => amount + selectedAmount(bucket), 0);
            const sessionTotal = buckets.reduce((count, bucket) => count + (data.session_periods?.[state.period]?.[bucket] || 0), 0);
            const peak = buckets.map((bucket) => ({ bucket, value: selectedAmount(bucket) })).sort((a, b) => b.value - a.value)[0];
            const current = data.current_usage;
            byId("today-label").textContent = `今日用量 · ${current.today.label}`;
            byId("today-value").textContent = number(current.today.value);
            byId("month-label").textContent = `本月用量 · ${current.month.label}`;
            byId("month-value").textContent = number(current.month.value);
            byId("total-value").textContent = number(total);
            byId("total-label").textContent = state.model === "*" ? `${labels[state.period]}区间用量` : "筛选模型用量";
            byId("session-count").textContent = number(sessionTotal);
            byId("peak-value").textContent = peak?.value ? peak.bucket : "-";
            byId("peak-note").textContent = peak?.value ? `${number(peak.value)} Credits` : "Credits";
            byId("model-count").textContent = modelNames().length;
            byId("undated-value").textContent = number(data.undated_credits);
            byId("trend-note").textContent = `${labels[state.period]} · ${state.model === "*" ? "全部模型" : state.model}`;
            byId("clear-selection").hidden = !state.bucket;
        }
        function render() {
            for (const button of document.querySelectorAll("[data-period]")) {
                button.setAttribute("aria-pressed", String(button.dataset.period === state.period));
            }
            populateModelFilter();
            renderMetrics();
            drawChart();
            renderTables();
        }
        document.querySelectorAll("[data-period]").forEach((button) => button.addEventListener("click", () => {
            state.period = button.dataset.period; state.bucket = null; render();
        }));
        byId("model-filter").addEventListener("change", (event) => { state.model = event.target.value; render(); });
        byId("clear-selection").addEventListener("click", () => { state.bucket = null; render(); });
        byId("status").textContent = data.message;
        byId("footer").textContent = `扫描 ${data.files_scanned} 个文件 · ${data.credit_records} 条有效记录 · ${data.malformed_records} 条无效记录 · Credits 字段：${(data.credit_fields || []).join(", ") || "未发现"}`;
        byId("generated-at").textContent = new Date(data.generated_at).toLocaleString("zh-CN");
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
        help="扫描指定文件或目录（可重复）；默认扫描 VS Code workspaceStorage 中的 Copilot 日志。",
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
