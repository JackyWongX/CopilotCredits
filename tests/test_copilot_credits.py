import json
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import copilot_credits


class CopilotCreditsTests(unittest.TestCase):
    def test_default_scan_includes_chat_sessions_with_copilot_credits(self):
        timestamp = 1792044758325
        with tempfile.TemporaryDirectory() as directory:
            session_dir = (
                Path(directory)
                / "Code"
                / "User"
                / "workspaceStorage"
                / "workspace-id"
                / "chatSessions"
            )
            session_dir.mkdir(parents=True)
            session = session_dir / "session.jsonl"
            session.write_text(
                json.dumps(
                    {
                        "kind": 2,
                        "v": [
                            {
                                "timestamp": timestamp,
                                "requestId": "request-1",
                                "copilotCredits": 1.25,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"APPDATA": directory}):
                with patch.object(copilot_credits.Path, "home", return_value=Path(directory)):
                    paths = copilot_credits.discover_default_paths()
                    result = copilot_credits.analyze_paths(paths)

        local_day = copilot_credits.parse_timestamp(timestamp).date()
        iso_year, iso_week, _ = local_day.isocalendar()
        self.assertIn(session_dir, paths)
        self.assertEqual(result["status"], "credits_found")
        self.assertEqual(result["periods"]["daily"], {local_day.isoformat(): 1.25})
        self.assertEqual(result["periods"]["monthly"], {local_day.strftime("%Y-%m"): 1.25})

    def test_default_scan_includes_copilot_agent_session_state(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            sessions = home / ".copilot" / "session-state"
            sessions.mkdir(parents=True)
            with patch.dict(os.environ, {"APPDATA": str(home)}):
                with patch.object(copilot_credits.Path, "home", return_value=home):
                    paths = copilot_credits.discover_default_paths()

        self.assertIn(sessions, paths)

    def test_agent_usage_credits_come_from_nano_aiu_not_per_request_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            sessions = Path(directory) / "session-state"
            sessions.mkdir()
            for session_id, nano_aiu in (("session-one", 1_000_000_000), ("session-two", 1_500_000_000)):
                events = (
                    {
                        "type": "session.start",
                        "timestamp": "2026-10-19T09:00:00Z",
                        "data": {"sessionId": session_id},
                    },
                    {
                        "type": "session.usage_record",
                        "timestamp": "2026-10-19T09:05:00Z",
                        "data": {
                            "usage": {
                                "model": "gpt-6-luna",
                                "modelDisplayName": "GPT-6 Luna",
                                "cost": 1,
                                "aiCreditsStatus": "complete",
                                "inputTokens": 1000,
                                "copilotUsage": {"totalNanoAiu": nano_aiu},
                            }
                        },
                    },
                )
                (sessions / f"{session_id}.jsonl").write_text(
                    "\n".join(json.dumps(event) for event in events),
                    encoding="utf-8",
                )

            result = copilot_credits.analyze_paths([sessions])

        self.assertEqual(result["periods"]["daily"], {"2026-10-19": 2.5})
        self.assertEqual(
            result["model_periods"]["daily"],
            {"2026-10-19": {"GPT-6 Luna": 2.5}},
        )
        self.assertEqual(result["session_periods"]["daily"], {"2026-10-19": 2})
        self.assertEqual(result["credit_fields"], ["usage.copilotUsage.totalNanoAiu"])
        self.assertEqual(result["credit_records"], 2)
        self.assertEqual(
            result["unpriced_usage_periods"]["daily"],
            {},
        )
        self.assertIn("Nano AIU", result["message"])

    def test_chat_subagent_credits_use_file_modified_date_when_timestamp_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "session.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        {
                            "kind": "subagent",
                            "modelName": "Claude Haiku 4.5",
                            "credits": 505.44,
                        },
                        {
                            "kind": "subagent",
                            "modelName": "GPT-6 Luna",
                            "credits": 2,
                            "timestamp": "2026-10-27T09:00:00Z",
                        },
                    )
                ),
                encoding="utf-8",
            )
            file_time = datetime(2026, 10, 28, 12).timestamp()
            os.utime(source, (file_time, file_time))

            result = copilot_credits.analyze_paths([source])

        self.assertEqual(
            result["periods"]["daily"],
            {"2026-10-27": 2, "2026-10-28": 505.44},
        )
        self.assertEqual(result["undated_credits"], 0)
        self.assertEqual(result["file_timestamp_records"], 1)
        self.assertIn("按日志文件修改日期归档", result["message"])

    def test_incomplete_agent_usage_cost_is_not_counted_as_credits(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "type": "session.usage_record",
                        "timestamp": "2026-10-19T09:05:00Z",
                        "data": {"usage": {"cost": 1, "aiCreditsStatus": "pending"}},
                    }
                ),
                encoding="utf-8",
            )

            result = copilot_credits.analyze_paths([source])

        self.assertEqual(result["status"], "no_credit_fields")
        self.assertEqual(result["periods"]["daily"], {})

    def test_completed_agent_usage_without_nano_aiu_is_reported_as_unpriced(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "type": "session.usage_record",
                        "timestamp": "2026-10-19T09:05:00Z",
                        "data": {
                            "usage": {
                                "cost": 1,
                                "aiCreditsStatus": "complete",
                                "copilotUsage": {},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = copilot_credits.analyze_paths([source])

        self.assertEqual(result["periods"]["daily"], {})
        self.assertEqual(
            result["unpriced_usage_periods"]["daily"],
            {"2026-10-19": 1},
        )

    def test_undated_completed_agent_usage_is_reported_and_makes_current_usage_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "type": "session.usage_record",
                            **({"timestamp": timestamp} if timestamp is not None else {}),
                            "data": {
                                "usage": {
                                    "aiCreditsStatus": "complete",
                                    "copilotUsage": {},
                                }
                            },
                        }
                    )
                    for timestamp in (None, "not-a-timestamp")
                ),
                encoding="utf-8",
            )

            result = copilot_credits.analyze_paths([source])
            current = copilot_credits._current_usage(
                {
                    **result,
                    "periods": {
                        "daily": {"2026-10-16": 2.5},
                        "monthly": {"2026-10": 12.5},
                    },
                },
                datetime(2026, 10, 16, 12).astimezone(),
            )

        self.assertEqual(result["undated_unpriced_usage_records"], 2)
        self.assertIn("2 条缺少可解析时间的记录", result["message"])
        self.assertIn("未分组的未计价用量事件: 2 条", copilot_credits.format_table(result))
        self.assertIsNone(current["today"]["value"])
        self.assertEqual(current["today"]["unpriced_records"], 2)
        self.assertIsNone(current["month"]["value"])
        self.assertEqual(current["month"]["unpriced_records"], 2)

    def test_task_credits_match_response_details_after_aggregating_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            sessions = Path(directory) / "session-state"
            sessions.mkdir()
            events = [
                {
                    "type": "session.start",
                    "timestamp": "2026-10-19T10:05:24Z",
                    "data": {"sessionId": "task-session"},
                }
            ]
            for nano_aiu in [162_482_607] * 13 + [162_482_609]:
                events.append(
                    {
                        "type": "session.usage_record",
                        "timestamp": "2026-10-19T10:06:00Z",
                        "data": {
                            "usage": {
                                "modelDisplayName": "GPT-6 Luna",
                                "cost": 1,
                                "aiCreditsStatus": "complete",
                                "copilotUsage": {"totalNanoAiu": nano_aiu},
                            }
                        },
                    }
                )
            (sessions / "task-session.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events),
                encoding="utf-8",
            )

            result = copilot_credits.analyze_paths([sessions])

        self.assertEqual(result["periods"]["daily"], {"2026-10-19": 2.2747565})
        self.assertEqual(
            result["model_periods"]["daily"],
            {"2026-10-19": {"GPT-6 Luna": 2.2747565}},
        )
        self.assertEqual(result["session_periods"]["daily"], {"2026-10-19": 1})

    def test_session_counts_use_creation_date_and_count_each_file_once(self):
        with tempfile.TemporaryDirectory() as directory:
            sessions = Path(directory) / "chatSessions"
            sessions.mkdir()
            (sessions / "session-one.jsonl").write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        {"kind": 0, "v": {"creationDate": "2026-10-12T08:00:00"}},
                        {"kind": 2, "v": [{"timestamp": "2026-10-12T09:00:00", "copilotCredits": 1.5}]},
                        {"kind": 2, "v": [{"timestamp": "2026-10-13T09:00:00", "copilotCredits": 2}]},
                    )
                ),
                encoding="utf-8",
            )
            (sessions / "session-two.jsonl").write_text(
                json.dumps({"kind": 0, "v": {"creationDate": "2026-10-13T10:00:00"}}),
                encoding="utf-8",
            )

            result = copilot_credits.analyze_paths([sessions])

        self.assertEqual(result["session_periods"]["daily"], {"2026-10-12": 1, "2026-10-13": 1})
        self.assertEqual(result["session_periods"]["weekly"], {"2026-W42": 2})
        self.assertEqual(result["periods"]["weekly"], {"2026-W42": 3.5})
        self.assertIn("每周", copilot_credits.format_table(result))
        self.assertIn("3.50", copilot_credits.format_table(result))
        self.assertIn("2 个会话", copilot_credits.format_table(result))

    def test_copilot_auto_is_attributed_to_gpt6_luna(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        {
                            "timestamp": "2026-10-12T10:00:00",
                            "modelId": "copilot/auto",
                            "copilotCredits": 1.5,
                        },
                        {
                            "timestamp": "2026-10-12T10:01:00",
                            "model": "Auto",
                            "copilotCredits": 2,
                        },
                        {
                            "timestamp": "2026-10-12T10:02:00",
                            "modelId": "copilot/auto",
                            "resolvedModel": "gpt-6-luna",
                            "copilotCredits": 3,
                        },
                    )
                ),
                encoding="utf-8",
            )

            result = copilot_credits.analyze_paths([source])

        self.assertEqual(
            result["model_periods"]["daily"],
            {"2026-10-12": {"GPT-6 Luna": 6.5}},
        )
        self.assertNotIn("copilot/auto", str(result["model_periods"]))

    def test_html_report_shows_undated_credits_as_separate_model_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "type": "session.usage_record",
                        "data": {
                            "usage": {
                                "modelDisplayName": "Claude Haiku 4.5",
                                "aiCreditsStatus": "complete",
                                "copilotUsage": {"totalNanoAiu": 505_440_000_000},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = copilot_credits.analyze_paths([source])
            report = Path(directory) / "report.html"
            copilot_credits.write_html_report(result, report)
            html = report.read_text(encoding="utf-8")

        self.assertEqual(
            result["undated_by_model"],
            {"Claude Haiku 4.5": 505.44},
        )
        self.assertIn("addModels(data.undated_by_model || {}, true);", html)
        self.assertIn("unknownDate ? `${name}（日期未知）` : name", html)

    def test_current_usage_uses_generation_day_and_month(self):
        result = {
            "periods": {
                "daily": {"2026-10-16": 2.5},
                "monthly": {"2026-10": 12.5},
            }
        }

        current = copilot_credits._current_usage(
            result,
            datetime(2026, 10, 16, 12).astimezone(),
        )

        self.assertEqual(
            current["today"],
            {"label": "2026-10-16", "value": 2.5, "unpriced_records": 0},
        )
        self.assertEqual(
            current["month"],
            {"label": "2026-10", "value": 12.5, "unpriced_records": 0},
        )

    def test_current_usage_is_unknown_when_logs_have_unpriced_usage(self):
        result = {
            "periods": {"daily": {}, "monthly": {}},
            "unpriced_usage_periods": {
                "daily": {"2026-10-16": 2},
                "monthly": {"2026-10": 2},
            },
        }

        current = copilot_credits._current_usage(
            result,
            datetime(2026, 10, 16, 12).astimezone(),
        )

        self.assertIsNone(current["today"]["value"])
        self.assertEqual(current["today"]["unpriced_records"], 2)
        self.assertIsNone(current["month"]["value"])
        self.assertEqual(current["month"]["unpriced_records"], 2)

    def test_current_usage_is_unknown_when_undated_usage_is_unpriced(self):
        result = {
            "periods": {
                "daily": {"2026-10-16": 2.5},
                "monthly": {"2026-10": 12.5},
            },
            "undated_unpriced_usage_records": 1,
        }

        current = copilot_credits._current_usage(
            result,
            datetime(2026, 10, 16, 12).astimezone(),
        )

        self.assertIsNone(current["today"]["value"])
        self.assertEqual(current["today"]["unpriced_records"], 1)
        self.assertIsNone(current["month"]["value"])
        self.assertEqual(current["month"]["unpriced_records"], 1)

    def test_jsonl_aggregates_daily_iso_week_and_month_and_skips_bad_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "events.jsonl"
            log.write_text(
                '\n'.join(
                    (
                        json.dumps(
                            {
                                "timestamp": "2026-10-12T10:00:00",
                                "modelId": "model-a",
                                "creditsUsed": 1.5,
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-10-13T10:00:00",
                                "modelId": "model-b",
                                "usage": {"credits": 2},
                            }
                        ),
                        "not valid json",
                    )
                ),
                encoding="utf-8",
            )

            result = copilot_credits.analyze_paths([log])

        self.assertEqual(result["status"], "credits_found")
        self.assertEqual(result["periods"]["daily"], {"2026-10-12": 1.5, "2026-10-13": 2})
        self.assertEqual(result["periods"]["monthly"], {"2026-10": 3.5})
        self.assertEqual(
            result["model_periods"]["daily"],
            {"2026-10-12": {"model-a": 1.5}, "2026-10-13": {"model-b": 2}},
        )
        self.assertEqual(result["malformed_records"], 1)

    def test_json_supports_nested_credit_usage_object(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "events.json"
            log.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "createdAt": "2026-02-03T12:00:00",
                                "usage": {"creditUsage": {"used": "0.75"}},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = copilot_credits.analyze_paths([log])

        self.assertEqual(result["periods"]["daily"], {"2026-02-03": 0.75})
        self.assertEqual(result["credit_records"], 1)

    def test_tokens_and_request_counts_do_not_count_as_credits(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "events.json"
            log.write_text(
                json.dumps(
                    {
                        "timestamp": "2026-10-12T10:00:00",
                        "inputTokens": 500,
                        "outputTokens": 250,
                        "toolRequests": 3,
                        "model": "example-model",
                    }
                ),
                encoding="utf-8",
            )

            result = copilot_credits.analyze_paths([log])

        self.assertEqual(result["status"], "no_credit_fields")
        self.assertIn("无可统计的 Credits 字段", result["message"])
        self.assertEqual(result["periods"]["daily"], {})
        self.assertNotIn("0 Credits", copilot_credits.format_table(result))

    def test_explicit_zero_is_a_real_credits_record(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "events.jsonl"
            log.write_text(
                json.dumps({"timestamp": "2026-10-12", "credits": 0}),
                encoding="utf-8",
            )

            result = copilot_credits.analyze_paths([log])

        self.assertEqual(result["status"], "credits_found")
        self.assertEqual(result["periods"]["daily"], {"2026-10-12": 0})

    def test_explicit_but_unparseable_credit_field_is_not_reported_as_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "events.json"
            log.write_text(json.dumps({"credits": None}), encoding="utf-8")

            result = copilot_credits.analyze_paths([log])

        self.assertEqual(result["status"], "credits_found")
        self.assertEqual(result["credit_records"], 0)
        self.assertIn("没有可解析的数值", result["message"])

    def test_html_report_contains_interactive_model_and_period_data(self):
        unsafe_model = "</script><script>alert(1)</script>"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "timestamp": "2026-10-12T10:00:00",
                        "modelId": unsafe_model,
                        "copilotCredits": 1.25,
                    }
                ),
                encoding="utf-8",
            )
            report = Path(directory) / "report.html"
            result = copilot_credits.analyze_paths([source])
            copilot_credits.write_html_report(result, report)
            html = report.read_text(encoding="utf-8")

        self.assertIn('id="period-control"', html)
        self.assertIn('id="model-filter"', html)
        self.assertIn('id="month-filter"', html)
        self.assertIn("const availableMonths = () =>", html)
        self.assertIn("Object.keys(data.periods.daily || {})", html)
        self.assertIn("state.period === \"daily\" && state.month", html)
        self.assertNotIn('<article class="metric">', html)
        self.assertIn('<strong class="metric-value" id="today-value">', html)
        self.assertIn('<strong class="metric-value" id="month-value">', html)
        self.assertIn('id="period-chart"', html)
        self.assertIn('const chartColors = { credits: "#ff3d8e", sessions: "#a855f7" };', html)
        self.assertIn("Math.max(640, buckets.length * 88 + 100, chart.parentElement.clientWidth)", html)
        self.assertIn("window.addEventListener(\"resize\", renderPeriodChart);", html)
        self.assertIn('id="chart-tooltip"', html)
        self.assertIn('id="model-chart"', html)
        self.assertIn('const creditsNumber = (value) => new Intl.NumberFormat("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })', html)
        self.assertIn('data-series": "credits"', html)
        self.assertIn('data-series": "sessions"', html)
        self.assertIn('data-value": creditValue', html)
        self.assertIn('data-value": sessionValue', html)
        self.assertIn("function showTooltip(bucket, clientX, clientY)", html)
        self.assertIn(".chart-tooltip-table { width: 100%; border-collapse: collapse; }", html)
        self.assertIn(".chart-tooltip-table td { max-width: 210px; font-variant-numeric: tabular-nums; text-align: left;", html)
        self.assertIn('table.className = "chart-tooltip-table";', html)
        self.assertIn('["Credits", creditsNumber(selectedAmount(bucket))]', html)
        self.assertIn('["会话数（全部模型）", number(sessionValue)]', html)
        self.assertIn('["模型筛选", modelLabel]', html)
        self.assertIn("cell.textContent = value;", html)
        self.assertIn("const creditPlotHeight = plotHeight * 0.62;", html)
        self.assertIn("const sessionPlotHeight = plotHeight - creditPlotHeight;", html)
        self.assertIn("const baselineY = margin.top + creditPlotHeight;", html)
        self.assertIn("const creditHeight = creditValue / creditScale * creditPlotHeight;", html)
        self.assertIn("const sessionHeight = sessionValue / sessionScale * sessionPlotHeight;", html)
        self.assertIn("const barWidth = Math.min(step * 0.8, 48, Math.max(12, 96 / Math.sqrt(buckets.length)));", html)
        self.assertIn("const barX = center - barWidth / 2;", html)
        self.assertIn("x: barX, y: baselineY - creditHeight,", html)
        self.assertIn("x: barX, y: baselineY,", html)
        self.assertIn("direction: ltr; width: 100%;", html)
        self.assertIn('role: "button"', html)
        self.assertIn('event.key === "Enter" || event.key === " "', html)
        self.assertIn('row.className = "model-chart-row"', html)
        self.assertIn('value.className = "model-chart-value"', html)
        self.assertIn(".model-chart-track { grid-column: 1 / -1; grid-row: 2;", html)
        self.assertIn(".model-chart-value { grid-column: 2; grid-row: 1;", html)
        self.assertNotIn('class="masthead"', html)
        self.assertNotIn('id="status"', html)
        self.assertNotIn('<h2>周期明细</h2>', html)
        self.assertNotIn('<h2>模型用量</h2>', html)
        self.assertIn('id="clear-selection"', html)
        self.assertIn('data-period="weekly"', html)
        self.assertIn('"session_periods"', html)
        self.assertNotIn('id="period-rows"', html)
        self.assertNotIn('id="model-rows"', html)
        self.assertNotIn('byId("model-rows")', html)
        self.assertNotIn("<table>", html)
        self.assertNotIn("function renderTables()", html)
        self.assertIn("#ff3d8e", html)
        self.assertIn('"current_usage"', html)
        self.assertIn('current.today.value === null ? "—" : creditsNumber(current.today.value)', html)
        self.assertIn('current.month.value === null ? "—" : creditsNumber(current.month.value)', html)
        self.assertIn('"model_periods"', html)
        self.assertNotIn(unsafe_model, html)
        self.assertIn("\\u003c/script\\u003e", html)

    def test_credit_display_rounds_half_up_to_two_decimal_places(self):
        self.assertEqual(copilot_credits._format_credits(1.005), "1.01")
        self.assertEqual(copilot_credits._format_credits(2.999), "3.00")

    def test_main_writes_report_then_opens_browser(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.json"
            source.write_text(
                json.dumps(
                    {
                        "timestamp": "2026-10-12T10:00:00",
                        "modelId": "model-a",
                        "copilotCredits": 2,
                    }
                ),
                encoding="utf-8",
            )
            report = Path(directory) / "report.html"
            output = io.StringIO()
            with patch.object(copilot_credits.webbrowser, "open", return_value=True) as open_browser:
                with redirect_stdout(output):
                    status = copilot_credits.main(
                        ["--path", str(source), "--output", str(report)]
                    )
            report_exists = report.is_file()

        self.assertEqual(status, 0)
        self.assertTrue(report_exists)
        open_browser.assert_called_once_with(report.resolve().as_uri())
        self.assertIn("已在默认浏览器中打开报告", output.getvalue())

    def test_no_open_option_skips_browser_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.json"
            source.write_text("{}", encoding="utf-8")
            report = Path(directory) / "report.html"
            with patch.object(copilot_credits.webbrowser, "open") as open_browser:
                with redirect_stdout(io.StringIO()):
                    status = copilot_credits.main(
                        ["--path", str(source), "--output", str(report), "--no-open"]
                    )
            report_exists = report.is_file()

        self.assertEqual(status, 0)
        self.assertTrue(report_exists)
        open_browser.assert_not_called()

    def test_cache_reuses_contributions_and_matches_cold_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "events.jsonl"
            source.write_text(
                "\n".join(
                    (
                        json.dumps({"timestamp": "2026-10-12", "model": "model-a", "credits": "1.25"}),
                        json.dumps({"creationDate": "2026-10-13", "credits": 2}),
                        "not valid json",
                    )
                ),
                encoding="utf-8",
            )
            cache_dir = root / "cache"
            cold = copilot_credits.analyze_paths([source])

            with patch.object(copilot_credits, "_load_file", wraps=copilot_credits._load_file) as load:
                first = copilot_credits.analyze_paths([source], cache_dir=cache_dir)
                self.assertEqual(load.call_count, 1)
            self.assertEqual(first["files_reparsed"], 1)
            self.assertEqual(first["cache_hits"], 0)

            with patch.object(copilot_credits, "_load_file", wraps=copilot_credits._load_file) as load:
                second = copilot_credits.analyze_paths([source], cache_dir=cache_dir)
                load.assert_not_called()
            self.assertEqual(second["files_reparsed"], 0)
            self.assertEqual(second["cache_hits"], 1)
            for result in (first, second):
                result.pop("cache_hits")
                result.pop("files_reparsed")
            self.assertEqual(first, cold)
            self.assertEqual(second, cold)

    def test_cache_reparses_changed_and_added_files_and_excludes_deleted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "sources"
            sources.mkdir()
            changed = sources / "changed.json"
            deleted = sources / "deleted.json"
            added = sources / "added.json"
            changed.write_text(json.dumps({"timestamp": "2026-10-12", "credits": 1}), encoding="utf-8")
            deleted.write_text(json.dumps({"timestamp": "2026-10-12", "credits": 2}), encoding="utf-8")
            cache_dir = root / "cache"
            copilot_credits.analyze_paths([sources], cache_dir=cache_dir)

            changed.write_text(
                json.dumps({"timestamp": "2026-10-12", "credits": "100.5"}), encoding="utf-8"
            )
            deleted.unlink()
            added.write_text(json.dumps({"timestamp": "2026-10-12", "credits": 4}), encoding="utf-8")
            with patch.object(copilot_credits, "_load_file", wraps=copilot_credits._load_file) as load:
                result = copilot_credits.analyze_paths([sources], cache_dir=cache_dir)

        self.assertEqual(load.call_count, 2)
        self.assertEqual(result["files_scanned"], 2)
        self.assertEqual(result["files_reparsed"], 2)
        self.assertEqual(result["cache_hits"], 0)
        self.assertEqual(result["periods"]["daily"], {"2026-10-12": 104.5})

    def test_cache_keeps_entries_for_other_scan_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir()
            second_root.mkdir()
            first_file = first_root / "first.json"
            second_file = second_root / "second.json"
            first_file.write_text(json.dumps({"credits": 1}), encoding="utf-8")
            second_file.write_text(json.dumps({"credits": 2}), encoding="utf-8")
            cache_dir = root / "cache"
            copilot_credits.analyze_paths([first_root], cache_dir=cache_dir)
            copilot_credits.analyze_paths([second_root], cache_dir=cache_dir)

            with patch.object(copilot_credits, "_load_file", wraps=copilot_credits._load_file) as load:
                result = copilot_credits.analyze_paths([first_root], cache_dir=cache_dir)

        load.assert_not_called()
        self.assertEqual(result["cache_hits"], 1)
        self.assertEqual(result["credit_records"], 1)
        self.assertEqual(result["undated_credits"], 1)

    def test_corrupt_cache_falls_back_to_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "events.json"
            source.write_text(json.dumps({"timestamp": "2026-10-12", "credits": 3}), encoding="utf-8")
            cache_dir = root / "cache"
            cache_dir.mkdir()
            (cache_dir / copilot_credits.CACHE_FILENAME).write_text("{broken", encoding="utf-8")

            with patch.object(copilot_credits, "_load_file", wraps=copilot_credits._load_file) as load:
                result = copilot_credits.analyze_paths([source], cache_dir=cache_dir)

        load.assert_called_once_with(source)
        self.assertEqual(result["files_reparsed"], 1)
        self.assertEqual(result["periods"]["daily"], {"2026-10-12": 3})


if __name__ == "__main__":
    unittest.main()
