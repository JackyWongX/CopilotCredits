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
                paths = copilot_credits.discover_default_paths()
                result = copilot_credits.analyze_paths(paths)

        local_day = copilot_credits.parse_timestamp(timestamp).date()
        iso_year, iso_week, _ = local_day.isocalendar()
        self.assertIn(session_dir, paths)
        self.assertEqual(result["status"], "credits_found")
        self.assertEqual(result["periods"]["daily"], {local_day.isoformat(): 1.25})
        self.assertEqual(result["periods"]["monthly"], {local_day.strftime("%Y-%m"): 1.25})

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

        self.assertEqual(current["today"], {"label": "2026-10-16", "value": 2.5})
        self.assertEqual(current["month"], {"label": "2026-10", "value": 12.5})

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
        self.assertNotIn('<article class="metric">', html)
        self.assertIn('<strong class="metric-value" id="today-value">', html)
        self.assertIn('<strong class="metric-value" id="month-value">', html)
        self.assertNotIn('id="trend-chart"', html)
        self.assertNotIn('id="chart-tooltip"', html)
        self.assertNotIn("Credits 趋势", html)
        self.assertNotIn('class="masthead"', html)
        self.assertNotIn('id="status"', html)
        self.assertNotIn('<h2>周期明细</h2>', html)
        self.assertNotIn('<h2>模型用量</h2>', html)
        self.assertIn('id="clear-selection"', html)
        self.assertIn('data-period="weekly"', html)
        self.assertIn('"session_periods"', html)
        self.assertIn(".period-bar-track { display: flex; justify-content: flex-end; }", html)
        self.assertIn("const creditMaximum = Math.max(...buckets.map(selectedAmount), 0);", html)
        self.assertIn("const sessionMaximum = Math.max(...buckets.map((bucket) => data.session_periods?.[state.period]?.[bucket] || 0), 0);", html)
        self.assertIn('track.className = "bar-track period-bar-track"', html)
        self.assertIn("maximum ? value / maximum * 100 : 0", html)
        self.assertIn("#ff3d8e", html)
        self.assertIn('"current_usage"', html)
        self.assertIn('"model_periods"', html)
        self.assertNotIn(unsafe_model, html)
        self.assertIn("\\u003c/script\\u003e", html)

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
