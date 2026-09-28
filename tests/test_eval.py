import importlib.util
from pathlib import Path
import unittest
import sys

spec = importlib.util.spec_from_file_location("nemo_eval", Path(__file__).resolve().parents[1] / "evals/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class EvaluationMetricsTests(unittest.TestCase):
    def test_deadline_retains_output_instead_of_discarding_it(self):
        import os
        code, stdout, stderr, timed_out, rss = runner.run_agent(
            [sys.executable,"-u","-c","import time,sys; print('evidence',flush=True); print('diagnostic',file=sys.stderr,flush=True); time.sleep(10)"],
            os.environ.copy(),0.2)
        self.assertTrue(timed_out)
        self.assertNotEqual(code,0)
        self.assertIn("evidence",stdout)
        self.assertIn("diagnostic",stderr)
        self.assertTrue(rss is None or rss > 0)

    def test_structured_failures_and_interrupted_usage_are_counted(self):
        events = [
            {"kind": "completion", "data": {"generated_tokens": 17, "complete": False,
                "token_count_source": "estimate", "usage": None, "wall_ms": 25}},
            {"kind": "tool_finished", "data": {"outcome": {"schema_version": 1,
                "status": "failed", "error_code": "check_failed"}, "duration_ms": 12}},
            {"kind": "tool_finished", "data": {"outcome": {"schema_version": 1,
                "status": "unavailable", "error_code": "no_tests_collected"}}},
            {"kind": "tool_finished", "data": {"outcome": {"schema_version": 1,
                "status": "success"}, "task_elapsed_ms": 60}},
            {"kind": "protocol_error", "data": {}},
        ]
        metrics = runner.summarize_events(events)
        self.assertEqual(metrics["generated_tokens"], 17)
        self.assertEqual(metrics["tool_errors"], 2)
        self.assertEqual(metrics["tool_errors_by_code"], {"check_failed": 1, "no_tests_collected": 1})
        self.assertEqual(metrics["first_successful_tool_ms"], 60)
        self.assertEqual(metrics["protocol_errors"], 1)

    def test_legacy_reports_remain_readable(self):
        metrics = runner.summarize_events([
            {"kind": "tool_finished", "data": {"result": "Error: missing"}},
            {"kind": "completion", "data": {"usage": {"prompt_tokens_details": {"cached_tokens": 42}}}},
        ])
        self.assertEqual(metrics["tool_errors"], 1)
        self.assertEqual(metrics["cached_prompt_tokens"], 42)


if __name__ == "__main__":
    unittest.main()
