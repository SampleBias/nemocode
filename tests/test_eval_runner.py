import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("nemo_eval", Path(__file__).resolve().parents[1] / "evals/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ToolOutcomeTests(unittest.TestCase):
    def test_failure_in_structured_checker_output_is_counted(self):
        events = [
            {"kind": "tool_finished", "data": {"outcome": {"schema_version": 1, "status": status}}}
            for status in ("failed", "timed_out", "unavailable", "success")
        ]
        self.assertEqual(runner.summarize_events(events)["tool_errors"], 3)


if __name__ == "__main__":
    unittest.main()
