import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("nemo_eval", Path(__file__).resolve().parents[1] / "evals/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ToolOutcomeTests(unittest.TestCase):
    def test_failure_in_structured_checker_output_is_counted(self):
        self.assertTrue(runner.tool_failed('Diagnostics source: ruff\n{"exit_code": 1}'))
        self.assertTrue(runner.tool_failed('{"exit_code": 0, "timed_out": true}'))
        self.assertTrue(runner.tool_failed('Error: missing dependency'))
        self.assertFalse(runner.tool_failed('{"exit_code": 0, "timed_out": false}'))
        self.assertFalse(runner.tool_failed('plain successful file content'))


if __name__ == "__main__":
    unittest.main()
