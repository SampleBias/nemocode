#!/usr/bin/env python3
"""Repeatable local harness evaluation. Validators stay outside agent workspaces.

This small public suite is a regression baseline, not an independent benchmark.
Supply --tasks with private held-out tasks for model selection/training decisions.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="target/release/nemocode")
    parser.add_argument("--tasks", type=Path, default=Path(__file__).with_name("tasks.json"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    tasks = json.loads(args.tasks.read_text())
    if args.list:
        print(json.dumps([task["id"] for task in tasks]))
        return
    binary = str(Path(args.binary).resolve())
    results = []
    for repetition in range(args.repeat):
        for task in tasks:
            with tempfile.TemporaryDirectory(prefix="nemo-eval-") as directory:
                root = Path(directory)
                workspace = root / "workspace"
                workspace.mkdir()
                for name, content in task["files"].items():
                    path = (workspace / name).resolve()
                    if not path.is_relative_to(workspace):
                        raise ValueError(f"fixture path escapes workspace: {name}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content)
                env = os.environ.copy()
                # Reproducible sampling. Users can override the model/base URL.
                env.update(NEMO_SEED=str(42 + repetition), NEMO_TASK_TIMEOUT_SECS=str(args.timeout), NEMO_PYTHON_LSP="off")
                started = time.monotonic()
                try:
                    process = subprocess.run(
                        [binary, "run", "--workspace", str(workspace), "--session-dir", str(root / "session"),
                         "--permission", "workspace", "--prompt", task["prompt"]],
                        env=env, capture_output=True, text=True, timeout=args.timeout + 10,
                    )
                    response = json.loads(process.stdout) if process.stdout.strip() else {}
                    checker = "import sys;sys.path.insert(0,sys.argv[1]);\n" + task["verify"]
                    verified = subprocess.run([sys.executable, "-I", "-c", checker, str(workspace)],
                                              cwd=workspace, capture_output=True, text=True, timeout=15)
                    events = []
                    journal = root / "session/events.jsonl"
                    if journal.exists():
                        events = [json.loads(line) for line in journal.read_text().splitlines()]
                    completions = [e["data"] for e in events if e["kind"] == "completion"]
                    tools = [e["data"] for e in events if e["kind"] == "tool_finished"]
                    record = {"id": task["id"], "repetition": repetition,
                              "passed": process.returncode == 0 and verified.returncode == 0,
                              "agent_ok": response.get("ok", False),
                              "generated_tokens": sum(e.get("generated_tokens", 0) for e in completions),
                              "tool_calls": len(tools),
                              "tool_errors": sum(str(e.get("result", "")).startswith("Error:") for e in tools),
                              "validation_error": verified.stderr[-2000:]}
                except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                    record = {"id": task["id"], "repetition": repetition, "passed": False, "error": str(error)}
                record["wall_seconds"] = round(time.monotonic() - started, 3)
                results.append(record)
                print(json.dumps(record), file=sys.stderr, flush=True)
    print(json.dumps({"model": os.environ.get("NEMO_MODEL", "bundled default"),
                      "passed": sum(r["passed"] for r in results), "total": len(results), "results": results}))
    if not all(r["passed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
