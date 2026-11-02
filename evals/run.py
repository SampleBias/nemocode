#!/usr/bin/env python3
"""Repeatable local harness evaluation. Validators stay outside agent workspaces.

This small public suite is a regression baseline, not an independent benchmark.
Supply --tasks with private held-out tasks for model selection/training decisions.
"""
import argparse
import contextlib
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tool_failed(result):
    if result.startswith("Error:"):
        return True
    for line in result.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "exit_code" in value:
            return value["exit_code"] != 0 or value.get("timed_out", False) or value.get("cancelled", False)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="target/release/nemocode")
    parser.add_argument("--tasks", type=Path, default=Path(__file__).with_name("tasks.json"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="Retain fixture workspaces, sessions, and CLI output for diagnosis")
    parser.add_argument("--config", type=Path, help="Use a NemoCode config file for each task")
    parser.add_argument("--base-url", help="Local model endpoint (overrides NEMO_BASE_URL)")
    parser.add_argument("--report", type=Path, help="Write the aggregate JSON report to a file")
    args = parser.parse_args()
    tasks = json.loads(args.tasks.read_text())
    if args.list:
        print(json.dumps([task["id"] for task in tasks]))
        return
    binary = str(Path(args.binary).resolve())
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for repetition in range(args.repeat):
        for task in tasks:
            context = (contextlib.nullcontext(tempfile.mkdtemp(prefix="nemo-eval-", dir=args.output_dir))
                       if args.output_dir else tempfile.TemporaryDirectory(prefix="nemo-eval-"))
            with context as directory:
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
                if args.base_url:
                    env["NEMO_BASE_URL"] = args.base_url
                config_args = ["--config", str(args.config.resolve())] if args.config else []
                started = time.monotonic()
                try:
                    process = subprocess.run(
                        [binary, "run", "--workspace", str(workspace), "--session-dir", str(root / "session"),
                         "--permission", "workspace", "--prompt", task["prompt"], *config_args],
                        env=env, capture_output=True, text=True, timeout=args.timeout + 10,
                    )
                    if args.output_dir:
                        (root / "stdout.json").write_text(process.stdout)
                        (root / "stderr.log").write_text(process.stderr)
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
                    runtime = next((e["data"] for e in events if e["kind"] == "runtime"), None)
                    record = {"id": task["id"], "repetition": repetition,
                              "passed": process.returncode == 0 and verified.returncode == 0,
                              "agent_ok": response.get("ok", False),
                              "agent_error": response.get("error"),
                              "agent_stderr": process.stderr[-4000:] if process.returncode else "",
                              "generated_tokens": sum(e.get("generated_tokens", 0) for e in completions),
                              "prompt_tokens": sum(e.get("prompt_tokens", 0) or 0 for e in completions),
                              "first_token_ms": completions[0].get("ttft_ms") if completions else None,
                              "runtime": runtime,
                              "tool_calls": len(tools),
                              "tool_errors": sum(tool_failed(str(e.get("result", ""))) for e in tools),
                              "validator_passed": verified.returncode == 0,
                              "agent_exit_code": process.returncode,
                              "validation_error": verified.stderr[-2000:]}
                    if record["passed"]:
                        record["outcome"] = "passed"
                    elif "budget" in str(record["agent_error"]).lower() or "timeout" in str(record["agent_error"]).lower():
                        record["outcome"] = "agent_timeout"
                    elif process.returncode != 0:
                        record["outcome"] = "agent_error"
                    else:
                        record["outcome"] = "validation_failed"
                except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                    record = {"id": task["id"], "repetition": repetition, "passed": False,
                              "outcome": "subprocess_timeout" if isinstance(error, subprocess.TimeoutExpired) else "invalid_agent_json",
                              "error": str(error)}
                record["wall_seconds"] = round(time.monotonic() - started, 3)
                if args.output_dir:
                    record["artifacts"] = str(root.resolve())
                results.append(record)
                print(json.dumps(record), file=sys.stderr, flush=True)
    report = json.dumps({"model": os.environ.get("NEMO_MODEL", "bundled default"),
                         "metadata": {"platform": platform.platform(), "python": sys.version,
                                      "binary_sha256": digest(binary), "tasks_sha256": digest(args.tasks),
                                      "config_sha256": digest(args.config) if args.config else None,
                                      "base_url": args.base_url or os.environ.get("NEMO_BASE_URL", "default"),
                                      "thinking_override": os.environ.get("NEMO_THINKING"),
                                      "timeout_seconds": args.timeout, "repeat": args.repeat,
                                      "seeds": [42 + i for i in range(args.repeat)]},
                         "passed": sum(r["passed"] for r in results), "total": len(results), "results": results})
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report + "\n")
    print(report)
    if not all(r["passed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
