#!/usr/bin/env python3
"""Repeatable local harness evaluation. Validators stay outside agent workspaces.

This small public suite is a regression baseline, not an independent benchmark.
Supply --tasks with private held-out tasks for model selection/training decisions.
"""
import argparse
from collections import Counter
import contextlib
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import time


def summarize_events(events):
    completions = [e["data"] for e in events if e.get("kind") == "completion"]
    requests = [e["data"] for e in events if e.get("kind") == "request"]
    tools = [e["data"] for e in events if e.get("kind") == "tool_finished"]
    errors = Counter()
    useful = []
    for tool in tools:
        outcome = tool.get("outcome")
        if not isinstance(outcome, dict):
            try:
                outcome = json.loads(tool.get("result", ""))
            except (ValueError, TypeError):
                outcome = {}
        if isinstance(outcome, dict) and outcome.get("schema_version") == 1:
            if outcome.get("status") != "success":
                errors[outcome.get("error_code") or outcome.get("status", "unknown")] += 1
            elif isinstance(tool.get("task_elapsed_ms"), (int, float)):
                useful.append(tool["task_elapsed_ms"])
        elif str(tool.get("result", "")).startswith("Error:"):
            errors["legacy_error"] += 1
    return {
        "model_requests": len(requests),
        "generated_tokens": sum(e.get("generated_tokens", 0) for e in completions),
        "estimated_token_completions": sum(e.get("token_count_source") == "estimate" for e in completions),
        "prompt_tokens": sum(e.get("prompt_tokens") or 0 for e in requests),
        "cached_prompt_tokens": sum((((e.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0) for e in completions),
        "tool_calls": len(tools), "tool_errors": sum(errors.values()), "tool_errors_by_code": dict(errors),
        "protocol_errors": sum(e.get("kind") == "protocol_error" for e in events),
        "repeated_tool_calls": sum(e.get("kind") == "repeated_tool" for e in events),
        "generation_wall_ms": sum(e.get("wall_ms", 0) for e in completions),
        "prefill_ms": sum((e.get("timings") or {}).get("prompt_ms", 0) for e in completions),
        "decode_ms": sum((e.get("timings") or {}).get("predicted_ms", 0) for e in completions),
        "tool_wall_ms": sum(e.get("duration_ms", 0) for e in tools),
        "ttft_ms": [e["ttft_ms"] for e in completions if e.get("ttft_ms") is not None],
        "first_successful_tool_ms": min(useful) if useful else None,
    }


def run_agent(command, env, timeout):
    """Retain partial output on timeout and sample CLI RSS (not model-server RSS)."""
    peak_rss = None
    timed_out = False
    with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, start_new_session=True) as process:
        deadline = time.monotonic() + timeout
        while True:
            try:
                status = Path(f"/proc/{process.pid}/status").read_text()
                rss = next(int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:"))
                peak_rss = max(peak_rss or 0, rss)
            except (OSError, StopIteration, ValueError):
                pass
            try:
                stdout, stderr = process.communicate(timeout=max(0.01, min(0.1, deadline - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    stdout, stderr = process.communicate()
                    break
        return process.returncode, stdout, stderr, timed_out, peak_rss


def load_events(session):
    events = []
    for filename in ("events.previous.jsonl", "events.jsonl"):
        path = session / filename
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    # A killed process can leave the final journal line incomplete.
                    continue
    return events


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
                record = {"id": task["id"], "repetition": repetition, "passed": False}
                try:
                    returncode, stdout, stderr, timed_out, peak_rss = run_agent(
                        [binary, "run", "--workspace", str(workspace), "--session-dir", str(root / "session"),
                         "--permission", "workspace", "--prompt", task["prompt"], *config_args],
                        env=env, timeout=args.timeout + 10,
                    )
                    if args.output_dir:
                        (root / "stdout.json").write_text(stdout)
                        (root / "stderr.log").write_text(stderr)
                    record.update(agent_returncode=returncode, timed_out=timed_out, agent_peak_rss_kib=peak_rss,
                                  agent_stderr=stderr[-4000:] if returncode else "")
                    try:
                        response = json.loads(stdout) if stdout.strip() else {}
                    except json.JSONDecodeError as error:
                        response = {}
                        record["response_error"] = str(error)
                    checker = "import sys;sys.path.insert(0,sys.argv[1]);\n" + task["verify"]
                    verified = subprocess.run([sys.executable, "-I", "-c", checker, str(workspace)],
                                              cwd=workspace, capture_output=True, text=True, timeout=15)
                    record.update({"passed": returncode == 0 and response.get("ok", False) and verified.returncode == 0,
                              "agent_ok": response.get("ok", False),
                              "agent_error": response.get("error"),
                              "task_outcome": response.get("task_outcome"),
                              "verification_status": response.get("verification_status"),
                              "behavior_passed": verified.returncode == 0,
                              "false_completion": response.get("task_outcome") == "completed" and verified.returncode != 0,
                              "validation_error": verified.stderr[-2000:]})
                except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                    record["error"] = str(error)
                events = load_events(root / "session")
                record.update(summarize_events(events))
                record["runtime"] = [e["data"] for e in events if e.get("kind") == "runtime"]
                record["config"] = next((e["data"]["config"] for e in events if e.get("kind") == "task_started"), None)
                record["wall_seconds"] = round(time.monotonic() - started, 3)
                if args.output_dir:
                    record["artifacts"] = str(root.resolve())
                results.append(record)
                print(json.dumps(record), file=sys.stderr, flush=True)
    report = json.dumps({"model": os.environ.get("NEMO_MODEL", "bundled default"),
                         "passed": sum(r["passed"] for r in results), "total": len(results), "results": results})
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report + "\n")
    print(report)
    if not all(r["passed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
