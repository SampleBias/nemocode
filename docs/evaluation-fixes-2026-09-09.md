# Follow-up evaluation and fixes

This follows the [September 9 build review](build-review-2026-09-09.md).
The original live run completed no tool calls and passed none of the four
public regression fixtures. The follow-up uses the same bundled GGUF and
llama.cpp build recorded in that review.

## Corrected failures

- **Tool generation:** the bundled template defaults to thinking. A local
  protocol probe with thinking disabled produced native tool calls, but could
  repeat them until the output limit. Requests now default to
  `NEMO_THINKING=off`, request one tool at a time, and stop at `</tool_call>`
  when the runtime advertises the matching XML template. Reasoning text is
  still never executed. `NEMO_THINKING=on` remains available explicitly.
- **Lost file evidence:** an overconservative preflight estimate silently
  removed recent read results even when the measured prompt fit. Compaction
  now follows prompt measurement, includes the same template options used for
  generation, and preserves the latest complete tool/result round. Oversized
  active context fails explicitly instead of discarding that evidence.
- **False completion after a rejected write:** attempted writes were added to
  the changed-file ledger even when the tool returned an error. Failed writes
  now remain unresolved until a successful file operation corrects them.
  A premature completion claim receives up to two correction attempts, then
  fails explicitly. Diagnostics alone cannot clear a failed write. Cancelled
  or expired operations are also recorded as errors.
- **Repeated exploration:** task state is attached to the current request
  without creating a fresh user turn after every tool result. Instructions
  explain reuse of read results and direct reads of named files. Tool schemas
  match the session permission; workspace sessions do not advertise shell or
  arbitrary Python execution. Location hints consistently identify the working
  directory used by all tools.
- **Evaluation visibility:** the runner accepts `--config`, `--base-url`,
  `--output-dir`, and `--report`. Retained workspaces, sessions, stdout, and
  stderr make unsuccessful runs inspectable. Request events record prompt
  counts, their measurement source, and the completion allowance. Headless
  execution suppresses spinner animation.

## Automated checks

All 22 unit tests, 8 integration tests, and 6 launcher tests pass. Formatting,
Clippy with warnings denied, shell syntax, and `git diff --check` also pass.
New integration coverage checks that recent tool results survive the CPU
profile's context budget and that an existing-file overwrite rejection cannot
be followed by a successful completion claim without a corrective edit.

## Live validation

The CPU-small profile uses two threads and an 8,192-token context. A follow-up
run with the original 180-second per-task limit still passed 0/4. It completed
19 tool calls, including one successful source edit, but did not finish a
fixture. The errors and partial changes are retained separately from the
original baseline. Generation speed varied with host load; no speedup claim is
made from these runs.

The subsequent diagnostic run allowed 600 seconds per task, below the normal
CLI's 900-second task deadline. It was stopped after the first fixture spent
243.534 seconds making nine unsuccessful tool calls involving incorrect paths.
Stopping the test server interrupted that request; the remaining three tasks
then failed to connect. This is an aborted diagnostic run, not a completed
600-second evaluation or evidence that a longer deadline fixes correctness.
External behavior validators remained outside the model's workspace and were
not changed.

To reproduce against a running local server:

```bash
python3 evals/run.py --binary target/release/nemocode \
  --config profiles/cpu-small.env --timeout 600 \
  --output-dir /tmp/nemocode-evaluation \
  --report /tmp/nemocode-evaluation.json
```

These four public fixtures are a smoke suite, not an independent coding
benchmark. The host has no pytest installation; synchronous Python syntax
diagnostics and the evaluation runner's external assertions are available.

## Checkpoint for resuming

Work paused at the user's request. The release executable was rebuilt after
the Rust changes, but the local tarball predates these follow-up fixes and
must be rebuilt with `bash scripts/package.sh` before distribution. The test
model server was stopped. No passing live suite has been established.

Next, test a short reasoning allowance instead of disabling thinking entirely.
The installed runtime advertises `--reasoning-budget N`; its shared library
also contains a `reasoning_budget` request field, but a request-level limit has
not yet been verified experimentally. A 96-token probe request was prepared
at `/tmp/nemocode-fix/bounded-thinking-request.json` and was not run. Do not
change the default based on that untested probe alone. The current default
`NEMO_THINKING=off` enables tool generation but remains an experimental
configuration with observed path-selection failures.

Retained reports:

- [Completed 180-second follow-up](validation/2026-09-09-eval-fixes-180s.json).
- [Aborted longer diagnostic](validation/2026-09-09-eval-fixes-600s-aborted.json).
- [Sampled runtime memory](validation/2026-09-09-eval-fixes-runtime.json).

Local diagnostic workspaces and sessions remain under `/tmp/ncf-full` and
`/tmp/ncf-long`; protocol probes and server logs are under
`/tmp/nemocode-fix`. These temporary paths may disappear after cleanup or a
restart, so the committed reports preserve the main outcomes.
