# Edge reliability and efficiency changes — September 28, 2026

Base: `6f5489f` on `master`, matching `origin/master` at the review's remote check.
The implementation and measurements below were prepared from that base.
The bundled GGUF, installed llama.cpp runtime,
hardware profiles, sampling defaults, and thinking default have not been changed.

## Implemented

### Reliable execution evidence

- Versioned tool outcomes distinguish success, failure, unavailable dependencies,
  cancellation, timeout, partial completion, stale revisions, and blocked actions.
  Process exit status, not a formatted string's presence, determines success.
- Checks retain their kind (behavior, syntax, lint, types), result, and source
  revisions. Failed checks remain unresolved until rerun; unrelated passing
  checks do not erase them. External edits invalidate successful old evidence.
  A successful alternative behavior checker can replace an unavailable checker,
  but cannot conceal a genuine failed test.
- Headless output separates execution completion from behavioral verification.
  Edits without behavior checks finish as `completed_unverified`. Failed or stale
  checks receive bounded recovery, then an incomplete result rather than a
  success claim. Syntax-only checks are never labeled behavioral verification.
- The authorized workspace root is independent of model-controlled cwd and
  persists in version-2 checkpoints. Version-1 checkpoints retain their saved
  boundary conservatively; uncertain recovered operations must be reverified.
- Successful creates/edits update the agent's revision ledger immediately.
  Creates cannot clobber a competing creator. Edit replacements use one expected
  source revision, reject ambiguous/overlapping matches, and commit once.
- Multi-file creation preflights deterministic conflicts. Per-file committed
  outcomes survive later failures; cross-file atomicity is not claimed.
- Incomplete native tool calls and prematurely ended streams cannot silently
  finish a task. Partial usage is retained; protocol regeneration is bounded.
  Reasoning text is still never executed as a tool call.
- No-progress detection uses normalized paths and execution evidence. Real file
  changes allow fresh attempts; identical failures across tool categories do not
  acquire unlimited retries simply because a call is not read-only.

### Less wasted model work

- Previously sent messages remain byte-stable between tool turns. Compact state
  updates are stored with their tool results, not moved between old messages.
  Deliberate compaction preserves a ledger of inspected ranges and revisions.
- Source ranges contain complete lines. Per-result JSON is bounded to a
  context-scaled 1–4 KiB allowance. Oversized evidence is stored under a
  content-addressed session ID and can be read with `read_tool_output`.
  Existing subprocess capture remains bounded (48 KiB per stream); artifacts
  preserve that captured output, not bytes already omitted by the process runner.
- `read_multiple_files` accepts either legacy `file_paths` or `files` with
  independent `start_line` / `line_count` ranges. Reads inside one batch are
  ordered and sequential; separate independent native calls retain bounded
  parallel execution. Mutations remain ordered.
- `edit_file` accepts legacy snippet arguments or an `edits` array (1–16
  replacements), reducing otherwise necessary model round trips.
- Search uses literal ripgrep over an already filtered file list, with a bounded
  Rust fallback, selected-input budget of 32 MiB, deadline, cancellation checks,
  and exact filename-match priority. No embedding model or background indexer.
- Location hints include a small actual filename listing rather than an example
  `src/` path. This targets nonexistent-path guesses observed in the live trace;
  it is not an automatic rewrite of the model's requested write destination.
- `run_unittest` provides dependency-free behavioral verification under the same
  project-code permission as pytest. Zero-test/all-skipped unittest runs are not
  passes. The prompt distinguishes diagnostics from behavioral tests.

Examples (tool arguments):

```json
{"files":[{"path":"app.py","start_line":40,"line_count":30},{"path":"test_app.py","start_line":1,"line_count":60}]}
```

```json
{"file_path":"app.py","edits":[{"original_snippet":"x = 1","new_snippet":"x = 2"},{"original_snippet":"y = 1","new_snippet":"y = 3"}]}
```

### Measurements that survive failure

The evaluator records structured error codes, protocol/repetition failures,
model requests, generated/prompt/cached tokens, time to first token, prefill and
decode timings when supplied by the server, tool duration, and first successful
tool latency. Interrupted estimated token counts are explicitly labeled.
Timeouts retain stdout/stderr and completed journal events. CLI RSS is measured
separately from model-server RSS. External fixture assertions remain outside the
agent workspace; a reported completion that fails those assertions is flagged.

## Validation

Automated checks pass: 37 Rust unit tests, 20 headless integration tests, and
9 Python tests. `cargo fmt --check`, strict Clippy, release build, launcher shell
syntax, evaluation listing, and `git diff --check` also pass. Integration tests
require loopback socket permission for their mock server.

Coverage includes stable request prefixes, successive edits, batched ranges and
replacements, rejected overlapping edits, Unicode source, concurrent no-clobber
creation, partial-outcome accounting, version-1 checkpoint migration, navigation
and resume, stale verification, empty unittest suites, real unittest assertions,
fragmented SSE/Unicode, malformed and unterminated calls, bounded loop recovery,
protected-path search/fallback, source-line integrity, artifact pagination, and
timeout output retention.

| Live metric | A: reliability candidate | B: efficiency candidate |
| --- | ---: | ---: |
| Fully completed tasks | 0/4 | 0/4 |
| External behavior assertions passed | 0/4 | 1/4 |
| Tool calls / tool errors | 18 / 8 | 21 / 4 |
| Model requests | 22 | 24 |
| Recorded prefill time | 386.8 s | 302.0 s |
| Recorded decode time | 216.2 s | 287.9 s |
| Peak model-server RSS | 3,190 MiB | 3,026 MiB |
| False completed-task reports | 0 | 0 |

Sources: [candidate A report](validation/2026-09-28-edge-A.json),
[candidate B report](validation/2026-09-28-edge-B.json). Reports retain binary
hashes, runtime command/configuration, per-task metrics, and artifact locations.
Generation timing excludes unfinished work for which the server supplied no
timing record. Lower prefill time is encouraging but is not a controlled causal
speedup measurement: prompts, request counts, paths, and execution paths differ.

B corrected `stable_unique` sufficiently for the external assertions to pass,
but did not complete its own verification before the deadline. Two other tasks
also exhausted their task budgets; `boundary_search` stopped after repeated
unchanged read results. No candidate establishes coding readiness. The final
source additionally hardens revision-sensitive loop retries and supports search
of a single file (a valid model request observed in B); those final changes have
regression coverage but are not included in B's live binary hash.

The live comparison uses the four existing public fixtures, seed 42, a 180-second
per-task budget, `profiles/cpu-small.env`, and isolated owned llama-server processes.
The model is the existing 4B `nemotron_h` Q4_K_M GGUF; runtime build is
`b10068-571d0d540`. CPU settings: two threads, 8,192-token context, one slot,
256/64 batch/micro-batch, q8_0 KV cache, no GPU offload, thinking off.
The host is a Ryzen 5 1600 with approximately 7.7 GiB RAM. A memory guard terminates
only the owned server if available system memory falls below 384 MiB.

Candidate A contains reliability changes before the efficiency work. Candidate B
adds the efficiency work and grounded location hints. This is not a pristine
base-commit benchmark, and the comparison changes several mechanisms together.
The host was not isolated from other workloads; development/checks overlapped
parts of validation. Four public tasks and one seed cannot establish general
coding quality or a causal speedup. Absolute cache totals also depend on how many
requests each task makes; they must not be read as a standalone performance score.

Both isolated servers were stopped after evaluation. Neither run triggered the
memory guard. Raw retained logs/workspaces are in
`/tmp/nemocode-validation-A2e48L/`; these temporary artifacts may disappear after
system cleanup. The compact reports linked above are kept in the repository.

## Limits and next experiments

- A revision snapshot is not execution coverage. Tests can miss behavior or be
  inadequate; independent assertions and held-out tasks are still necessary.
- Native-write revision checks are optimistic, not an OS-wide lock against
  arbitrary concurrent writers. Multi-file commits are not transactional.
- The read ledger records inspected ranges, not a semantic summary of every
  symbol or constraint. Compaction may still require rereading exact code.
- Search is bounded, so `truncated: true` is not evidence that no other match
  exists. Filename priority is deliberately simple, not a learned relevance ranker.
- Artifact storage is bounded to 1 MiB per result and 64 MiB per session.
  Storage exhaustion is reported rather than silently discarding evidence.
- A compact/conditional tool catalog, checker capability discovery, automatic
  outline caching, richer AST retrieval, and extra within-batch concurrency are
  deferred until measured against the stable-catalog baseline.
- No default reasoning/sampling changes, quantization changes, runtime upgrade,
  speculative decoding, fine-tuning, downloads, publication, or weight edits.
  Next, compare small reasoning budgets and compact catalogs on held-out tasks;
  keep training separate from these harness correctness fixes.
