# Harness and edge implementation

The implementation is organized around bounded resources and observable outcomes.
The September 8 delivery lands the planned harness pieces. The follow-up
[build review](build-review-2026-09-09.md) records fixes and validation; remaining
work includes hardware measurements, release publication, and broader coverage.

The [follow-up evaluation checkpoint](evaluation-fixes-2026-09-09.md) records
additional tool/context fixes and the unfinished live-model investigation.
Automated checks pass, but live coding readiness is not yet established.

Hardware profiles are starting points for measurement, not performance guarantees.
No model training or external publication is performed by these changes.

## Shipped

- Shared subprocess execution with concurrent bounded capture, timeouts, and cancellation (`src/process.rs`). Unix only.
- Versioned Python diagnostics and consistent project Python selection (`src/lsp`, `src/python_tools.rs`).
- Validated tool arguments, workspace-scoped paths, atomic writes, and stale-edit SHA-256 checks (`src/storage.rs`).
- Bounded file-read cache, lexical retrieval (`search_code`, `file_outline`, `index_directory`), task state, and durable session journals (`src/retrieval.rs`, `src/session.rs`).
- Cancellation-aware streaming, per-task token/time budgets, sampling (`NEMO_TEMPERATURE`, `NEMO_TOP_P`, `NEMO_SEED`), and token accounting.
- Portable CLI: `chat`, headless `run`, `doctor`, `status`, `config`, `bench`, `runtime`, `model` (`src/runtime.rs`).
- Permission modes: `read-only`, `workspace` (file tools + project checkers), `trusted` (bash and `run_python`).
- Relocatable packaging (`scripts/package.sh`), hardware profiles (`profiles/`), CI checks (`.github/workflows/check.yml`), and a small public eval suite (`evals/`).
- Source launches check the release build for freshness; packaged launchers resolve their own bundled CLI. Sessions retain an absolute journal directory across navigation, and retrieval rejects symlinks outside its root or into protected files.

## What's next

1. **Measure the hardware profiles.** Run `nemocode bench` and device RSS/VRAM notes on a CPU-only box (`profiles/cpu-small.env`) and a machine with an accelerator (`profiles/balanced.env`). Replace the current starting-point values only after those measurements.
2. **Record a live eval baseline.** `evals/run.py --list` is what CI runs. A quality baseline needs a running `llama-server` plus the bundled GGUF: `python3 evals/run.py --binary target/release/nemocode`. Keep any private held-out tasks outside this repo.
3. **Publish a relocatable 0.1 package.** `scripts/package.sh` already writes `dist/`. Cut a tagged GitHub release from the CI artifact (`nemocode-linux-x86_64`) and verify `nemocode runtime install` / `nemocode model install` on a machine without a checkout.
4. **Interactive session resume.** `--resume SESSION_DIR` works for `run`. The REPL should list sessions and resume one without constructing the session directory by hand.
5. **Windows subprocess support.** `src/process.rs` currently bails on non-Unix hosts.
6. **Eval coverage.** The four public tasks are a regression smoke set, not a coding benchmark. Deterministic integration tests now cover permission denials, stale-edit rejection, session navigation, and Git fallback symlinks. Add live stale-edit recovery and larger multi-file refactors before treating pass rate as a model-quality signal.
