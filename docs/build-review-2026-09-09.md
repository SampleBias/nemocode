# September 9 build review

The subsequent [evaluation and fix report](evaluation-fixes-2026-09-09.md)
records the investigation of this review's live failures. Results below are
the original baseline and are retained for comparison.

Reviewed the September 8 commit `9538cbf74d79f58801d1507f3c493ccac9eec4f6`
and its documented harness/optimization delivery. The working tree was clean
at the start. Its release executable was dated July 19 and could be launched
without checking whether it matched the current source.

## Findings corrected

1. **Stale source executable:** `start-nemo.sh` previously preferred any existing
   release executable over rebuilding. Source launches now run Cargo's freshness
   check and stop if the build fails. The target directory is explicit so a
   `CARGO_TARGET_DIR` override cannot leave the launcher running another build.
2. **Relocated launcher:** a directly invoked packaged launcher now resolves
   `bin/nemocode` within its own bundle without requiring a `PATH` entry.
3. **Session navigation:** a relative session directory previously broke journal
   writes after `change_directory`. It is now canonicalized when opened.
4. **Retrieval boundaries:** Git fallback enumeration could expose symlinked
   files outside the search root or inside protected paths. Both lexical and
   canonical paths are now checked before indexing or reading file contents.
5. **Release checksum:** checksum sidecars now reference the archive basename,
   so validation works after downloading or moving both files together. The
   bundle also includes the documentation linked from its README.
6. **Build checks and cleanup:** resolved 16 Clippy warnings, removed unused
   `dotenvy` and the obsolete checkout-only launcher helper, and aligned
   `.env.example` with the current continuation/tool-round defaults, task
   budgets, permission modes, sampling, and runtime options.

## Automated validation

Toolchain: `rustc 1.97.1 (8bab26f4f 2026-07-14)` on Linux x86_64.

| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check` | Pass |
| `cargo clippy --locked --all-targets -- -D warnings` | Pass |
| `cargo test --locked` | 21 unit tests and 7 integration tests pass |
| `bash -n start-nemo.sh scripts/package.sh` | Pass |
| Python launcher unittest suite | 6 tests pass |
| `python3 evals/run.py --list` | Four fixtures discovered |
| `bash scripts/package.sh` | Optimized release tarball built |
| Relocated package smoke check | Checksum, direct launcher, config, and binary identity pass |

The new launcher and session regressions were reproduced against the original
code before applying their fixes. Integration tests require local loopback
sockets; the restricted execution sandbox denies those sockets, so the suite
was run outside it. No code change was made to bypass that restriction.

## Live CPU measurement

Device: AMD Ryzen 5 1600, 12 logical CPUs, approximately 7.7 GiB system RAM.
Used `profiles/cpu-small.env` without changing its values: two CPU threads,
8,192-token context, 256/64 batch sizes, q8_0 KV cache, and zero GPU layers.
The already-installed llama.cpp runtime is build `b10068` (`571d0d540`).
The local bundled model's recorded SHA-256 is
`67514959734347b17cce57ab75e9e91cd469e57830acbd8ee2a4363060431741`.
This records local artifact identity; it is not verification against a
publisher-supplied checksum.

| Benchmark case | Wall time | Generated tokens/sec | Cached prompt tokens |
| --- | ---: | ---: | ---: |
| First request after runtime startup | 23.449 s | 4.163 | 0 |
| Repeated prompt | 20.192 s | 4.183 | 26 |
| Reasoning prompt | 33.025 s | 4.262 | 6 |

These are three latency probes, not a comparison with yesterday's executable
or an independent coding-quality benchmark. The repeated request's prompt
evaluation took 0.830 s versus 4.170 s initially.

To repeat with the default localhost port, start the runtime in one terminal:

```bash
NEMO_CONFIG="$PWD/profiles/cpu-small.env" ./start-nemo.sh --runtime-start
```

Run the probes and fixtures in another terminal from the checkout:

```bash
NEMO_CONFIG="$PWD/profiles/cpu-small.env" target/release/nemocode bench
NEMO_CONFIG="$PWD/profiles/cpu-small.env" python3 evals/run.py --binary target/release/nemocode --timeout 180
```

This review used a temporary localhost port, then stopped its own test server.

The live regression run passed **0 of 4** tasks with the existing 180-second
per-task evaluation limit. No fixture completed a tool call or changed its
source files. Three runs reached the time limit; one failed earlier, with its
original CLI error unavailable. This CPU configuration is not validated for practical
coding use by this run.

| Fixture | Result | Wall time | Completed tool calls |
| --- | --- | ---: | ---: |
| `empty_average` | Fail; reached time limit | 180.137 s | 0 |
| `stable_unique` | Fail; reached time limit | 180.101 s | 0 |
| `two_file_normalization` | Fail before time limit | 101.304 s | 0 |
| `boundary_search` | Fail; reached time limit | 180.089 s | 0 |

Sampled peak model-server RSS was 4,050,280 KiB (about 3.86 GiB). This is the
server process only, sampled once per second; it does not include the OS,
harness, or other applications. Generation counts in interrupted fixtures
are harness estimates when final server usage is unavailable.

Raw results: [benchmark](validation/2026-09-09-bench-cpu-small.json) and
[coding fixtures](validation/2026-09-09-eval-cpu-small.json).

The eval runner now includes `agent_error` and a bounded stderr tail in future
failure records. Previously it discarded the actual CLI failure and retained
only the downstream Python assertion, making early failures hard to diagnose.

A [fresh repeat of `two_file_normalization`](validation/2026-09-09-diagnostic-normalization.json)
also failed, this time at 180.083 s with `task time budget exhausted; session
saved`. Its stderr showed the reasoning stream entering the tool-markup parser
without completing a native tool call. The harness intentionally does not
execute function markup found in reasoning. Model/runtime tool formatting and
generation latency both need investigation; increasing the timeout alone has
not been demonstrated to resolve the problem. The earlier 101-second failure
was not reproduced, and its exact cause remains unknown.

## Remaining delivery work

- Resolve the failed live fixtures and record a passing run before declaring
  the optimization delivery complete. This review establishes build and
  regression-test success, not live coding readiness.
- Measure an accelerator profile on suitable hardware. No accelerator
  performance or VRAM claim is established by this CPU run.
- Validate fresh runtime/model downloads on a machine without a checkout.
  This review uses existing local runtime and model artifacts.
- Publish a tagged release separately; this review only builds the local tarball.
- Add interactive session selection, Windows subprocess support, and broader
  live recovery/refactor coverage as listed in `implementation.md`.

The existing tarball script is relocatable; it does not promise byte-identical
archives across builds.
