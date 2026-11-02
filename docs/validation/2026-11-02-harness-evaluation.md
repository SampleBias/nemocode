# Harness evaluation, 2026-11-02

One run per task on an Intel Core i7-6700 CPU using the two-thread CPU-small
profile, an 8,192-token context, and a 180-second task deadline. The server
was llama.cpp `b10107-c0bc8591e`; the bundled Q4_K_M GGUF had SHA-256
`67514959734347b17cce57ab75e9e91cd469e57830acbd8ee2a4363060431741`.
Both public-suite runs used seed 42 and thinking disabled. Compilation ran
concurrently for portions of both runs, so latency comparisons are unreliable.

| Fixture | Baseline | Updated harness |
| --- | --- | --- |
| empty_average | timeout, validator failed | timeout, validator failed |
| stable_unique | timeout, validator failed | timeout, validator passed |
| two_file_normalization | timeout, validator failed | timeout, validator failed |
| boundary_search | agent finished, validator failed | agent finished, validator passed |

The end-to-end score was 0/4 before and 1/4 after. The correct `stable_unique`
edit did not count as a pass because the agent timed out before finishing. A
single run on four public examples cannot establish a causal or general quality
improvement. The model ignored a correct path suggestion on `empty_average`
and continued exploring; further work is needed on planning and tool selection.

A separate `empty_average` probe enabled thinking with a server reasoning budget
of 96 tokens and the same deadline. It timed out with an unchanged file after
attempting to create an existing file. This one probe does not justify changing
the default or establish which reasoning setting is best.

The full local reports and session artifacts were saved in
`/tmp/nemocode-baseline`, `/tmp/nemocode-optimized`, and
`/tmp/nemocode-thinking`, with aggregate reports beside those directories.
These temporary artifacts may be removed by the system. Use repeated runs and
a separate held-out task set before selecting settings or claiming readiness.
