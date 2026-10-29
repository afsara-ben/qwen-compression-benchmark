# Implementation validation

Validation was performed on 2026-10-29. GPU smoke work was stopped at the user's request; the implementation and distribution were then completed. No new full benchmark workload was run for this implementation task, and no smoke score is included in the 112-cell grid.

## Completed checks

- The CPU suite passed **17 tests** covering the 15/16 versus 16/16 RULER gate, filters, competing workers, stale ownership and publication fencing, isolated bundle imports, duplicate imports, archive traversal rejection, GPU assignments, mocked A100/H100 memory eligibility, complete-result reuse, arithmetic, runtime precision flags, process interruption, all 34 bundled results, and NVFP4 conversion/resume on small tensors. A subsequent focused test covers recovery of an interrupted BFCL response append.
- A fresh runtime and separate BFCL environment were installed from the pinned sources and dependency constraints, without the old experiment environments. Repeating setup reused the completed installation. The resolved package inventories are retained in `validation/`.
- Full original Qwen3-30B and Qwen3.6-35B checkpoints were converted with the generalized CPU converter: 49 and 41 output shards respectively. Conversion receipts, without weights, are retained in `validation/conversions/`.
- Every bundled result is checked for protocol identity, source/tokenizer revision, cache/weight precision, launch settings, expert counts, actual per-example counts, evidence checksums and score aggregation. This includes all 17 BFCL categories and all 13 RULER subtasks. The four additional Baseline GSM8K/IFEval results also have saved input-comparison evidence in `assets/baseline-validation.json`.
- HumanEval isolation accepted valid arithmetic and rejected assertion failures, filesystem access, socket creation and an infinite loop in the new environment. Its small validation log is retained.

## GPU checks and limits

Available hardware was an RTX PRO 6000 Blackwell Server Edition with 96 GB VRAM, compute capability 12.0 and driver 580.95.05. The assigned visibility permitted one local GPU; tests respected that assignment. No A100 or H100 was available.

Before the user stopped GPU testing, **seven Qwen3.6 configurations** completed server startup, precision/residency audits, a semantic generation check, a prompt-likelihood check, and one 32k example from each of RULER's 13 subtasks: KV, E, E+KV, W, W+KV, E+W and E+W+KV. Their evidence is retained under `validation/smokes/`. This is implementation evidence, not a full 6,500-example workload result.

The initial clean-install tests exposed missing CUDA linker aliases and lm-eval optional API/RULER dependencies. The installer now creates the local CUDA library aliases and explicitly installs the API, RULER and IFEval extras. Subsequent completed checks used those fixes. Remaining smoke checks, including the Qwen3-30B retry, were stopped and the project-owned GPU processes were released.

The deployment launcher conservatively requires driver **580.126.20 or newer**, as specified in the plan. The local direct acceptance harness exercised the runtime on the available older 580-series driver; it did not change or bypass the launcher's deployment requirement. Actual A100/H100 backend behavior, speed and memory fit are **not claimed as verified**. The included automatic startup checks and optional `--smoke` command support validation on those nodes.

## Result status

The bundled report contains **34/112 completed workload scores**, including **6/16 RULER cells**. It leaves **78 full runs**: 10 RULER and 68 other workloads. The seven partial GPU smoke suites above do not change those counts. Blank scores and partial averages remain blank.

Shared-directory coordination and isolated-node bundle merging were tested with independent registries on one host. A real two-node shared filesystem was not available for this implementation task; it must provide atomic rename/mkdir and working POSIX advisory locking. No estimate of H100 completion time is inferred from these checks.
