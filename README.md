# Portable Qwen compression benchmark

Run the two models across eight compression configurations and seven workloads with one entry point. The launcher installs isolated Python environments, downloads pinned checkpoints, creates NVFP4 artifacts, resumes evaluations, and updates `results.md`, `results.json`, and `results.csv` in the state directory.

Download the [complete standalone package](https://github.com/afsara-ben/qwen-compression-benchmark/releases/tag/v1.0.0), or clone this repository. The Git checkout omits large Arrow, compressed response and NLTK files; its first normal launch automatically downloads the checksum-pinned release package and extracts only the missing benchmark assets. `--check` reports missing assets without downloading them. The complete standalone package already contains those assets.

```bash
git clone https://github.com/afsara-ben/qwen-compression-benchmark.git
cd qwen-compression-benchmark
```

```bash
bash run.sh
```

The distribution includes **34 validated completed results**, their evidence, and frozen RULER inputs. See the bundled [initial results table](results.md). It contains **no model weights or installed environments**. A normal campaign needs **78 new runs**. A fresh campaign needs all **112 runs**. Smoke tests never count as completed benchmark results.

## Host requirements

- Linux x86-64; Bash; a system Python 3.10 or newer for bootstrap; Git, curl, a C/C++ compiler and standard build tools.
- An assigned NVIDIA Ampere-or-newer GPU. The intended deployment is **one H100 80 GB on each of two separate nodes**. Each worker runs a complete model on one GPU; there is no cross-node tensor parallelism.
- NVIDIA driver **580.126.20 or newer** is the project's conservative deployment requirement for its CUDA 13.0 Update 3 stack. The launcher does not install or change drivers. See [NVIDIA's release notes](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html).
- `libseccomp.so.2` (`libseccomp2` on Debian/Ubuntu) and kernel Landlock support for HumanEval's isolated code execution. A missing HumanEval prerequisite is reported separately and does not stop RULER or other eligible workloads.
- Internet access to PyPI, Python distributions, GitHub, and Hugging Face during setup and dataset download. Credentials, if required by a hosting service, remain outside the distribution.
- Recommended **128 GB host RAM and 500 GB free SSD per node** when keeping the full model cache. Actual usage depends on selected models and cached files. Downloads and conversion check available disk space before writing large artifacts.

For a Debian/Ubuntu host, an administrator can install the ordinary system dependencies with:

```bash
sudo apt-get install python3 git curl build-essential libseccomp2
```

Python **3.12.15**, vLLM **0.32.0**, PyTorch **2.13.0**, Transformers **5.19.0**, the local CUDA toolkit, and benchmark dependencies are installed automatically under the work directory. A system CUDA toolkit, system pip, and system `venv` package are not required. The BFCL evaluator has a separate environment to preserve its dependency versions. The pinned installer, constraints, source commits, and compatibility patch are in `locks/`, `config/experiment.json`, and `patches/`.

NVFP4 uses **weight-only W4A16 with BF16 activations and Marlin**, including on A100/H100. This is the Marlin execution path described in the [vLLM compression documentation](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-scheme/). Every server startup audits the actual backend, expert counts, GPU residency and cache precision, then checks generation and prompt log probabilities. Deployment smoke tests remain necessary on the target hardware.

## Quick start on two nodes

Copy this project to both nodes. Prefer local SSD paths for `--work-dir`; it contains models, environments, compilation caches, response caches, and logs. The default is `work/HOSTNAME` under the project. The default state directory is `WORK_DIR/state`.

With shared storage, use the **same state directory** on both nodes:

```bash
# Run on node A, then run the same command on node B.
bash run.sh --work-dir /local-ssd/qwen-benchmark --state-dir /shared/qwen-benchmark
```

Each node discovers its assigned, idle GPUs and claims compatible pending jobs. It respects `CUDA_VISIBLE_DEVICES` and scheduler assignments, checks compute processes, and uses per-GPU project locks. It never terminates unrelated processes. Optional `--gpus 0` selects ordinal zero within the visible inventory; full GPU UUIDs are also accepted. Shared storage must support atomic directory creation/rename and POSIX advisory locks across nodes.

No shared storage is required. Use one model selection per node initially:

```bash
# Node A
bash run.sh --work-dir /local-ssd/qwen-benchmark --models qwen3_30b
# Node B
bash run.sh --work-dir /local-ssd/qwen-benchmark --models qwen3_6_35b
```

Each isolated node stops at the RULER barrier until it has the other node's results. Export, transfer the files yourself, and import them on the peer:

```bash
# Node A: export; copy node-a-results.tar.gz to node B.
bash run.sh --work-dir /local-ssd/qwen-benchmark --export-results node-a-results.tar.gz
# Node B: export; copy node-b-results.tar.gz to node A.
bash run.sh --work-dir /local-ssd/qwen-benchmark --export-results node-b-results.tar.gz

# Node A, after receiving B's bundle:
bash run.sh --work-dir /local-ssd/qwen-benchmark --import-results node-b-results.tar.gz
bash run.sh --work-dir /local-ssd/qwen-benchmark --models qwen3_30b
# Node B, after receiving A's bundle:
bash run.sh --work-dir /local-ssd/qwen-benchmark --import-results node-a-results.tar.gz
bash run.sh --work-dir /local-ssd/qwen-benchmark --models qwen3_6_35b
```

Import/export commands do not start evaluation unless `--run` is also supplied. Transfer final bundles again to consolidate the complete table. Reimporting the same result is harmless; conflicting protocols are rejected. First valid completed evidence for a cell is retained, with its original hardware provenance.

## Runs and the global RULER gate

A run is **one model × one configuration × one workload**, not an individual task or example. There are **2 × 8 × 7 = 112 runs**. RULER has 13 subtasks, MMLU 57 subjects, MuSR three subsets, and BFCLv3 17 categories within their respective workload cells.

| Configuration | New RULER runs | New other workload runs | Total new runs |
|---|---:|---:|---:|
| Baseline | 2 | 8 | 10 |
| E | 2 | 12 | 14 |
| W | 2 | 12 | 14 |
| KV | 0 | 0 | 0 |
| W+KV | 2 | 12 | 14 |
| E+KV | 0 | 0 | 0 |
| E+W | 2 | 12 | 14 |
| E+W+KV | 0 | 12 | 12 |
| **Total** | **10** | **68** | **78** |

The original 30 confirmed scores are joined by four earlier Baseline GSM8K/IFEval scores. Their saved document IDs, inputs, targets, rendered requests, task settings and sample counts were checked against matching later runs. See `assets/baseline-validation.json`; original summaries are retained alongside normalized provenance.

**Phase 1:** all 16 valid RULER results must exist globally. Six are bundled, leaving ten full runs, each with **13 × 500 = 6,500 examples**: **65,000 new RULER examples**.

**Phase 2:** only then may GSM8K, MMLU, MuSR, IFEval, HumanEval and BFCLv3 run. Work is grouped by model/configuration where possible. BFCL's different serving capacity requires a server reload. Filters cannot bypass the global gate. Workers release their GPUs while waiting for a peer at the barrier; if no peer is active, they exit with saved state for a later invocation.

## Fixed scientific protocol

| Model ID | Original checkpoint | Expert-compressed checkpoint |
|---|---|---|
| `qwen3_30b` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | `SamsungSAILMontreal/Qwen3-30B-A3B-Instruct-2507-REAP`: 128 → 96 experts, 25% |
| `qwen3_6_35b` | `Qwen/Qwen3.6-35B-A3B` | `RangerX/Qwen3.6-35B-REAP-Pruned-ratio-0.2`: 256 → 205 experts, nominal 20% |

Exact checkpoint and dataset commit IDs are frozen in `config/experiment.json`. Published REAP checkpoints are downloaded; pruning is not repeated. Both expert and weight compression always use the pinned original model tokenizer.

| Configuration | Experts | Weights | Attention KV cache |
|---|---|---|---|
| Baseline | Original | BF16 | BF16 |
| E | REAP | BF16 | BF16 |
| W | Original | NVFP4 | BF16 |
| KV | Original | BF16 | FP8 |
| W+KV | Original | NVFP4 | FP8 |
| E+KV | REAP | BF16 | FP8 |
| E+W | REAP | NVFP4 | BF16 |
| E+W+KV | REAP | NVFP4 | FP8 |

Qwen3.6 uses text-only serving with FP32 recurrent states in every configuration. FP8 attention scales retain the original unit-scale policy. Decoding is greedy, non-thinking, seed 17. No CPU offload or precision substitution is allowed.

The converter reads layer and expert counts from the source configuration. It preserves the original block-16 NVFP4 maximum-based recipe, FP8 block scales, FP32 global scales, fused projection scaling, excluded embeddings/norms/vision/MTP weights, model-specific router handling, Qwen3.6 REAP loader fix, and canonical Marlin ordering. It needs no calibration dataset. The resulting weights include unquantized tensors and quantization scales, so their footprint is not simply one quarter of BF16.

| Workload | Evaluator/task | Reported metric | Coverage |
|---|---|---|---|
| GSM8K | lm-eval `gsm8k_cot` | Flexible extracted exact match | 1,319 questions, 8-shot |
| MMLU | lm-eval `mmlu` | Accuracy | 14,042 questions, 57 subjects, 5-shot |
| MuSR | lm-eval `leaderboard_musr` | Normalized accuracy | 756 questions, 3 subsets |
| IFEval | lm-eval `ifeval` | Prompt-level strict accuracy | 541 prompts |
| HumanEval | lm-eval `humaneval_instruct` | pass@1 | 164 problems |
| BFCLv3 | Pinned official BFCL evaluator | Official overall accuracy | 4,441 cases, 17 categories |
| RULER | lm-eval `ruler` | Equal mean of 13 subtask scores | 6,500 examples |

GSM8K saves 2,638 evidence rows because both answer-filter views are logged; each question is generated once. Workload-specific generation and few-shot settings are preserved from the pinned tasks. GSM8K/IFEval allow 2,048 output tokens. BFCL allows 4,096 output tokens with a **131,072-token serving ceiling**, retaining benchmark-native inputs. Some imported successful BFCL responses originated at a smaller ceiling; the full output allowance fit, and capacity failures were rerun. Their evidence records that history.

**RULER alone is the long-context experiment:** 32,768 total tokens including chat formatting and the output allowance. Frozen input banks are included for both original tokenizers. The runner rejects over-capacity requests rather than truncating inputs. Other workloads use native inputs and generation limits; a 32k ordinary server ceiling does not pad or extend their prompts.

## Controls, recovery and output

```bash
bash run.sh --check
bash run.sh --prepare-only --models qwen3_30b --configs W E+W
bash run.sh --smoke --models qwen3_30b --configs W W+KV
bash run.sh --report-only
bash run.sh --models qwen3_6_35b --configs Baseline E W --workloads RULER
```

Append your `--work-dir` and `--state-dir` to each command when using custom locations. `--check` inspects prerequisites and frozen assets without installing environments or downloading models. `--prepare-only` installs and creates selected artifacts without evaluation. `--smoke` tests generation, likelihoods, runtime precision and one 32k example from each of RULER's 13 subtasks, without publishing scores. `--report-only` needs only the bootstrap Python.

Use `--fresh` once to create a separate campaign without the 34 bundled scores. It prints the exact new work/state paths. Resume using those paths **without `--fresh`**, which would create another new campaign. Advanced users may specify `--runtime-dir`, or compatible `--python` and `--bfcl-python` interpreters.

Interrupt with Ctrl-C or SIGTERM. Completed results and incremental response caches remain on disk; restart with the same command and paths. Completed downloads, verified conversion shards, and validated result cells are reused. Partial caches include the scientific protocol, source hardware, runtime code and memory profile identity, so a different backend/profile cannot silently reuse unfinished responses. Only project-owned child process groups are stopped.

The initial profile allows four requests and 8,192 batched tokens. On memory failures the worker tries two requests/4,096 tokens, one request/2,048 tokens with eager execution, then one request/1,024 tokens/eager at 95% memory utilization. Context and precision stay fixed. Each cell has bounded attempts and a later recovery pass; another eligible RULER cell can run after a failure. A 40 GB A100 can do suitable quantized jobs, while BF16 jobs remain pending for larger hardware. Required pending RULER cells keep Phase 2 closed.

Reports are updated during execution and written atomically:

- `STATE_DIR/results.md`, `results.json`, `results.csv`: all 16 rows, seven workload columns, remaining work, measured weights-only GB and reduction, and provenance. An average appears only when all seven scores exist.
- `STATE_DIR/records/CELL/`: validated summary, per-example evidence, audit, launch metadata and checksums. RULER subtask scores are in its summary.
- `STATE_DIR/issues/`, `hardware-check.json`, `workers/`: failures, eligibility and current workers.
- `WORK_DIR/results/`: attempts, incremental response caches and request journals. `WORK_DIR/logs/`: server, preparation and evaluation logs.
- `WORK_DIR/models/`, `artifacts/`: downloaded checkpoints and resumable converted weights, each with integrity receipts.

Weight footprint means unique loaded parameter storage, including scales and backend padding, excluding attention cache, recurrent states and other runtime buffers. Decimal GB is used. KV does not change the weights-only number. Screenshot values are historical references, not imported measurements. PPL and undefined composite scores are excluded.

Exit status: **0** for complete evaluation or a successful auxiliary command; **3** for saved pending work (memory, filters, unavailable workers, or missing peer results); **2** for setup/validation errors; **1** for failed preparation/smoke; **130** for an orderly interruption. A partial grid is never reported as complete.

## Validation and project layout

Run the CPU acceptance suite after environment preparation:

```bash
WORK_DIR/environment/venv/bin/python -m pytest -q tests
```

`qbench/` contains installation, conversion, GPU selection, coordination, execution, validation and reporting. `runtime/` contains the pinned-protocol adapters, loader/audit extensions and HumanEval isolation. `tools/` contains maintainer utilities; normal use requires only `run.sh`. `assets/` contains frozen banks and completed evidence. `VALIDATION.md` records what was actually tested and which hardware checks remain for deployment.

The existing results and local acceptance evidence are from Blackwell hardware. A100/H100 GPU selection is covered by mocked inventories; actual A100/H100 backend performance and memory fit must be established by `--smoke` on those nodes. Mixed hardware provenance is visible in the report; use a fresh campaign for a complete repeat on new hardware.

Third-party notices and licenses are listed in `THIRD_PARTY_NOTICES.md` and `vendor/`.
