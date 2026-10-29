# Qwen BF16 · RULER 32k

Run the two original BF16 models **sequentially on one H100 80 GB**:

```bash
git clone https://github.com/afsara-ben/qwen-compression-benchmark.git
cd qwen-compression-benchmark
bash run_bf16_ruler.sh
```

The script installs one Python environment, downloads the pinned original checkpoints and frozen benchmark inputs, runs Qwen3-30B-A3B-Instruct-2507, releases its GPU memory, then runs Qwen3.6-35B-A3B. It evaluates RULER only. There is no campaign coordination or phase barrier.

- BF16 weights and BF16 attention KV cache; Qwen3.6 retains FP32 recurrent states.
- 32,768 **total tokens**, including chat formatting and the task output allowance. Overlength inputs are rejected.
- lm-eval; 13 subtasks × 500 examples = **6,500 examples per model**, **13,000 total**.
- Seed 17, greedy decoding, thinking disabled; the same frozen inputs and pinned evaluator as the earlier experiment.
- One sequence at a time, tensor parallelism 1, eager execution, no CPU offload.

Use a local SSD for the large downloads:

```bash
RULER_WORK_DIR=/local-ssd/ruler bash run_bf16_ruler.sh
```

The script respects `CUDA_VISIBLE_DEVICES` and uses the first assigned GPU. Run on an idle H100. Host prerequisites are Linux x86-64, Bash, `flock`, Python 3.10+, Git, compiler/build tools, internet access and an NVIDIA driver compatible with the pinned CUDA 13.0 stack (580.126.20+). Python 3.12.15, vLLM 0.32.0, PyTorch 2.13.0, Transformers 5.19.0 and lm-eval install automatically. No BFCL environment or model conversion is installed or run by this entry point.

## Saved results

Default output: **`work/ruler-bf16/results.md`**. With a custom work directory: **`$RULER_WORK_DIR/results.md`**.

The Markdown file updates **every 30 seconds during download/loading/evaluation**, after each model, and on graceful shutdown. It shows status, cached response counts and the final RULER scores. The final score is the equal mean of all 13 subtask scores; it appears only after the full model evaluation completes. JSON summaries, per-example evidence, response caches and logs are retained alongside it.

Interrupt with Ctrl+C and rerun the same command with the same work directory to resume. Completed model results are skipped; saved responses from an interrupted model are reused. If one model fails, its logs are kept and the other is attempted. The script never terminates unrelated GPU processes.

To run just one model later:

```bash
bash run_bf16_ruler.sh qwen3
bash run_bf16_ruler.sh qwen36
```

No GPU evaluation or smoke test was run while adding this script. The original eight-configuration runner remains available for later work; its documentation is in [FULL_BENCHMARK.md](FULL_BENCHMARK.md).
