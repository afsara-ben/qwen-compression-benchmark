# Qwen compression benchmark

Updated: 2026-10-29T06:36:43.887575+00:00

**34/112 workload runs complete; 78 remaining. RULER: 6/16. Phase: RULER.**

E = REAP (Qwen3: 25%; Qwen3.6: 20% nominal). W = NVFP4 W4A16/BF16 activations. KV = FP8 attention cache.
RULER uses 32,768 total tokens. Other workloads use native inputs. Weight sizes include scales and backend padding, exclude KV/cache/runtime buffers, and use decimal GB.
A dash means unavailable, never zero. Companion KV rows share the same measured weights. Averages require all seven scores.

| Model | Configuration | Weights GB (reduction) | GSM8K | MMLU | MuSR | IFEval | HumanEval | BFCLv3 | RULER | Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | Baseline | 61.064 (0.00%) | 0.9219 | — | — | 0.8318 | — | — | — | — |
| Qwen3-30B-A3B | E | 46.562 (23.75%) | — | — | — | — | — | — | — | — |
| Qwen3-30B-A3B | W | — | — | — | — | — | — | — | — | — |
| Qwen3-30B-A3B | KV | 61.064 (0.00%) | 0.9219 | 0.8092 | 0.4034 | 0.8244 | 0.9329 | 0.6668 | 0.9362 | 0.7850 |
| Qwen3-30B-A3B | W+KV | — | — | — | — | — | — | — | — | — |
| Qwen3-30B-A3B | E+KV | 46.562 (23.75%) | 0.9310 | 0.7025 | 0.4048 | 0.7930 | 0.7988 | 0.6310 | 0.9346 | 0.7422 |
| Qwen3-30B-A3B | E+W | 13.992 (77.09%) | — | — | — | — | — | — | — | — |
| Qwen3-30B-A3B | E+W+KV | 13.992 (77.09%) | — | — | — | — | — | — | 0.9275 | — |
| Qwen3.6-35B-A3B | Baseline | 69.321 (0.00%) | 0.9598 | — | — | 0.8281 | — | — | — | — |
| Qwen3.6-35B-A3B | E | 56.478 (18.53%) | — | — | — | — | — | — | — | — |
| Qwen3.6-35B-A3B | W | — | — | — | — | — | — | — | — | — |
| Qwen3.6-35B-A3B | KV | 69.321 (0.00%) | 0.9606 | 0.8433 | 0.4272 | 0.8503 | 0.9634 | 0.6908 | 0.9458 | 0.8116 |
| Qwen3.6-35B-A3B | W+KV | — | — | — | — | — | — | — | — | — |
| Qwen3.6-35B-A3B | E+KV | 56.478 (18.53%) | 0.9598 | 0.8266 | 0.4352 | 0.8041 | 0.9451 | 0.6826 | 0.9453 | 0.7998 |
| Qwen3.6-35B-A3B | E+W | 17.373 (74.94%) | — | — | — | — | — | — | — | — |
| Qwen3.6-35B-A3B | E+W+KV | 17.373 (74.94%) | — | — | — | — | — | — | 0.9463 | — |

## Result provenance

| Model / configuration / workload | Origin | GPU | Elapsed | Evidence |
|---|---|---|---:|---|
| qwen3_30b / Baseline / GSM8K | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.24 h | [record](assets/completed/qwen3_30b--Baseline--GSM8K/record.json) |
| qwen3_30b / Baseline / IFEval | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.24 h | [record](assets/completed/qwen3_30b--Baseline--IFEval/record.json) |
| qwen3_30b / E+KV / BFCLv3 | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.13 h | [record](assets/completed/qwen3_30b--E+KV--BFCLv3/record.json) |
| qwen3_30b / E+KV / GSM8K | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.25 h | [record](assets/completed/qwen3_30b--E+KV--GSM8K/record.json) |
| qwen3_30b / E+KV / HumanEval | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.05 h | [record](assets/completed/qwen3_30b--E+KV--HumanEval/record.json) |
| qwen3_30b / E+KV / IFEval | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.25 h | [record](assets/completed/qwen3_30b--E+KV--IFEval/record.json) |
| qwen3_30b / E+KV / MMLU | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 1.41 h | [record](assets/completed/qwen3_30b--E+KV--MMLU/record.json) |
| qwen3_30b / E+KV / MuSR | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.06 h | [record](assets/completed/qwen3_30b--E+KV--MuSR/record.json) |
| qwen3_30b / E+KV / RULER | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 4.11 h | [record](assets/completed/qwen3_30b--E+KV--RULER/record.json) |
| qwen3_30b / E+W+KV / RULER | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 4.11 h | [record](assets/completed/qwen3_30b--E+W+KV--RULER/record.json) |
| qwen3_30b / KV / BFCLv3 | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.33 h | [record](assets/completed/qwen3_30b--KV--BFCLv3/record.json) |
| qwen3_30b / KV / GSM8K | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.25 h | [record](assets/completed/qwen3_30b--KV--GSM8K/record.json) |
| qwen3_30b / KV / HumanEval | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.05 h | [record](assets/completed/qwen3_30b--KV--HumanEval/record.json) |
| qwen3_30b / KV / IFEval | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.22 h | [record](assets/completed/qwen3_30b--KV--IFEval/record.json) |
| qwen3_30b / KV / MMLU | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 1.41 h | [record](assets/completed/qwen3_30b--KV--MMLU/record.json) |
| qwen3_30b / KV / MuSR | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.06 h | [record](assets/completed/qwen3_30b--KV--MuSR/record.json) |
| qwen3_30b / KV / RULER | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 4.12 h | [record](assets/completed/qwen3_30b--KV--RULER/record.json) |
| qwen3_6_35b / Baseline / GSM8K | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.20 h | [record](assets/completed/qwen3_6_35b--Baseline--GSM8K/record.json) |
| qwen3_6_35b / Baseline / IFEval | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.17 h | [record](assets/completed/qwen3_6_35b--Baseline--IFEval/record.json) |
| qwen3_6_35b / E+KV / BFCLv3 | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.71 h | [record](assets/completed/qwen3_6_35b--E+KV--BFCLv3/record.json) |
| qwen3_6_35b / E+KV / GSM8K | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.20 h | [record](assets/completed/qwen3_6_35b--E+KV--GSM8K/record.json) |
| qwen3_6_35b / E+KV / HumanEval | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.03 h | [record](assets/completed/qwen3_6_35b--E+KV--HumanEval/record.json) |
| qwen3_6_35b / E+KV / IFEval | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.18 h | [record](assets/completed/qwen3_6_35b--E+KV--IFEval/record.json) |
| qwen3_6_35b / E+KV / MMLU | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 1.83 h | [record](assets/completed/qwen3_6_35b--E+KV--MMLU/record.json) |
| qwen3_6_35b / E+KV / MuSR | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.08 h | [record](assets/completed/qwen3_6_35b--E+KV--MuSR/record.json) |
| qwen3_6_35b / E+KV / RULER | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.73 h | [record](assets/completed/qwen3_6_35b--E+KV--RULER/record.json) |
| qwen3_6_35b / E+W+KV / RULER | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.73 h | [record](assets/completed/qwen3_6_35b--E+W+KV--RULER/record.json) |
| qwen3_6_35b / KV / BFCLv3 | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 1.21 h | [record](assets/completed/qwen3_6_35b--KV--BFCLv3/record.json) |
| qwen3_6_35b / KV / GSM8K | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.19 h | [record](assets/completed/qwen3_6_35b--KV--GSM8K/record.json) |
| qwen3_6_35b / KV / HumanEval | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.02 h | [record](assets/completed/qwen3_6_35b--KV--HumanEval/record.json) |
| qwen3_6_35b / KV / IFEval | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.16 h | [record](assets/completed/qwen3_6_35b--KV--IFEval/record.json) |
| qwen3_6_35b / KV / MMLU | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 1.82 h | [record](assets/completed/qwen3_6_35b--KV--MMLU/record.json) |
| qwen3_6_35b / KV / MuSR | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0.08 h | [record](assets/completed/qwen3_6_35b--KV--MuSR/record.json) |
| qwen3_6_35b / KV / RULER | imported validated study | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.74 h | [record](assets/completed/qwen3_6_35b--KV--RULER/record.json) |

## Pending work and issues

RULER pending: 10; other workload cells pending: 68.
Workers cannot start other workloads until all 16 RULER cells validate. Imported scores may come from different hardware; use --fresh for a new hardware-specific campaign.

| Configuration | RULER remaining | Other workloads remaining |
|---|---:|---:|
| Baseline | 2 | 8 |
| E | 2 | 12 |
| W | 2 | 12 |
| KV | 0 | 0 |
| W+KV | 2 | 12 |
| E+KV | 0 | 0 |
| E+W | 2 | 12 |
| E+W+KV | 0 | 12 |


## Protocol

Model/dataset revisions, task settings, precision and runtime identities are recorded in each result. BFCL uses the pinned official v3 evaluator; the other workloads use lm-eval.
No CPU offload, precision substitution, PPL, or undefined composite scores. Original screenshot scores are not imported.
