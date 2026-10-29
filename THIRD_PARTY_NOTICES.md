# Third-party notices

The benchmark orchestration and experiment adapters are provided as source for this research project. No ownership of upstream models, datasets, generated model responses, or vendored third-party code is transferred.

- **lm-evaluation-harness**, EleutherAI and contributors: MIT license; see `vendor/lm-eval-LICENSE.md`. The project installs the exact commit in `config/experiment.json` and applies `patches/lm-eval.patch`. RULER task implementations incorporate NVIDIA's Apache-2.0 code and notices from that upstream tree.
- **Berkeley Function Calling Leaderboard / Gorilla**, its authors and contributors: see `vendor/BFCL-LICENSE`. The evaluator and data are fetched at the pinned revision. Result bundles retain the official evaluator's category evidence.
- **vLLM**, vLLM contributors: Apache-2.0. `runtime/marlin_order.py` retains its SPDX notice; the installed vLLM distribution carries its complete notices and license. Other compatibility adapters use vLLM's public and version-specific interfaces.
- **Qwen and REAP checkpoints** are downloaded from their named Hugging Face repositories. Consult each repository's license and model card at the pinned revision before redistribution. This project does not include their weights.
- **Benchmark datasets and frozen RULER inputs** retain their respective upstream terms. The source dataset revisions, RULER recipe provenance and file hashes are recorded under `config/` and `assets/`. RULER question-answer banks include material derived from SQuAD and HotpotQA; needle/haystack banks follow the pinned harness's recipes.
- **NLTK data** is included to preserve the original evaluation environment. Its resource directories retain their available README and license files. NLTK itself is installed as a pinned dependency.
- Python, uv, PyTorch, Transformers, Hugging Face libraries, NVIDIA CUDA libraries, FlashInfer and other dependencies retain their own distribution licenses. The pinned uv wheel is verified against its SHA-256 before its bootstrap binary is used.

Completed-result assets are experimental evidence, including generated responses and benchmark references. Their presence does not imply permission to redistribute upstream content beyond the applicable licenses.
