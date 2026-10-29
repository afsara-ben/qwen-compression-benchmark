#!/usr/bin/env python3
"""Evaluate one pinned workload against an existing quality-study vLLM server."""
import argparse
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from datetime import datetime, timezone

ROOT = Path(os.environ['QBENCH_WORK_DIR'])
os.environ.setdefault('HF_HOME', str(ROOT/'cache/hf'))
os.environ.setdefault('HF_DATASETS_CACHE', str(ROOT/'cache/datasets'))
os.environ.setdefault('NLTK_DATA', str(ROOT/'cache/nltk'))
os.environ.setdefault('HF_HUB_OFFLINE', '0')
os.environ.update(TOKENIZERS_PARALLELISM='false', RAYON_NUM_THREADS='2',
                  OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('HF_ALLOW_CODE_EVAL', '1')

from lm_eval import evaluator
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.api.task import ConfigurableTask
from quality_server import paths, server_context_capacity

TASKS = {'GSM8K':'gsm8k_cot', 'MMLU':'mmlu', 'MuSR':'leaderboard_musr',
         'IFEval':'ifeval', 'HumanEval':'humaneval_instruct', 'RULER':'ruler'}


def serial(value):
    if hasattr(value, 'item'): return value.item()
    if isinstance(value, (set, tuple)): return list(value)
    return str(value)


def save(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=serial) + '\n')
    temporary.replace(path)


class StrictAPI(LocalCompletionsAPI):
    def __init__(self, journal, **kwargs):
        self.journal = Path(journal)
        self.client_stops = None
        capacity = int(kwargs['max_length'])
        super().__init__(**kwargs)
        self.max_length = capacity

    def record(self, value):
        with self.journal.open('a') as f:
            f.write(json.dumps(value, default=serial) + '\n')

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(chat_history, tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt, enable_thinking=False)

    def generate_until(self, requests, **kwargs):
        for request in requests:
            context, generation = request.args
            allowance = generation.get('max_gen_toks', self._max_gen_toks)
            length = len(self.tok_encode(context))
            if length + allowance > self.max_length:
                raise ValueError(f'Context would be truncated: {length}+{allowance}>{self.max_length}')
        return super().generate_until(requests, **kwargs)

    def batch_loglikelihood_requests(self, chunks):
        chunks = list(chunks)
        for chunk in chunks:
            for _, context, continuation in chunk:
                if len(context)+len(continuation)+1 > self.max_length:
                    raise ValueError(f'Likelihood request exceeds server capacity {self.max_length}')
        return super().batch_loglikelihood_requests(chunks)

    def _create_payload(self, messages, generate=False, **kwargs):
        payload = super()._create_payload(messages, generate=generate, **kwargs)
        stops = payload.get('stop')
        if generate and isinstance(stops, list) and len(stops) > 4:
            if self.client_stops is not None and self.client_stops != stops:
                raise ValueError('Mixed oversized stop lists require per-request tracking')
            self.client_stops = list(stops)
            payload['stop'] = stops[:4]
        self.record(dict(event='request', time=time.time(), generate=generate,
                         request_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                         prompt_lengths=[len(x) for x in messages], max_tokens=payload['max_tokens']))
        return payload

    def parse_generations(self, outputs, **kwargs):
        self.record(dict(event='response', time=time.time(), generate=True, output=outputs))
        parsed = super().parse_generations(outputs, **kwargs)
        if self.client_stops:
            for i, text in enumerate(parsed):
                if text is not None:
                    positions = [text.find(s) for s in self.client_stops if s and s in text]
                    if positions: parsed[i] = text[:min(positions)]
        return parsed

    def parse_logprobs(self, outputs, **kwargs):
        result = super().parse_logprobs(outputs, **kwargs)
        self.record(dict(event='response', time=time.time(), generate=False, scores=result))
        return result


def cache_ruler_data():
    original = ConfigurableTask.download
    def download(self, dataset_kwargs=None, **kwargs):
        fn = self.config.custom_dataset
        if callable(fn) and ('ruler' in str(getattr(fn, '__module__', ''))):
            import datasets
            from qbench.common import PROJECT, read
            model_id = os.environ['QBENCH_MODEL_ID']
            banks = read(PROJECT/'assets/ruler-bank-index.json')
            entry = banks[model_id][self.config.task]
            dest = PROJECT/'assets/ruler-banks'/entry['directory']
            if not (dest/'dataset_dict.json').exists():
                raise RuntimeError('Frozen RULER bank missing: '+str(dest))
            self.dataset = datasets.load_from_disk(str(dest))
            return
        else:
            revisions = json.loads((ROOT/'state/dataset-revisions.json').read_text())
            revision = revisions.get(self.config.dataset_path)
            if revision:
                dataset_kwargs = dict(dataset_kwargs or {}, revision=revision)
            original(self, dataset_kwargs=dataset_kwargs, **kwargs)
    ConfigurableTask.download = download


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--attempt', required=True)
    p.add_argument('--workload', choices=TASKS, required=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    dest = ROOT/'results'/args.attempt/args.workload/('pilot' if args.limit else 'full')
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'summary.json').exists():
        print('Already complete:', dest, flush=True)
        return
    _, tokenizer = paths(args.gpu, False)
    server_capacity = server_context_capacity(args.gpu)
    ruler = args.workload == 'RULER'
    context_capacity = 32768 if ruler else server_capacity
    if server_capacity < context_capacity:
        raise ValueError(f'RULER requires 32768 total-token capacity; server has {server_capacity}')
    metadata = {'tokenizer':tokenizer}
    if ruler: metadata['max_seq_lengths'] = [32768]
    started = time.time()
    save(dest/'status.json', dict(status='running', started_at=datetime.now(timezone.utc).isoformat(), args=vars(args)))
    cache_ruler_data()
    if args.workload == 'HumanEval':
        os.environ['QUALITY_CODE_VERDICT_PATH'] = str(dest/'code-verdicts.jsonl')
        from quality_code_eval import install
        install()
    model = StrictAPI(journal=dest/'requests.jsonl', model='quality-model',
        base_url=os.environ['QBENCH_ENDPOINT']+'/v1/completions',
        tokenizer=tokenizer, tokenizer_backend='huggingface', tokenized_requests=True,
        num_concurrent=int(os.environ.get('QBENCH_CONCURRENCY','4')), batch_size=1, seed=17, max_length=context_capacity,
        max_gen_toks=2048, timeout=1800, max_retries=3)
    gen = {'temperature':0, 'do_sample':False}
    if args.workload in ['GSM8K','IFEval']: gen['max_gen_toks'] = 2048
    result = evaluator.simple_evaluate(model=model,
        tasks=[TASKS[args.workload]], batch_size=1, limit=args.limit,
        num_fewshot=5 if args.workload == 'MMLU' else None,
        use_cache=str(dest/'responses'), log_samples=True, apply_chat_template=True,
        fewshot_as_multiturn=True, gen_kwargs=gen, bootstrap_iters=1000,
        random_seed=17, numpy_random_seed=17, torch_random_seed=17, fewshot_random_seed=17,
        confirm_run_unsafe_code=True,
        metadata=metadata)
    samples = result.pop('samples', {})
    with gzip.open(dest/'samples.json.gz', 'wt') as f:
        json.dump(samples, f, default=serial)
    result['sample_counts_saved'] = {k:len(v) for k,v in samples.items()}
    result['elapsed_s'] = time.time()-started
    result['study_protocol'] = dict(thinking=False, seed=17,
        workload=args.workload, pilot=bool(args.limit),
        context_policy='ruler_32k_total' if ruler else 'benchmark_native_prompts_and_output_limits',
        server_context_capacity_tokens=server_capacity)
    if ruler: result['study_protocol']['context_total_tokens'] = 32768
    result['study_protocol']['dataset_revisions'] = json.loads((ROOT/'state/dataset-revisions.json').read_text())
    result['study_protocol']['extended_stop_policy'] = 'All benchmark stop strings retained; lists above the API limit use four server stops plus all stops before response caching/scoring.'
    if args.workload == 'RULER':
        components = {k:v['32768,none'] for k,v in result['results'].items() if k!='ruler' and '32768,none' in v}
        if len(components)!=13 or not all(0<=x<=1 for x in components.values()):
            raise ValueError(f'Invalid RULER component metrics: {components}')
        result['study_protocol']['ruler_component_mean'] = sum(components.values())/13
    save(dest/'summary.json', result)
    save(dest/'status.json', dict(status='complete', elapsed_s=result['elapsed_s'],
                                sample_counts=result['sample_counts_saved']))
    print(json.dumps({'workload':args.workload,'results':result['results'],'elapsed_s':result['elapsed_s']},default=serial), flush=True)


if __name__ == '__main__':
    main()
