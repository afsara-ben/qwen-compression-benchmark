"""Filesystem coordination; no shared SQLite database or node-local absolute paths."""
from __future__ import annotations

import os
import fcntl
import gzip
import json
from collections import Counter
from contextlib import contextmanager
import shutil
import socket
import threading
import time
import uuid
from pathlib import Path

from .common import all_cells, canonical_task_configs, digest, fingerprint, manifest, now, read, safe_relative, sha, write


def validate_examples(directory, cell, files, expected):
    """Count actual evidence records; a summary's declared count is insufficient."""
    counts=Counter();seen=set()
    if cell['workload']=='BFCLv3':
        paths=[directory/name for name in files if name.startswith('responses/') and '_result.json' in name]
        for path in paths:
            opener=gzip.open if path.suffix=='.gz' else open
            with opener(path,'rt') as stream:
                for line in stream:
                    if not line.strip():continue
                    sample=json.loads(line); identity=sample['id']
                    if identity in seen or 'traceback' in sample or str(sample.get('result','')).startswith('Error during inference:'):
                        raise ValueError('Duplicate or failed BFCL response evidence')
                    seen.add(identity);counts[identity.rsplit('_',1)[0]]+=1
    else:
        def count(task,sample):
            identity=(task,sample['doc_id'],sample.get('filter','none'))
            if identity in seen:
                raise ValueError('Duplicate per-example evidence')
            seen.add(identity);counts[task]+=1
        if 'samples.jsonl.gz' in files:
            with gzip.open(directory/'samples.jsonl.gz','rt') as stream:
                for line in stream:
                    sample=json.loads(line);count(sample['task'],sample)
        else:
            with gzip.open(directory/'samples.json.gz','rt') as stream:
                samples=json.load(stream)
            for task,rows in samples.items():
                for sample in rows:count(task,sample)
    if dict(counts)!=expected:
        raise ValueError('Actual per-example counts do not match the full workload')


def validate_record(directory, hashes=True):
    directory = Path(directory)
    record = read(directory / 'record.json')
    cells = {x['id']: x for x in all_cells()}
    cell = cells.get(record.get('cell_id'))
    if not cell or record.get('protocol_fingerprint') != fingerprint(cell):
        raise ValueError('Unknown cell or incompatible scientific protocol')
    if record.get('status') != 'complete':
        raise ValueError('Only complete records can enter the result grid')
    files = record.get('files', {})
    if not {'summary.json', 'audit.json', 'launch.json'} <= files.keys():
        raise ValueError('Missing summary, audit or launch evidence')
    if cell['workload'] == 'BFCLv3':
        if not any(k.startswith('responses/') for k in files) or not any(k.startswith('scores/') for k in files):
            raise ValueError('Missing BFCL per-example/category evidence')
    elif not {'samples.json.gz', 'samples.jsonl.gz'} & files.keys():
        raise ValueError('Missing per-example evidence')
    for name, checksum in files.items():
        path = directory / safe_relative(name)
        if not path.is_file() or path.is_symlink():
            raise ValueError('Missing or symbolic evidence file: ' + name)
        if hashes and sha(path) != checksum:
            raise ValueError('Evidence checksum mismatch: ' + name)
    spec = manifest()
    cfg, model = spec['configurations'][cell['configuration']], spec['models'][cell['model']]
    expected_experts = model['reap' if cfg['expert'] else 'base']['experts']
    audit = read(directory / 'audit.json')['results'][0]
    if audit.get('non_cuda_parameters') or len(audit['experts']) != model['layers'] or {x['count'] for x in audit['experts']} != {expected_experts}:
        raise ValueError('Expert count or weight residency mismatch')
    groups = [g for g in audit['cache_groups'] if g['spec'] == 'FullAttentionSpec']
    if not groups or any(g['dtype'] != ('torch.uint8' if cfg['kv'] == 'fp8' else 'torch.bfloat16') for g in groups):
        raise ValueError('Attention cache precision mismatch')
    if cfg['weights'] == 'nvfp4':
        modules = [m for m in audit.get('nvfp4_modules', []) if m['method'] == 'ModelOptNvFp4FusedMoE']
        if len(modules) != model['layers'] or not all(m.get('use_a16') and 'MARLIN' in m['backend'] for m in modules):
            raise ValueError('NVFP4 W4A16/Marlin audit mismatch')
        if not audit.get('marlin_canonical_order'):
            raise ValueError('Marlin ordering policy missing')
    elif audit['quantization_methods'].get('ModelOptNvFp4FusedMoE'):
        raise ValueError('Unexpected quantized weights')
    elif set(audit['parameter_dtypes']) - {'torch.bfloat16', 'torch.float32'}:
        raise ValueError('Non-BF16 weights in an unquantized configuration')
    if record.get('weight_storage_bytes') != audit['weight_storage_bytes']:
        raise ValueError('Weight measurement mismatch')
    launch = read(directory / 'launch.json')
    identity = launch.get('benchmark_identity', {})
    if identity.get('model') != cell['model'] or identity.get('configuration') != cell['configuration']:
        raise ValueError('Model launch identity mismatch')
    checkpoint = model['reap' if cfg['expert'] else 'base']
    if identity.get('checkpoint') != checkpoint or identity.get('tokenizer') != model['base']:
        raise ValueError('Checkpoint or tokenizer revision mismatch')
    command = launch.get('command', [])
    def option(name):
        return command[command.index(name)+1] if name in command else None
    expected_options = {'--dtype':'bfloat16', '--seed':'17', '--cpu-offload-gb':'0',
                        '--tensor-parallel-size':'1', '--pipeline-parallel-size':'1',
                        '--kv-cache-dtype':'fp8' if cfg['kv']=='fp8' else 'auto'}
    if model['family'] == 1:
        expected_options.update({'--mamba-cache-dtype':'float32', '--mamba-ssm-cache-dtype':'float32'})
        if '--language-model-only' not in command:
            raise ValueError('Qwen3.6 text-only serving was not enabled')
    if cfg['weights'] == 'nvfp4':
        expected_options.update({'--quantization':'modelopt_fp4', '--linear-backend':'marlin', '--moe-backend':'marlin'})
    elif option('--quantization') is not None:
        raise ValueError('Unexpected weight quantization in launch command')
    if any(option(name) != value for name,value in expected_options.items()):
        raise ValueError('Launch precision, offload, parallelism or seed mismatch')
    if cell['workload']=='RULER' and option('--max-model-len') != '32768':
        raise ValueError('RULER launch context mismatch')
    if not audit.get('kv_scales') or any(float(x)!=1.0 for value in audit['kv_scales'].values()
            for x in (value if isinstance(value,list) else [value])):
        raise ValueError('Attention KV scale policy mismatch')
    summary = read(directory / 'summary.json')
    protocol = summary.get('study_protocol', {})
    if protocol.get('pilot') is not False or protocol.get('thinking') is not False:
        raise ValueError('Pilot or missing full non-thinking protocol')
    if cell['workload'] != 'BFCLv3':
        if protocol.get('seed') != 17 or protocol.get('dataset_revisions') != spec['dataset_revisions']:
            raise ValueError('Seed/dataset revisions mismatch')
    else:
        if (protocol.get('official_revision') != spec['sources']['bfcl']['revision']
                or protocol.get('output_allowance_tokens') != 4096
                or protocol.get('server_context_capacity_tokens') != 131072):
            raise ValueError('BFCL protocol mismatch')
        if summary.get('inference_errors'):
            raise ValueError('BFCL contains infrastructure errors')
    task = spec['workloads'][cell['workload']]
    if cell['workload'] != 'BFCLv3':
        normalized = canonical_task_configs(summary['configs'])
        if digest(normalized) != task['task_config_digest_by_model'][cell['model']]:
            raise ValueError('Task/few-shot/generation/scoring configuration mismatch')
    if summary.get('sample_counts_saved') != task['sample_counts']:
        raise ValueError('Incomplete or mismatched sample counts')
    validate_examples(directory,cell,files,task['sample_counts'])
    score = summary['results'][task['task']][task['metric']]
    if not isinstance(score, (int, float)) or not 0 <= score <= 1 or score != record.get('score'):
        raise ValueError('Invalid score')
    if cell['workload'] == 'RULER':
        subs = {k: v['32768,none'] for k, v in summary['results'].items() if k != 'ruler' and '32768,none' in v}
        if protocol.get('context_total_tokens') != 32768 or len(subs) != 13:
            raise ValueError('Incomplete RULER suite or incorrect context')
        if any(not 0 <= v <= 1 for v in subs.values()) or abs(sum(subs.values()) / 13 - score) > 1e-12:
            raise ValueError('RULER aggregation mismatch')
    if cell['workload']=='BFCLv3':
        categories=summary['category_scores']
        if set(categories)!=set(task['sample_counts']):
            raise ValueError('BFCL category coverage mismatch')
        for name,value in categories.items():
            if (value['total_count']!=task['sample_counts'][name] or not 0<=value['correct_count']<=value['total_count']
                    or abs(value['accuracy']-value['correct_count']/value['total_count'])>1e-12):
                raise ValueError('BFCL category score/count mismatch')
        acc=lambda name:categories[name]['accuracy']
        simple=sum(acc(name) for name in ('simple','java','javascript'))/3
        nonlive=(simple+sum(acc(name) for name in ('multiple','parallel','parallel_multiple','irrelevance')))/5
        live=[value for name,value in categories.items() if name.startswith('live_')]
        live_score=sum(value['correct_count'] for value in live)/sum(value['total_count'] for value in live)
        multi=sum(value['accuracy'] for name,value in categories.items() if name.startswith('multi_turn_'))/4
        if abs((nonlive+live_score+multi)/3-score)>1e-12:
            raise ValueError('BFCL overall score mismatch')
    if not record.get('hardware') or not record.get('runtime_fingerprint'):
        raise ValueError('Missing hardware/runtime provenance')
    return record


class Claim:
    def __init__(self, registry, cell, token):
        self.registry, self.cell, self.token = registry, cell, token
        self.directory = registry.root / 'claims' / cell['id']
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def owned(self):
        try:
            return read(self.directory / 'owner.json')['token'] == self.token
        except (FileNotFoundError, ValueError, KeyError):
            return False

    def _heartbeat(self):
        while not self.stop_event.wait(15):
            if not self.owned():
                return
            try:
                os.utime(self.directory / 'owner.json', None)
            except FileNotFoundError:
                return

    def release(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        with self.registry.guard(self.cell['id']):
            if self.owned():
                shutil.rmtree(self.directory, ignore_errors=True)


class Registry:
    def __init__(self, root):
        self.root = Path(root).resolve()
        for name in ('records', 'claims', 'claim-locks', 'issues', 'events', 'workers'):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.verified = {}

    @contextmanager
    def guard(self, cell_id):
        # Serialize expiry checks and replacement so two stale-lease reclaimers
        # cannot accidentally remove a newly acquired owner.
        safe_relative(cell_id)
        with (self.root/'claim-locks'/(cell_id+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            yield

    def records(self):
        result = {}
        for path in sorted((self.root / 'records').glob('*/record.json')):
            stat = path.stat()
            key = (str(path), stat.st_mtime_ns, stat.st_size)
            try:
                if key not in self.verified:
                    self.verified[key] = validate_record(path.parent)
                value = self.verified[key]
                result[value['cell_id']] = value
            except (ValueError, KeyError, OSError, TypeError, IndexError, EOFError) as exc:
                self.issue(path.parent.name, 'invalid_result', str(exc))
        return result

    def phase(self):
        records = self.records()
        ruler = [x for x in all_cells() if x['workload'] == 'RULER']
        if not all(x['id'] in records for x in ruler):
            return 'RULER'
        return 'complete' if len(records) == 112 else 'other'

    def pending(self, models=None, configs=None, workloads=None):
        records, phase = self.records(), self.phase()
        return [x for x in all_cells() if x['id'] not in records
                and (x['workload'] == 'RULER' if phase == 'RULER' else x['workload'] != 'RULER')
                and (not models or x['model'] in models)
                and (not configs or x['configuration'] in configs)
                and (not workloads or x['workload'] in workloads)]

    def claim(self, cell, ttl=900):
        with self.guard(cell['id']):
            return self._claim_locked(cell,ttl)

    def _claim_locked(self, cell, ttl):
        if cell['id'] in self.records():
            return None
        if self.phase() == 'RULER' and cell['workload'] != 'RULER':
            return None
        directory = self.root / 'claims' / cell['id']
        for _ in range(2):
            try:
                directory.mkdir()
                token = uuid.uuid4().hex
                write(directory / 'owner.json', dict(token=token, host=socket.gethostname(), pid=os.getpid(), started=now()))
                return Claim(self, cell, token)
            except FileExistsError:
                try:
                    stamp = (directory / 'owner.json').stat().st_mtime if (directory / 'owner.json').exists() else directory.stat().st_mtime
                    if time.time() - stamp <= ttl:
                        return None
                    tomb = directory.with_name(directory.name + '.expired-' + uuid.uuid4().hex)
                    directory.rename(tomb)
                    shutil.rmtree(tomb)
                except FileNotFoundError:
                    pass
        return None

    def publish(self, directory, claim=None):
        record = validate_record(directory)
        if claim and (not claim.owned() or claim.cell['id'] != record['cell_id']):
            raise RuntimeError('Job ownership was lost; result retained locally, not published')
        dest = self.root / 'records' / record['cell_id']
        if dest.exists():
            old = validate_record(dest)
            if old['protocol_fingerprint'] != record['protocol_fingerprint']:
                raise ValueError('Conflicting result protocol')
            return old
        temporary = dest.with_name('.' + dest.name + '-' + uuid.uuid4().hex)
        shutil.copytree(directory, temporary)
        try:
            with self.guard(record['cell_id']):
                if claim and not claim.owned():
                    raise RuntimeError('Job ownership lost while publishing')
                try:
                    temporary.rename(dest)
                except OSError:
                    if not dest.exists():
                        raise
                    validate_record(dest)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        (self.root / 'issues' / (record['cell_id'] + '.json')).unlink(missing_ok=True)
        path=dest/'record.json';stat=path.stat()
        self.verified[(str(path),stat.st_mtime_ns,stat.st_size)]=record
        return record

    def issue(self, cell_id, kind, message):
        safe_relative(cell_id)
        write(self.root / 'issues' / (cell_id + '.json'), dict(cell_id=cell_id, kind=kind, message=message, time=now()))
