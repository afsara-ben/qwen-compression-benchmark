from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def canonical_task_configs(configs):
    """Remove filesystem metadata and process-specific callable addresses only."""
    def normalize(value):
        if isinstance(value, dict):
            return {k:normalize(v) for k,v in value.items()}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        if isinstance(value, str):
            return re.sub(r'<function ([^>]+) at 0x[0-9a-fA-F]+>',r'<function \1>',value)
        return value
    return normalize({k:{field:value for field,value in config.items() if field!='metadata'} for k,config in configs.items()})


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def recover_jsonl(path):
    """Retain complete response records after an interrupted append.

    Only an invalid final line is removed, with its original bytes preserved.
    Corruption before the final line remains a hard error.
    """
    path=Path(path);records=[];tail=None;offset=0;needs_newline=False
    with path.open('rb') as stream:
        while True:
            offset=stream.tell();line=stream.readline()
            if not line:break
            if not line.strip():continue
            try:
                records.append(json.loads(line))
                needs_newline=not line.endswith(b'\n')
            except (ValueError,UnicodeError):
                rest=stream.read()
                if rest.strip():
                    raise ValueError('Malformed JSONL record before the final line: '+str(path))
                tail=line+rest
                break
    if tail is not None:
        path.with_name(path.name+'.interrupted-'+uuid.uuid4().hex).write_bytes(tail)
        with path.open('r+b') as stream:stream.truncate(offset)
    elif needs_newline:
        with path.open('ab') as stream:stream.write(b'\n')
    return records


def manifest():
    return read(PROJECT / 'config/experiment.json')


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Unsafe relative path: ' + str(value))
    return path


def cell_id(model, config, workload):
    return '--'.join((model, config, workload))


def all_cells():
    spec = manifest()
    return [dict(id=cell_id(m, c, w), model=m, configuration=c, workload=w)
            for m in spec['models'] for c in spec['configurations'] for w in spec['workloads']]


def protocol(model, config, workload):
    spec = manifest()
    c, m = spec['configurations'][config], spec['models'][model]
    return dict(schema_version=1, model=model, configuration=config, workload=workload,
                checkpoint=m['reap' if c['expert'] else 'base'], tokenizer=m['base'],
                precision=c, recipe=spec['quantization'], generation=spec['generation'],
                task=spec['workloads'][workload], sources=spec['sources'],
                dataset_revisions=spec['dataset_revisions'], ruler_banks=spec['ruler_bank_digests'][model]
                if workload == 'RULER' else None)


def fingerprint(cell):
    return digest(protocol(cell['model'], cell['configuration'], cell['workload']))
