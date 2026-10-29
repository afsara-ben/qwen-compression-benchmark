from __future__ import annotations

import argparse
import fcntl
import math
import os
from pathlib import Path
import struct
import shutil
import time

from .common import manifest, now, read, sha, write


def checkpoint(work, model, expert):
    return Path(work)/'models'/model/('reap' if expert else 'base')


def artifact(work, model, expert):
    return Path(work)/'artifacts'/model/('reap-NVFP4' if expert else 'base-NVFP4')


def download(work, model, expert):
    from huggingface_hub import HfApi, snapshot_download
    spec = manifest()['models'][model]['reap' if expert else 'base']
    dest = checkpoint(work, model, expert)
    dest.mkdir(parents=True, exist_ok=True)
    receipt = dest/'download.json'
    if receipt.exists():
        r = read(receipt)
        if r['checkpoint'] == spec and all((dest/f['name']).is_file() and (dest/f['name']).stat().st_size == f['bytes']
                and (dest/f['name']).stat().st_mtime_ns == f['mtime_ns'] for f in r['files']):
            return dest
    info = HfApi().model_info(spec['repo'], revision=spec['revision'], files_metadata=True)
    if info.sha != spec['revision']:
        raise ValueError('Repository did not resolve to the pinned revision')
    remote = {f.rfilename:f.lfs.sha256 for f in info.siblings if getattr(f, 'lfs', None)}
    needed=sum(int(f.size or 0) for f in info.siblings if f.rfilename.endswith(('.json','.safetensors','.txt','.model','.jinja'))
               and (not (dest/f.rfilename).is_file() or (dest/f.rfilename).stat().st_size != int(f.size or 0)))
    if shutil.disk_usage(dest).free < needed + 10*1024**3:
        raise OSError(f'Insufficient disk: {needed/1e9:.1f} GB of checkpoint data plus 10 GiB download headroom required')
    snapshot_download(spec['repo'], revision=spec['revision'], local_dir=dest,
                      allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'], max_workers=4)
    cfg = read(dest/'config.json')
    text = cfg.get('text_config', cfg)
    if text['num_experts'] != spec['experts'] or text['num_experts_per_tok'] != 8:
        raise ValueError('Checkpoint expert configuration mismatch')
    index = read(dest/'model.safetensors.index.json')['weight_map']
    files, found = [], set()
    for filename in sorted(set(index.values())):
        p = dest/filename
        checksum = sha(p)
        if filename not in remote or checksum != remote[filename]:
            raise ValueError('Checkpoint LFS checksum mismatch: '+filename)
        with p.open('rb') as stream:
            size = struct.unpack('<Q',stream.read(8))[0]
            import json
            headers = json.loads(stream.read(size))
        keys = set(headers)-{'__metadata__'}
        if any(index.get(k) != filename for k in keys):
            raise ValueError('Safetensors index mismatch')
        found.update(keys)
        files.append(dict(name=filename, bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns, sha256=checksum))
    if found != set(index):
        raise ValueError('Missing checkpoint tensors')
    for name in ('config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        p = dest/name
        if p.exists():
            files.append(dict(name=name, bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns, sha256=sha(p)))
    write(receipt, dict(checkpoint=spec, status='verified', files=files, verified_at=now()))
    return dest


def prepare(work, model, config):
    work = Path(work)
    (work/'state').mkdir(parents=True, exist_ok=True)
    lock = (work/'state/prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    c = manifest()['configurations'][config]
    for attempt in range(3):
        try:
            base = download(work, model, False)
            selected = download(work, model, True) if c['expert'] else base
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5*(2**attempt))
    if c['weights'] == 'nvfp4':
        from .convert import convert
        output = artifact(work, model, c['expert'])
        estimate=manifest()['models'][model]['weight_bytes']['nvfp4_reap' if c['expert'] else 'nvfp4_base']*1.3
        present=sum(p.stat().st_size for p in output.glob('*.safetensors')) if output.exists() else 0
        if shutil.disk_usage(work).free < max(0,estimate-present)+2*1024**3:
            raise OSError('Insufficient disk for NVFP4 conversion; completed shards are retained')
        convert(selected, output)
        selected = output
    write(work/'state/dataset-revisions.json', manifest()['dataset_revisions'])
    lock.close()
    return selected, base


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    prepare(args.work_dir, args.model, args.config)
