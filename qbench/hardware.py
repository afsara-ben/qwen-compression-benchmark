from __future__ import annotations

import csv
import ctypes
import uuid
import fcntl
import io
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .common import manifest


@dataclass
class GPU:
    index: str
    uuid: str
    name: str
    total_mib: int
    free_mib: int
    capability: float
    driver: str
    busy: bool = False

    def to_dict(self):
        return asdict(self)


def parse_inventory(text, processes='', visible=None, overrides=None, slurm_visible=None):
    rows = []
    active = {row[0].strip() for row in csv.reader(io.StringIO(processes)) if row}
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        index, uid, name, total, free, cap, driver = (x.strip() for x in row)
        rows.append(GPU(index, uid, name, int(float(total)), int(float(free)), float(cap), driver, uid in active))
    def select(pool, selector):
        if selector is None:
            return pool
        tokens = [x.strip() for x in selector.split(',') if x.strip()]
        return [g for g in pool if any(t == g.index or g.uuid == t or (t.startswith('GPU-') and g.uuid.startswith(t)) for t in tokens)]
    # CUDA IDs refer to the physical inventory here. User overrides may also use
    # ordinal positions within the already restricted visible inventory.
    rows = select(rows, visible if visible is not None else slurm_visible)
    if overrides:
        selected = set()
        for token in overrides.split(','):
            token = token.strip()
            if token.isdigit() and int(token) < len(rows):
                selected.add(rows[int(token)].uuid)
            else:
                selected.update(g.uuid for g in rows if g.uuid == token or g.uuid.startswith(token))
        rows = [g for g in rows if g.uuid in selected]
    return rows


def discover(overrides=None):
    command = ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free,compute_cap,driver_version', '--format=csv,noheader,nounits']
    text = subprocess.check_output(command, text=True, timeout=20)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True, timeout=20)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    # Ask the driver for the visible UUIDs so integer CUDA IDs remain correct
    # under PCI ordering, scheduler remapping and container device isolation.
    try:
        cuda=ctypes.CDLL('libcuda.so.1')
        count=ctypes.c_int()
        if cuda.cuInit(0)!=0 or cuda.cuDeviceGetCount(ctypes.byref(count))!=0:
            raise RuntimeError('CUDA enumeration unavailable')
        uuids=[]
        for ordinal in range(count.value):
            device=ctypes.c_int(); raw=(ctypes.c_ubyte*16)()
            if cuda.cuDeviceGet(ctypes.byref(device),ordinal)!=0 or cuda.cuDeviceGetUuid(ctypes.byref(raw),device)!=0:
                raise RuntimeError('CUDA UUID enumeration unavailable')
            uuids.append('GPU-'+str(uuid.UUID(bytes=bytes(raw))))
        if visible is not None or os.environ.get('NVIDIA_VISIBLE_DEVICES') is not None:
            visible=','.join(uuids)
    except (OSError,AttributeError,RuntimeError):
        pass
    return parse_inventory(text, processes, visible, overrides,
                           os.environ.get('SLURM_STEP_GPUS', os.environ.get('SLURM_JOB_GPUS')))


def required_bytes(cell, context=None):
    spec = manifest()
    model, config = spec['models'][cell['model']], spec['configurations'][cell['configuration']]
    weights = model['weight_bytes'][config['weights'] + ('_reap' if config['expert'] else '_base')]
    tokens = context or spec['workloads'][cell['workload']]['context_capacity']
    if model['family'] == 0:
        cache = tokens * 48 * 4 * 128 * 2 * (1 if config['kv'] == 'fp8' else 2)
    else:
        # Conservative allowance for hybrid shared-pool padding + FP32 states.
        cache = int(tokens / 32768 * (0.42e9 if config['kv'] == 'fp8' else 0.75e9)) + int(0.15e9)
    return weights + cache + 4 * 1024**3


def fits(gpu, cell):
    return gpu.capability >= 8.0 and required_bytes(cell) <= min(gpu.free_mib, gpu.total_mib * .95) * 1024**2


class GPULock:
    def __init__(self, gpu, directory=None):
        root = Path(directory or os.environ.get('QBENCH_GPU_LOCK_DIR', f'/tmp/qwen-benchmark-gpus-{os.getuid()}'))
        root.mkdir(parents=True, exist_ok=True)
        self.file = (root / (gpu.uuid + '.lock')).open('a+')
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.file.close()
            raise

    def close(self):
        self.file.close()
