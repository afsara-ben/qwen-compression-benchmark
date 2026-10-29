"""Acceptance checks for fixed launch settings and owned-process recovery."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch

import pytest

from qbench.common import all_cells, canonical_task_configs, digest, manifest, read, recover_jsonl, sha, write
from qbench.runner import monitored, server_command, PROFILES
from qbench.registry import Registry, validate_record
from test_workflow import fixture


def test_all_sixteen_launch_precisions_and_contexts():
    for cell in all_cells():
        cfg=manifest()['configurations'][cell['configuration']]
        for profile in PROFILES:
            command=server_command('python','weights','tokenizer',cell,'http://127.0.0.1:8000',profile)
            option=lambda name:command[command.index(name)+1]
            assert option('--cpu-offload-gb')=='0'
            assert option('--max-model-len')==('131072' if cell['workload']=='BFCLv3' else '32768')
            assert option('--kv-cache-dtype')==('fp8' if cfg['kv']=='fp8' else 'auto')
            assert ('--quantization' in command)==(cfg['weights']=='nvfp4')
            assert ('--enforce-eager' in command)==profile[3]
            if cfg['weights']=='nvfp4':
                assert option('--linear-backend')==option('--moe-backend')=='marlin'
            if cell['model']=='qwen3_6_35b':
                assert option('--mamba-cache-dtype')==option('--mamba-ssm-cache-dtype')=='float32'


def test_task_identity_ignores_only_nonscientific_metadata():
    a={'task':{'process_results':'<function process_results at 0x1234>', 'metadata':{'tokenizer':'/first'}, 'num_fewshot':5}}
    b={'task':{'process_results':'<function process_results at 0xabcd>', 'metadata':{'tokenizer':'/second'}, 'num_fewshot':5}}
    assert digest(canonical_task_configs(a))==digest(canonical_task_configs(b))
    b['task']['num_fewshot']=0
    assert digest(canonical_task_configs(a))!=digest(canonical_task_configs(b))


def test_pilot_and_wrong_recurrent_precision_rejected(tmp_path):
    cell=next(c for c in all_cells() if c['model']=='qwen3_6_35b' and c['workload']=='RULER')
    directory=fixture(tmp_path,cell)
    summary=read(directory/'summary.json');summary['study_protocol']['pilot']=True
    write(directory/'summary.json',summary)
    record=read(directory/'record.json');record['files']['summary.json']=sha(directory/'summary.json');write(directory/'record.json',record)
    with pytest.raises(ValueError,match='Pilot'):validate_record(directory)
    summary['study_protocol']['pilot']=False;write(directory/'summary.json',summary)
    launch=read(directory/'launch.json');i=launch['command'].index('--mamba-ssm-cache-dtype')
    launch['command'][i+1]='float16';write(directory/'launch.json',launch)
    for name in ('summary.json','launch.json'):record['files'][name]=sha(directory/name)
    write(directory/'record.json',record)
    with pytest.raises(ValueError,match='precision'):validate_record(directory)


def test_interrupt_stops_owned_child_preserves_unrelated_process(tmp_path):
    other=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
    stopped=threading.Event();failure=[]
    marker=tmp_path/'owned.pid'
    code='import os,time,pathlib; pathlib.Path('+repr(str(marker))+').write_text(str(os.getpid())); time.sleep(30)'
    def run():
        try:monitored([sys.executable,'-c',code],tmp_path/'log',dict(os.environ),stopped)
        except InterruptedError:failure.append('stopped')
    thread=threading.Thread(target=run);thread.start()
    try:
        deadline=time.monotonic()+5
        while not marker.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert marker.exists()
        owned=int(marker.read_text());stopped.set();thread.join(timeout=5)
        assert failure==['stopped'] and not thread.is_alive()
        with pytest.raises(ProcessLookupError):os.kill(owned,0)
        assert other.poll() is None
    finally:
        stopped.set();thread.join(timeout=5);other.terminate();other.wait(timeout=5)


def test_completed_cells_are_not_claimed_again(tmp_path):
    cell=next(c for c in all_cells() if c['workload']=='RULER')
    registry=Registry(tmp_path/'state');registry.publish(fixture(tmp_path/'fixture',cell))
    second=Registry(tmp_path/'state')
    assert second.claim(cell) is None
    assert cell['id'] not in {c['id'] for c in second.pending()}


def test_full_averages_use_all_seven_real_scores(tmp_path):
    from qbench.bundles import import_seed
    from qbench.report import build
    registry=Registry(tmp_path/'state');import_seed(registry)
    report=build(registry)
    for row in report['rows']:
        scores=[row[w] for w in manifest()['workloads']]
        if row['configuration'] in ('KV','E+KV'):
            assert row['Avg.']==pytest.approx(sum(scores)/7)
        else:
            assert row['Avg.'] is None
    assert report['remaining']==78 and report['ruler_completed']==6


def test_interrupted_response_append_preserves_complete_records(tmp_path):
    path=tmp_path/'responses.jsonl'
    path.write_bytes(b'{"id":"simple_0"}\n{"id":"simple_1"')
    assert recover_jsonl(path)==[{'id':'simple_0'}]
    assert path.read_bytes()==b'{"id":"simple_0"}\n'
    assert len(list(tmp_path.glob('*.interrupted-*')))==1
    path.write_bytes(b'{"id":"simple_0"}')
    assert recover_jsonl(path)==[{'id':'simple_0'}]
    assert path.read_bytes().endswith(b'\n')
    path.write_bytes(b'broken\n{"id":"simple_0"}\n')
    with pytest.raises(ValueError):recover_jsonl(path)
