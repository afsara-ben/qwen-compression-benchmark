from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.request
import uuid
from functools import lru_cache

from .bootstrap import environment
from .common import PROJECT, digest, fingerprint, manifest, now, read, sha, write
from .hardware import GPULock, discover, fits
from .prepare import artifact, checkpoint
from .registry import Registry, validate_record
from .report import build

PROFILES = [(4,8192,.90,False), (2,4096,.90,False), (1,2048,.90,True), (1,1024,.95,True)]


@lru_cache(maxsize=4)
def runtime_versions(python):
    code="import json,sys,importlib.metadata as m; print(json.dumps(dict(python=sys.version.split()[0],**{n:m.version(n) for n in ['vllm','torch','transformers','flashinfer-python','numpy']})))"
    versions=json.loads(subprocess.check_output([str(python),'-c',code],text=True,timeout=30))
    for name,expected in {'vllm':'0.32.0','torch':'2.13.0','transformers':'5.19.0','flashinfer-python':'0.7.0.post1'}.items():
        if versions[name].split('+')[0] != expected:
            raise RuntimeError(f'Incompatible {name}: {versions[name]}, expected {expected}')
    if not versions['python'].startswith('3.12.'):
        raise RuntimeError('The evaluation runtime must use Python 3.12')
    return versions


def request(endpoint, path, payload=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(endpoint+'/'+path, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
        return json.loads(body) if body else None


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    # Every child is launched in its own session; never signal a GPU's arbitrary PIDs.
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def monitored(command, logfile, env, stopped, claim=None):
    if stopped.is_set() or (claim and not claim.owned()):
        raise InterruptedError('Stopped or job lease lost before launch')
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('a') as stream:
        process = subprocess.Popen(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            while process.poll() is None:
                if stopped.wait(1) or (claim and not claim.owned()):
                    raise InterruptedError('Stopped or job lease lost')
            if process.returncode:
                raise RuntimeError(f'Command exited {process.returncode}; see {logfile}')
        finally:
            stop_process(process)


def server_command(python, model_path, tokenizer, cell, endpoint, profile):
    spec = manifest(); model = spec['models'][cell['model']]; config = spec['configurations'][cell['configuration']]
    concurrency, batched, utilization, eager = profile
    port = endpoint.rsplit(':',1)[1]
    command = [str(python), '-m', 'vllm.entrypoints.openai.api_server', '--model', str(model_path),
        '--tokenizer', str(tokenizer), '--served-model-name', 'quality-model', '--host', '127.0.0.1', '--port', port,
        '--dtype','bfloat16','--seed','17','--generation-config','vllm','--tensor-parallel-size','1',
        '--pipeline-parallel-size','1','--data-parallel-size','1','--distributed-executor-backend','uni',
        '--no-enable-expert-parallel','--max-model-len',str(spec['workloads'][cell['workload']]['context_capacity']),
        '--max-num-seqs',str(concurrency),'--max-num-batched-tokens',str(batched),
        '--attention-config','{"backend":"FLASHINFER"}', '--enable-chunked-prefill','--no-enable-prefix-caching',
        '--gpu-memory-utilization',str(utilization),'--cpu-offload-gb','0','--kv-cache-dtype',
        'fp8' if config['kv']=='fp8' else 'auto','--max-logprobs','1','--no-enable-log-requests',
        '--worker-extension-cls','quality_reap_loader.QualityReapAudit' if model['family']==1 and config['expert'] else 'quality_audit.QualityAudit']
    if eager:
        command += ['--enforce-eager']
    else:
        command += ['--compilation-config','{"cudagraph_mode":"FULL_AND_PIECEWISE"}']
    if model['family']==1:
        command += ['--language-model-only','--mamba-cache-dtype','float32','--mamba-ssm-cache-dtype','float32','--gdn-prefill-backend','triton']
    else:
        command += ['--block-size','16']
    if config['weights']=='nvfp4':
        command += ['--quantization','modelopt_fp4','--linear-backend','marlin','--moe-backend','marlin']
    return command


class Server:
    def __init__(self, python, work, gpu, cell, profile, stopped):
        self.work, self.cell, self.gpu, self.profile = work, cell, gpu, profile
        spec = manifest(); c = spec['configurations'][cell['configuration']]; m = spec['models'][cell['model']]
        self.model = artifact(work,cell['model'],c['expert']) if c['weights']=='nvfp4' else checkpoint(work,cell['model'],c['expert'])
        self.tokenizer = checkpoint(work,cell['model'],False)
        self.runtime = dict(gpu={k:v for k,v in gpu.to_dict().items() if k not in ('free_mib','busy','index')},
                            profile=profile, versions=runtime_versions(python),
                            engine='FLASHINFER', quantized_backend='marlin', code={p.name:sha(p) for p in (PROJECT/'runtime').glob('*.py')},
                            server_code=sha(__file__))
        self.runtime_id = digest(self.runtime)
        self.identity = (cell['model'],cell['configuration'],spec['workloads'][cell['workload']]['context_capacity'])
        self.attempt = f"{cell['id']}--{fingerprint(cell)[:12]}--{self.runtime_id[:16]}"
        self.directory = work/'results'/self.attempt
        self.directory.mkdir(parents=True,exist_ok=True)
        with socket.socket() as s:
            s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
        self.endpoint = f'http://127.0.0.1:{port}'
        self.env = environment(python,work)
        self.env.update(CUDA_VISIBLE_DEVICES=gpu.uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID', VLLM_SERVER_DEV_MODE='1',
                        HF_HUB_OFFLINE='1', VLLM_PLUGINS='qbench' if c['weights']=='nvfp4' else '',
                        MOECBENCH_MARLIN_CANONICAL_ORDER='1' if c['weights']=='nvfp4' else '0',
                        QBENCH_ENDPOINT=self.endpoint, QBENCH_MODEL_ID=cell['model'],
                        QBENCH_MODEL_PATH=str(self.model), QBENCH_TOKENIZER_PATH=str(self.tokenizer),
                        QBENCH_CONCURRENCY=str(profile[0]))
        self.env.pop('VLLM_MARLIN_INPUT_DTYPE',None)
        if c['weights']=='nvfp4':
            self.env['VLLM_USE_DEEP_GEMM']='0'
        command=server_command(python,self.model,self.tokenizer,cell,self.endpoint,profile)
        self.launch=dict(command=command, started_at=now(), runtime=self.runtime,
            benchmark_identity=dict(model=cell['model'],configuration=cell['configuration'],
                checkpoint=m['reap' if c['expert'] else 'base'],tokenizer=m['base']))
        write(self.directory/'launch.json',self.launch)
        self.logfile = work/'logs'/(self.attempt+'-server.log')
        self.logfile.parent.mkdir(parents=True,exist_ok=True)
        self.log = self.logfile.open('a')
        self.process = subprocess.Popen(command,stdout=self.log,stderr=subprocess.STDOUT,env=self.env,start_new_session=True)
        try:
            deadline=time.monotonic()+3600
            while time.monotonic()<deadline:
                if stopped.wait(2):
                    raise InterruptedError('Stopping server startup')
                if self.process.poll() is not None:
                    raise RuntimeError(f'Server startup failed; see {self.logfile}')
                try:
                    request(self.endpoint,'health',timeout=2)
                    break
                except Exception:
                    pass
            else:
                raise TimeoutError('Server startup exceeded one hour')
            self.audit=request(self.endpoint,'collective_rpc',{'method':'quality_audit'})['results'][0]
            write(self.directory/'audit.json',{'results':[self.audit]})
            expected=m['reap' if c['expert'] else 'base']['experts']
            if self.audit['non_cuda_parameters'] or {x['count'] for x in self.audit['experts']}!={expected}:
                raise RuntimeError('CPU offload or wrong expert count')
            if c['kv']=='fp8' and not self.audit['kv_dtype'].startswith('fp8'):
                raise RuntimeError('FP8 cache not active')
            if any(float(x)!=1.0 for v in self.audit['kv_scales'].values() for x in (v if isinstance(v,list) else [v])):
                raise RuntimeError('KV scale policy differs from the pinned unit-scale policy')
            if c['weights']=='nvfp4':
                modules=[x for x in self.audit['nvfp4_modules'] if x['method']=='ModelOptNvFp4FusedMoE']
                if len(modules)!=m['layers'] or not all(x['use_a16'] and 'MARLIN' in x['backend'] for x in modules):
                    raise RuntimeError('NVFP4 W4A16/Marlin not active')
            smoke=request(self.endpoint,'v1/chat/completions',dict(model='quality-model',messages=[dict(role='user',content='What is 6 multiplied by 7? Reply with only the number.')],temperature=0,max_tokens=32,chat_template_kwargs={'enable_thinking':False}))
            write(self.directory/'smoke.json',smoke)
            if '42' not in (smoke['choices'][0]['message']['content'] or ''):
                raise RuntimeError('Semantic smoke failed')
            likelihood=request(self.endpoint,'v1/completions',dict(model='quality-model',prompt='One plus one is two.',max_tokens=1,echo=True,logprobs=1,temperature=0))
            write(self.directory/'likelihood-smoke.json',likelihood)
            probs=likelihood['choices'][0]['logprobs']['token_logprobs']
            if not probs or not all(math.isfinite(x) for x in probs if x is not None):
                raise RuntimeError('Prompt log-probability check failed')
        except BaseException:
            self.close()
            raise

    def close(self):
        stop_process(getattr(self,'process',None))
        if hasattr(self,'log'):
            self.log.close()


def evaluate(server, cell, python, bfcl_python, stopped, claim=None, smoke=False):
    spec=manifest(); workload=cell['workload']; family=spec['models'][cell['model']]['family']
    # Sharing a server never implies sharing responses across workloads/protocols.
    attempt=f"{cell['id']}--{fingerprint(cell)[:12]}--{server.runtime_id[:16]}"+('-smoke' if smoke else '')
    directory=server.work/'results'/attempt
    directory.mkdir(parents=True,exist_ok=True)
    write(directory/'launch.json',server.launch)
    write(directory/'audit.json',{'results':[server.audit]})
    write(directory/'protocol.json',dict(protocol_fingerprint=fingerprint(cell),runtime_fingerprint=server.runtime_id))
    env=dict(server.env, CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='0')
    if workload=='BFCLv3':
        command=[bfcl_python,PROJECT/'runtime/quality_bfcl.py']
    else:
        command=[python,PROJECT/'runtime/quality_eval.py','--workload',workload]
    command += ['--gpu',str(family),'--attempt',attempt]
    if smoke:
        command += ['--limit','1']
    monitored(command,server.work/'logs'/(attempt+'.log'),env,stopped,claim)
    full=directory/workload/('pilot' if smoke else 'full')
    if smoke:
        write(directory/'smoke-complete.json',dict(cell_id=cell['id'],status='smoke_only',time=now()))
        return None
    summary=read(full/'summary.json'); task=spec['workloads'][workload]
    export=directory/'export'
    export.mkdir(exist_ok=True)
    for source,target in [(full/'summary.json','summary.json'),(directory/'launch.json','launch.json'),(directory/'audit.json','audit.json')]:
        shutil.copy2(source,export/target)
    for name in ('samples.json.gz','code-verdicts.jsonl'):
        if (full/name).exists():
            shutil.copy2(full/name,export/name)
    if workload=='BFCLv3':
        for name in ('scores','responses'):
            shutil.copytree(full/name,export/name,dirs_exist_ok=True)
    files={str(p.relative_to(export)):sha(p) for p in export.rglob('*') if p.is_file() and p.name!='record.json'}
    record=dict(schema_version=1,status='complete',cell_id=cell['id'],model=cell['model'],configuration=cell['configuration'],workload=workload,
        protocol_fingerprint=fingerprint(cell),runtime_fingerprint=server.runtime_id,
        score=summary['results'][task['task']][task['metric']],weight_storage_bytes=server.audit['weight_storage_bytes'],
        elapsed_s=summary.get('elapsed_s',0),hardware=server.gpu.to_dict(),origin='new evaluation',files=files)
    write(export/'record.json',record)
    validate_record(export)
    return export


def worker(args, gpu, python, bfcl_python, stopped, smoke_cells=None):
    registry=Registry(args.state_dir)
    work=Path(args.work_dir).resolve()
    worker_id=socket.gethostname()+'-'+str(os.getpid())+'-'+gpu.uuid
    status_path=registry.root/'workers'/(worker_id+'.json')
    server=None; gpu_lock=None; attempted={}; prepared=set(); smoke_done=set()
    last=None
    try:
        gpu_lock=GPULock(gpu)
        # Recheck after locking, so two local launchers cannot both adopt an idle device.
        current=next(g for g in discover() if g.uuid==gpu.uuid)
        if current.busy:
            return
        while not stopped.is_set():
            pending=smoke_cells if smoke_cells is not None else registry.pending(args.models,args.configs,args.workloads)
            eligible=[c for c in pending if fits(gpu,c) and attempted.get(c['id'],0)<2 and c['id'] not in smoke_done]
            for cell in pending:
                if not fits(gpu,cell):
                    registry.issue(cell['id'],'insufficient_gpu_memory',f'{gpu.name}: this worker cannot fit weights/cache; CPU offload is disabled. Another capable worker may run this cell.')
            if not eligible:
                break
            # Keep the current model resident for the other workloads when possible.
            eligible.sort(key=lambda c:(attempted.get(c['id'],0), 0 if last==(c['model'],c['configuration']) else 1,
                                        0 if checkpoint(work,c['model'],False).exists() else 1))
            cell=eligible[0]
            claim=None if smoke_cells is not None else registry.claim(cell)
            if smoke_cells is None and claim is None:
                attempted[cell['id']]=2
                continue
            write(status_path,dict(status='running',cell=cell,hardware=gpu.to_dict(),updated_at=now()))
            try:
                key=(cell['model'],cell['configuration'])
                if key not in prepared:
                    monitored([python,'-m','qbench.prepare','--work-dir',work,'--model',cell['model'],'--config',cell['configuration']],
                        work/'logs'/('prepare-'+cell['model']+'-'+cell['configuration']+'.log'),environment(python,work),stopped,claim)
                    prepared.add(key)
                desired=(cell['model'],cell['configuration'],manifest()['workloads'][cell['workload']]['context_capacity'])
                if server and (server.identity!=desired or server.process.poll() is not None):
                    server.close(); server=None
                failure=None
                for profile in PROFILES:
                    try:
                        if server is None:
                            occupied=next(g for g in discover() if g.uuid==gpu.uuid)
                            if occupied.busy:
                                raise RuntimeError('Selected GPU acquired an unrelated compute process; refusing to launch')
                            server=Server(python,work,gpu,cell,profile,stopped)
                        exported=evaluate(server,cell,python,bfcl_python,stopped,claim,smoke=smoke_cells is not None)
                        if exported:
                            registry.publish(exported,claim)
                        else:
                            smoke_done.add(cell['id'])
                        failure=None
                        break
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        failure=exc
                        log=(server.logfile if server else None)
                        content=log.read_text(errors='replace')[-12000:] if log and log.exists() else str(exc)
                        if server:
                            server.close(); server=None
                        # Startup attempts also log CUDA OOM through the referenced server log.
                        latest=sorted((work/'logs').glob(cell['id']+'*server.log'),key=lambda p:p.stat().st_mtime)
                        if latest:
                            content+='\n'+latest[-1].read_text(errors='replace')[-12000:]
                        if not any(x in content.lower() for x in ('out of memory','insufficient memory','no available memory','larger than the maximum number of tokens','kv cache memory')):
                            break
                if failure:
                    raise failure
                last=key
                build(registry)
                print(f"Completed {'smoke ' if smoke_cells is not None else ''}{cell['id']}",flush=True)
            except InterruptedError:
                break
            except Exception as exc:
                attempted[cell['id']]=attempted.get(cell['id'],0)+1
                registry.issue(cell['id'],'attempt_failed',repr(exc))
                print(f"Failed {cell['id']}: {exc}",flush=True)
            finally:
                if claim:
                    claim.release()
        write(status_path,dict(status='idle',hardware=gpu.to_dict(),updated_at=now()))
    finally:
        if server:
            server.close()
        if gpu_lock:
            gpu_lock.close()
        status_path.unlink(missing_ok=True)
        build(registry)
    return len(smoke_done) if smoke_cells is not None else None
