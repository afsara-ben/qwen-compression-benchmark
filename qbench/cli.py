from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import threading
import time

from .common import PROJECT, all_cells, digest, manifest, now, read, write
from .registry import Registry
from .report import build


def parse(argv=None):
    p=argparse.ArgumentParser(description='Eight Qwen compression configurations; global RULER-first evaluation, automatic idle GPU selection.')
    p.add_argument('--work-dir',type=Path,default=PROJECT/'work'/socket.gethostname(),help='Node-local models, environments, response caches and logs (default: work/HOSTNAME)')
    p.add_argument('--state-dir',type=Path,help='Shared result registry, or node-local state when isolated')
    p.add_argument('--runtime-dir',type=Path,help='Default: WORK_DIR/environment')
    p.add_argument('--models',nargs='+',choices=list(manifest()['models']))
    p.add_argument('--configs',nargs='+',choices=list(manifest()['configurations']))
    p.add_argument('--workloads',nargs='+',choices=list(manifest()['workloads']))
    p.add_argument('--gpus',help='Comma-separated visible ordinal indices or GPU UUIDs; default: all idle eligible GPUs')
    modes=p.add_mutually_exclusive_group()
    for flag,helptext in [('check','Inspect prerequisites and pending work without installing/downloading'),
                          ('prepare-only','Install dependencies and prepare selected model artifacts; no evaluation'),
                          ('smoke','Validate selected model configurations with one RULER example per subtask; do not score the grid'),
                          ('report-only','Regenerate reports without GPU or heavyweight dependencies')]:
        modes.add_argument('--'+flag,action='store_true',help=helptext)
    p.add_argument('--fresh',action='store_true',help='Create an independent campaign, without importing the bundled completed results')
    p.add_argument('--import-results',type=Path,action='append',default=[])
    p.add_argument('--export-results',type=Path)
    p.add_argument('--run',action='store_true',help='Continue evaluation after a result import/export action')
    p.add_argument('--python',type=Path,help='Advanced: use an already prepared compatible runtime instead of installing')
    p.add_argument('--bfcl-python',type=Path,help='Advanced: compatible isolated BFCL interpreter')
    args=p.parse_args(argv)
    args.work_dir=args.work_dir.resolve()
    args.state_dir=(args.state_dir or args.work_dir/'state').resolve()
    if args.fresh:
        import uuid
        tag='fresh-'+now().replace(':','').replace('+','_')+'-'+uuid.uuid4().hex[:6]
        args.work_dir=args.work_dir/tag
        args.state_dir=args.state_dir/tag
    args.runtime_dir=(args.runtime_dir or args.work_dir/'environment').resolve()
    return args


def host_checks(work):
    errors=[]
    workload_errors={}
    for program in ('git','curl','cc','nvidia-smi'):
        if not shutil.which(program):
            errors.append('Missing executable: '+program)
    if sys.platform!='linux' or os.uname().machine!='x86_64':
        errors.append('This pinned project supports Linux x86-64')
    try:
        ctypes.CDLL('libseccomp.so.2')
        if ctypes.CDLL(None,use_errno=True).syscall(444,0,0,1)<1:
            workload_errors['HumanEval']='Landlock unavailable: HumanEval isolation cannot run'
    except OSError:
        workload_errors['HumanEval']='libseccomp2 unavailable'
    free=shutil.disk_usage(work).free
    ram=None
    try:
        ram=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemTotal:')))*1024
    except (OSError,StopIteration):
        pass
    return dict(errors=errors,workload_errors=workload_errors,free_disk_GB=round(free/1e9,1),host_RAM_GB=round(ram/1e9,1) if ram else None,
                recommendations='128 GB host RAM and 500 GB free SSD for a full model cache; actual requirements depend on filters and existing downloads.')


def report_line(registry):
    r=build(registry)
    print(f"{r['completed']}/112 complete; RULER {r['ruler_completed']}/16; {r['remaining']} remaining; phase={r['phase']}",flush=True)
    print('Report:',registry.root/'results.md',flush=True)
    return r


def main(argv=None):
    args=parse(argv)
    from .assets import ensure_assets
    ensure_assets(download=not args.check)
    stopped=threading.Event()
    def stop(signum,frame):
        print('Saving progress and stopping owned processes...',flush=True)
        stopped.set()
    signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop)
    args.work_dir.mkdir(parents=True,exist_ok=True)
    registry=Registry(args.state_dir)
    campaign=registry.root/'campaign.json'
    contract=digest(manifest())
    if campaign.exists():
        if read(campaign)['experiment_digest']!=contract:
            raise ValueError('State directory belongs to a different experiment; choose a new state directory')
    else:
        write(campaign,dict(experiment_digest=contract,import_seed=not args.fresh,created_at=now()))
    from .bundles import import_seed,import_bundle,export_bundle,verify_banks
    if read(campaign)['import_seed']:
        import_seed(registry)
    for path in args.import_results:
        print('Imported cells:',import_bundle(registry,path),flush=True)
    if args.export_results:
        print('Exported:',export_bundle(registry,args.export_results),flush=True)
    status=report_line(registry)
    if args.fresh:
        print(f'Resume this campaign with: bash run.sh --work-dir {args.work_dir} --state-dir {args.state_dir}',flush=True)
    if args.report_only or ((args.import_results or args.export_results) and not args.run):
        return 0
    if status['completed']==112 and not args.smoke and not args.prepare_only and not args.check:
        return 0
    from .hardware import discover,fits
    checks=host_checks(args.work_dir)
    try:
        gpus=discover(args.gpus)
    except Exception as exc:
        gpus=[]; checks['errors'].append('GPU discovery failed: '+str(exc))
    for gpu in gpus:
        version=tuple(int(x) for x in gpu.driver.split('.'))
        if version<(580,126,20):
            checks['errors'].append(f'{gpu.name}: driver {gpu.driver}; pinned CUDA setup requires >=580.126.20')
    checks['gpus']=[g.to_dict() for g in gpus]
    checks['pending_jobs_that_fit']={g.uuid:sum(fits(g,c) for c in registry.pending(args.models,args.configs,args.workloads)) for g in gpus}
    write(registry.root/'hardware-check.json',checks)
    build(registry)
    print(f"Host RAM: {checks['host_RAM_GB']} GB; free work-disk space: {checks['free_disk_GB']} GB; assigned GPUs: {len(gpus)}",flush=True)
    if args.check:
        verify_banks()
        print(json.dumps(checks,indent=2))
        return 2 if checks['errors'] or checks['workload_errors'] else 0
    if checks['errors'] and not args.prepare_only:
        for error in checks['errors']:
            print(error,file=sys.stderr)
        return 2
    if not args.prepare_only:
        gpus=[g for g in gpus if not g.busy and g.capability>=8.0]
        if not gpus:
            print('No idle assigned compatible GPU is available. State is saved; rerun when one becomes available.',file=sys.stderr)
            return 3
        if not args.smoke and not registry.pending(args.models,args.configs,args.workloads):
            print('No selected work is eligible in the current phase. Import peer RULER results or broaden filters; the global RULER gate remains enforced.',flush=True)
            return 3
    verify_banks()
    if args.python:
        python=args.python.resolve(); bfcl=args.bfcl_python.resolve() if args.bfcl_python else python
    else:
        from .bootstrap import install
        try:
            python,bfcl=install(args.runtime_dir,stopped)
        except InterruptedError:
            report_line(registry)
            return 130
    from .bootstrap import environment
    from .runner import monitored,worker
    (args.work_dir/'state').mkdir(exist_ok=True)
    write(args.work_dir/'state/dataset-revisions.json',manifest()['dataset_revisions'])
    if args.prepare_only:
        seen=set(); failures=[]
        for model in args.models or manifest()['models']:
            for config in args.configs or manifest()['configurations']:
                c=manifest()['configurations'][config]; key=(model,c['expert'],c['weights'])
                if key in seen:
                    continue
                seen.add(key)
                try:
                    monitored([python,'-m','qbench.prepare','--work-dir',args.work_dir,'--model',model,'--config',config],
                        args.work_dir/'logs'/f'prepare-{model}-{config}.log',environment(python,args.work_dir),stopped)
                except InterruptedError:
                    return 130
                except Exception as exc:
                    failures.append(str(exc));print(exc,file=sys.stderr)
        return 1 if failures else 0
    smoke_cells=None
    if args.smoke:
        smoke_cells=[c for c in all_cells() if c['workload']=='RULER' and (not args.models or c['model'] in args.models)
                     and (not args.configs or c['configuration'] in args.configs)]
        if args.workloads and args.workloads!=['RULER']:
            raise ValueError('Smoke mode uses RULER boundary examples; other workload filters cannot bypass the phase gate')
    def ticker():
        while not stopped.wait(30):
            report_line(registry)
    ticker_thread=threading.Thread(target=ticker,daemon=True);ticker_thread.start()
    try:
        for phase_round in range(2):
            before=registry.phase()
            assignments=None
            if smoke_cells is not None:
                assignments=[[] for _ in gpus]
                for cell in smoke_cells:
                    compatible=[i for i,g in enumerate(gpus) if fits(g,cell)]
                    if compatible:
                        idx=min(compatible,key=lambda i:len(assignments[i]));assignments[idx].append(cell)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                futures=[pool.submit(worker,args,g,python,bfcl,stopped,assignments[i] if assignments is not None else None) for i,g in enumerate(gpus)]
                outcomes=[]
                for future in futures:
                    try:
                        outcomes.append(future.result())
                    except Exception as exc:
                        print('Worker failed:',exc,file=sys.stderr)
            if stopped.is_set():
                return 130
            if smoke_cells is not None:
                return 0 if sum(x or 0 for x in outcomes)==len(smoke_cells) else 1
            if registry.phase()=='complete':
                return 0
            # Wait without holding GPU resources if remote workers still have live leases.
            while registry.phase()==before=='RULER' and not stopped.is_set():
                leases=[p for p in (registry.root/'claims').glob('*/owner.json') if time.time()-p.stat().st_mtime<900]
                if not leases:
                    break
                stopped.wait(15)
            if registry.phase()==before:
                break
        print('Remaining work is saved. Check results.md for failures, memory limits, filters, or missing peer RULER results.',flush=True)
        return 3
    finally:
        stopped.set();ticker_thread.join(timeout=2)
        report_line(registry)


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,RuntimeError,OSError) as exc:
        print('Benchmark error:',exc,file=sys.stderr)
        raise SystemExit(2)
