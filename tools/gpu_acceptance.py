"""Exercise all selected real model/cache combinations after model preparation.

Writes smoke evidence only. Never publishes a workload score to the main grid.
"""
import argparse
import concurrent.futures
from pathlib import Path
import sys
import threading
import time
import signal

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qbench.common import all_cells, manifest, now, read, write
from qbench.hardware import GPULock, discover
from qbench.runner import Server, evaluate, PROFILES


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work-dir',required=True,type=Path)
    p.add_argument('--python',required=True,type=Path)
    p.add_argument('--bfcl-python',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--configs',nargs='+',default=['Baseline','KV','E','E+KV','E+W','E+W+KV','W','W+KV'])
    p.add_argument('--models',nargs='+',choices=list(manifest()['models']))
    p.add_argument('--resume',action='store_true',help='Retain passed smoke evidence from the same output file')
    p.add_argument('--boundary',action='store_true',help='Also evaluate one 32k example from each RULER subtask')
    args=p.parse_args()
    gpus=[g for g in discover() if not g.busy]
    if not gpus:
        raise RuntimeError('No idle assigned GPU is available')
    stopped=threading.Event();results=[];mutex=threading.Lock()
    signal.signal(signal.SIGINT,lambda *_:stopped.set())
    signal.signal(signal.SIGTERM,lambda *_:stopped.set())
    if args.resume and args.output.exists():
        results=[r for r in read(args.output)['results'] if r['status']=='passed'
                 and (not args.boundary or r.get('boundary')) and Path(r['evidence']).exists()]
    passed={r['cell_id'] for r in results}
    (args.work_dir/'state').mkdir(parents=True,exist_ok=True)
    write(args.work_dir/'state/dataset-revisions.json',manifest()['dataset_revisions'])
    def lane(model,gpu):
        lock=GPULock(gpu)
        try:
            for config in args.configs:
                cell=next(c for c in all_cells() if c['model']==model and c['configuration']==config and c['workload']=='RULER')
                if stopped.is_set():break
                if cell['id'] in passed:continue
                server=None;began=time.time()
                try:
                    server=Server(args.python,args.work_dir,gpu,cell,PROFILES[0],stopped)
                    if args.boundary:
                        evaluate(server,cell,args.python,args.bfcl_python,stopped,smoke=True)
                    row=dict(cell_id=cell['id'],status='passed',gpu=gpu.to_dict(),
                        weight_storage_bytes=server.audit['weight_storage_bytes'],elapsed_s=time.time()-began,
                        evidence=str(server.directory),boundary=args.boundary)
                except Exception as exc:
                    row=dict(cell_id=cell['id'],status='failed',error=repr(exc),elapsed_s=time.time()-began)
                finally:
                    if server:server.close()
                with mutex:
                    results.append(row);write(args.output,dict(updated_at=now(),results=results))
                print(row,flush=True)
        finally:lock.close()
    models=args.models or list(manifest()['models'])
    def gpu_lane(index,gpu):
        for model in models[index::len(gpus)]:lane(model,gpu)
    with concurrent.futures.ThreadPoolExecutor(len(gpus)) as pool:
        fs=[pool.submit(gpu_lane,i,gpu) for i,gpu in enumerate(gpus)]
        for f in fs:f.result()
    if stopped.is_set():return 130
    selected={c['id'] for c in all_cells() if c['workload']=='RULER' and c['model'] in models and c['configuration'] in args.configs}
    return 0 if selected<={r['cell_id'] for r in results if r['status']=='passed'} else 1


if __name__=='__main__':raise SystemExit(main())
