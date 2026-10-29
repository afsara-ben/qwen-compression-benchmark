"""Use official HumanEval tests and pass@k estimator in isolated processes."""
import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from quality_server import ROOT


def check(program):
    root=ROOT/'cache/code-sandboxes'; root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        env={'HOME':tmp,'TMPDIR':tmp,'LANG':'C.UTF-8','OPENBLAS_NUM_THREADS':'1'}
        try:
            result=subprocess.run([sys.executable,'-I','-S',str(Path(__file__).with_name('quality_code_worker.py'))],
                input=json.dumps({'program':program}),text=True,stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,cwd=tmp,env=env,timeout=5)
        except subprocess.TimeoutExpired:
            return False,'timed out'
        if result.returncode==120:
            raise RuntimeError('HumanEval execution isolation failed; no score emitted')
        return result.returncode==0, 'passed' if result.returncode==0 else 'failed'


class IsolatedCodeEval:
    def compute(self, references, predictions, k=None, **kwargs):
        pairs=[(i,j,candidate+'\n'+references[i]) for i,candidates in enumerate(predictions)
               for j,candidate in enumerate(candidates)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            verdicts=list(pool.map(check,[p[2] for p in pairs]))
        results={i:[] for i in range(len(predictions))}
        for (i,j,_),(passed,reason) in zip(pairs,verdicts):
            results[i].append((j,dict(task_id=i,completion_id=j,passed=passed,result=reason)))
        if os.environ.get('QUALITY_CODE_VERDICT_PATH'):
            with open(os.environ['QUALITY_CODE_VERDICT_PATH'],'a') as f:
                for (i,j,program),(passed,reason) in zip(pairs,verdicts):
                    f.write(json.dumps({'task_index':i,'completion_index':j,'passed':passed,
                        'result':reason,'problem_count':len(references),
                        'program_sha256':hashlib.sha256(program.encode()).hexdigest()})+'\n')
        scores={}
        for kk in (k or [1]):
            if any(len(candidates)<kk for candidates in predictions): continue
            values=[]
            for i,candidates in enumerate(predictions):
                n=len(candidates); c=sum(item[1]['passed'] for item in results[i])
                values.append(1.0 if n-c<kk else 1-math.comb(n-c,kk)/math.comb(n,kk))
            scores[f'pass@{kk}']=sum(values)/len(values)
        return scores,results


def install():
    import evaluate
    original=evaluate.load
    def load(path,*args,**kwargs):
        return IsolatedCodeEval() if path=='code_eval' else original(path,*args,**kwargs)
    evaluate.load=load


if __name__=='__main__':
    cases=[('assert 2+2==4',True),('assert False',False),
           ("open('/etc/passwd').read()",False),
           ("import socket; socket.socket()",False),
           ('while True: pass',False),
           ('import math; assert math.sqrt(9)==3',True)]
    for program,expected in cases:
        got=check(program); print(program,got,flush=True); assert got[0]==expected
