"""Install only into project-owned environments; never modify the host driver."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import fcntl
import urllib.request
import zipfile
import signal
import time
from functools import partial

from .common import PROJECT, digest, manifest, read, sha, write


def run(command, stopped=None, **kwargs):
    print('+ ' + ' '.join(map(str, command)), flush=True)
    if stopped and stopped.is_set():
        raise InterruptedError('Installation interrupted')
    process=subprocess.Popen(list(map(str,command)),start_new_session=True,**kwargs)
    try:
        while process.poll() is None:
            if stopped and stopped.wait(.5):
                raise InterruptedError('Installation interrupted; rerun to resume')
            if not stopped:
                time.sleep(.5)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode,command)
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL); process.wait()


def cuda_layout(python):
    """Expose the standard CUDA toolkit linker layout inside this environment."""
    cuda=Path(python).parent.parent/'lib/python3.12/site-packages/nvidia/cu13'
    if not (cuda/'lib').is_dir():
        raise RuntimeError('Pinned CUDA toolkit libraries are missing')
    if not (cuda/'lib64').exists():
        (cuda/'lib64').symlink_to('lib',target_is_directory=True)
    for name in ('cudart','nvrtc'):
        link=cuda/'lib'/('lib'+name+'.so')
        if not link.exists():
            matches=sorted((cuda/'lib').glob('lib'+name+'.so.*'))
            if not matches:
                raise RuntimeError('CUDA linker library missing: '+name)
            link.symlink_to(matches[0].name)


def install(root=None, stopped=None):
    run=partial(globals()['run'],stopped=stopped)
    root = Path(root or PROJECT / '.runtime').resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.install.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    spec = manifest()
    identity = digest(dict(locks={p.name:sha(p) for p in sorted((PROJECT/'locks').iterdir()) if p.is_file()},
                           sources=spec['sources'], bootstrap=sha(__file__), python='3.12.15'))
    ready = root / 'ready.json'
    python, bfcl = root/'venv/bin/python', root/'bfcl-venv/bin/python'
    if ready.exists() and read(ready).get('identity') == identity and python.exists() and bfcl.exists():
        lock.close()
        return python, bfcl
    tools = root/'tools'
    tools.mkdir(exist_ok=True)
    uv_binary = tools/'uv'
    uv_spec = read(PROJECT/'locks/bootstrap.json')
    if not uv_binary.exists():
        wheel = tools/'uv.whl'
        urllib.request.urlretrieve(uv_spec['url'],wheel)
        if sha(wheel)!=uv_spec['sha256']:
            raise RuntimeError('uv bootstrap wheel checksum mismatch')
        with zipfile.ZipFile(wheel) as archive:
            name=next(n for n in archive.namelist() if n.endswith('/scripts/uv'))
            with archive.open(name) as src,uv_binary.open('wb') as dst:
                shutil.copyfileobj(src,dst)
        uv_binary.chmod(0o755)
        wheel.unlink()
    uv = [uv_binary]
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(root/'python'), UV_CACHE_DIR=str(root/'uv-cache'))
    run(uv+['python', 'install', '3.12.15'], env=env)
    if not python.exists():
        run(uv+['venv', '--python', '3.12.15', root/'venv'], env=env)
    sources = root/'sources'
    sources.mkdir(exist_ok=True)
    for name, source in spec['sources'].items():
        directory = sources/name
        if not (directory/'.git').exists():
            directory.mkdir(exist_ok=True)
            run(['git', 'init', directory])
            run(['git', '-C', directory, 'remote', 'add', 'origin', source['url']])
            run(['git', '-C', directory, 'fetch', '--depth', '1', 'origin', source['revision']])
            run(['git', '-C', directory, 'checkout', '--detach', 'FETCH_HEAD'])
        revision = subprocess.check_output(['git','-C',str(directory),'rev-parse','HEAD'], text=True).strip()
        if revision != source['revision']:
            raise RuntimeError('Source revision mismatch: '+name)
        if name == 'lm_eval':
            patch = PROJECT/'patches/lm-eval.patch'
            if sha(patch) != source['patch_sha256']:
                raise RuntimeError('lm-eval patch checksum mismatch')
            already = subprocess.run(['git','-C',str(directory),'apply','--reverse','--check',str(patch)], capture_output=True)
            if already.returncode:
                run(['git','-C',directory,'apply','--check',patch])
                run(['git','-C',directory,'apply',patch])
    run(uv+['pip','install','--python',python,'--constraint',PROJECT/'locks/runtime-constraints.txt',
            '-r',PROJECT/'locks/runtime-requirements.txt','-e',str(sources/'lm_eval')+'[api,ruler,ifeval]','-e',PROJECT], env=env)
    cuda_layout(python)
    if not bfcl.exists():
        run(uv+['venv','--python','3.12.15',root/'bfcl-venv'], env=env)
    run(uv+['pip','install','--python',bfcl,'--constraint',PROJECT/'locks/bfcl-constraints.txt',
            '-e',sources/'bfcl/berkeley-function-call-leaderboard'], env=env)
    # Match the original BFCL SDK isolation without duplicating the CUDA stack.
    runtime_site = root/'venv/lib/python3.12/site-packages'
    bfcl_site = root/'bfcl-venv/lib/python3.12/site-packages'
    (bfcl_site/'engine_environment.pth').write_text(str(runtime_site)+'\n')
    run([python,'-c',"import vllm, torch, transformers, lm_eval; assert vllm.__version__ == '0.32.0'; print(torch.__version__, transformers.__version__)"], env=env)
    run([bfcl,'-c',"from bfcl_eval.constants.category_mapping import VERSION_PREFIX; assert VERSION_PREFIX == 'BFCL_v3'"], env=env)
    run(uv+['pip','freeze','--python',python], env=env, stdout=(root/'installed-runtime.txt').open('w'))
    run(uv+['pip','freeze','--python',bfcl], env=env, stdout=(root/'installed-bfcl.txt').open('w'))
    write(ready, dict(identity=identity, python=str(python), bfcl_python=str(bfcl)))
    lock.close()
    return python, bfcl


def environment(python, work):
    env = dict(os.environ)
    site = Path(python).parent.parent/'lib/python3.12/site-packages'
    cuda = site/'nvidia/cu13'
    env.update(PYTHONPATH=str(PROJECT)+os.pathsep+str(PROJECT/'runtime'),
               QBENCH_WORK_DIR=str(work), HF_HOME=str(Path(work)/'cache/hf'),
               HF_DATASETS_CACHE=str(Path(work)/'cache/datasets'),
               NLTK_DATA=str(PROJECT/'assets/nltk'),
               VLLM_CACHE_ROOT=str(Path(work)/'cache/vllm'), TRITON_CACHE_DIR=str(Path(work)/'cache/triton'),
               FLASHINFER_WORKSPACE_BASE=str(Path(work)/'cache/flashinfer'),
               HF_HUB_DISABLE_TELEMETRY='1', VLLM_NO_USAGE_STATS='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8')
    if cuda.exists():
        env.update(CUDA_HOME=str(cuda), PATH=str(cuda/'bin')+os.pathsep+str(Path(python).parent)+os.pathsep+env.get('PATH',''),
                   LD_LIBRARY_PATH=str(cuda/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH',''))
    return env
