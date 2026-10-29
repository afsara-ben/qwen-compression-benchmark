#!/usr/bin/env bash
# Two original BF16 models, RULER only, sequentially on one H100 80 GB.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: bash run_bf16_ruler.sh [both|qwen3|qwen36]
Default: both models, sequentially on one GPU. RULER only, 32,768 total tokens.
RULER_WORK_DIR=/local-ssd/ruler selects the cache/results directory (default: work/ruler-bf16).
CUDA_VISIBLE_DEVICES selects the assigned GPU; otherwise the first visible GPU is used.
Rerun the same command to resume. Results: $RULER_WORK_DIR/results.md (updated every 30s).
Requires Linux, Python 3.10+ for setup, Git, build tools, and an H100 80 GB with a
CUDA 13-compatible driver (580.126.20+). Python 3.12 and runtime packages install automatically.
HELP
  exit 0
fi
case "${1:-both}" in both|qwen3|qwen36) ;; *) echo "Use both, qwen3, or qwen36" >&2; exit 2 ;; esac
export RULER_SELECTION="${1:-both}"
export RULER_WORK_DIR="${RULER_WORK_DIR:-$PWD/work/ruler-bf16}"
mkdir -p "$RULER_WORK_DIR"
# One launcher per output directory; the lock is released even after interruption.
exec 9>"$RULER_WORK_DIR/.run.lock"
flock -n 9 || { echo "A RULER run already owns this output directory." >&2; exit 2; }

# Install just the inference/lm-eval environment. No BFCL, conversions, or campaign registry.
python3 - <<'SETUP'
import hashlib, json, os, shutil, subprocess, urllib.request, zipfile
from pathlib import Path
from qbench.bootstrap import cuda_layout
from qbench.common import PROJECT, manifest, sha

root = Path(os.environ['RULER_WORK_DIR']).resolve()
runtime = root / 'runtime'
runtime.mkdir(parents=True, exist_ok=True)
spec = manifest()['sources']['lm_eval']
identity = hashlib.sha256((sha(PROJECT/'locks/runtime-constraints.txt') + spec['revision'] +
                          sha(PROJECT/'patches/lm-eval.patch') + 'bf16-ruler-env-v1').encode()).hexdigest()
ready = runtime/'ready.txt'
python = runtime/'venv/bin/python'
if not (python.exists() and ready.exists() and ready.read_text() == identity):
    uv = runtime/'uv'
    if not uv.exists():
        pin = json.loads((PROJECT/'locks/bootstrap.json').read_text())
        wheel = runtime/'uv.whl'
        urllib.request.urlretrieve(pin['url'], wheel)
        if sha(wheel) != pin['sha256']:
            raise RuntimeError('Installer checksum mismatch')
        with zipfile.ZipFile(wheel) as archive:
            name = next(n for n in archive.namelist() if n.endswith('/scripts/uv'))
            with archive.open(name) as src, uv.open('wb') as dst:
                shutil.copyfileobj(src, dst)
        uv.chmod(0o755)
        wheel.unlink()
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(runtime/'python'), UV_CACHE_DIR=str(runtime/'uv-cache'))
    def run(*args):
        subprocess.run(list(map(str, args)), env=env, check=True)
    run(uv, 'python', 'install', '3.12.15')
    if not python.exists():
        run(uv, 'venv', '--python', '3.12.15', runtime/'venv')
    source = runtime/'lm-eval'
    if not (source/'.git').exists():
        run('git', 'init', source)
        run('git', '-C', source, 'remote', 'add', 'origin', spec['url'])
    if subprocess.run(['git', '-C', str(source), 'rev-parse', '--verify', 'HEAD'], capture_output=True).returncode:
        run('git', '-C', source, 'fetch', '--depth', '1', 'origin', spec['revision'])
        run('git', '-C', source, 'checkout', '--detach', 'FETCH_HEAD')
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != spec['revision']:
        raise RuntimeError('lm-eval revision mismatch')
    patch = PROJECT/'patches/lm-eval.patch'
    if sha(patch) != spec['patch_sha256']:
        raise RuntimeError('lm-eval patch checksum mismatch')
    if subprocess.run(['git', '-C', str(source), 'apply', '--reverse', '--check', str(patch)], capture_output=True).returncode:
        run('git', '-C', source, 'apply', patch)
    run(uv, 'pip', 'install', '--python', python, '-c', PROJECT/'locks/runtime-constraints.txt',
        'vllm==0.32.0', 'torch==2.13.0', 'transformers==5.19.0', 'flashinfer-python==0.7.0.post1',
        'cuda-toolkit[nvcc]==13.0.3.0', 'tenacity', '-e', str(source)+'[api,ruler]')
    cuda_layout(python)
    ready.write_text(identity)
print('RULER environment ready.', flush=True)
SETUP

exec "$RULER_WORK_DIR/runtime/venv/bin/python" - <<'RUN'
import hashlib, json, os, signal, socket, sqlite3, subprocess, sys, threading, time
from pathlib import Path
from huggingface_hub import snapshot_download
from qbench.assets import ensure_assets
from qbench.bootstrap import environment
from qbench.common import PROJECT, atomic_text, manifest, now, read, sha, write
from qbench.runner import monitored, request, server_command, stop_process

work = Path(os.environ['RULER_WORK_DIR']).resolve()
spec = manifest()
models = list(spec['models'])
selection = os.environ['RULER_SELECTION']
selected = models if selection == 'both' else [models[0 if selection == 'qwen3' else 1]]
stopped = threading.Event()
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda *_: stopped.set())
(work/'state').mkdir(exist_ok=True)
write(work/'state/dataset-revisions.json', {})
env = environment(sys.executable, work)
env.update(VLLM_PLUGINS='', HF_HUB_OFFLINE='0', VLLM_SERVER_DEV_MODE='0')
for key in ('MOECBENCH_MARLIN_CANONICAL_ORDER', 'VLLM_MARLIN_INPUT_DTYPE'):
    env.pop(key, None)
# TP=1 uses the first assigned device; never expand CUDA_VISIBLE_DEVICES.
os.environ.update({k: v for k, v in env.items() if k in ('HF_HOME', 'HF_HUB_OFFLINE')})
try:
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'], text=True).strip()
except (OSError, subprocess.CalledProcessError):
    raise SystemExit('An NVIDIA GPU and nvidia-smi are required.')
profile = (1, 2048, .95, True)  # One sequence, eager execution, no CPU offload.
states = {m: {'status': 'pending', 'score': None, 'responses': 0} for m in models}
guard = threading.Lock()

def update(model, **values):
    with guard:
        states[model].update(values)

def report():
    with guard:
        rows = {m: dict(v) for m, v in states.items()}
    for model, row in rows.items():
        if row.get('directory'):
            for db in Path(row['directory']).glob('responses*.db'):
                try:
                    with sqlite3.connect(db.as_uri()+'?mode=ro', uri=True, timeout=1) as conn:
                        row['responses'] = conn.execute('SELECT COUNT(*) FROM unnamed').fetchone()[0]
                except sqlite3.Error:
                    pass
    lines = ['# BF16 RULER results', '', 'Updated: '+now(), '',
             'BF16 weights and attention KV cache; 32,768 total tokens; seed 17; non-thinking greedy decoding.',
             '13 subtasks × 500 examples per model. Scores appear only after all 6,500 examples are scored.', '',
             '| Model | Status | Cached responses / 6,500 | RULER |', '|---|---|---:|---:|']
    for model, row in rows.items():
        score = '—' if row['score'] is None else f"{row['score']:.4f}"
        lines.append(f"| {spec['models'][model]['label']} | {row['status']} | {row['responses']} | {score} |")
    for model, row in rows.items():
        if row.get('error'):
            lines += ['', spec['models'][model]['label']+': '+row['error']]
    lines += ['', 'Rerun the same command to resume. Raw summaries, samples and response caches are in `results/`.', '']
    atomic_text(work/'results.md', '\n'.join(lines))
    write(work/'results.json', dict(updated_at=now(), models=rows, hardware=hardware))

def ticker():
    while not stopped.wait(30):
        try:
            report()
        except Exception as exc:
            print('Report update will retry:', exc, file=sys.stderr, flush=True)

report()
thread = threading.Thread(target=ticker, daemon=True)
thread.start()
failures = False
try:
    ensure_assets()  # Restore the same frozen RULER inputs used in the earlier runs.
    for model in selected:
        if stopped.is_set():
            break
        base = spec['models'][model]['base']
        identity = dict(checkpoint=base, profile=profile, hardware=hardware,
                        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                        protocol=spec['workloads']['RULER'], banks=spec['ruler_bank_digests'][model],
                        evaluator=sha(PROJECT/'runtime/quality_eval.py'),
                        launcher=sha(PROJECT/'run_bf16_ruler.sh'))
        attempt = 'bf16-ruler-'+model+'-'+hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        directory = work/'results'/attempt/'RULER/full'
        directory.mkdir(parents=True, exist_ok=True)
        update(model, directory=str(directory))
        server = None
        try:
            if not (directory/'summary.json').exists():
                update(model, status='downloading checkpoint')
                report()
                path = snapshot_download(base['repo'], revision=base['revision'], cache_dir=work/'cache/hf/hub',
                                         ignore_patterns=['*.gguf', '*.bin', '*.pt', '*.pth'])
                if stopped.is_set():
                    raise InterruptedError('Stopped; rerun to resume')
                for entry in read(PROJECT/'assets/ruler-bank-index.json')[model].values():
                    for name, expected in entry['files'].items():
                        if sha(PROJECT/'assets/ruler-banks'/entry['directory']/name) != expected:
                            raise RuntimeError('Frozen RULER bank checksum mismatch')
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    port = sock.getsockname()[1]
                endpoint = f'http://127.0.0.1:{port}'
                cell = dict(model=model, configuration='Baseline', workload='RULER')
                command = server_command(sys.executable, path, path, cell, endpoint, profile)
                # The simple run needs no worker audit extension or startup smoke evaluation.
                index = command.index('--worker-extension-cls')
                del command[index:index+2]
                child_env = dict(env, QBENCH_ENDPOINT=endpoint, QBENCH_MODEL_ID=model,
                                 QBENCH_MODEL_PATH=path, QBENCH_TOKENIZER_PATH=path, QBENCH_CONCURRENCY='1')
                write(directory/'launch.json', dict(identity=identity, command=command, started_at=now()))
                logs = work/'logs'
                logs.mkdir(exist_ok=True)
                update(model, status='loading model')
                report()
                with (logs/(model+'-server.log')).open('a') as log:
                    server = subprocess.Popen(command, env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic()+3600
                while True:
                    if stopped.wait(2):
                        raise InterruptedError('Stopped; rerun to resume')
                    if server.poll() is not None:
                        raise RuntimeError('Server exited; see logs/'+model+'-server.log')
                    try:
                        request(endpoint, 'health', timeout=2)
                        break
                    except Exception:
                        if time.monotonic() > deadline:
                            raise TimeoutError('Server startup exceeded one hour')
                update(model, status='evaluating')
                report()
                monitored([sys.executable, PROJECT/'runtime/quality_eval.py', '--gpu', str(spec['models'][model]['family']),
                           '--attempt', attempt, '--workload', 'RULER'], logs/(model+'-eval.log'),
                          dict(child_env, CUDA_VISIBLE_DEVICES=''), stopped)
            result = read(directory/'summary.json')
            if result.get('sample_counts_saved') != spec['workloads']['RULER']['sample_counts']:
                raise RuntimeError('Incomplete RULER sample counts')
            score = result['study_protocol']['ruler_component_mean']
            if not 0 <= score <= 1:
                raise RuntimeError('Invalid RULER score')
            update(model, status='complete', score=score, responses=6500)
            print(model, 'RULER:', f'{score:.4f}', flush=True)
        except InterruptedError as exc:
            update(model, status='interrupted', error=str(exc))
            break
        except Exception as exc:
            failures = True
            update(model, status='failed; rerun to resume', error=str(exc))
            print(model+': '+str(exc), file=sys.stderr, flush=True)
        finally:
            stop_process(server)  # Release the first model before loading the second.
            report()
finally:
    interrupted = stopped.is_set()
    stopped.set()
    thread.join(timeout=2)
    report()
print('Saved:', work/'results.md', flush=True)
sys.exit(130 if interrupted else 1 if failures else 0)
RUN
