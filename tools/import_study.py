"""Build the portable, checksummed evidence bundle from the prior local study.

This maintainer tool is not needed to run the distributed project.
"""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qbench.common import PROJECT, cell_id, digest, fingerprint, manifest, read, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-root', required=True, type=Path)
    args = parser.parse_args()
    study = args.study_root
    spec = manifest()
    bank_index = {m:{} for m in spec['models']}
    for source in sorted((study/'cache/ruler-banks').iterdir()):
        if not (source/'provenance.json').is_file():
            continue
        provenance = read(source/'provenance.json')
        tokenizer = provenance['metadata']['tokenizer']
        model = 'qwen3_6_35b' if 'qwen36' in tokenizer else 'qwen3_30b'
        task = provenance['task']
        target = PROJECT/'assets/ruler-banks'/model/task
        shutil.copytree(source, target, dirs_exist_ok=True)
        files = {str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
        bank_index[model][task] = dict(directory=f'{model}/{task}', files=files, digest=digest(files),
                                      recipe=provenance['recipe'])
    assert all(len(v)==13 for v in bank_index.values())
    write(PROJECT/'assets/ruler-bank-index.json', bank_index)
    spec['ruler_bank_digests'] = {m:{t:v['digest'] for t,v in tasks.items()} for m,tasks in bank_index.items()}
    write(PROJECT/'config/experiment.json', spec)
    verified_baselines = []
    selections = []
    for family, model in enumerate(spec['models']):
        selections += [(model, 'KV', f'gpu{family}-missing-kv-01', w) for w in spec['workloads']]
        reap = 'gpu0-missing-e-kv-01' if family==0 else 'gpu1-missing-e-kv-03'
        selections += [(model, 'E+KV', reap, w) for w in spec['workloads']]
        selections += [(model, 'E+W+KV', f'gpu{family}-ruler-e-w-kv-nvfp4-01', 'RULER')]
        baseline = 'gpu0-baseline-02' if family==0 else 'gpu1-baseline-03'
        for workload in ('GSM8K','IFEval'):
            path = study/'results'/baseline/workload/'full'
            reference = study/'results'/f'gpu{family}-missing-kv-01'/workload/'full'
            a, b = read(path/'summary.json'), read(reference/'summary.json')
            def task_config(d):
                return {k:{x:y for x,y in v.items() if x!='metadata'} for k,v in d['configs'].items()}
            assert task_config(a)==task_config(b)
            assert a['study_protocol']['seed']==b['study_protocol']['seed']==17
            sa, sb = json.load(gzip.open(path/'samples.json.gz','rt')), json.load(gzip.open(reference/'samples.json.gz','rt'))
            assert sa.keys()==sb.keys()
            for task in sa:
                assert len(sa[task])==len(sb[task])
                assert all(all(x.get(k)==y.get(k) for k in ('doc_id','doc','target','arguments')) for x,y in zip(sa[task],sb[task]))
            verified_baselines.append(dict(model=model, workload=workload, baseline_summary_sha256=sha(path/'summary.json'),
                matched_reference_summary_sha256=sha(reference/'summary.json'),
                baseline_samples_sha256=sha(path/'samples.json.gz'), reference_samples_sha256=sha(reference/'samples.json.gz'),
                checks=['identical task config excluding filesystem metadata','identical doc IDs, full documents, targets and rendered generation arguments','seed 17 and full sample counts']))
            selections.append((model, 'Baseline', baseline, workload))
    write(PROJECT/'assets/baseline-validation.json', verified_baselines)
    for model, config, attempt, workload in selections:
        source = study/'results'/attempt
        full = source/workload/'full'
        cell = dict(id=cell_id(model,config,workload), model=model, configuration=config, workload=workload)
        target = PROJECT/'assets/completed'/cell['id']
        target.mkdir(parents=True, exist_ok=True)
        summary = read(full/'summary.json')
        shutil.copy2(full/'summary.json', target/'original-summary.json')
        if config=='Baseline':
            summary['study_protocol'].update(dataset_revisions=spec['dataset_revisions'],
                context_policy='benchmark_native_prompts_and_output_limits',
                compatibility_validation='Full saved inputs/targets/generation arguments match pinned reference; see baseline-validation.json')
            shutil.copy2(PROJECT/'assets/baseline-validation.json', target/'baseline-validation.json')
        write(target/'summary.json', summary)
        shutil.copy2(source/'audit.json', target/'audit.json')
        launch = read(source/'launch.json')
        launch['benchmark_identity'] = dict(model=model, configuration=config,
            checkpoint=spec['models'][model]['reap' if spec['configurations'][config]['expert'] else 'base'],
            tokenizer=spec['models'][model]['base'])
        write(target/'launch.json', launch)
        if workload=='RULER':
            import ijson
            with gzip.open(full/'samples.json.gz','rb') as stream, gzip.open(target/'samples.jsonl.gz','wt') as out:
                for task, examples in ijson.kvitems(stream,''):
                    for example in examples:
                        value = {k:v for k,v in example.items() if k not in ('doc','arguments')}
                        value.update(task=task, bank=f'{model}/{task}', source_sample_archive_sha256=None)
                        out.write(json.dumps(value, default=float)+'\n')
            write(target/'sample-evidence.json', dict(original_archive_sha256=sha(full/'samples.json.gz'),
                reconstruction='Full documents are in frozen RULER banks indexed by task and doc_id; prompt hashes and outputs are retained.'))
        elif (full/'samples.json.gz').exists():
            shutil.copy2(full/'samples.json.gz', target/'samples.json.gz')
        if workload=='BFCLv3':
            for name in ('scores','responses'):
                for source_file in (full/name).rglob('*'):
                    if source_file.is_file():
                        dest=target/name/source_file.relative_to(full/name)
                        dest=dest.with_name(dest.name+'.gz')
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        with source_file.open('rb') as src,gzip.open(dest,'wb') as out:
                            shutil.copyfileobj(src,out)
        if (full/'code-verdicts.jsonl').exists():
            shutil.copy2(full/'code-verdicts.jsonl',target/'code-verdicts.jsonl')
        audit = read(source/'audit.json')['results'][0]
        task = spec['workloads'][workload]
        score = summary['results'][task['task']][task['metric']]
        files = {str(f.relative_to(target)):sha(f) for f in sorted(target.rglob('*')) if f.is_file() and f.name!='record.json'}
        record = dict(schema_version=1, status='complete', cell_id=cell['id'], model=model, configuration=config, workload=workload,
            protocol_fingerprint=fingerprint(cell), runtime_fingerprint=digest({'launch':launch,'audit':audit}),
            score=score, weight_storage_bytes=audit['weight_storage_bytes'], elapsed_s=summary.get('elapsed_s',0),
            hardware=dict(name='NVIDIA RTX PRO 6000 Blackwell Server Edition', memory_mib=97887, gpu_uuid=audit['gpu']),
            origin='imported validated study', files=files)
        write(target/'record.json',record)
        print('Imported',cell['id'],flush=True)
    from qbench.registry import validate_record
    for p in (PROJECT/'assets/completed').iterdir():
        validate_record(p)
    write(PROJECT/'assets/manifest.json', {'schema_version':1, 'completed_scores':len(selections),
        'files':{str(f.relative_to(PROJECT/'assets')):sha(f) for f in sorted((PROJECT/'assets').rglob('*')) if f.is_file() and f!=PROJECT/'assets/manifest.json'}})
    print('Validated seed results:',len(selections))


if __name__ == '__main__':
    main()
