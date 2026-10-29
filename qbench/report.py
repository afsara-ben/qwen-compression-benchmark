from __future__ import annotations

import csv
import io
from .common import all_cells, atomic_text, manifest, now, read, write


def build(registry):
    spec, records = manifest(), registry.records()
    ruler_count = sum(x['workload'] == 'RULER' for x in records.values())
    issues=[]
    for path in sorted((registry.root/'issues').glob('*.json')):
        if path.stem in records:continue
        try:issues.append(read(path))
        except FileNotFoundError:pass  # A peer resolved this issue during reporting.
    rows = []
    for model, m in spec['models'].items():
        for config in spec['configurations']:
            row = dict(model=m['label'], configuration=config, weight_GB=None, reduction_percent=None)
            matched = [v for v in records.values() if v['model'] == model and v['configuration'] == config]
            # KV changes do not change weight storage; audited companion rows
            # are valid weight measurements, never manufactured quality scores.
            c = spec['configurations'][config]
            weights = [v['weight_storage_bytes'] for v in records.values() if v['model'] == model
                       and spec['configurations'][v['configuration']]['expert'] == c['expert']
                       and spec['configurations'][v['configuration']]['weights'] == c['weights']]
            if weights:
                row['weight_GB'] = weights[-1] / 1e9
                row['reduction_percent'] = 100 * (1 - weights[-1] / m['weight_bytes']['bf16_base'])
            for w in spec['workloads']:
                value = next((v for v in matched if v['workload'] == w), None)
                row[w] = value['score'] if value else None
            row['Avg.'] = sum(row[w] for w in spec['workloads']) / 7 if all(row[w] is not None for w in spec['workloads']) else None
            rows.append(row)
    status = dict(updated_at=now(), phase=registry.phase(), completed=len(records), total=112,
                  ruler_completed=ruler_count, ruler_total=16, remaining=112-len(records), rows=rows,
                  pending=[c for c in all_cells() if c['id'] not in records], issues=issues)
    write(registry.root / 'results.json', status)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    atomic_text(registry.root / 'results.csv', stream.getvalue())
    lines = ['# Qwen compression benchmark', '', f"Updated: {status['updated_at']}", '',
             f"**{len(records)}/112 workload runs complete; {112-len(records)} remaining. RULER: {ruler_count}/16. Phase: {status['phase']}.**", '',
             'E = REAP (Qwen3: 25%; Qwen3.6: 20% nominal). W = NVFP4 W4A16/BF16 activations. KV = FP8 attention cache.',
             'RULER uses 32,768 total tokens. Other workloads use native inputs. Weight sizes include scales and backend padding, exclude KV/cache/runtime buffers, and use decimal GB.',
             'A dash means unavailable, never zero. Companion KV rows share the same measured weights. Averages require all seven scores.', '',
             '| Model | Configuration | Weights GB (reduction) | ' + ' | '.join(spec['workloads']) + ' | Avg. |',
             '|---|---|---:' + '|---:' * 8 + '|']
    for row in rows:
        weight = f"{row['weight_GB']:.3f} ({row['reduction_percent']:.2f}%)" if row['weight_GB'] is not None else '—'
        values = ['—' if row[w] is None else f'{row[w]:.4f}' for w in [*spec['workloads'], 'Avg.']]
        lines.append('| ' + ' | '.join([row['model'], row['configuration'], weight, *values]) + ' |')
    lines += ['', '## Result provenance', '', '| Model / configuration / workload | Origin | GPU | Elapsed | Evidence |', '|---|---|---|---:|---|']
    for key, r in sorted(records.items()):
        gpu = r['hardware'].get('name', str(r['hardware']))
        elapsed = f"{r.get('elapsed_s', 0)/3600:.2f} h"
        lines.append(f"| {r['model']} / {r['configuration']} / {r['workload']} | {r['origin']} | {gpu} | {elapsed} | [record](records/{key}/record.json) |")
    lines += ['', '## Pending work and issues', '', f"RULER pending: {16-ruler_count}; other workload cells pending: {96-(len(records)-ruler_count)}.",
              'Workers cannot start other workloads until all 16 RULER cells validate. Imported scores may come from different hardware; use --fresh for a new hardware-specific campaign.', '']
    lines += ['| Configuration | RULER remaining | Other workloads remaining |', '|---|---:|---:|']
    for config in spec['configurations']:
        pending=[c for c in status['pending'] if c['configuration']==config]
        n=sum(c['workload']=='RULER' for c in pending)
        lines.append(f'| {config} | {n} | {len(pending)-n} |')
    lines.append('')
    for issue in issues:
        lines.append('- ' + issue['cell_id'] + ': ' + issue['kind'] + ' — ' + issue['message'].replace('\n', ' ').replace('|', '\\|'))
    hardware=registry.root/'hardware-check.json'
    if hardware.exists():
        checks=read(hardware)
        lines += ['', '## Most recent worker hardware check', '',
                  '| GPU | Assigned UUID | Free / total MiB | Driver | Busy |', '|---|---|---:|---|---|']
        for g in checks.get('gpus',[]):
            lines.append(f"| {g['name']} | {g['uuid']} | {g['free_mib']} / {g['total_mib']} | {g['driver']} | {g['busy']} |")
        lines += ['', 'This snapshot belongs to the most recent launcher; result records retain each source node and runtime.', '']
        for error in checks.get('errors',[]):
            lines.append('- '+error)
        for w,error in checks.get('workload_errors',{}).items():
            lines.append('- '+w+': '+error)
    lines += ['', '## Protocol', '', 'Model/dataset revisions, task settings, precision and runtime identities are recorded in each result. BFCL uses the pinned official v3 evaluator; the other workloads use lm-eval.',
              'No CPU offload, precision substitution, PPL, or undefined composite scores. Original screenshot scores are not imported.', '']
    atomic_text(registry.root / 'results.md', '\n'.join(lines))
    return status
