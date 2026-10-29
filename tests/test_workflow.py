"""CPU-only integration tests using real saved summaries in isolated directories."""
import copy
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from qbench.bundles import export_bundle, import_bundle
from qbench.common import PROJECT, all_cells, fingerprint, manifest, read, sha, write
from qbench.hardware import GPU, parse_inventory, fits
from qbench.registry import Registry, validate_record
from qbench.report import build


def fixture(root, cell):
    """Synthetic receipt for scheduler testing; never used as a benchmark result."""
    source=PROJECT/'assets/completed/qwen3_30b--KV--RULER'
    dest=Path(root)/cell['id'];dest.mkdir(parents=True)
    for name in ['summary.json','audit.json','launch.json','samples.jsonl.gz']:
        shutil.copy2(source/name,dest/name)
    spec=manifest();m=spec['models'][cell['model']];c=spec['configurations'][cell['configuration']]
    r=read(source/'record.json');a=read(dest/'audit.json')['results'][0];l=read(dest/'launch.json')
    a['experts']=[dict(name=str(i),count=m['reap' if c['expert'] else 'base']['experts']) for i in range(m['layers'])]
    for group in a['cache_groups']:
        group['dtype']='torch.uint8' if c['kv']=='fp8' else 'torch.bfloat16'
    a['nvfp4_modules']=[]
    if c['weights']=='nvfp4':
        a['nvfp4_modules']=[dict(name=str(i),method='ModelOptNvFp4FusedMoE',use_a16=True,backend='MARLIN') for i in range(m['layers'])]
        a['marlin_canonical_order']=True
    l['benchmark_identity']=dict(model=cell['model'],configuration=cell['configuration'],
        checkpoint=m['reap' if c['expert'] else 'base'],tokenizer=m['base'])
    from qbench.runner import server_command,PROFILES
    l['command']=server_command('python','model','tokenizer',cell,'http://127.0.0.1:8000',PROFILES[0])
    write(dest/'audit.json',{'results':[a]});write(dest/'launch.json',l)
    r.update(cell_id=cell['id'],model=cell['model'],configuration=cell['configuration'],workload=cell['workload'],
             protocol_fingerprint=fingerprint(cell),hardware={'name':'SYNTHETIC TEST FIXTURE'},origin='unit-test fixture',
             files={f.name:sha(f) for f in dest.iterdir() if f.is_file()})
    write(dest/'record.json',r)
    return dest


class Workflow(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.registry=Registry(self.root/'state')
        self.rulers=[x for x in all_cells() if x['workload']=='RULER']

    def tearDown(self):
        self.temp.cleanup()

    def test_gate_at_fifteen_and_sixteen_and_filters(self):
        for cell in self.rulers[:15]:
            self.registry.publish(fixture(self.root/'fixtures',cell))
        self.assertEqual(self.registry.phase(),'RULER')
        self.assertEqual(self.registry.pending(workloads=['GSM8K']),[])
        gsm=next(x for x in all_cells() if x['workload']=='GSM8K')
        self.assertIsNone(self.registry.claim(gsm))
        self.registry.publish(fixture(self.root/'fixtures',self.rulers[15]))
        self.assertEqual(self.registry.phase(),'other')
        self.assertEqual(len(self.registry.pending()),96)
        claim=self.registry.claim(gsm);self.assertIsNotNone(claim);claim.release()

    def test_claim_race_and_stale_fencing(self):
        cell=self.rulers[0]
        with ThreadPoolExecutor(2) as pool:
            claims=list(pool.map(lambda _: Registry(self.registry.root).claim(cell),range(2)))
        winners=[x for x in claims if x]
        self.assertEqual(len(winners),1)
        first=winners[0]
        os.utime(first.directory/'owner.json',(time.time()-100,time.time()-100))
        replacement=self.registry.claim(cell,ttl=1)
        self.assertIsNotNone(replacement);self.assertFalse(first.owned())
        with self.assertRaises(RuntimeError):
            self.registry.publish(fixture(self.root/'fixtures',cell),first)
        first.release();self.assertTrue(replacement.owned());replacement.release()

    def test_checksums_protocol_and_pilots_rejected(self):
        dest=fixture(self.root/'fixtures',self.rulers[0])
        (dest/'summary.json').write_text('{}')
        with self.assertRaises(ValueError):validate_record(dest)
        r=read(dest/'record.json');r['protocol_fingerprint']='wrong';write(dest/'record.json',r)
        with self.assertRaises(ValueError):validate_record(dest)

    def test_portable_bundle_merge_idempotence(self):
        self.registry.publish(fixture(self.root/'fixtures',self.rulers[0]))
        bundle=export_bundle(self.registry,self.root/'peer.tar.gz')
        peer=Registry(self.root/'peer-state')
        self.assertEqual(import_bundle(peer,bundle),1)
        import_bundle(peer,bundle)
        self.assertEqual(len(peer.records()),1)
        self.assertEqual(peer.phase(),'RULER')

    def test_tar_traversal_rejected(self):
        bundle=self.root/'bad.tar.gz'
        with tarfile.open(bundle,'w:gz') as archive:
            info=tarfile.TarInfo('../outside');info.size=1;archive.addfile(info,io.BytesIO(b'x'))
        with self.assertRaises(ValueError):import_bundle(self.registry,bundle)
        self.assertFalse((self.root/'outside').exists())

    def test_report_partial_averages_and_weights(self):
        self.registry.publish(fixture(self.root/'fixtures',self.rulers[0]))
        report=build(self.registry)
        self.assertEqual(report['completed'],1)
        self.assertTrue(all(row['Avg.'] is None for row in report['rows']))
        self.assertIn('1/16',(self.registry.root/'results.md').read_text())

    def test_seed_inventory_validated(self):
        seeds=list((PROJECT/'assets/completed').iterdir())
        self.assertEqual(len(seeds),34)
        records=[validate_record(p) for p in seeds]
        self.assertEqual(sum(r['workload']=='RULER' for r in records),6)


class Hardware(unittest.TestCase):
    inventory='0, GPU-aaa, NVIDIA A100, 40960, 40000, 8.0, 580.126.20\n1, GPU-bbb, NVIDIA H100, 81920, 80000, 9.0, 580.126.20\n'

    def test_allocations_overrides_and_busy_devices(self):
        g=parse_inventory(self.inventory,'GPU-aaa, 42\n',visible='1',overrides='0')
        self.assertEqual([x.uuid for x in g],['GPU-bbb']);self.assertFalse(g[0].busy)
        self.assertTrue(parse_inventory(self.inventory,'GPU-aaa, 42\n')[0].busy)
        self.assertEqual(parse_inventory(self.inventory,visible=''),[])
        self.assertEqual([x.uuid for x in parse_inventory(self.inventory,slurm_visible='1')],['GPU-bbb'])

    def test_forty_gb_defers_bf16_but_fits_nvfp4(self):
        a100,h100=parse_inventory(self.inventory)
        baseline=next(c for c in all_cells() if c['configuration']=='Baseline' and c['workload']=='RULER')
        w=next(c for c in all_cells() if c['configuration']=='W' and c['workload']=='RULER')
        self.assertFalse(fits(a100,baseline));self.assertTrue(fits(h100,baseline));self.assertTrue(fits(a100,w))


if __name__=='__main__':unittest.main()
