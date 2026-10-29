import io
import json
from pathlib import Path
import tarfile
import tempfile

from .common import PROJECT, digest, manifest, now, read, safe_relative, sha
from .registry import validate_record


def import_seed(registry):
    release = read(PROJECT/'assets/manifest.json')
    existing=registry.records()
    for path in sorted((PROJECT/'assets/completed').iterdir()):
        if not path.is_dir():
            continue
        relative = str((path/'record.json').relative_to(PROJECT/'assets'))
        if sha(path/'record.json') != release['files'][relative]:
            raise ValueError('Bundled seed record changed: '+relative)
        if path.name in existing:
            continue
        registry.publish(path)


def verify_banks():
    index=read(PROJECT/'assets/ruler-bank-index.json')
    expected=manifest()['ruler_bank_digests']
    if set(index)!=set(expected):
        raise ValueError('RULER tokenizer bank coverage mismatch')
    for model,tasks in index.items():
        if len(tasks)!=13 or set(tasks)!=set(expected[model]):
            raise ValueError('Expected 13 RULER banks per tokenizer')
        for task,entry in tasks.items():
            if digest(entry['files'])!=expected[model][task]:
                raise ValueError(f'RULER bank identity mismatch: {model}/{task}')
            directory=PROJECT/'assets/ruler-banks'/safe_relative(entry['directory'])
            for filename,checksum in entry['files'].items():
                if sha(directory/safe_relative(filename))!=checksum:
                    raise ValueError(f'RULER bank checksum mismatch: {model}/{task}/{filename}')
    release=read(PROJECT/'assets/manifest.json')
    for name,checksum in release['files'].items():
        if name.startswith('nltk/') and sha(PROJECT/'assets'/safe_relative(name))!=checksum:
            raise ValueError('NLTK resource checksum mismatch: '+name)


def export_bundle(registry, destination):
    destination=Path(destination).resolve()
    destination.parent.mkdir(parents=True,exist_ok=True)
    records=registry.records()
    temporary=destination.with_suffix(destination.suffix+'.partial')
    with tarfile.open(temporary,'w:gz') as archive:
        data=json.dumps(dict(schema_version=1,created_at=now(),cells=sorted(records))).encode()
        info=tarfile.TarInfo('bundle.json'); info.size=len(data)
        archive.addfile(info,io.BytesIO(data))
        for cell in sorted(records):
            directory=registry.root/'records'/cell
            record=validate_record(directory)
            for filename in ['record.json',*record['files']]:
                archive.add(directory/safe_relative(filename),arcname=f'records/{cell}/{filename}',recursive=False)
    temporary.replace(destination)
    return destination


def import_bundle(registry, source):
    with tempfile.TemporaryDirectory(prefix='.import-',dir=registry.root) as tmp:
        root=Path(tmp)
        with tarfile.open(source,'r:gz') as archive:
            seen=set()
            for member in archive:
                relative=safe_relative(member.name)
                if member.name in seen or not member.isfile():
                    raise ValueError('Bundle contains duplicate or non-file entries')
                seen.add(member.name)
                if relative.parts[0] not in ('bundle.json','records'):
                    raise ValueError('Unexpected bundle entry')
                target=root/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as src,target.open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src,dst)
        bundle=read(root/'bundle.json')
        if bundle.get('schema_version')!=1:
            raise ValueError('Unsupported bundle schema')
        directories=[root/'records'/safe_relative(c) for c in bundle['cells']]
        # Validate the entire bundle before publishing any cell.
        for directory in directories:
            validate_record(directory)
        for directory in directories:
            registry.publish(directory)
    return len(directories)
