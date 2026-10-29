"""Restore large, checksum-pinned assets omitted from the lightweight Git clone."""
import fcntl
import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
import urllib.error

from .common import PROJECT, read, safe_relative, sha


def ensure_assets(download=True):
    expected=read(PROJECT/'assets/manifest.json')['files']
    missing=lambda:[name for name in expected if not (PROJECT/'assets'/safe_relative(name)).is_file()]
    if not missing():return
    if not download:
        raise RuntimeError('Large release assets are absent from this Git clone. A normal bash run.sh invocation downloads them automatically; --check does not download data.')
    with (PROJECT/'.assets.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not missing():return
        spec=read(PROJECT/'config/release-assets.json')
        archive=PROJECT/'.asset-download.tar.gz.partial'
        offset=archive.stat().st_size if archive.exists() else 0
        print('Downloading frozen benchmark assets from the pinned GitHub release...',flush=True)
        if not offset or sha(archive)!=spec['sha256']:
            headers={'Range':f'bytes={offset}-'} if offset else {}
            req=urllib.request.Request(spec['url'],headers=headers)
            try:
                response=urllib.request.urlopen(req,timeout=60)
            except urllib.error.HTTPError as exc:
                if exc.code!=416:raise
                response=urllib.request.urlopen(spec['url'],timeout=60)
            with response:
                mode='ab' if offset and response.status==206 else 'wb'
                with archive.open(mode) as stream:shutil.copyfileobj(response,stream,1<<20)
        if sha(archive)!=spec['sha256']:
            archive.unlink(missing_ok=True)
            raise RuntimeError('Release archive checksum mismatch; rerun to download again')
        with tarfile.open(archive,'r:gz') as source:
            for member in source:
                parts=member.name.split('/')
                if parts[:2]!=['qwen-compression-benchmark','assets'] or not member.isfile():continue
                relative=safe_relative('/'.join(parts[2:]));name=str(relative)
                if name not in expected:continue
                target=PROJECT/'assets'/relative
                if target.is_file():continue
                target.parent.mkdir(parents=True,exist_ok=True)
                fd,temp=tempfile.mkstemp(prefix='.release-',dir=target.parent)
                try:
                    h=hashlib.sha256()
                    with os.fdopen(fd,'wb') as out,source.extractfile(member) as data:
                        for block in iter(lambda:data.read(1<<20),b''):
                            out.write(block);h.update(block)
                    if h.hexdigest()!=expected[name]:raise RuntimeError('Asset checksum mismatch: '+name)
                    os.replace(temp,target)
                finally:
                    if os.path.exists(temp):os.unlink(temp)
        if missing():raise RuntimeError('Release archive did not supply all required assets')
        archive.unlink()
        print('Frozen RULER banks and completed-result evidence restored.',flush=True)
