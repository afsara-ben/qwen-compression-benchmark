"""Build a transferable source/data archive; never include weights or environments."""
import argparse
from pathlib import Path
import sys
import tarfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qbench.common import PROJECT,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    roots=['run.sh','README.md','VALIDATION.md','THIRD_PARTY_NOTICES.md','pyproject.toml','.gitignore',
           'qbench','runtime','config','locks','patches','tests','tools','assets','vendor','validation',
           'results.md','results.json','results.csv']
    target=args.output.resolve();target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.partial')
    def include(member):
        parts=Path(member.name).parts
        if any(p in ('__pycache__','.pytest_cache') for p in parts) or member.name.endswith('.pyc'):
            return None
        if member.issym() or member.islnk() or member.name.endswith(('.safetensors','.pt','.bin')):
            raise ValueError('Refusing unexpected model/binary/link artifact: '+member.name)
        return member
    with tarfile.open(temporary,'w:gz',compresslevel=6) as archive:
        for name in roots:
            source=PROJECT/name
            if source.exists():
                archive.add(source,arcname='qwen-compression-benchmark/'+name,filter=include)
    temporary.replace(target)
    checksum=sha(target)
    target.with_name(target.name+'.sha256').write_text(checksum+'  '+target.name+'\n')
    print(f'{target}: {target.stat().st_size/1e6:.1f} MB; sha256 {checksum}')


if __name__=='__main__':main()
