"""Prepare a self-contained final checkpoint upload without modifying weights."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if a.output.exists():raise SystemExit('Refusing to overwrite package')
    a.output.mkdir(parents=True)
    source=root/'build/v17a'
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    for name in ['weights.f32','config.json','manifest.json']:
        shutil.copyfile(source/name,a.output/name)
    shutil.copyfile(root/'publication/huggingface/README.md',a.output/'README.md')
    shutil.copyfile(root/'LICENSE',a.output/'LICENSE')
    with zipfile.ZipFile(a.output/'runtime.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted((root/'chat1gb_autoregressive').glob('*.py')):
            z.write(file,'chat1gb_autoregressive/'+file.name)
        for file in sorted((root/'versions/autoregressive_v17a').glob('*.py')):
            z.write(file,'versions/autoregressive_v17a/'+file.name)
        for name in ['config.json','manifest.json']:
            z.write(source/name,'versions/autoregressive_v17a/model/'+name)
        for name in ['LICENSE','requirements.txt','ARCHITECTURE.md','CAPABILITIES.md']:
            z.write(root/name,name)
    assert (a.output/'weights.f32').stat().st_size==1_000_000_000
    for name,expected in manifest['sha256'].items():assert sha(a.output/name)==expected
    with zipfile.ZipFile(a.output/'runtime.zip') as z:
        assert z.testzip() is None
        assert hashlib.sha256(z.read('chat1gb_autoregressive/engine.py')).hexdigest()=='f56d29cd23ba2b3018c4ba6a7ad6928dcb28c437e213929802aa4cba90329e10'
    entries=[dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(a.output.iterdir())]
    (a.output/'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['name']}\n" for r in entries),encoding='utf-8')
    (root/'publication/huggingface-package.json').write_text(json.dumps(dict(version='v17a',files=entries),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(entries,indent=2))


if __name__=='__main__':main()
