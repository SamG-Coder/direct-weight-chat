"""Download our published model and verify its bytes and standalone runtime."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime,timezone
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download',type=Path,required=True)
    p.add_argument('--python',required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    expected=json.loads((root/'publication/huggingface-package.json').read_text(encoding='utf-8'))
    repo='samfromlife/direct-weight-chat-v17a'
    info=HfApi().model_info(repo,files_metadata=True)
    assert not info.private
    if a.download.exists():raise SystemExit('Refusing to overwrite the fresh-download directory')
    snapshot_download(repo,revision=info.sha,local_dir=a.download,
                      allow_patterns=[x['name'] for x in expected['files']]+['SHA256SUMS.txt'],max_workers=3)
    files=[]
    for entry in expected['files']:
        f=a.download/entry['name'];actual=sha(f)
        assert actual==entry['sha256'] and f.stat().st_size==entry['bytes'],entry['name']
        files.append(dict(**entry,download_verified=True))
    checksums=(a.download/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines()
    assert {x.split('  ',1)[1]:x.split('  ',1)[0] for x in checksums}=={x['name']:x['sha256'] for x in files}
    with zipfile.ZipFile(a.download/'runtime.zip') as z:
        assert z.testzip() is None
        for name in z.namelist():assert (a.download/name).resolve().is_relative_to(a.download.resolve())
        z.extractall(a.download)
    program='''import json
from chat1gb_autoregressive.engine import Model
m=Model(directory='.',device='cpu')
rows=[m.generate(p) for p in ['A kelv can fly. What can a kelv do?','A kelv cannot fly. Can kelv fly?','149 - 370']]
print(json.dumps(rows))
'''
    run=subprocess.run([a.python,'-c',program],cwd=a.download,capture_output=True,text=True,encoding='utf-8',check=True)
    rows=json.loads(run.stdout)
    assert [x['text'] for x in rows]==['Kelv can fly.','No.','-221.']
    assert all(x['stopped_on_eos'] for x in rows)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),repository=f'https://huggingface.co/{repo}',
                revision=info.sha,public=True,files=files,checksum_file_matches=True,
                extracted_runtime_crc_verified=True,standalone_cpu_smoke_passed=True,
                turns=rows,model_weights_unchanged=sha(a.download/'weights.f32')==next(x['sha256'] for x in expected['files'] if x['name']=='weights.f32'))
    assert result['model_weights_unchanged']
    (root/'publication/huggingface-verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(repository=result['repository'],revision=info.sha,verified_files=len(files),cpu_smoke='3/3'),indent=2))


if __name__=='__main__':main()
