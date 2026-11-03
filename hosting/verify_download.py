"""Check the public Space repository and its actual static serving endpoint."""
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
expected = json.loads((ROOT / 'publication/hosted-chat-package.json').read_text(encoding='utf-8'))
repo = 'SamGCoder/direct-weight-chat'
info = HfApi().space_info(repo, files_metadata=True)
dest = ROOT / 'build/hosted-chat-fresh-download'
snapshot_download(repo, repo_type='space', revision=info.sha, local_dir=dest,
                  allow_patterns=[e['name'] for e in expected['files']], max_workers=3)
sha = lambda data: hashlib.sha256(data).hexdigest()
for entry in expected['files']:
    data = (dest / entry['name']).read_bytes()
    assert len(data) == entry['bytes'] and sha(data) == entry['sha256'], entry['name']
served = []
for name in ['index.html', 'app.mjs', 'runtime.mjs', 'worker.mjs', 'config.json', 'weights.sparse.zip']:
    with urllib.request.urlopen('https://samgcoder-direct-weight-chat.static.hf.space/' + name) as response:
        data = response.read()
        kind = response.headers.get('Content-Type')
    entry = next(e for e in expected['files'] if e['name'] == name)
    normalized = data
    if name == 'index.html':
        # Hugging Face injects its Space variables and social preview metadata.
        # Remove only the two observed additions before comparing our full HTML.
        normalized = re.sub(rb'<head><script>window\.huggingface=\{variables:\{[^<]*\}\};</script>', b'<head>', normalized)
        normalized = re.sub(rb'<meta property="og:title"[^<]*(?:<meta (?:property="og:(?:description|image)"|name="twitter:(?:card|image)")[^<]*){4}</head>', b'</head>', normalized)
    assert sha(normalized) == entry['sha256'], name
    if name.endswith('.mjs'):
        assert 'javascript' in kind, (name, kind)
    served.append(dict(name=name, bytes=len(data), sha256=sha(data), content_type=kind,
                       matches_uploaded_source_after_hf_metadata=sha(normalized) == entry['sha256']))
report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
              repository='https://huggingface.co/spaces/' + repo, revision=info.sha,
              public=not info.private, sdk=info.sdk, stage=info.runtime.stage,
              all_published_assets_match=True, served_assets=served,
              hf_injects_space_variables_and_social_metadata_in_html=True,
              unchanged_weights_sha256=next(e['sha256'] for e in expected['files'] if e['name'] == 'weights.sparse.zip'))
(ROOT / 'publication/hosted-chat-download-verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(revision=info.sha, public=not info.private, stage=info.runtime.stage,
                     verified_repo_files=len(expected['files']), verified_served_assets=len(served))))
