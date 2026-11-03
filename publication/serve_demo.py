"""Loopback presentation server; model inference runs in an offline subprocess.

The server transports user text and model output. It performs no reasoning,
answer assembly, retrieval or fallback. Captured replay is explicitly separate.
"""
import argparse
import json
import mimetypes
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python',required=True,help='Existing Python environment with PyTorch')
    p.add_argument('--model',default=str(ROOT/'build/v17a-compact'))
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    p.add_argument('--port',type=int,default=8768)
    a=p.parse_args()
    worker=subprocess.Popen([a.python,'-m','publication.model_worker','--model',a.model,'--device',a.device],
                            cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,
                            encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    ready=json.loads(worker.stdout.readline())
    lock=threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def respond(self,status,data,kind='application/json'):
            payload=data if isinstance(data,bytes) else json.dumps(data).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type',kind+'; charset=utf-8')
            self.send_header('Content-Length',str(len(payload)))
            self.send_header('Cache-Control','no-store')
            self.end_headers();self.wfile.write(payload)

        def do_GET(self):
            path=urlsplit(self.path).path
            if path=='/api/health':return self.respond(200,dict(**ready,version='v17a-compact' if ready.get('compact') else 'v17a'))
            if path=='/transcript.json':file=ROOT/'publication/demo-transcript-compact.json'
            else:
                base=ROOT/'publication/demo_ui'
                file=(base/('index.html' if path=='/' else path.lstrip('/'))).resolve()
                if not file.is_relative_to(base.resolve()):return self.respond(403,dict(error='Invalid path'))
            if not file.is_file():return self.respond(404,dict(error='Not found'))
            self.respond(200,file.read_bytes(),mimetypes.guess_type(file.name)[0] or 'application/octet-stream')

        def do_POST(self):
            if self.path not in ['/api/message','/api/reset']:return self.respond(404,dict(error='Not found'))
            if self.headers.get('Origin') not in [None,f'http://127.0.0.1:{a.port}',f'http://localhost:{a.port}']:
                return self.respond(403,dict(error='Local origin required'))
            size=int(self.headers.get('Content-Length','0'))
            if size>20000:return self.respond(413,dict(error='Message too long'))
            try:
                data=json.loads(self.rfile.read(size) or '{}')
                if self.path=='/api/reset':request=dict(action='reset')
                else:
                    prompt=data.get('prompt')
                    if not isinstance(prompt,str) or not prompt.strip():raise ValueError('Enter a message')
                    request=dict(prompt=prompt)
                with lock:
                    worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush()
                    response=json.loads(worker.stdout.readline())
                self.respond(200,response)
            except Exception as error:self.respond(400,dict(ok=False,error=str(error)))

    class LocalServer(ThreadingHTTPServer):
        allow_reuse_address=False

    server=LocalServer(('127.0.0.1',a.port),Handler)
    print(f'Demo ready at http://127.0.0.1:{a.port} | frozen v17a | {ready["device"]}',flush=True)
    try:server.serve_forever()
    finally:
        server.server_close();worker.terminate();worker.wait(timeout=10)


if __name__=='__main__':main()
