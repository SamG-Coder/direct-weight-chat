"""Local static test server with explicit module MIME types on Windows."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, '.mjs': 'text/javascript', '.js': 'text/javascript'}

if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', 8770), partial(Handler, directory='build/huggingface-space')).serve_forever()
