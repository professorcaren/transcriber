# Local static server with cross-origin isolation headers (multithreaded WASM).
# Run from the repo root: python3 tools/serve.py  -> http://127.0.0.1:8765/?models=local
import http.server, functools, sys
class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, '.mjs': 'text/javascript', '.wasm': 'application/wasm'}
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
http.server.ThreadingHTTPServer(('127.0.0.1', port), functools.partial(H, directory='.')).serve_forever()
