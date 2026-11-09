#!/usr/bin/env python3
"""Serves the browser viewer: web/ at /, and the saved runs in python/results/watch/ at /runs/
(with /runs/index.json listing them). Nothing else in the repo is reachable: reference/ and
harness/ hold the decompiled game and must never be served.

    python3 web/serve.py              # this machine only: http://localhost:8765/
    python3 web/serve.py --lan        # the local network too: http://<this machine>:8765/

A run by link: http://localhost:8765/?run=/runs/a20_win_L7WC.json
"""
import argparse
import http.server
import json
import os
import posixpath
import socket
import urllib.parse

WEB = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(WEB), "python", "results", "watch")


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".js": "text/javascript"}

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=WEB, **kw)

    def end_headers(self):
        # Revalidate every file, so an edited app.js or a rebuilt sts.wasm shows on the next load.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def translate_path(self, path):
        path = urllib.parse.unquote(urllib.parse.urlsplit(path).path)
        if path == "/runs" or path.startswith("/runs/"):
            rest = posixpath.normpath(path[len("/runs"):]).lstrip("/")
            full = os.path.realpath(os.path.join(RUNS, rest))
            # Stay inside the runs folder.
            return full if full == os.path.realpath(RUNS) or full.startswith(os.path.realpath(RUNS) + os.sep) else os.path.join(RUNS, "\0")
        return super().translate_path(path)

    def do_GET(self):
        if urllib.parse.urlsplit(self.path).path == "/runs/index.json":
            names = sorted(f for f in os.listdir(RUNS) if f.endswith(".json")) if os.path.isdir(RUNS) else []
            body = json.dumps(names).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--lan", action="store_true", help="listen on the local network too, not only this machine")
    a = p.parse_args()
    if not os.path.exists(os.path.join(WEB, "sts.wasm")):
        raise SystemExit("web/sts.wasm is missing: run web/build.sh")
    host = "0.0.0.0" if a.lan else "127.0.0.1"
    server = http.server.ThreadingHTTPServer((host, a.port), Handler)
    print(f"http://localhost:{a.port}/", flush=True)
    if a.lan:
        print(f"http://{socket.gethostname()}:{a.port}/ (on the local network)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
