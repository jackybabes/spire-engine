#!/bin/sh
# Builds the engine if needed and serves the viewer (web/serve.py): this machine only, or the
# local network with --lan. http://localhost:8765/
cd "$(dirname "$0")"
[ -f sts.wasm ] || ./build.sh
exec python3 serve.py "$@"
