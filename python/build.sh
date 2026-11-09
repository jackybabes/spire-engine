#!/bin/sh
# Builds the `sts` Python module into this folder (no maturin needed): python3 -c "import sts".
set -e
cd "$(dirname "$0")/.."
cargo build --release -p sts-py
# Copy, then rename over the old module: a running Python process keeps the file it loaded.
cp target/release/libsts.dylib python/sts.abi3.so.tmp 2>/dev/null || cp target/release/libsts.so python/sts.abi3.so.tmp
mv python/sts.abi3.so.tmp python/sts.abi3.so
echo "built python/sts.abi3.so"
