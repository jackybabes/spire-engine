#!/bin/sh
# Builds the engine for the browser (crates/sts-wasm) into web/sts.wasm.
set -e
cd "$(dirname "$0")/.."
cargo build --release --target wasm32-unknown-unknown -p sts-wasm
cp target/wasm32-unknown-unknown/release/sts_wasm.wasm web/sts.wasm
echo "built web/sts.wasm"
