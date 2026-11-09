// The engine (crates/sts-wasm, built into web/sts.wasm by web/build.sh) behind one function:
// call({cmd: ...}) returns the reply as an object, or throws its error. call.loadNet(bytes) hands
// it a release network's weights file (an ArrayBuffer) and returns the file's metadata.

/// The current engine (web/build_site.py points it at the hashed file it uploads). A network that
/// needs an older one names it in networks.json (`engine`).
export const ENGINE = "sts.wasm";

export async function loadEngine(url = ENGINE) {
  const { instance } = await WebAssembly.instantiateStreaming(fetch(url), {});
  const e = instance.exports;
  const enc = new TextEncoder();
  const dec = new TextDecoder();
  const reply = (n) => {
    const r = JSON.parse(dec.decode(new Uint8Array(e.memory.buffer, e.out_ptr(), n)));
    if (r && r.error) throw new Error(r.error);
    return r;
  };
  // Copies `bytes` into the engine's memory, runs `fn(ptr, len)` and frees them.
  const withBytes = (bytes, fn) => {
    const p = e.alloc(bytes.length);
    new Uint8Array(e.memory.buffer, p, bytes.length).set(bytes);
    const n = fn(p, bytes.length);
    e.dealloc(p, bytes.length);
    return reply(n);
  };
  function call(cmd) {
    return withBytes(enc.encode(JSON.stringify(cmd)), e.call);
  }
  call.loadNet = (buffer) => withBytes(new Uint8Array(buffer), e.load_net);
  return call;
}
